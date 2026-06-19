"""LangChain 版 FastAPI 路由。

对照手写版：api/routes.py

新增端点（前缀 /api/v1/langchain）：
  POST /api/v1/langchain/query        → LcGenerator.generate()
  POST /api/v1/langchain/query/batch  → LcGenerator.batch_generate()
  GET  /api/v1/langchain/query/stream → LcGenerator.stream_generate()
  POST /api/v1/langchain/agent        → LcAgent.run()

使用方式：
  curl http://localhost:8000/api/v1/langchain/query \
    -H "Content-Type: application/json" \
    -d '{"query": "Qdrant怎么配置？"}'
"""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from enterprise_kb.api.dependencies import (
    get_wiki_navigator,
    get_hybrid_retriever,
    get_fusion_service,
)
from enterprise_kb.core.wiki_navigator import WikiNavigator
from enterprise_kb.core.retriever import HybridRetriever
from enterprise_kb.core.fusion import ContextFusion
from enterprise_kb.interfaces.base import RetrievedChunk
from enterprise_kb.langchain.lc_generator import LcGenerator
from enterprise_kb.langchain.lc_agent import LcAgent

router = APIRouter(prefix="/api/v1/langchain", tags=["LangChain"])


# ── 请求/响应模型 ────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict] = []


class AgentRequest(BaseModel):
    query: str


class AgentResponse(BaseModel):
    answer: str
    steps: int = 0
    trace: list[dict] = []


# ── 单例 ──────────────────────────────────────────────

_generator: LcGenerator | None = None
_agent: LcAgent | None = None


def get_lc_generator() -> LcGenerator:
    global _generator
    if _generator is None:
        _generator = LcGenerator()
    return _generator


def get_lc_agent() -> LcAgent:
    global _agent
    if _agent is None:
        _agent = LcAgent()
    return _agent


# ── 路由 ──────────────────────────────────────────────


@router.post("/query", response_model=QueryResponse)
async def langchain_query(
    req: QueryRequest,
    wiki_nav: WikiNavigator = Depends(get_wiki_navigator),
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    fusion: ContextFusion = Depends(get_fusion_service),
    generator: LcGenerator = Depends(get_lc_generator),
):
    """LangChain 版 RAG 问答（非流式）。

    和手写版 /api/v1/query 功能对等，但 LLM 生成走 ChatOpenAI。
    """
    # 1. 检索 Wiki
    wiki_results = await wiki_nav.query(req.query)
    wiki_chunks = [
        RetrievedChunk(content=p.content, source="wiki", document_id=p.path)
        for p, _ in (wiki_results or [])
    ]

    # 2. 检索 HybridRAG
    rag_chunks = await retriever.retrieve(req.query)

    # 3. 融合
    all_chunks = wiki_chunks + rag_chunks
    fused = fusion.fuse(all_chunks, max_chunks=req.top_k)

    # 4. 生成（LangChain 版）
    answer = await generator.generate(req.query, fused)

    return QueryResponse(
        answer=answer,
        sources=[
            {"id": c.document_id, "source": c.source}
            for c in fused[:3]
        ],
    )


@router.post("/query/batch", response_model=list[QueryResponse])
async def langchain_batch_query(
    reqs: list[QueryRequest],
    generator: LcGenerator = Depends(get_lc_generator),
):
    """LangChain 版批量 RAG 问答。

    演示 Runnable 接口的 batch 调用。
    实际使用 need 先检索好所有上下文中饭后一次性生成。
    """
    # 简化版——不检索，只用于展示 batch 能力
    responses = await generator.batch_generate([
        (r.query, [])
        for r in reqs
    ])
    return [QueryResponse(answer=a) for a in responses]


@router.get("/query/stream")
async def langchain_stream_query(
    q: str,
    generator: LcGenerator = Depends(get_lc_generator),
):
    """LangChain 版流式 RAG 问答。

    演示 Runnable 接口的 stream 调用。
    逐 token 推送 SSE 事件。
    """
    async def event_stream():
        async for token in generator.stream_generate(q, []):
            if token:
                yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )


@router.post("/agent", response_model=AgentResponse)
async def langchain_agent(
    req: AgentRequest,
    agent: LcAgent = Depends(get_lc_agent),
):
    """LangChain 版 Agent 问答。

    和手写版 /api/v1/agent/chat 功能对等。
    一次调用 = 完整 Think→Act→Observe 循环。
    """
    result = await agent.run(req.query)
    return AgentResponse(
        answer=result["answer"],
        steps=result["steps"],
        trace=result["trace"],
    )
