"""FastAPI 路由定义 — 知识库系统 RESTful API。

端点一览：
- ``POST /api/v1/ingest`` — 文档入库
- ``POST /api/v1/query`` — 查询知识库
- ``GET  /api/v1/wiki`` — Wiki 页面导航/搜索
- ``GET  /api/v1/wiki/{path:path}`` — 获取特定 Wiki 页面
- ``GET  /api/v1/health`` — 健康检查
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from enterprise_kb.agent import AgentLoop, MemoryManager
from enterprise_kb.api.dependencies import (
    get_context_fusion,
    get_generator,
    get_hybrid_retriever,
    get_router,
    get_wiki_navigator,
)
from enterprise_kb.api.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AgentStep,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    WikiPageResponse,
    WikiSearchResponse,
)
from enterprise_kb.core.chunker import Chunker
from enterprise_kb.core.document_processor import Document, DocumentProcessor
from enterprise_kb.core.fusion import ContextFusion
from enterprise_kb.core.generator import VLLMGenerator
from enterprise_kb.core.retriever import HybridRetriever
from enterprise_kb.core.router import SmartRouter
from enterprise_kb.core.wiki_navigator import WikiNavigator
from enterprise_kb.utils.logger import logger

router = APIRouter(prefix="/api/v1", tags=["Knowledge Base"])


# ─── 健康检查 ───


@router.get("/health", response_model=HealthResponse)
async def health_check() -> dict[str, Any]:
    """系统健康检查端点。"""
    return {
        "status": "ok",
        "version": "0.1.0",
        "service": "Enterprise KB",
    }


# ─── 文档入库 ───


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    request: IngestRequest,
    wiki_nav: WikiNavigator = Depends(get_wiki_navigator),
    hybrid_retriever: HybridRetriever = Depends(get_hybrid_retriever),
) -> dict[str, Any]:
    """文档入库：解析 → 分块 → 嵌入 → 存储。

    支持 PDF / Word / Markdown 文件。
    文件路径通过 ``file_paths`` 参数传递。
    """
    processor = DocumentProcessor()
    chunker = Chunker()
    total_chunks = 0
    total_docs = 0
    errors: list[str] = []

    for file_path in request.file_paths:
        try:
            doc = await processor.process_file(file_path)
            file_type = doc.file_type

            # 分块
            chunks = await chunker.chunk(
                text=doc.content,
                metadata=doc.metadata,
                file_type=file_type,
            )

            # 注入 HybridRAG
            ingested = await hybrid_retriever.ingest(chunks)
            total_chunks += ingested
            total_docs += 1

            logger.info("Ingested %s → %d chunks", file_path, ingested)

        except Exception as exc:
            err_msg = f"{file_path}: {exc}"
            errors.append(err_msg)
            logger.error("Failed to ingest %s: %s", file_path, exc)

    await processor.close()

    return {
        "status": "partial" if errors else "ok",
        "documents_processed": total_docs,
        "chunks_ingested": total_chunks,
        "errors": errors,
    }


# ─── 知识库查询 ───


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    router_service: SmartRouter = Depends(get_router),
    wiki_nav: WikiNavigator = Depends(get_wiki_navigator),
    hybrid_retriever: HybridRetriever = Depends(get_hybrid_retriever),
    fusion: ContextFusion = Depends(get_context_fusion),
    generator: VLLMGenerator = Depends(get_generator),
) -> dict[str, Any]:
    """查询知识库系统。

    智能路由自动判断查询类型，分发到 Wiki / HybridRAG / 两者并行。
    最终结果通过 vLLM 生成回答。
    """
    query = request.query
    top_k = request.top_k
    use_generator = request.use_generator

    # 1. 意图分类
    intent = router_service.classify(query)
    logger.info("Query intent: %s — %s", intent.value, query[:80])

    # 2. 分发引擎
    results = []

    if router_service.should_use_wiki(query):
        logger.info("Routing to Wiki engine: %s", query[:60])
        wiki_result = await wiki_nav.query(query, top_k=top_k)
        results.append(wiki_result)

    if router_service.should_use_hybrid_rag(query):
        logger.info("Routing to HybridRAG engine: %s", query[:60])
        rag_result = await hybrid_retriever.retrieve(query, top_k=top_k)
        results.append(rag_result)

    # 3. 融合
    fused_chunks = fusion.fuse(results, max_chunks=15)

    # 4. 生成答案
    if use_generator and fused_chunks:
        answer = await generator.generate(
            query=query,
            context_chunks=fused_chunks,
        )
    elif fused_chunks:
        answer = fused_chunks[0].content[:1000]
    else:
        answer = "未检索到与查询相关的信息。"

    sources = [
        {
            "content": c.content[:300],
            "source": c.source,
            "document_id": c.document_id,
            "score": round(c.score, 4),
            "metadata": c.metadata,
        }
        for c in fused_chunks[:5]
    ]

    return {
        "query": query,
        "answer": answer,
        "intent": intent.value,
        "sources": sources,
        "total_chunks": len(fused_chunks),
    }


# ─── Wiki 导航 ───


@router.get("/wiki", response_model=WikiSearchResponse)
async def search_wiki(
    q: str = "",
    tag: str = "",
    wiki_nav: WikiNavigator = Depends(get_wiki_navigator),
) -> dict[str, Any]:
    """搜索 Wiki 知识库页面。

    参数：
    - ``q`` — 全文搜索关键词（可选）
    - ``tag`` — 按标签过滤（可选）
    """
    if tag:
        pages = await wiki_nav.get_by_tag(tag)
    elif q:
        pages = await wiki_nav.search(q)
    else:
        # 返回所有页面
        pages = []
        if wiki_nav._loaded or True:
            for rel in list(wiki_nav._pages.keys()):
                page = await wiki_nav.get_page(rel)
                if page:
                    pages.append(page)

    return {
        "total": len(pages),
        "pages": [
            {
                "title": p.title,
                "category": p.category,
                "tags": p.tags,
                "path": str(p.path.relative_to(wiki_nav._root)) if p.path else "",
            }
            for p in pages
        ],
    }


@router.get("/wiki/{path:path}", response_model=WikiPageResponse)
async def get_wiki_page(
    path: str,
    wiki_nav: WikiNavigator = Depends(get_wiki_navigator),
) -> dict[str, Any]:
    """获取指定 Wiki 页面的完整内容。

    ``path`` 是相对于 wiki/ 目录的路径，例如 ``categories/architecture.md``。
    """
    # 自动补全 .md 后缀
    if not path.endswith(".md"):
        path = path + ".md"

    page = await wiki_nav.get_page(path)
    if not page:
        raise HTTPException(status_code=404, detail=f"Wiki page not found: {path}")

    return {
        "title": page.title,
        "category": page.category,
        "tags": page.tags,
        "content": page.content,
        "toc": page.html_toc,
        "path": path,
    }


# ─── Agent 端点 ───

_memory_mgr = MemoryManager()


@router.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(
    req: AgentChatRequest,
    wiki_nav: WikiNavigator = Depends(get_wiki_navigator),
    hybrid: HybridRetriever = Depends(get_hybrid_retriever),
) -> dict[str, Any]:
    """AI Agent 对话端点。

    支持 Function Calling 工具调用、对话记忆、多步推理。
    工具清单：wiki_search / rag_search / calculator / get_time
    """
    memory = _memory_mgr.get_or_create(req.session_id)

    # 每次请求创建新的 AgentLoop（确保工具状态刷新）
    loop = AgentLoop(max_steps=req.max_steps)

    result = await loop.run(
        query=req.query,
        memory=memory,
    )

    return {
        "answer": result["answer"],
        "tool_calls": result["tool_calls"],
        "steps": result["steps"],
        "session_id": req.session_id,
        "trace": [
            AgentStep(**t) for t in result["trace"]
        ],
    }
