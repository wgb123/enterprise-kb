"""LangChain 版生成器 — ChatOpenAI 封装。

对照手写版：core/generator.py → VLLMGenerator
"""

from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from enterprise_kb.config import settings
from enterprise_kb.interfaces.base import RetrievedChunk


DEFAULT_SYSTEM_PROMPT = """你是一个企业知识库助手。请根据提供的上下文信息回答用户问题。

规则：
1. 只使用上下文中的信息回答，不要编造事实
2. 如果上下文不足以回答问题，请说"根据现有知识库无法回答这个问题"
3. 在回答中引用来源，使用 [来源: 文档名称] 格式
4. 如果多个来源的信息有冲突，优先采信 Wiki 知识库的内容
5. 用清晰、结构化的语言组织答案
6. 如果上下文包含代码块，在答案中保持代码格式"""


PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("human", "## 用户问题\n{query}\n\n## 参考上下文\n{context}\n\n请根据上述上下文回答用户问题。"
              "如果上下文不足以给出完整答案，请说明缺少哪些信息。"
              "引用来源时使用 [来源: 文档名称] 格式。"),
])


class LcGenerator:
    """LangChain 版答案生成器。

    使用 ChatOpenAI（兼容 DeepSeek / vLLM 等任何 OpenAI 兼容 API）。
    支持 invoke、batch、stream 三种调用方式（Runnable 接口）。
    """

    def __init__(
        self,
        api_url: str = "",
        api_key: str = "",
        model_name: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        self.api_url = (api_url or settings.vllm_api_url).rstrip("/")
        self.api_key = api_key or settings.vllm_api_key
        self.model_name = model_name or settings.vllm_model_name
        self.temperature = temperature or settings.vllm_temperature
        self.max_tokens = max_tokens or settings.vllm_max_tokens

        # ChatOpenAI 支持任意 OpenAI 兼容 API
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            base_url=self.api_url,
            api_key=self.api_key,
        )

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        """将检索块格式化为结构化的上下文文本。（同手写版）"""
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            source_label = chunk.source.upper()
            doc_id = chunk.document_id.split("/")[-1] if chunk.document_id else "unknown"
            title = chunk.metadata.get("title", doc_id)
            parts.append(
                f"[上下文 {i}] (来源: {source_label}, 文档: {title})\n"
                f"{chunk.content}\n"
            )
        return "\n---\n".join(parts)

    # ── Runnable 接口三件套 ──────────────────────────

    async def generate(
        self,
        query: str,
        context_chunks: list[RetrievedChunk],
        system_prompt: Optional[str] = None,
    ) -> str:
        """生成基于上下文的答案。（同手写版 generator.generate）"""
        context_text = self._format_context(context_chunks)

        messages = PROMPT_TEMPLATE.format_messages(
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
            query=query,
            context=context_text,
        )

        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as exc:
            return self._fallback_response(query, context_chunks, reason=str(exc))

    async def batch_generate(
        self,
        queries: list[tuple[str, list[RetrievedChunk]]],
        system_prompt: Optional[str] = None,
    ) -> list[str]:
        """批量生成答案。"""
        messages_batch = []
        for query, chunks in queries:
            context_text = self._format_context(chunks)
            messages_batch.append(PROMPT_TEMPLATE.format_messages(
                system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
                query=query,
                context=context_text,
            ))
        responses = await self.llm.abatch(messages_batch)
        return [r.content for r in responses]

    async def stream_generate(
        self,
        query: str,
        context_chunks: list[RetrievedChunk],
        system_prompt: Optional[str] = None,
    ):
        """流式生成答案。（逐 token 推送）"""
        context_text = self._format_context(context_chunks)
        messages = PROMPT_TEMPLATE.format_messages(
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
            query=query,
            context=context_text,
        )

        async for chunk in self.llm.astream(messages):
            yield chunk.content

    @staticmethod
    def _fallback_response(
        query: str,
        context_chunks: list[RetrievedChunk],
        reason: str = "",
    ) -> str:
        """降级回复。（同手写版）"""
        if not context_chunks:
            return f"（{reason}）当前无法生成答案，且没有检索到相关上下文。"
        top = context_chunks[0]
        source = f"[来源: {top.document_id}]"
        return (
            f"**（{reason}——采用降级模式，直接返回最相关的上下文）**\n\n"
            f"> {top.content[:800]}\n\n{source}"
        )
