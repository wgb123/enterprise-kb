"""答案生成层 — 调用本地 vLLM 生成最终答案。

基于检索到的上下文，通过 prompt 模板注入，让 LLM 生成
引用来源的、忠实于上下文的最终答案。
"""

import json
from typing import AsyncGenerator, Optional

import httpx

from enterprise_kb.config import settings
from enterprise_kb.interfaces.base import BaseGenerator, RetrievedChunk
from enterprise_kb.utils.logger import logger


DEFAULT_SYSTEM_PROMPT = """你是一个企业知识库助手。请根据提供的上下文信息回答用户问题。

规则：
1. 只使用上下文中的信息回答，不要编造事实
2. 如果上下文不足以回答问题，请说"根据现有知识库无法回答这个问题"
3. 在回答中引用来源，使用 [来源: 文档名称] 格式
4. 如果多个来源的信息有冲突，优先采信 Wiki 知识库的内容
5. 用清晰、结构化的语言组织答案
6. 如果上下文包含代码块，在答案中保持代码格式"""


class VLLMGenerator(BaseGenerator):
    """基于 vLLM（OpenAI 兼容 API）的答案生成器。

    通过调用本地 vLLM 的 chat/completions 接口生成答案。
    支持自定义系统提示词和温度参数。
    """

    def __init__(
        self,
        api_url: str = "",
        api_key: str = "",
        model_name: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> None:
        self.api_url = (api_url or settings.vllm_api_url).rstrip("/") + "/chat/completions"
        self.api_key = api_key or settings.vllm_api_key
        self.model_name = model_name or settings.vllm_model_name
        self.temperature = temperature or settings.vllm_temperature
        self.max_tokens = max_tokens or settings.vllm_max_tokens
        self.timeout = settings.vllm_timeout

    async def generate(
        self,
        query: str,
        context_chunks: list[RetrievedChunk],
        system_prompt: Optional[str] = None,
    ) -> str:
        """生成基于上下文的答案。

        Args:
            query: 用户原始查询。
            context_chunks: 检索并融合后的上下文块列表。
            system_prompt: 可选的系统提示词覆盖。

        Returns:
            生成的答案字符串。
        """
        context_text = self._format_context(context_chunks)
        user_prompt = self._build_user_prompt(query, context_text)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["choices"][0]["message"]["content"]
                logger.info(
                    "Generated answer (%d tokens) for: %s",
                    data.get("usage", {}).get("completion_tokens", 0),
                    query[:60],
                )
                return answer

        except httpx.TimeoutException:
            logger.warning("vLLM request timed out after %ds", self.timeout)
            return self._fallback_response(
                query,
                context_chunks,
                reason="vLLM 请求超时",
            )
        except httpx.HTTPStatusError as exc:
            logger.warning("vLLM HTTP error: %s", exc)
            return self._fallback_response(
                query,
                context_chunks,
                reason=f"vLLM 服务异常 ({exc.response.status_code})",
            )
        except Exception as exc:
            logger.exception("vLLM generation failed")
            return self._fallback_response(
                query,
                context_chunks,
                reason=f"生成失败: {exc}",
            )

    async def generate_stream(
        self,
        query: str,
        context_chunks: list[RetrievedChunk],
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """流式生成基于上下文的答案，逐 token 产出。

        Args:
            query: 用户原始查询。
            context_chunks: 检索并融合后的上下文块列表。
            system_prompt: 可选的系统提示词覆盖。

        Yields:
            每个生成 token 的内容字符串。
        """
        context_text = self._format_context(context_chunks)
        user_prompt = self._build_user_prompt(query, context_text)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", self.api_url, json=payload, headers=headers
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                token = (
                                    data.get("choices", [{}])[0]
                                    .get("delta", {})
                                    .get("content", "")
                                )
                                if token:
                                    yield token
                            except json.JSONDecodeError:
                                continue

        except httpx.TimeoutException:
            logger.warning("vLLM stream timed out after %ds", self.timeout)
            yield self._fallback_response(
                query,
                context_chunks,
                reason="vLLM 请求超时",
            )
        except httpx.HTTPStatusError as exc:
            logger.warning("vLLM stream HTTP error: %s", exc)
            yield self._fallback_response(
                query,
                context_chunks,
                reason=f"vLLM 服务异常 ({exc.response.status_code})",
            )
        except Exception as exc:
            logger.exception("vLLM stream generation failed")
            yield self._fallback_response(
                query,
                context_chunks,
                reason=f"流式生成失败: {exc}",
            )

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        """将检索到的块格式化为结构化的上下文文本。"""
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

    @staticmethod
    def _build_user_prompt(query: str, context: str) -> str:
        """构建用户侧 prompt。"""
        return (
            f"## 用户问题\n{query}\n\n"
            f"## 参考上下文\n{context}\n\n"
            "请根据上述上下文回答用户问题。如果上下文不足以给出完整答案，"
            "请说明缺少哪些信息。引用来源时使用 [来源: 文档名称] 格式。"
        )

    @staticmethod
    def _fallback_response(
        query: str,
        context_chunks: list[RetrievedChunk],
        reason: str = "",
    ) -> str:
        """当 vLLM 不可用时的降级回复。

        直接返回最相关的上下文块作为答案。
        """
        if not context_chunks:
            return f"（{reason}）当前无法生成答案，且没有检索到相关上下文。"

        top = context_chunks[0]
        source = f"[来源: {top.document_id}]"

        return (
            f"**（{reason}——采用降级模式，直接返回最相关的上下文）**\n\n"
            f"根据检索结果，以下是与「{query}」最相关的内容：\n\n"
            f">{top.content[:800]}\n\n"
            f"{source}"
        )
