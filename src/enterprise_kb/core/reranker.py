"""BGE-Reranker-v2-m3 重排序模块。

从初步检索的结果中重新排序，提升 top-K 结果的准确率。
"""

import asyncio
from typing import Any, Optional

from enterprise_kb.config import settings
from enterprise_kb.utils.logger import logger


class BGEReranker:
    """BGE-Reranker-v2-m3 重排序器。

    对查询和候选文档对的交叉编码器得分进行排序。
    支持本地 FlagEmbedding 推理和 API 调用两种模式。
    """

    def __init__(
        self,
        model_name: str = "",
        device: str = "",
        top_k: int = 10,
    ) -> None:
        self.model_name = model_name or settings.reranker_model_name
        self.device = device or settings.reranker_model_device
        self.top_k = top_k or settings.reranker_top_k
        self._model: Optional[object] = None
        self._semaphore = asyncio.Semaphore(1)

    async def _ensure_model(self) -> None:
        """惰性加载重排序模型。"""
        if self._model is not None:
            return

        logger.info("Loading reranker model: %s (device=%s)", self.model_name, self.device)
        loop = asyncio.get_running_loop()

        def _load() -> object:
            from FlagEmbedding import FlagReranker

            return FlagReranker(
                model_name_or_path=self.model_name,
                use_fp16=self.device == "cuda",
                devices=self.device if self.device != "cpu" else None,
            )

        try:
            self._model = await loop.run_in_executor(None, _load)
            logger.info("Reranker model loaded successfully")
        except Exception as exc:
            logger.warning(
                "Failed to load local reranker: %s. Falling back to API mode.", exc
            )
            self._model = None

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """对候选文档进行重排序。

        Args:
            query: 原始查询字符串。
            candidates: 候选文档列表，每个元素须包含 ``text`` 或 ``content`` 键。
            top_k: 返回的最大结果数，默认为配置值。

        Returns:
            按相关性降序排列的文档列表，每项新增 ``rerank_score`` 键。
        """
        if not candidates:
            return []

        top_k = top_k or self.top_k
        await self._ensure_model()

        # 提取文本
        texts = []
        for c in candidates:
            text = c.get("text") or c.get("content") or ""
            texts.append(text)

        scores = await self._compute_scores(query, texts)

        # 将分数附加到原文档上
        results = []
        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = round(float(score), 6)
            results.append(doc)

        # 按重排序分数降序排列
        results.sort(key=lambda x: x["rerank_score"], reverse=True)

        return results[:top_k]

    async def _compute_scores(
        self,
        query: str,
        texts: list[str],
    ) -> list[float]:
        """计算查询与每个文本的相关性分数。"""
        if self._model is not None:
            return await self._compute_local(query, texts)
        return await self._compute_api(query, texts)

    async def _compute_local(
        self,
        query: str,
        texts: list[str],
    ) -> list[float]:
        """使用本地 CrossEncoder 模型计算分数。"""
        loop = asyncio.get_running_loop()

        pairs = [[query, text] for text in texts]

        def _run() -> list[float]:
            return self._model.compute_score(pairs, normalize=True)  # type: ignore[union-attr]

        return await loop.run_in_executor(None, _run)

    async def _compute_api(
        self,
        query: str,
        texts: list[str],
    ) -> list[float]:
        """通过 HTTP API 计算重排序分数。"""
        import httpx

        api_url = settings.vllm_api_url.rstrip("/") + "/rerank"
        api_key = settings.vllm_api_key
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": texts,
            }
            resp = await client.post(api_url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # OpenAI-compatible rerank response format
            results = data.get("results", [])
            indexed = {r["index"]: r["relevance_score"] for r in results}
            return [indexed.get(i, 0.0) for i in range(len(texts))]
