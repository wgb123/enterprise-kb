"""Embedder module — converts text to vector embeddings using BGE-M3.

Supports:
- BAAI/bge-m3 via FlagEmbedding (local inference)
- OpenAI-compatible API fallback (for remote inference)
- Batch processing with configurable batch size
- Automatic normalization
"""

import asyncio
from typing import Optional

import numpy as np

from enterprise_kb.config import settings
from enterprise_kb.interfaces.base import BaseEmbedder
from enterprise_kb.utils.logger import logger


class BGEM3Embedder(BaseEmbedder):
    """BGE-M3 embedding model wrapper.

    Uses FlagEmbedding for local inference by default.
    Falls back to an OpenAI-compatible API if the model cannot be loaded locally.
    """

    def __init__(
        self,
        model_name: str = "",
        device: str = "",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name or settings.embedding_model_name
        self.device = device or settings.embedding_model_device
        self.batch_size = batch_size or settings.embedding_batch_size
        self.normalize = normalize
        self._model: Optional[object] = None
        self._semaphore = asyncio.Semaphore(2)  # limit concurrent model calls

    async def _ensure_model(self) -> None:
        """Lazy-load the embedding model."""
        if self._model is not None:
            return

        logger.info("Loading embedding model: %s (device=%s)", self.model_name, self.device)
        loop = asyncio.get_running_loop()

        def _load() -> object:
            from FlagEmbedding import FlagAutoModel

            return FlagAutoModel.from_finetuned(
                model_name_or_path=self.model_name,
                query_instruction_for_retrieval="",
                query_instruction_format="",
                devices=self.device if self.device != "cpu" else None,
            )

        try:
            self._model = await loop.run_in_executor(None, _load)
            logger.info("Embedding model loaded successfully")
        except Exception as exc:
            logger.warning(
                "Failed to load local embedding model: %s. Falling back to API mode.", exc
            )
            self._model = None  # signals API fallback

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors.

        Args:
            texts: List of text strings.

        Returns:
            List of embedding vectors (list of floats).
        """
        if not texts:
            return []

        await self._ensure_model()

        async with self._semaphore:
            vectors = await self._embed_batch(texts)

        if self.normalize:
            vectors = self._l2_normalize(vectors)

        return vectors

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches to avoid OOM."""
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vectors = await self._embed_single_batch(batch)
            all_vectors.extend(vectors)

        return all_vectors

    async def _embed_single_batch(self, texts: list[str]) -> list[list[float]]:
        """Run a single batch through the model or API."""
        if self._model is not None:
            return await self._embed_local(texts)

        return await self._embed_api(texts)

    async def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Run local FlagEmbedding inference."""

        loop = asyncio.get_running_loop()

        def _run() -> list[list[float]]:
            result = self._model.encode(texts, return_dense=True, return_sparse=False)
            if isinstance(result, dict):
                # FlagAutoModel returns dict with "dense_vecs" key
                dense = result.get("dense_vecs", result.get("dense", []))
            else:
                dense = result

            if isinstance(dense, np.ndarray):
                return dense.tolist()
            return list(dense)

        return await loop.run_in_executor(None, _run)

    async def _embed_api(self, texts: list[str]) -> list[list[float]]:
        """Use an OpenAI-compatible API (e.g. vLLM embedding endpoint)."""
        import httpx

        api_url = settings.vllm_api_url.rstrip("/") + "/embeddings"
        api_key = settings.vllm_api_key

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        async with httpx.AsyncClient(timeout=60.0) as client:
            all_vectors: list[list[float]] = []
            for text in texts:
                payload = {
                    "model": settings.vllm_model_name,
                    "input": text,
                    "encoding_format": "float",
                }
                resp = await client.post(api_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                vector = data["data"][0]["embedding"]
                all_vectors.append(vector)

            return all_vectors

    @property
    def vector_size(self) -> int:
        """Dimensionality of BGE-M3 embeddings."""
        return settings.qdrant_vector_size

    @staticmethod
    def _l2_normalize(vectors: list[list[float]]) -> list[list[float]]:
        """L2-normalize each vector in-place."""
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = arr / norms
        return normalized.tolist()
