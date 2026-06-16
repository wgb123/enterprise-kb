"""HybridRAG retriever — 向量检索 + BM25关键词检索 + RRF融合。

实现了 B 类记忆层的完整检索管道。
"""

from typing import Any, Optional

from qdrant_client.models import PointStruct

from enterprise_kb.config import settings
from enterprise_kb.core.embedder import BGEM3Embedder
from enterprise_kb.core.reranker import BGEReranker
from enterprise_kb.interfaces.base import BaseRetriever, QueryResult, RetrievedChunk
from enterprise_kb.storage.bm25_index import BM25Index
from enterprise_kb.storage.qdrant_store import QdrantStore
from enterprise_kb.utils.logger import logger


class HybridRetriever(BaseRetriever):
    """HybridRAG 检索器。

    结合向量检索（Qdrant）和关键词检索（BM25），通过 RRF
    （Reciprocal Rank Fusion）融合结果，最后经 BGE-Reranker 重排序。
    """

    def __init__(
        self,
        embedder: Optional[BGEM3Embedder] = None,
        vector_store: Optional[QdrantStore] = None,
        bm25_index: Optional[BM25Index] = None,
        reranker: Optional[BGEReranker] = None,
    ) -> None:
        self.embedder = embedder or BGEM3Embedder()
        self.vector_store = vector_store or QdrantStore()
        self.bm25_index = bm25_index or BM25Index()
        self.reranker = reranker or BGEReranker()
        self._ingested = False

    # ── 文档入库 ──

    async def ingest(
        self,
        chunks: list[dict[str, Any]],
    ) -> int:
        """将分块后的文档注入向量存储和 BM25 索引。

        Args:
            chunks: 文档块列表，每项包含 ``text``、``metadata``、``chunk_id``。

        Returns:
            入库的文档块数量。
        """
        if not chunks:
            return 0

        # 1. 确保 Qdrant 集合存在
        await self.vector_store.ensure_collection()

        # 2. 生成嵌入向量
        texts = [c["text"] for c in chunks]
        logger.info("Generating embeddings for %d chunks ...", len(texts))
        vectors = await self.embedder.embed(texts)

        # 3. 构建 Qdrant points
        points: list[PointStruct] = []
        for i, chunk in enumerate(chunks):
            point_id = abs(hash(chunk["chunk_id"])) % (2**63)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vectors[i],
                    payload={
                        "text": chunk["text"],
                        "chunk_id": chunk["chunk_id"],
                        **chunk.get("metadata", {}),
                    },
                )
            )

        # 4. 写入 Qdrant
        inserted = await self.vector_store.upsert(points)

        # 5. 重建 BM25 索引（增量）
        self.bm25_index.build(chunks)

        self._ingested = True
        logger.info("Ingested %d chunks into HybridRAG storage", inserted)
        return inserted

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> QueryResult:
        """执行 HybridRAG 检索。

        流程：
        1. 向量检索 → N 条
        2. BM25 检索 → N 条
        3. RRF 融合
        4. BGE-Reranker 重排序

        Args:
            query: 用户查询。
            top_k: 最终返回的结果数，默认使用配置值。

        Returns:
            :class:`QueryResult` 包含检索到的文档块。
        """
        top_k = top_k or settings.retrieval_final_top_k
        vector_top_k = settings.retrieval_vector_top_k
        bm25_top_k = settings.retrieval_bm25_top_k

        # 1. 向量检索
        query_vector = (await self.embedder.embed([query]))[0]
        vector_results = await self.vector_store.search(
            query_vector=query_vector,
            top_k=vector_top_k,
        )

        # 2. BM25 检索
        bm25_results = self.bm25_index.search(query=query, top_k=bm25_top_k)

        # 3. RRF 融合
        fused = self._rrf_fuse(
            vector_results=vector_results,
            bm25_results=bm25_results,
            k=settings.retrieval_rrf_k,
        )

        if not fused:
            logger.warning("HybridRAG returned no results for query: %s", query)
            return QueryResult(query=query, chunks=[], engine="hybrid_rag")

        # 4. 重排序
        reranked = await self.reranker.rerank(
            query=query,
            candidates=fused,
            top_k=top_k * 2,  # 多留一些给重排序裁剪
        )

        # 5. 转为统一格式
        chunks = []
        for doc in reranked[:top_k]:
            chunks.append(
                RetrievedChunk(
                    content=doc.get("text") or doc.get("content", ""),
                    source="hybrid_rag",
                    document_id=str(doc.get("chunk_id", "")),
                    chunk_id=str(doc.get("chunk_id", "")),
                    score=doc.get("rerank_score", doc.get("score", 0.0)),
                    metadata=doc.get("metadata", {}),
                )
            )

        return QueryResult(query=query, chunks=chunks, engine="hybrid_rag")

    # ── RRF 融合 ──

    @staticmethod
    def _rrf_fuse(
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion 融合两种检索结果。

        基于文档 ID（chunk_id）合并，相同文档的 RRF 分数相加。
        """
        rrf_scores: dict[str, dict] = {}

        # 向量检索排名
        for rank, doc in enumerate(vector_results, start=1):
            doc_id = str(doc.get("payload", {}).get("chunk_id", doc.get("id", rank)))
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    "text": doc.get("payload", {}).get("text", ""),
                    "chunk_id": doc_id,
                    "metadata": doc.get("payload", {}),
                    "vector_score": doc.get("score", 0.0),
                    "bm25_score": 0.0,
                    "rrf_score": 0.0,
                }
            rrf_scores[doc_id]["rrf_score"] += 1.0 / (k + rank)

        # BM25 检索排名
        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = str(doc.get("chunk_id", f"bm25_{rank}"))
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    "text": doc.get("text", ""),
                    "chunk_id": doc_id,
                    "metadata": doc.get("metadata", {}),
                    "vector_score": 0.0,
                    "bm25_score": doc.get("score", 0.0),
                    "rrf_score": 0.0,
                }
            rrf_scores[doc_id]["bm25_score"] = doc.get("score", 0.0)
            rrf_scores[doc_id]["rrf_score"] += 1.0 / (k + rank)

        # 按 RRF 分数降序排列
        sorted_docs = sorted(
            rrf_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        # 将 rrf_score 设为文档的主 score
        for doc in sorted_docs:
            doc["score"] = doc["rrf_score"]

        return sorted_docs
