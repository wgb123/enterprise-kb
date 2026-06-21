"""Qdrant vector store — async client wrapper.

Handles collection management, point upsertion, and vector search.
"""

from typing import Any, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from enterprise_kb.config import settings
from enterprise_kb.utils.logger import logger


class QdrantStore:
    """Async Qdrant client wrapper for document storage and retrieval.

    Usage:
        store = QdrantStore()
        await store.ensure_collection()
        await store.upsert(points)
        hits = await store.search(query_vector, top_k=10)
    """

    def __init__(
        self,
        host: str = "",
        port: int = 0,
        collection_name: str = "",
        vector_size: int = 0,
        api_key: str = "",
    ) -> None:
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.vector_size = vector_size or settings.qdrant_vector_size
        self.api_key = api_key or settings.qdrant_api_key
        self._client: Optional[AsyncQdrantClient] = None

    async def _get_client(self) -> AsyncQdrantClient:
        """Get or create the async Qdrant client."""
        if self._client is None:
            url = settings.qdrant_url
            logger.info("Connecting to Qdrant at %s", url)
            self._client = AsyncQdrantClient(
                url=url,
                api_key=self.api_key or None,
                timeout=30,
            )
            # Test connection
            try:
                await self._client.get_collections()
                logger.info("Qdrant connection established")
            except Exception as exc:
                logger.warning("Qdrant connection failed: %s. Using offline mode.", exc)
        return self._client

    async def ensure_collection(self) -> None:
        """Create the collection if it does not exist."""
        client = await self._get_client()
        collections = await client.get_collections()
        existing = {c.name for c in collections.collections}

        if self.collection_name in existing:
            logger.info("Collection '%s' already exists", self.collection_name)
            return

        logger.info(
            "Creating collection '%s' (vector_size=%d, distance=Cosine)",
            self.collection_name,
            self.vector_size,
        )
        await client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
            # HNSW index config — good balance for <1M vectors
            hnsw_config={
                "m": 16,
                "ef_construct": 200,
            },
            optimizers_config={
                "default_segment_number": 2,
            },
        )

        logger.info("Collection '%s' created", self.collection_name)

    async def delete_collection(self) -> None:
        """Delete the collection and all its data."""
        client = await self._get_client()
        await client.delete_collection(collection_name=self.collection_name)
        logger.info("Collection '%s' deleted", self.collection_name)

    async def upsert(self, points: list[PointStruct]) -> int:
        """Insert or update points in the collection.

        Args:
            points: List of Qdrant :class:`PointStruct` objects.

        Returns:
            Number of points upserted.
        """
        if not points:
            return 0

        client = await self._get_client()
        await client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.debug("Upserted %d points to '%s'", len(points), self.collection_name)
        return len(points)

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        score_threshold: Optional[float] = None,
        filter_condition: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Search for the nearest vectors.

        Args:
            query_vector: Query embedding vector.
            top_k: Maximum number of results.
            score_threshold: Optional minimum score threshold.
            filter_condition: Optional dict filter (converted to Qdrant Filter).

        Returns:
            List of result dicts with keys: ``id``, ``score``, ``payload``.
        """
        client = await self._get_client()

        qdrant_filter = None
        if filter_condition:
            conditions = []
            for key, value in filter_condition.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
            qdrant_filter = Filter(must=conditions)  # type: ignore[arg-type]

        # qdrant-client >=1.18 uses query_points() instead of the removed search()
        hits = await client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload or {},
            }
            for hit in hits.points
        ]

    async def scroll_all(
        self,
        batch_size: int = 100,
    ) -> list[dict[str, Any]]:
        """Scroll through all points in the collection.

        Args:
            batch_size: Records per scroll request.

        Returns:
            All points with payload.
        """
        client = await self._get_client()
        all_points: list[dict] = []
        next_offset: Optional[int] = None

        while True:
            result = await client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = result
            for p in points:
                all_points.append({
                    "id": p.id,
                    "payload": p.payload or {},
                })
            if next_offset is None:
                break

        return all_points

    async def count(self) -> int:
        """Get the total number of points in the collection."""
        client = await self._get_client()
        result = await client.count(collection_name=self.collection_name)
        return result.count

    async def delete_points(
        self,
        point_ids: list[int],
    ) -> None:
        """Delete specific points by ID.

        Args:
            point_ids: List of point IDs to delete.
        """
        client = await self._get_client()
        await client.delete(
            collection_name=self.collection_name,
            points_selector=point_ids,
            wait=True,
        )

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Qdrant client closed")
