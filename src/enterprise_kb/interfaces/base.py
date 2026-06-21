"""Abstract base classes for the knowledge base system.

Defines the contract that every engine component must implement.
This enables the smart router and fusion layer to treat all
retrieval sources polymorphically.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RetrievedChunk:
    """A single chunk retrieved from any memory layer."""

    content: str
    source: str  # e.g. "wiki", "hybrid_rag", "graph_rag"
    document_id: str
    chunk_id: str
    score: float = 0.0
    vector_score: float = 0.0  # score from vector search (Qdrant), 0 if unmatched
    bm25_score: float = 0.0   # score from keyword search (BM25), 0 if unmatched
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Unified query result from any engine."""

    query: str
    chunks: list[RetrievedChunk]
    engine: str  # which engine produced this result
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseChunker(ABC):
    """Abstract chunker — splits a document into retrievable pieces."""

    @abstractmethod
    async def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Split text into chunks with metadata.

        Args:
            text: Document text to split.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of dicts with keys: ``text``, ``metadata``, ``chunk_id``.
        """
        ...


class BaseEmbedder(ABC):
    """Abstract embedder — converts text to vectors."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors.

        Args:
            texts: List of text strings.

        Returns:
            List of embedding vectors (list of floats).
        """
        ...

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...


class BaseRetriever(ABC):
    """Abstract retriever — searches for relevant chunks given a query."""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10) -> QueryResult:
        """Retrieve relevant chunks for a query.

        Args:
            query: User query string.
            top_k: Maximum number of chunks to return.

        Returns:
            A :class:`QueryResult` containing retrieved chunks.
        """
        ...


class BaseGenerator(ABC):
    """Abstract generator — produces a final answer from retrieved context."""

    @abstractmethod
    async def generate(
        self,
        query: str,
        context_chunks: list[RetrievedChunk],
        system_prompt: str | None = None,
    ) -> str:
        """Generate an answer based on query and context.

        Args:
            query: Original user query.
            context_chunks: Retrieved and fused context chunks.
            system_prompt: Optional system prompt override.

        Returns:
            Generated answer string.
        """
        ...


class BaseEngine(ABC):
    """Abstract engine — a complete retrieval+generation pipeline."""

    @abstractmethod
    async def query(self, query: str, top_k: int = 10) -> QueryResult:
        """Run the full pipeline: retrieve + optional generate.

        Args:
            query: User query string.
            top_k: Maximum chunks to retrieve.

        Returns:
            :class:`QueryResult` with retrieved chunks.
        """
        ...
