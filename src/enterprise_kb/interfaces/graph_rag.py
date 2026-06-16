"""GraphRAG interface (C-class memory — reserved, not yet implemented).

This module defines the contract for a future Graph-based Retrieval
Augmented Generation engine. When implemented, it will:
- Parse documents into entity-relation graphs
- Store graph structure in a knowledge graph database (e.g. Neo4j)
- Perform graph traversal + community summarization for queries
- Support both local (entity-centric) and global (community-centric) search

Usage:
    Register implementation via the factory pattern:
    >>> from enterprise_kb.core.router import EngineRegistry
    >>> EngineRegistry.register("graph_rag", MyGraphRAGEngine())
"""

from typing import Any, Protocol

from enterprise_kb.interfaces.base import BaseEngine, QueryResult, RetrievedChunk


class GraphEntity(Protocol):
    """Protocol for a graph entity (node)."""

    id: str
    name: str
    type: str  # e.g. "concept", "person", "document"
    properties: dict[str, Any]


class GraphRelation(Protocol):
    """Protocol for a graph relation (edge)."""

    source_id: str
    target_id: str
    relation_type: str  # e.g. "mentions", "cites", "relates_to"
    weight: float
    properties: dict[str, Any]


class GraphKnowledgeBase(Protocol):
    """Protocol for the underlying graph database.

    Expected methods (to be implemented later):
    - query_entities(query: str, top_k: int) -> list[GraphEntity]
    - query_relations(entity_id: str, depth: int) -> list[GraphRelation]
    - community_search(query: str, top_k: int) -> list[RetrievedChunk]
    - insert_graph(entities: list[GraphEntity], relations: list[GraphRelation]) -> None
    """

    ...


class GraphRAGEngine(BaseEngine):
    """Reserved GraphRAG engine — raises NotImplementedError on use.

    This class satisfies the BaseEngine interface at import time.
    Actual implementation will be added as a separate task.
    """

    _implemented: bool = False

    async def query(self, query: str, top_k: int = 10) -> QueryResult:
        """GraphRAG is not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        msg = (
            "GraphRAG (C-class memory) is not yet implemented. "
            "Please use A-class (Wiki) or B-class (HybridRAG) engines."
        )
        raise NotImplementedError(msg)

    @classmethod
    def is_available(cls) -> bool:
        """Check if the GraphRAG engine is implemented."""
        return cls._implemented

    @classmethod
    def mark_available(cls) -> None:
        """Mark GraphRAG as available after implementation is complete."""
        cls._implemented = True

    @property
    def name(self) -> str:
        return "graph_rag"
