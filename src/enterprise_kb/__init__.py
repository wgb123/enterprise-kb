"""Enterprise Layered Knowledge Base System.

A hierarchical memory system with three layers:
- A: Compiled Memory (DeepWiki) — local Markdown knowledge base
- B: HybridRAG Retrieval Memory — vector + BM25 + reranking
- C: GraphRAG (reserved interface, not yet implemented)
"""

__version__ = "0.1.0"
