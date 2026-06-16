"""BM25 keyword index — complementary retrieval to vector search.

Uses rank_bm25 with optional persistence via pickle.
BM25 excels at exact keyword matching (names, IDs, technical terms)
that pure semantic search may miss.
"""

import pickle
import tempfile
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from enterprise_kb.config import settings
from enterprise_kb.utils.logger import logger


class BM25Index:
    """BM25 keyword index for exact-term retrieval.

    Builds a BM25 index over tokenized document chunks.
    Supports persistence to disk for re-use across sessions.

    Usage:
        index = BM25Index()
        index.build(chunks)
        results = index.search("query terms")
        index.save()
    """

    def __init__(self, persist_path: Optional[str | Path] = None) -> None:
        self._corpus: list[str] = []
        self._metadata: list[dict] = []
        self._index: Optional[BM25Okapi] = None
        # Default persist location
        self._persist_path = Path(
            str(persist_path or Path(tempfile.gettempdir()) / "enterprise_kb_bm25.pkl")
        )

    @property
    def is_built(self) -> bool:
        """Whether the BM25 index has been built."""
        return self._index is not None

    @property
    def size(self) -> int:
        """Number of documents in the index."""
        return len(self._corpus)

    def build(self, documents: list[dict[str, str | dict]]) -> None:
        """Build the BM25 index from a list of chunk documents.

        Args:
            documents: List of dicts with keys ``text`` (str) and ``metadata`` (dict).
                       Each dict must at least contain ``text``.
        """
        self._corpus = []
        self._metadata = []

        for doc in documents:
            text = doc.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            self._corpus.append(text)
            self._metadata.append(doc.get("metadata", {}))

        if not self._corpus:
            logger.warning("BM25 index built with empty corpus")
            self._index = BM25Okapi(corpus=[[]])
            return

        tokenized = [self._tokenize(t) for t in self._corpus]
        self._index = BM25Okapi(corpus=tokenized)
        logger.info("BM25 index built: %d documents", len(self._corpus))

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[dict[str, float | str | dict]]:
        """Search the BM25 index.

        Args:
            query: Raw query string (will be tokenized internally).
            top_k: Maximum results to return.

        Returns:
            List of dicts with keys: ``text``, ``metadata``, ``score``.
        """
        if not self._index or not self._corpus:
            logger.warning("BM25 index is empty — nothing to search")
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append({
                "text": self._corpus[idx],
                "metadata": self._metadata[idx],
                "score": float(scores[idx]),
            })

        return results

    def add_documents(self, documents: list[dict]) -> None:
        """Add more documents and rebuild the index (full rebuild).

        Args:
            documents: Same format as :meth:`build`.
        """
        self.build(self._corpus_texts() + documents)

    def save(self, path: Optional[str | Path] = None) -> None:
        """Persist the BM25 index to disk.

        Args:
            path: Override save path. Defaults to ``self._persist_path``.
        """
        save_path = Path(path or self._persist_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "corpus": self._corpus,
            "metadata": self._metadata,
            "index_state": {
                "doc_freqs": self._index.doc_freqs if self._index else [],
                "doc_len": self._index.doc_len if self._index else [],
                "avgdl": self._index.avgdl if self._index else 0,
                "corpus_size": self._index.corpus_size if self._index else 0,
            },
        }

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        logger.info("BM25 index saved to %s (%d docs)", save_path, len(self._corpus))

    def load(self, path: Optional[str | Path] = None) -> bool:
        """Load a persisted BM25 index from disk.

        Args:
            path: Override load path. Defaults to ``self._persist_path``.

        Returns:
            True if loaded successfully.
        """
        load_path = Path(path or self._persist_path)
        if not load_path.is_file():
            logger.warning("BM25 index file not found: %s", load_path)
            return False

        try:
            with open(load_path, "rb") as f:
                data = pickle.load(f)

            self._corpus = data["corpus"]
            self._metadata = data["metadata"]
            tokenized = [self._tokenize(t) for t in self._corpus]
            self._index = BM25Okapi(corpus=tokenized)

            logger.info("BM25 index loaded from %s (%d docs)", load_path, len(self._corpus))
            return True
        except Exception as exc:
            logger.warning("Failed to load BM25 index: %s", exc)
            return False

    # ── Internal helpers ──

    def _corpus_texts(self) -> list[dict]:
        """Return current corpus as document dicts."""
        return [
            {"text": t, "metadata": m}
            for t, m in zip(self._corpus, self._metadata)
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lower-case tokenizer with basic filtering."""
        import re

        # Split on non-alphanumeric (keep CJK character sequences)
        tokens = re.findall(r"[a-zA-Z0-9_\-]+|[^\s\W]{2,}", text.lower())
        # Filter very short tokens (likely noise)
        return [t for t in tokens if len(t) > 1]
