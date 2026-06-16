"""Semantic and fixed-size chunker module.

Supports:
- Fixed-size chunking with configurable overlap
- Markdown-aware chunking (by heading level)
- Semantic-sentence chunking (sliding window over sentences)
- Strategy registry for per-document-type chunking
"""

import hashlib
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

import tiktoken

from enterprise_kb.config import settings
from enterprise_kb.interfaces.base import BaseChunker
from enterprise_kb.utils.logger import logger


class ChunkingStrategy(ABC):
    """Abstract chunking strategy."""

    @abstractmethod
    async def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        ...


class FixedSizeChunker(ChunkingStrategy):
    """Fixed-token-size chunking with sliding window overlap.

    Uses tiktoken for accurate token counting.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name

    async def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Split text into fixed-size overlapping chunks."""
        enc = tiktoken.get_encoding(self.encoding_name)
        tokens = enc.encode(text)
        chunks: list[dict] = []

        if len(tokens) <= self.chunk_size:
            chunk_text = enc.decode(tokens)
            chunks.append(self._make_chunk(chunk_text, metadata, 0))
            return chunks

        start = 0
        idx = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(self._make_chunk(chunk_text, metadata, idx))
            idx += 1

            if end >= len(tokens):
                break
            start += self.chunk_size - self.chunk_overlap

        return chunks

    @staticmethod
    def _make_chunk(text: str, metadata: dict | None, index: int) -> dict:
        chunk_id = hashlib.md5(text.encode()).hexdigest()[:12]
        return {
            "text": text.strip(),
            "metadata": metadata or {},
            "chunk_id": f"fixed_{index}_{chunk_id}",
        }


class MarkdownHeaderChunker(ChunkingStrategy):
    """Markdown-aware chunker that splits by heading levels.

    Preserves the heading hierarchy for each chunk so that
    chunk-level context is self-contained.
    """

    def __init__(
        self,
        max_chunk_tokens: int = 512,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.max_chunk_tokens = max_chunk_tokens
        self.encoding_name = encoding_name

    async def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Split Markdown by ``##`` headings, then sub-split oversized sections."""
        enc = tiktoken.get_encoding(self.encoding_name)
        sections = self._split_by_heading(text)
        chunks: list[dict] = []
        idx = 0

        for heading, body in sections:
            section_text = f"{heading}\n{body}" if heading else body
            tokens = enc.encode(section_text)

            if len(tokens) <= self.max_chunk_tokens:
                chunks.append(self._make_chunk(section_text, metadata, idx))
                idx += 1
            else:
                # Sub-split the section using a fixed-size approach
                sub_chunks = await FixedSizeChunker(
                    chunk_size=self.max_chunk_tokens,
                    chunk_overlap=64,
                    encoding_name=self.encoding_name,
                ).chunk(body, metadata)
                for sc in sub_chunks:
                    sc["text"] = f"{heading}\n{sc['text']}" if heading else sc["text"]
                    sc["chunk_id"] = f"md_{idx}_{sc['chunk_id']}"
                    groups = re.match(r"^(#+)\s+(.+)$", heading) if heading else None
                    if groups:
                        sc["metadata"]["heading"] = heading
                        sc["metadata"]["heading_level"] = len(groups.group(1))
                    chunks.append(sc)
                    idx += 1

        return chunks

    @staticmethod
    def _split_by_heading(text: str) -> list[tuple[str, str]]:
        """Split Markdown into (heading, body) pairs.

        Lines before the first heading are grouped under an empty heading.
        """
        lines = text.splitlines()
        sections: list[tuple[str, str]] = []
        current_heading = ""
        current_body: list[str] = []

        for line in lines:
            if re.match(r"^##+\s+", line):
                if current_body or current_heading:
                    sections.append((current_heading, "\n".join(current_body).strip()))
                    current_body = []
                current_heading = line
            else:
                current_body.append(line)

        if current_body or current_heading:
            sections.append((current_heading, "\n".join(current_body).strip()))

        return sections

    @staticmethod
    def _make_chunk(text: str, metadata: dict | None, index: int) -> dict:
        chunk_id = hashlib.md5(text.encode()).hexdigest()[:12]
        heading = ""
        heading_level = 0
        m = re.match(r"^(#+)\s+(.+)$", text.split("\n")[0]) if text.strip() else None
        if m:
            heading = text.split("\n")[0]
            heading_level = len(m.group(1))

        return {
            "text": text.strip(),
            "metadata": {
                **(metadata or {}),
                "heading": heading,
                "heading_level": heading_level,
            },
            "chunk_id": f"md_{index}_{chunk_id}",
        }


class SemanticSentenceChunker(ChunkingStrategy):
    """Sentence-aware chunker using sliding window over sentences.

    Preserves sentence boundaries — never cuts in the middle of a sentence.
    """

    def __init__(
        self,
        max_sentences_per_chunk: int = 8,
        overlap_sentences: int = 2,
    ) -> None:
        self.max_sentences = max_sentences_per_chunk
        self.overlap = overlap_sentences

    async def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Split text by sentence boundaries."""
        from enterprise_kb.utils.text_utils import split_sentences

        sentences = split_sentences(text)
        chunks: list[dict] = []
        idx = 0

        if len(sentences) <= self.max_sentences:
            chunk_text = " ".join(sentences)
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            chunks.append({
                "text": chunk_text,
                "metadata": metadata or {},
                "chunk_id": f"sem_{idx}_{chunk_id}",
            })
            return chunks

        start = 0
        while start < len(sentences):
            end = min(start + self.max_sentences, len(sentences))
            chunk_text = " ".join(sentences[start:end])
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            chunks.append({
                "text": chunk_text,
                "metadata": metadata or {},
                "chunk_id": f"sem_{idx}_{chunk_id}",
            })
            idx += 1
            if end >= len(sentences):
                break
            start += self.max_sentences - self.overlap

        return chunks


class Chunker(BaseChunker):
    """Main chunker facade — selects strategy based on file type.

    Default strategies:
    - ``.md`` / ``.markdown`` → :class:`MarkdownHeaderChunker`
    - ``.pdf`` → :class:`SemanticSentenceChunker`
    - ``.docx`` → :class:`FixedSizeChunker`
    - other → :class:`FixedSizeChunker`
    """

    def __init__(self) -> None:
        self._strategies: dict[str, ChunkingStrategy] = {
            ".md": MarkdownHeaderChunker(
                max_chunk_tokens=settings.chunk_size,
            ),
            ".markdown": MarkdownHeaderChunker(
                max_chunk_tokens=settings.chunk_size,
            ),
            "default": FixedSizeChunker(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            ),
        }
        # PDF uses semantic sentence chunking
        self._strategies[".pdf"] = SemanticSentenceChunker(
            max_sentences_per_chunk=8,
            overlap_sentences=2,
        )
        self._strategies[".docx"] = FixedSizeChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def register_strategy(self, file_type: str, strategy: ChunkingStrategy) -> None:
        """Register a custom chunking strategy for a file type.

        Args:
            file_type: File extension (e.g. ``.rst``).
            strategy: Custom :class:`ChunkingStrategy` instance.
        """
        self._strategies[file_type] = strategy

    async def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        file_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Chunk text using the strategy appropriate for the file type.

        Args:
            text: Document text.
            metadata: Metadata to attach to each chunk.
            file_type: File extension to select strategy (e.g. ``.md``).
                       Falls back to default if ``None`` or unrecognized.

        Returns:
            List of chunks with keys: ``text``, ``metadata``, ``chunk_id``.
        """
        strategy = self._strategies.get(file_type or "") or self._strategies["default"]
        return await strategy.chunk(text, metadata)
