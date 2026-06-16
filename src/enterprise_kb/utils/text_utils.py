"""Text processing utilities: cleaning, splitting, deduplication."""

import re
from typing import Optional


def clean_text(text: str, max_length: Optional[int] = None) -> str:
    """Normalize whitespace and optionally truncate.

    Args:
        text: Raw input text.
        max_length: Optional maximum character length.

    Returns:
        Cleaned text.
    """
    # Collapse multiple whitespace / newlines
    text = re.sub(r"\s+", " ", text).strip()
    if max_length and len(text) > max_length:
        text = text[:max_length] + "..."
    return text


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple heuristics.

    Handles common abbreviations (Dr., Mr., etc.) and ellipsis.

    Args:
        text: Input text.

    Returns:
        List of sentences.
    """
    # Protect common abbreviations
    text = re.sub(r"\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|vs|etc)\.", r"\1<DOT>", text)
    # Protect numbers (1. 2. etc.)
    text = re.sub(r"\b(\d+)\.", r"\1<DOT>", text)
    # Split on sentence boundaries
    parts = re.split(r"(?<=[.!?])\s+", text)
    # Restore dots
    result = [p.replace("<DOT>", ".") for p in parts if p.strip()]
    return result


def deduplicate_chunks(chunks: list[str], similarity_threshold: float = 0.85) -> list[str]:
    """Simple hash-based near-deduplication for text chunks.

    Uses a rolling hash of normalized text to remove near-duplicates.
    Useful when multiple retrieval sources return overlapping content.

    Args:
        chunks: Input text chunks.
        similarity_threshold: Jaccard similarity threshold (0-1).

    Returns:
        Deduplicated chunks.
    """
    if not chunks:
        return []

    def _shingles(text: str, k: int = 5) -> set[tuple[str, ...]]:
        words = text.lower().split()
        if len(words) < k:
            return {tuple(words)}
        return {tuple(words[i : i + k]) for i in range(len(words) - k + 1)}

    selected: list[str] = []
    selected_shingle_sets: list[set] = []

    for chunk in chunks:
        shingles = _shingles(chunk)
        if not shingles:
            continue

        is_dup = False
        for existing in selected_shingle_sets:
            if not shingles or not existing:
                continue
            intersection = shingles & existing
            union = shingles | existing
            jaccard = len(intersection) / len(union)
            if jaccard >= similarity_threshold:
                is_dup = True
                break

        if not is_dup:
            selected.append(chunk)
            selected_shingle_sets.append(shingles)

    return selected


def truncate_to_tokens(text: str, max_tokens: int, encoding_name: str = "cl100k_base") -> str:
    """Truncate text to a maximum number of tokens.

    Args:
        text: Input text.
        max_tokens: Maximum token count.
        encoding_name: tiktoken encoding name.

    Returns:
        Truncated text.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding(encoding_name)
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])
    except ImportError:
        # Fallback: rough character estimate
        return text[: max_tokens * 4]
