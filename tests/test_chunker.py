"""分块模块单元测试。"""

import pytest

from enterprise_kb.core.chunker import (
    Chunker,
    FixedSizeChunker,
    MarkdownHeaderChunker,
    SemanticSentenceChunker,
)


@pytest.mark.asyncio
class TestFixedSizeChunker:
    """固定大小分块器测试。"""

    async def test_short_text(self):
        """短文本应返回单个块。"""
        chunker = FixedSizeChunker(chunk_size=512, chunk_overlap=128)
        chunks = await chunker.chunk("Short text.")
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Short text."

    async def test_long_text(self):
        """长文本应被分成多个重叠块。"""
        text = " ".join(["word"] * 1000)
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = await chunker.chunk(text)
        assert len(chunks) > 1
        for c in chunks:
            assert "chunk_id" in c
            assert c["chunk_id"].startswith("fixed_")


@pytest.mark.asyncio
class TestMarkdownHeaderChunker:
    """Markdown 分块器测试。"""

    async def test_split_by_header(self):
        """应按 ## 标题分割。"""
        text = "# Top\nIntro.\n\n## Section 1\nContent A\n\n## Section 2\nContent B"
        chunker = MarkdownHeaderChunker(max_chunk_tokens=512)
        chunks = await chunker.chunk(text)
        assert len(chunks) >= 2
        # 至少一个块应包含 Section 1 标题
        titles = [c["metadata"].get("heading", "") for c in chunks]
        assert any("Section" in h for h in titles)

    async def test_single_section(self):
        """单一标题应生成一个块。"""
        text = "## Only Section\nSome content."
        chunker = MarkdownHeaderChunker(max_chunk_tokens=512)
        chunks = await chunker.chunk(text)
        assert len(chunks) == 1


@pytest.mark.asyncio
class TestSemanticSentenceChunker:
    """语义分块器测试。"""

    async def test_sentence_boundaries(self):
        """不应在句子中间切分。"""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunker = SemanticSentenceChunker(max_sentences_per_chunk=3, overlap_sentences=0)

        # 5句，每块3句，无重叠 → 2块
        chunks = await chunker.chunk(text)
        assert len(chunks) == 2
        assert "First sentence" in chunks[0]["text"]
        assert "Third sentence" in chunks[0]["text"]

    async def test_short_text(self):
        """短文本应保持为单个块。"""
        chunker = SemanticSentenceChunker(max_sentences_per_chunk=8, overlap_sentences=2)
        chunks = await chunker.chunk("Only one sentence.")
        assert len(chunks) == 1


@pytest.mark.asyncio
class TestChunker:
    """Chunker 主门面测试。"""

    async def test_markdown_strategy(self, sample_markdown: str):
        """Markdown 文件应使用 MarkdownHeaderChunker。"""
        chunker = Chunker()
        chunks = await chunker.chunk(sample_markdown, file_type=".md")
        assert len(chunks) > 0
        # 应保留标题信息
        headings = [c["metadata"].get("heading", "") for c in chunks]
        assert any("第一节" in h for h in headings)

    async def test_default_strategy(self):
        """未识别的文件类型应使用默认策略。"""
        chunker = Chunker()
        chunks = await chunker.chunk("Hello world test content here.", file_type=".txt")
        assert len(chunks) == 1

    async def test_empty_text(self):
        """空文本应返回空列表。"""
        chunker = Chunker()
        chunks = await chunker.chunk("", file_type=".md")
        assert len(chunks) == 0
