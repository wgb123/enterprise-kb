"""文档处理器单元测试。"""

from pathlib import Path

import pytest

from enterprise_kb.core.document_processor import Document, DocumentProcessor


@pytest.mark.asyncio
class TestDocumentProcessor:
    """DocumentProcessor 核心功能测试。"""

    async def test_parse_markdown(self, tmp_path: Path):
        """应正确解析 Markdown 文件。"""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello\n\nThis is a **test** document.", encoding="utf-8")

        processor = DocumentProcessor()
        doc = await processor.process_file(md_file)

        assert isinstance(doc, Document)
        assert doc.file_type == ".md"
        assert "Hello" in doc.content or "test" in doc.content
        assert doc.metadata["file_name"] == "test.md"

        await processor.close()

    async def test_parse_unknown_type(self, tmp_path: Path):
        """不支持的格式应引发异常。"""
        unknown = tmp_path / "test.xyz"
        unknown.write_bytes(b"\x00\x01\x02")

        processor = DocumentProcessor()
        with pytest.raises(Exception):
            await processor.process_file(unknown)

        await processor.close()

    async def test_file_not_found(self):
        """文件不存在应引发 FileNotFoundError。"""
        processor = DocumentProcessor()
        with pytest.raises(FileNotFoundError):
            await processor.process_file("/nonexistent/file.pdf")
        await processor.close()

    async def test_document_repr(self):
        """Document 的 repr 应包含关键信息。"""
        doc = Document(
            content="test content",
            metadata={"key": "val"},
            file_path="/path/test.md",
            file_type=".md",
        )
        rep = repr(doc)
        assert "test.md" in rep
        assert ".md" in rep
        assert "test content" in rep or "12" in rep  # len("test content")=12
