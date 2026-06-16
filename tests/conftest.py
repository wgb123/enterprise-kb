"""pytest 配置和 fixtures。"""

from pathlib import Path
from typing import AsyncGenerator

import pytest
import pytest_asyncio

from enterprise_kb.config import Settings


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """返回测试用的配置对象（禁用外部依赖）。"""
    return Settings(
        APP_NAME="Enterprise KB Test",
        DEBUG=True,
        LOG_LEVEL="DEBUG",
        QDRANT_HOST="localhost",
        QDRANT_PORT=6333,
        QDRANT_COLLECTION_NAME="test_enterprise_kb",
        WIKI_ROOT="./wiki",
        EMBEDDING_MODEL_NAME="BAAI/bge-m3",
        EMBEDDING_MODEL_DEVICE="cpu",
    )


@pytest.fixture(scope="session")
def wiki_root() -> Path:
    """返回测试 Wiki 根目录路径。"""
    return Path(__file__).parent.parent / "wiki"


@pytest.fixture
def sample_markdown() -> str:
    """示例 Markdown 文档。"""
    return """# 测试文档

## 第一节
这是第一段内容。
包含一些测试文字。

## 第二节
这是第二段内容。
更多测试文字在这里。

### 子节
子节的内容。
"""


@pytest.fixture
def sample_chunks() -> list[dict]:
    """示例文档分块数据。"""
    return [
        {
            "text": "企业知识库系统采用分层记忆架构。",
            "chunk_id": "chunk_001",
            "metadata": {"file_name": "architecture.md", "file_type": ".md"},
        },
        {
            "text": "HybridRAG 结合向量检索和关键词检索。",
            "chunk_id": "chunk_002",
            "metadata": {"file_name": "architecture.md", "file_type": ".md"},
        },
        {
            "text": "Qdrant 是高性能向量数据库。",
            "chunk_id": "chunk_003",
            "metadata": {"file_name": "operations.md", "file_type": ".md"},
        },
    ]
