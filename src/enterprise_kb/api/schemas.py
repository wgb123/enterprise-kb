"""Pydantic v2 请求/响应模型定义。"""

from typing import Any

from pydantic import BaseModel, Field


# ─── 请求模型 ───


class IngestRequest(BaseModel):
    """文档入库请求体。"""

    file_paths: list[str] = Field(
        ...,
        description="待处理的文件路径列表（PDF / .docx / .md）",
        min_length=1,
    )


class QueryRequest(BaseModel):
    """知识库查询请求体。"""

    query: str = Field(
        ...,
        description="用户查询内容",
        min_length=1,
        max_length=4096,
    )
    top_k: int = Field(
        default=10,
        description="每个引擎返回的最大结果数",
        ge=1,
        le=50,
    )
    use_generator: bool = Field(
        default=True,
        description="是否使用 vLLM 生成最终答案",
    )


# ─── 响应模型 ───


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str
    version: str
    service: str


class IngestResponse(BaseModel):
    """文档入库响应。"""

    status: str  # "ok" | "partial"
    documents_processed: int
    chunks_ingested: int
    errors: list[str]


class SourceItem(BaseModel):
    """检索来源项。"""

    content: str
    source: str  # "wiki" | "hybrid_rag"
    document_id: str
    score: float
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    """知识库查询响应。"""

    query: str
    answer: str
    intent: str  # "wiki" | "document" | "hybrid" | "general"
    sources: list[SourceItem]
    total_chunks: int


class WikiPageInfo(BaseModel):
    """Wiki 页面信息。"""

    title: str
    category: str
    tags: list[str]
    path: str


class WikiSearchResponse(BaseModel):
    """Wiki 搜索响应。"""

    total: int
    pages: list[WikiPageInfo]


class WikiPageResponse(BaseModel):
    """Wiki 页面完整内容响应。"""

    title: str
    category: str
    tags: list[str]
    content: str
    toc: str
    path: str
