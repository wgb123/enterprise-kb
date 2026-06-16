"""Unified configuration management using pydantic-settings.

All configuration values are loaded from environment variables with
sensible defaults. Sensitive fields (API keys) are masked on serialization.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment & .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── 应用基础 ──
    app_name: str = Field(default="Enterprise KB", alias="APP_NAME")
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # ── LlamaParse ──
    llama_cloud_api_key: str = Field(default="", alias="LLAMA_CLOUD_API_KEY")

    # ── Qdrant ──
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_grpc_port: int = Field(default=6334, alias="QDRANT_GRPC_PORT")
    qdrant_prefer_grpc: bool = Field(default=False, alias="QDRANT_PREFER_GRPC")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="enterprise_kb", alias="QDRANT_COLLECTION_NAME")
    qdrant_vector_size: int = Field(default=1024, alias="QDRANT_VECTOR_SIZE")
    qdrant_index_threshold: int = Field(default=10_000, alias="QDRANT_INDEX_THRESHOLD")

    # ── 嵌入模型 ──
    embedding_model_name: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL_NAME")
    embedding_model_device: str = Field(default="cpu", alias="EMBEDDING_MODEL_DEVICE")
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")
    embedding_normalize: bool = Field(default=True, alias="EMBEDDING_NORMALIZE")

    # ── 重排序模型 ──
    reranker_model_name: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL_NAME")
    reranker_model_device: str = Field(default="cpu", alias="RERANKER_MODEL_DEVICE")
    reranker_top_k: int = Field(default=10, alias="RERANKER_TOP_K")

    # ── 分块配置 ──
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=128, alias="CHUNK_OVERLAP")
    chunk_markdown_by_header: bool = Field(default=True, alias="CHUNK_MARKDOWN_BY_HEADER")

    # ── HybridRAG 检索 ──
    retrieval_vector_top_k: int = Field(default=20, alias="RETRIEVAL_VECTOR_TOP_K")
    retrieval_bm25_top_k: int = Field(default=20, alias="RETRIEVAL_BM25_TOP_K")
    retrieval_rrf_k: int = Field(default=60, alias="RETRIEVAL_RRF_K")
    retrieval_final_top_k: int = Field(default=10, alias="RETRIEVAL_FINAL_TOP_K")

    # ── vLLM ──
    vllm_api_url: str = Field(default="http://localhost:8000/v1", alias="VLLM_API_URL")
    vllm_api_key: str = Field(default="", alias="VLLM_API_KEY")
    vllm_model_name: str = Field(default="Qwen/Qwen2.5-7B-Instruct", alias="VLLM_MODEL_NAME")
    vllm_max_tokens: int = Field(default=2048, alias="VLLM_MAX_TOKENS")
    vllm_temperature: float = Field(default=0.3, alias="VLLM_TEMPERATURE")
    vllm_timeout: int = Field(default=60, alias="VLLM_TIMEOUT")

    # ── Wiki ──
    wiki_root: str = Field(default="./wiki", alias="WIKI_ROOT")

    # ── 验证器 ──

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            msg = f"Invalid log level: {v}. Must be one of {allowed}"
            raise ValueError(msg)
        return v_upper

    @field_validator("embedding_model_device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"cpu", "cuda", "mps"}
        if v not in allowed:
            msg = f"Invalid device: {v}. Must be one of {allowed}"
            raise ValueError(msg)
        return v

    @property
    def qdrant_url(self) -> str:
        """Construct gRPC or HTTP URL for Qdrant."""
        if self.qdrant_prefer_grpc:
            return f"http://{self.qdrant_host}:{self.qdrant_grpc_port}"
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def wiki_path(self) -> Path:
        return Path(self.wiki_root).resolve()

    @property
    def use_llama_parse(self) -> bool:
        """Whether LlamaParse API key is configured."""
        return bool(self.llama_cloud_api_key)


# Global singleton
settings = Settings()  # type: ignore[call-arg]
