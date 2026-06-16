"""配置文件单元测试。"""

from enterprise_kb.config import Settings


class TestSettings:
    """验证配置系统正常工作。"""

    def test_default_values(self):
        """默认值应被正确填充。"""
        settings = Settings()
        assert settings.app_name == "Enterprise KB"
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.qdrant_host == "localhost"
        assert settings.qdrant_port == 6333
        assert settings.chunk_size == 512
        assert settings.chunk_overlap == 128
        assert settings.retrieval_rrf_k == 60

    def test_custom_values(self):
        """环境变量应覆盖默认值。"""
        settings = Settings(
            APP_NAME="Custom KB",
            LOG_LEVEL="DEBUG",
            QDRANT_PORT=9999,
            CHUNK_SIZE=256,
        )
        assert settings.app_name == "Custom KB"
        assert settings.log_level == "DEBUG"
        assert settings.qdrant_port == 9999
        assert settings.chunk_size == 256

    def test_properties(self):
        """属性方法应正确计算派生值。"""
        settings = Settings(
            QDRANT_HOST="myhost",
            QDRANT_PORT=6333,
            QDRANT_PREFER_GRPC=False,
        )
        assert settings.qdrant_url == "http://myhost:6333"

        settings_grpc = Settings(
            QDRANT_HOST="myhost",
            QDRANT_PORT=6333,
            QDRANT_GRPC_PORT=6334,
            QDRANT_PREFER_GRPC=True,
        )
        assert settings_grpc.qdrant_url == "http://myhost:6334"

    def test_use_llama_parse(self):
        """仅当 API key 配置时返回 True。"""
        settings_without = Settings(LLAMA_CLOUD_API_KEY="")
        assert settings_without.use_llama_parse is False

        settings_with = Settings(LLAMA_CLOUD_API_KEY="sk-xxx")
        assert settings_with.use_llama_parse is True
