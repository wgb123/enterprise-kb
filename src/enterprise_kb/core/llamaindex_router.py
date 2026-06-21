"""LlamaIndex 版路由 — 基于 LLM 语义的意图分类。

继承 BaseRouter 抽象基类，通过 LLMSingleSelector + RouterQueryEngine
实现语义级路由。与 SmartRouter（关键词路由）共享同一接口，可自由切换。

使用方式：
  # 在 .env 中配置（默认 keyword，切换为 llamaindex）：
  ROUTER_TYPE=llamaindex
  LLM_API_URL=https://api.deepseek.com/v1
  LLM_API_KEY=sk-xxx

实现原理（见学习笔记 5.2 节）：
  LLMSingleSelector 构造格式化 prompt，让 LLM 读每个引擎的 description
  和用户查询，选最合适的引擎 → 框架自动路由查询到对应引擎
"""

from typing import Optional

from enterprise_kb.config import settings
from enterprise_kb.interfaces.router import BaseRouter, QueryIntent
from enterprise_kb.utils.logger import logger


# LlamaIndex 是可选依赖，只在配置为 llamaindex 模式时才需要
try:
    from llama_index.core.query_engine import RouterQueryEngine
    from llama_index.core.tools import QueryEngineTool
    from llama_index.core.selectors import LLMSingleSelector
    from llama_index.core import Document, SummaryIndex
    from llama_index.llms.openai import OpenAI as LlamaOpenAI

    _HAS_LLAMA_INDEX = True
except ImportError:
    _HAS_LLAMA_INDEX = False
    RouterQueryEngine = None  # type: ignore
    QueryEngineTool = None  # type: ignore
    LLMSingleSelector = None  # type: ignore
    Document = None  # type: ignore
    SummaryIndex = None  # type: ignore
    LlamaOpenAI = None  # type: ignore


class LlamaIndexRouter(BaseRouter):
    """LlamaIndex LLM 语义路由。

    内部使用 LLMSingleSelector + RouterQueryEngine。
    首次调用时惰性构建，构建过程会创建两个虚拟 QueryEngine（Wiki / RAG
    描述各一份），以及一个 OpenAI 兼容的 LLM 实例。

    每条查询：
      1. LLM 读取查询 + 两个引擎的 description
      2. LLM 判断走哪个引擎（或同时走两个）
      3. 返回对应的 QueryIntent
    """

    def __init__(
        self,
        api_url: str = "",
        api_key: str = "",
        model_name: str = "",
        temperature: float = 0,
    ):
        if not _HAS_LLAMA_INDEX:
            raise ImportError(
                "LlamaIndex 未安装。请运行: pip install 'enterprise-kb[llamaindex]'"
                " 或手动: pip install llama-index-core llama-index-llms-openai"
            )

        self.api_url = api_url or settings.vllm_api_url
        self.api_key = api_key or settings.vllm_api_key
        self.model_name = model_name or settings.vllm_model_name
        self.temperature = temperature

        self._engine: Optional[RouterQueryEngine] = None

    def _ensure_engine(self) -> RouterQueryEngine:
        """惰性初始化 RouterQueryEngine（只在首次查询时构建）。"""
        if self._engine is not None:
            return self._engine

        # ── 创建 LLM ──
        llm = LlamaOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            api_base=self.api_url.rstrip("/") + "/v1",
            temperature=self.temperature,
        )

        # ── 创建两个虚拟 QueryEngine（只有描述，查询时自动路由） ──

        # Wiki 引擎的描述
        wiki_doc = Document(
            text="Wiki 知识库包含系统架构、开发指南、运维部署、三层记忆等内部知识。",
            metadata={"source": "wiki"},
        )
        wiki_index = SummaryIndex.from_documents([wiki_doc])
        wiki_engine = wiki_index.as_query_engine(llm=llm)

        wiki_tool = QueryEngineTool.from_defaults(
            query_engine=wiki_engine,
            description=(
                "用于回答关于系统架构、开发指南、运维部署、Wiki 知识库等内部文档的问题。"
                "包括架构设计、三层记忆（A/B/C类）、环境配置（.env）、Qdrant 部署、"
                "开发指南（development.md）、运维手册（operations.md）等。"
                "如果查询与系统架构、技术决策、内部开发配置相关，请选择此工具。"
            ),
        )

        # RAG 引擎的描述
        rag_doc = Document(
            text="HybridRAG 知识库包含海量外部文档、通用技术知识、PDF/Word 等自动导入的文档内容。",
            metadata={"source": "hybrid_rag"},
        )
        rag_index = SummaryIndex.from_documents([rag_doc])
        rag_engine = rag_index.as_query_engine(llm=llm)

        rag_tool = QueryEngineTool.from_defaults(
            query_engine=rag_engine,
            description=(
                "用于回答关于外部知识、通用技术问题、具体技术文档内容的问题。"
                "包括编程语言、框架、算法、数据库、产品文档、PDF 文档中的详细内容等。"
                "如果查询需要外部知识或具体技术名词的解释、文档中的详细信息，请选择此工具。"
            ),
        )

        # ── 构建 RouterQueryEngine ──
        self._engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(llm=llm),
            query_engine_tools=[wiki_tool, rag_tool],
            verbose=settings.debug,
        )

        logger.info(
            "LlamaIndexRouter initialized (model=%s, api=%s)",
            self.model_name,
            self.api_url,
        )
        return self._engine

    def _llm_classify(self, query: str) -> str:
        """通过 LLM 判断查询意图。（同步包装异步即可，但此处用同步路径）

        注意：RouterQueryEngine 的 selector 实际上是同步的 prompt → LLM → response。
        此处为了统一接口，使用同步方式调用 ChatOpenAI 的 complete。
        """
        engine = self._ensure_engine()

        # RouterQueryEngine 内部 selector 会自动做语义判断
        # 但我们只需要知道 LLM 选了哪个引擎，不需要实际执行查询
        # 因此直接调用 selector 来获取选择结果
        try:
            from llama_index.core.base.base_selector import SelectorResult

            selector_result: SelectorResult = engine._selector.select(
                engine._metadatas,
                query,
            )

            if not selector_result.selections:
                return "general"

            # 返回所有被选中的引擎索引
            selected_indices = [s.index for s in selector_result.selections]
            tool_names = []
            for idx in selected_indices:
                if idx < len(engine._metadatas):
                    tool_names.append(engine._metadatas[idx].name or f"tool_{idx}")

            if not tool_names:
                return "general"

            return ",".join(tool_names)

        except Exception as exc:
            logger.warning("LlamaIndex selector failed, falling back: %s", exc)
            return "general"

    # ── BaseRouter 接口实现 ──────────────────────────

    def classify(self, query: str) -> QueryIntent:
        """通过 LLM 语义判断查询意图。"""
        result = self._llm_classify(query)

        # 根据 LLM 选择的工具判断意图
        if result == "general":
            return QueryIntent.GENERAL
        if "wiki" in result and ("rag" in result or "tool_1" in result):
            return QueryIntent.HYBRID
        if "wiki" in result or "tool_0" in result:
            return QueryIntent.WIKI
        if "rag" in result or "tool_1" in result:
            return QueryIntent.DOCUMENT

        return QueryIntent.GENERAL

    def should_use_wiki(self, query: str) -> bool:
        intent = self.classify(query)
        return intent in (QueryIntent.WIKI, QueryIntent.HYBRID, QueryIntent.GENERAL)

    def should_use_hybrid_rag(self, query: str) -> bool:
        intent = self.classify(query)
        return intent in (QueryIntent.DOCUMENT, QueryIntent.HYBRID, QueryIntent.GENERAL)
