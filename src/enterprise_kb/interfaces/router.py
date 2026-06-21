"""路由器抽象基类 — 定义统一的意图分类接口。

SmartRouter（关键词）和 LlamaIndexRouter（LLM 语义路由）都继承此类，
确保 routes.py 中无需关心底层实现，通过配置自由切换。
"""

from abc import ABC, abstractmethod
from enum import Enum


class QueryIntent(str, Enum):
    """查询意图类型。两边共用同一枚举。"""

    WIKI = "wiki"
    DOCUMENT = "document"
    HYBRID = "hybrid"
    GENERAL = "general"


class BaseRouter(ABC):
    """路由器的抽象基类。

    子类必须实现三个方法：classify, should_use_wiki, should_use_hybrid_rag。
    routes.py 通过这个接口来分发查询。
    """

    @abstractmethod
    def classify(self, query: str) -> QueryIntent:
        """对查询进行分类，返回对应的路由意图。"""

    @abstractmethod
    def should_use_wiki(self, query: str) -> bool:
        """判断是否应该使用 Wiki 引擎。"""

    @abstractmethod
    def should_use_hybrid_rag(self, query: str) -> bool:
        """判断是否应该使用 HybridRAG 引擎。"""
