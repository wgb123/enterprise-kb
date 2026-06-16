"""FastAPI 依赖注入 — 管理各模块的生命周期。

在应用启动时初始化所有服务实例，通过 FastAPI 的
``Depends`` 系统注入到路由处理器中。
"""

from functools import lru_cache

from fastapi import Request

from enterprise_kb.config import settings
from enterprise_kb.core.fusion import ContextFusion
from enterprise_kb.core.generator import VLLMGenerator
from enterprise_kb.core.retriever import HybridRetriever
from enterprise_kb.core.router import SmartRouter
from enterprise_kb.core.wiki_navigator import WikiNavigator


@lru_cache(maxsize=1)
def get_wiki_navigator() -> WikiNavigator:
    """获取（单例）Wiki 导航器实例。

    Wiki 页面在首次访问时惰性加载。
    """
    return WikiNavigator(wiki_root=settings.wiki_root)


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridRetriever:
    """获取（单例）HybridRAG 检索器实例。"""
    return HybridRetriever()


@lru_cache(maxsize=1)
def get_router() -> SmartRouter:
    """获取（单例）智能路由器实例。"""
    return SmartRouter()


@lru_cache(maxsize=1)
def get_context_fusion() -> ContextFusion:
    """获取（单例）上下文融合器实例。"""
    return ContextFusion()


@lru_cache(maxsize=1)
def get_generator() -> VLLMGenerator:
    """获取（单例）答案生成器实例。"""
    return VLLMGenerator()
