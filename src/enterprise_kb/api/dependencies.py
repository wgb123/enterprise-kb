"""FastAPI 依赖注入 — 管理各模块的生命周期。

在应用启动时初始化所有服务实例，通过 FastAPI 的
``Depends`` 系统注入到路由处理器中。

路由模式通过 ``ROUTER_TYPE`` 环境变量切换：
  ``keyword``（默认）— SmartRouter，关键词匹配，毫秒级
  ``llamaindex`` — LlamaIndexRouter，LLM 语义路由，需安装依赖
"""

from functools import lru_cache

from fastapi import Request

from enterprise_kb.config import settings
from enterprise_kb.interfaces.router import BaseRouter
from enterprise_kb.core.fusion import ContextFusion
from enterprise_kb.core.generator import VLLMGenerator
from enterprise_kb.core.retriever import HybridRetriever
from enterprise_kb.core.router import SmartRouter
from enterprise_kb.core.wiki_navigator import WikiNavigator
from enterprise_kb.utils.logger import logger


def _create_router() -> BaseRouter:
    """根据配置创建对应的路由器实例。

    读取 ``settings.router_type`` 来决定使用哪种路由策略。
    """
    router_type = settings.router_type.lower()

    if router_type == "llamaindex":
        from enterprise_kb.core.llamaindex_router import LlamaIndexRouter

        logger.info("Using LlamaIndexRouter (LLM semantic routing)")
        return LlamaIndexRouter()

    logger.info("Using SmartRouter (keyword routing)")
    return SmartRouter()


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
def get_router() -> BaseRouter:
    """获取（单例）路由器实例。

    路由策略由 ``ROUTER_TYPE`` 环境变量控制：
    - ``keyword`` → SmartRouter（关键词匹配）
    - ``llamaindex`` → LlamaIndexRouter（LLM 语义路由）
    """
    return _create_router()


@lru_cache(maxsize=1)
def get_context_fusion() -> ContextFusion:
    """获取（单例）上下文融合器实例。"""
    return ContextFusion()


@lru_cache(maxsize=1)
def get_generator() -> VLLMGenerator:
    """获取（单例）答案生成器实例。"""
    return VLLMGenerator()
