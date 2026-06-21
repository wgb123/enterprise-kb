"""智能路由模块 — 基于关键词的意图分类。

继承 BaseRouter 抽象基类，通过关键词匹配实现快速路由。
与 LlamaIndexRouter（LLM 语义路由）共享同一接口，可自由切换。

支持四种路由意图：
1. Wiki 查询（A类）→ WikiNavigator
2. 文档检索（B类）→ HybridRetriever
3. 混合查询 → 并行分发两个引擎
4. 通用 → 混合模式做最佳覆盖
"""

import re
from typing import Optional

from enterprise_kb.interfaces.router import BaseRouter, QueryIntent
from enterprise_kb.utils.logger import logger


class SmartRouter(BaseRouter):
    """智能路由 — 根据查询内容自动选择引擎。

    判断依据：
    - Wiki 路由信号：包含 "wiki"、"知识库"、"架构"、"开发指南" 等关键词
    - 文档检索信号：包含 "文档"、"文件"、"PDF"、"查找" 等关键词
    - 混合信号：两者兼备
    - 通用：自由文本，需要结合多个引擎
    """

    # Wiki 路由关键词（命中任一即走 Wiki 引擎）
    WIKI_SIGNALS = [
        "wiki", "知识库", "索引", "架构", "开发指南", "运维手册",
        "编译式记忆", "deepwiki", "编译式",
        # 内部链接跳转信号
        "[[", "]]", "categories/", "index.md",
        # 标签信号
        "#architecture", "#development", "#operations",
    ]

    # 文档检索关键词
    DOCUMENT_SIGNALS = [
        "文档", "文件", "pdf", "word", "markdown", "docx",
        "查找", "搜索", "检索", "query", "find",
        "相似", "相关", "类似",
    ]

    def classify(self, query: str) -> QueryIntent:
        """对查询进行分类，返回对应的路由意图。

        Args:
            query: 用户原始查询。

        Returns:
            :class:`QueryIntent` 枚举值。
        """
        q_lower = query.lower()

        is_wiki = any(signal.lower() in q_lower for signal in self.WIKI_SIGNALS)
        is_doc = any(signal.lower() in q_lower for signal in self.DOCUMENT_SIGNALS)

        # Wiki 内部链接语法（[[ ]]）强制走 Wiki
        if "[[" in query and "]]" in query:
            return QueryIntent.WIKI

        # 混合模式
        if is_wiki and is_doc:
            return QueryIntent.HYBRID

        # 纯 Wiki
        if is_wiki:
            return QueryIntent.WIKI

        # 纯文档检索
        if is_doc:
            return QueryIntent.DOCUMENT

        # 默认：通用查询，走混合模式做最佳覆盖
        return QueryIntent.GENERAL

    def should_use_wiki(self, query: str) -> bool:
        """判断是否应该使用 Wiki 引擎。"""
        intent = self.classify(query)
        return intent in (QueryIntent.WIKI, QueryIntent.HYBRID, QueryIntent.GENERAL)

    def should_use_hybrid_rag(self, query: str) -> bool:
        """判断是否应该使用 HybridRAG 引擎。"""
        intent = self.classify(query)
        return intent in (QueryIntent.DOCUMENT, QueryIntent.HYBRID, QueryIntent.GENERAL)
