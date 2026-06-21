"""智能路由模块单元测试。"""

import pytest

from enterprise_kb.interfaces.router import QueryIntent
from enterprise_kb.core.router import SmartRouter


class TestSmartRouter:
    """路由分类器测试。"""

    def setup_method(self):
        self.router = SmartRouter()

    def test_wiki_intent(self):
        """Wiki 关键词应被分类为 WIKI。"""
        assert self.router.classify("wiki 架构概览") == QueryIntent.WIKI
        assert self.router.classify("查看知识库索引") == QueryIntent.WIKI
        assert self.router.classify("#architecture") == QueryIntent.WIKI
        assert self.router.classify("[[categories/architecture.md]]") == QueryIntent.WIKI
        assert self.router.classify("编译式记忆是什么") == QueryIntent.WIKI

    def test_document_intent(self):
        """文档检索关键词应被分类为 DOCUMENT。"""
        assert self.router.classify("查找PDF文档") == QueryIntent.DOCUMENT
        assert self.router.classify("搜索相似文件") == QueryIntent.DOCUMENT
        assert self.router.classify("与XX相关的文档") == QueryIntent.DOCUMENT

    def test_hybrid_intent(self):
        """同时包含 Wiki 和文档信号应分类为 HYBRID。"""
        intent = self.router.classify("查找 Wiki 中的架构文档")
        assert intent == QueryIntent.HYBRID

    def test_general_intent(self):
        """无特定关键词应分类为 GENERAL。"""
        assert self.router.classify("今天天气怎么样") == QueryIntent.GENERAL
        assert self.router.classify("什么是数据库") == QueryIntent.GENERAL

    def test_should_use_checks(self):
        """should_use_* 方法应正确判断。"""
        # Wiki 查询
        assert self.router.should_use_wiki("wiki 架构") is True
        assert self.router.should_use_hybrid_rag("wiki 架构") is False

        # 文档查询
        assert self.router.should_use_wiki("查找文档") is False
        assert self.router.should_use_hybrid_rag("查找文档") is True

        # 通用查询
        assert self.router.should_use_wiki("什么是数据库") is True
        assert self.router.should_use_hybrid_rag("什么是数据库") is True
