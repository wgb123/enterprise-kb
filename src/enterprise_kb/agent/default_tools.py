"""内置工具注册 — 系统预置的工具函数集合。

运行 `register_default_tools()` 后，Agent 即可使用以下工具：
- wiki_search: 搜索 A 类知识库
- rag_search: 检索 B 类 HybridRAG
- calculator: 数学计算
- get_time: 获取当前时间
"""

from __future__ import annotations

import asyncio
import datetime
import json
import operator
from typing import Any

from enterprise_kb.agent.tool_registry import ToolRegistry, ToolSpec
from enterprise_kb.utils.logger import logger
from enterprise_kb.core.router import SmartRouter


# ═══════════════════════════════════════════════
# 工具函数（handler）
# ═══════════════════════════════════════════════

# ── 获取当前时间 ──


async def handler_get_time() -> str:
    """返回当前日期和时间。"""
    now = datetime.datetime.now()
    return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%A')})"


# ── 数学计算 ──

# 安全的计算白名单
_SAFE_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
}


async def handler_calculator(expression: str) -> str:
    """安全执行四则运算。

    使用白名单模式，只允许基本算术运算符。
    不支持 eval 或任意 Python 代码执行。

    Args:
        expression: 数学表达式，如 "12345 * 6789" 或 "(100 + 200) / 3"
    """
    import ast
    import math

    # 允许的节点类型
    ALLOWED_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
        ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    )

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        # 安全检查：禁止非白名单节点
        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED_NODES):
                return f"表达式包含不安全的操作: {type(node).__name__}"

        result = eval(  # nosec
            compile(tree, "<string>", "eval"),
            {"__builtins__": {}},
            {"math": math},
        )
        return f"{expression} = {result}"

    except ZeroDivisionError:
        return "错误: 除以零"
    except SyntaxError as exc:
        return f"表达式语法错误: {exc}"
    except Exception as exc:
        return f"计算失败: {exc}"


# ── Wiki 搜索（需要外部注入 WikiNavigator） ──


def create_handler_wiki_search(wiki_nav: Any) -> Any:
    """创建 Wiki 搜索 handler。

    Args:
        wiki_nav: WikiNavigator 实例。
    """

    async def _search(query: str, max_results: int = 5) -> str:
        """在本地知识库中搜索。

        Args:
            query: 搜索关键词。
            max_results: 最多返回结果数。
        """
        try:
            results = await wiki_nav.query(query)
            if not results:
                return f"在知识库中未找到与「{query}」相关的内容。"

            parts = []
            for i, res in enumerate(results[:max_results], 1):
                parts.append(
                    f"[结果 {i}] 标题: {res.get('title', '未知')}\n"
                    f"路径: {res.get('path', '')}\n"
                    f"摘要: {res.get('content', '')[:300]}"
                )
            return "\n\n".join(parts)

        except Exception as exc:
            return f"Wiki 搜索失败: {exc}"

    return _search


# ── RAG 检索（需要外部注入 HybridRetriever） ──


def create_handler_rag_search(hybrid_retriever: Any) -> Any:
    """创建 RAG 搜索 handler。

    Args:
        hybrid_retriever: HybridRetriever 实例。
    """

    async def _search(query: str, top_k: int = 5) -> str:
        """在文档库中检索相关内容。

        Args:
            query: 检索关键词。
            top_k: 返回条数。
        """
        try:
            chunks = await hybrid_retriever.retrieve(query)
            if not chunks:
                return f"在文档库中未找到与「{query}」相关的内容。"

            parts = []
            for i, ch in enumerate(chunks[:top_k], 1):
                source = ch.metadata.get("source", ch.source) if hasattr(ch, "metadata") else ch.source
                content = ch.content[:300] if hasattr(ch, "content") else str(ch)[:300]
                parts.append(
                    f"[结果 {i}] 来源: {source}\n"
                    f"内容: {content}"
                )
            return "\n\n".join(parts)

        except Exception as exc:
            return f"RAG 检索失败: {exc}"

    return _search


# ═══════════════════════════════════════════════
# 注册函数
# ═══════════════════════════════════════════════


def register_default_tools(
    wiki_nav: Any = None,
    hybrid_retriever: Any = None,
) -> int:
    """注册系统内置工具。

    Args:
        wiki_nav: WikiNavigator 实例（可选，wiki_search 需要）。
        hybrid_retriever: HybridRetriever 实例（可选，rag_search 需要）。

    Returns:
        已注册的工具数量。
    """
    registry = ToolRegistry()

    # 1. 获取时间（无参数）
    registry.register(ToolSpec(
        name="get_time",
        description="获取当前日期和时间。当用户询问「现在几点」「今天日期」等时使用。",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=handler_get_time,
    ))

    # 2. 数学计算
    registry.register(ToolSpec(
        name="calculator",
        description="执行数学计算。支持加减乘除、幂运算、括号。用于需要精确计算的场景。",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 '12345 * 6789'、'(100 + 200) / 3'",
                }
            },
            "required": ["expression"],
        },
        handler=handler_calculator,
    ))

    # 3. Wiki 搜索（需注入 navigator）
    if wiki_nav is not None:
        registry.register(ToolSpec(
            name="wiki_search",
            description="在本地知识库（DeepWiki）中搜索相关内容。用于查找公司架构文档、开发指南、运维手册等内部资料。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，如「架构设计」「开发环境」「Qdrant 部署」",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最多返回的结果数量（默认 5）",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            handler=create_handler_wiki_search(wiki_nav),
        ))

    # 4. RAG 检索（需注入 retriever）
    if hybrid_retriever is not None:
        registry.register(ToolSpec(
            name="rag_search",
            description="在大规模文档库中检索相关内容。用于查找文档、技术资料、产品手册等外部知识。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回条数（默认 5）",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            handler=create_handler_rag_search(hybrid_retriever),
        ))

    count = registry.count
    logger.info("已注册 %d 个内置工具: %s", count,
                [t.name for t in registry.list_tools()])
    return count
