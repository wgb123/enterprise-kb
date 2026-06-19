"""LangChain 版工具定义 — @tool + StructuredTool。

对照手写版：
  agent/default_tools.py  →  4 个工具
  agent/tool_executor.py  →  LangChain 自带执行器
  agent/tool_registry.py  →  @tool 装饰器自动注册
"""

import ast
import operator
from datetime import datetime
from typing import Any

from langchain_core.tools import tool, StructuredTool

from enterprise_kb.core.wiki_navigator import WikiNavigator
from enterprise_kb.core.retriever import HybridRetriever


# ── 1. get_time ──────────────────────────────────────

@tool
def get_time() -> str:
    """获取当前时间。用于回答"现在几点""今天几号"等时间相关问题。"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


# ── 2. calculator（AST 安全沙箱）─────────────────────

# 允许的运算符映射
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expr: str) -> str:
    """AST 白名单 eval——只允许四则运算、幂、取模。禁止 __import__ 等危险操作。"""
    tree = ast.parse(expr.strip(), mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Expression):
            continue
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError(f"不支持的常量类型: {type(node.value).__name__}")
            continue
        if type(node) not in (ast.BinOp, ast.UnaryOp, ast.Expression, ast.Constant):
            raise ValueError(f"不支持的语法: {type(node).__name__}")
        if isinstance(node, (ast.BinOp, ast.UnaryOp)):
            op_type = type(node.op) if isinstance(node, ast.BinOp) else type(node.op)
            if op_type not in _ALLOWED_OPS:
                raise ValueError(f"不支持的运算符: {op_type.__name__}")

    result = eval(expr, {"__builtins__": {}}, {})
    return str(result)


@tool
def calculator(expression: str) -> str:
    """执行数学计算。支持加减乘除、幂运算、括号、取模。"""
    try:
        return _safe_eval(expression)
    except Exception as exc:
        return f"计算错误: {exc}"


# ── 3. wiki_search ──────────────────────────────────

async def _handler_wiki_search(query: str) -> str:
    """搜索本地 Wiki 知识库（A 类编译式记忆）。"""
    navigator = WikiNavigator()
    results = await navigator.query(query)
    if not results:
        return "本地 Wiki 知识库中没有找到相关内容。"
    parts = []
    for page, score in results[:5]:
        parts.append(
            f"--- 页面: {page.path} (匹配度: {score:.2f}) ---\n"
            f"{page.content[:500]}..."
        )
    return "\n\n".join(parts)


wiki_search = StructuredTool.from_function(
    func=_handler_wiki_search,
    name="wiki_search",
    description=(
        "在本地知识库（DeepWiki）中搜索相关内容。"
        "包含了公司架构文档、开发指南、运维手册等内部资料。"
        "当你需要查找内部文档、架构说明、环境配置时使用。"
    ),
    args_schema={"query": {"type": "string", "description": "搜索关键词"}},
)


# ── 4. rag_search ────────────────────────────────────

async def _handler_rag_search(query: str) -> str:
    """搜索向量知识库（B 类 HybridRAG）。"""
    retriever = HybridRetriever()
    results = await retriever.retrieve(query)
    if not results:
        return "向量知识库中没有找到相关内容。"
    parts = []
    for chunk in results[:5]:
        source = f"[来源: {chunk.document_id}]" if chunk.document_id else "[未知来源]"
        parts.append(f"{source}\n{chunk.content[:400]}...")
    return "\n\n---\n\n".join(parts)


rag_search = StructuredTool.from_function(
    func=_handler_rag_search,
    name="rag_search",
    description=(
        "在向量知识库（HybridRAG）中检索文档内容。"
        "包含了自动导入的海量文档（PDF/Word/Markdown）。"
        "当你需要搜索大量文档中的具体内容、技术细节、参考信息时使用。"
    ),
    args_schema={"query": {"type": "string", "description": "搜索关键词"}},
)


# ── 获取默认工具列表 ────────────────────────────────

def get_default_tools() -> list:
    """返回 4 个默认工具的列表，可直接传给 AgentExecutor。"""
    return [get_time, calculator, wiki_search, rag_search]
