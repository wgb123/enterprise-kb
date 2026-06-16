"""工具注册系统 — 集中管理 Agent 可调用的工具。

每个工具包含：
- 名称、描述（LLM 选择依据）
- 参数 JSON Schema（OpenAI Function Calling 格式）
- 执行函数（实际业务逻辑）
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional


@dataclass
class ToolSpec:
    """工具定义规范。

    Attributes:
        name: 工具名称，LLM 通过此名称调用。
        description: 自然语言描述，LLM 据此判断何时使用。
        parameters: JSON Schema 格式的参数定义。
        handler: 实际执行的异步函数。
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Coroutine[Any, Any, str]]
    # 可选配置
    timeout: float = 30.0
    retry_on_error: bool = False


class ToolRegistry:
    """工具注册表 — 单例模式，全局只此一份。"""

    _instance: Optional["ToolRegistry"] = None
    _tools: dict[str, ToolSpec] = field(default_factory=dict)

    def __new__(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    def register(self, spec: ToolSpec) -> None:
        """注册一个工具。

        Args:
            spec: 工具定义。
        """
        if spec.name in self._tools:
            raise ValueError(f"工具 '{spec.name}' 已注册")
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        """按名称获取工具。"""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        """返回所有已注册的工具。"""
        return list(self._tools.values())

    def unregister(self, name: str) -> None:
        """注销一个工具。"""
        self._tools.pop(name, None)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """生成 OpenAI Function Calling 格式的工具列表。

        直接传入 chat/completions 的 tools 参数。
        """
        result = []
        for spec in self._tools.values():
            result.append({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            })
        return result

    @property
    def count(self) -> int:
        return len(self._tools)
