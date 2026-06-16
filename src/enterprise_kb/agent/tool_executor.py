"""工具执行引擎 — 解析 LLM 返回的 tool_call，执行并返回结果。

核心流程：
1. 从 LLM 响应中提取 tool_call（名称 + 参数）
2. 查 ToolRegistry 找到对应的 handler
3. 执行 handler，捕获异常
4. 格式化为 tool message 返回给 LLM
"""

from __future__ import annotations

import json
import traceback
from typing import Any

from enterprise_kb.agent.tool_registry import ToolRegistry
from enterprise_kb.utils.logger import logger


class ToolExecutionError(Exception):
    """工具执行时的异常。"""


class ToolExecutor:
    """工具执行器。

    职责单一：接收 tool_call 请求 → 执行 → 返回结果字符串。
    """

    def __init__(self) -> None:
        self._registry = ToolRegistry()

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """执行一个工具调用。

        Args:
            tool_name: 工具名称。
            arguments: 解析后的参数字典。

        Returns:
            工具执行的文本结果。
        """
        spec = self._registry.get(tool_name)
        if spec is None:
            raise ToolExecutionError(f"未知工具: {tool_name}")

        logger.debug("执行工具 %s, 参数: %s", tool_name, arguments)

        try:
            # 检查 handler 是 async 还是 sync
            if inspect.iscoroutinefunction(spec.handler):
                result = await spec.handler(**arguments)
            else:
                result = spec.handler(**arguments)

            # 确保返回字符串
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
            return result

        except Exception as exc:
            error_msg = f"工具 '{tool_name}' 执行失败: {exc}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise ToolExecutionError(error_msg) from exc

    async def execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """批量执行多个工具调用（并行）。

        每个 tool_calls 元素格式：
        {
            "id": "call_xxx",
            "function": {"name": "...", "arguments": "{...}"}
        }

        Returns:
            tool_message 列表（逐条返回给 LLM）。
        """
        import asyncio

        tasks = []
        for tc in tool_calls:
            call_id = tc["id"]
            func = tc["function"]
            name = func["name"]
            try:
                args = json.loads(func["arguments"])
            except json.JSONDecodeError:
                args = {}

            tasks.append(self._execute_single(call_id, name, args))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        tool_messages = []
        for r in results:
            if isinstance(r, dict):
                tool_messages.append(r)
            else:
                # 异常场景
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": getattr(r, "call_id", ""),
                    "content": f"工具执行异常: {r}",
                })

        return tool_messages

    async def _execute_single(
        self,
        call_id: str,
        name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            result = await self.execute(name, args)
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "content": result,
            }
        except ToolExecutionError as exc:
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "content": str(exc),
            }


# Monkey-patch inspect for the import
import inspect  # noqa: E811
