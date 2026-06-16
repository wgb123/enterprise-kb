"""Agent 主循环 — Think → Act → Observe 的自主推理引擎。

核心流程：
1. 接收用户查询 + 对话记忆
2. 构建 messages（系统提示 + 历史 + 当前查询）
3. 调用 LLM（带 tools 参数，启用 Function Calling）
4. 判断响应：
   - 包含 tool_calls → 执行工具 → 结果注入 → 继续循环
   - 纯文本回复 → 返回结果，结束循环
5. 步数保护（默认 max 10 步）
"""

from __future__ import annotations

import json
from typing import Any, Optional

from enterprise_kb.agent.tool_executor import ToolExecutor
from enterprise_kb.agent.tool_registry import ToolRegistry
from enterprise_kb.agent.memory import ConversationMemory
from enterprise_kb.config import settings
from enterprise_kb.utils.logger import logger

# ── Agent 系统提示 ──

AGENT_SYSTEM_PROMPT = """你是一个企业智能助手（AI Agent）。你有能力使用多种工具来获取信息并完成任务。

## 行为准则

1. **思考优先**：在调用工具前，先思考需要哪些信息。
2. **使用工具**：如果用户问题需要外部信息（搜索、计算、查询等），请选择合适的工具。
3. **观察结果**：工具返回的结果是新的观察，据此决定下一步行动。
4. **给出答案**：获取足够信息后，用清晰的语言回答用户问题。
5. **诚实透明**：如果工具无法完成任务，诚实地告知用户。
6. **单步高效**：优先用最少的工具调用解决问题，避免不必要的步骤。

## 工具使用规则

- 一次可以调用多个工具（如果它们独立且都需要）
- 工具的参数请严格按照 JSON Schema 填写
- 如果工具返回错误，尝试用其他方式解决
- 最多执行 10 步工具调用后必须给出最终答案"""


class AgentLoop:
    """Agent 主循环。

    基于「Think → Act → Observe」模式的自主推理引擎。
    不依赖任何框架（LangChain/LlamaIndex），原生 Python 实现。
    """

    def __init__(
        self,
        max_steps: int = 10,
        model_name: str = "",
        api_url: str = "",
        api_key: str = "",
        temperature: float = 0.3,
    ) -> None:
        self.max_steps = max_steps
        self.model_name = model_name or settings.vllm_model_name
        self.api_url = (api_url or settings.vllm_api_url).rstrip("/") + "/chat/completions"
        self.api_key = api_key or settings.vllm_api_key
        self.temperature = temperature or settings.vllm_temperature
        self.timeout = settings.vllm_timeout

        self._registry = ToolRegistry()
        self._executor = ToolExecutor()

        self._step_count = 0
        self._tools_used: list[str] = []

    async def run(
        self,
        query: str,
        memory: Optional[ConversationMemory] = None,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """执行一次 Agent 查询。

        Args:
            query: 用户查询。
            memory: 对话记忆（可选，传入则自动管理历史）。
            system_prompt: 自定义系统提示（可选）。

        Returns:
            {
                "answer": str,           # 最终答案
                "tool_calls": [str],     # 使用的工具列表
                "steps": int,            # 推理步数
                "trace": [dict],         # 完整推理轨迹
            }
        """
        import httpx

        self._step_count = 0
        self._tools_used = []
        trace: list[dict[str, Any]] = []

        # 1. 构建消息列表
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt or AGENT_SYSTEM_PROMPT},
        ]

        # 2. 注入对话历史
        if memory:
            messages.extend(memory.get_full_history())

        # 3. 注入当前查询
        messages.append({"role": "user", "content": query})
        if memory:
            memory.add_message("user", query)

        tools = self._registry.to_openai_tools()

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                while self._step_count < self.max_steps:
                    self._step_count += 1

                    # 4. 调用 LLM
                    payload = {
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": self.temperature,
                        "stream": False,
                    }
                    if tools:
                        payload["tools"] = tools

                    logger.debug(
                        "Agent step %d/%d, messages: %d",
                        self._step_count, self.max_steps, len(messages),
                    )

                    resp = await client.post(
                        self.api_url, json=payload, headers=headers,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    choice = data["choices"][0]
                    msg = choice["message"]

                    # 5. 检查是否有 tool_calls
                    tool_calls = msg.get("tool_calls")
                    if not tool_calls:
                        # LLM 直接回复 → 结束循环
                        answer = msg.get("content", "")
                        messages.append({"role": "assistant", "content": answer})
                        if memory:
                            memory.add_message("assistant", answer)

                        trace.append({
                            "step": self._step_count,
                            "action": "final_answer",
                            "output": answer[:200],
                        })
                        break

                    # 6. 有 tool_calls → 执行工具
                    # 先记录 assistant 的 tool_calls 消息
                    assistant_msg = {
                        "role": "assistant",
                        "content": msg.get("content") or None,
                    }
                    if tool_calls:
                        assistant_msg["tool_calls"] = [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"],
                                },
                            }
                            for tc in tool_calls
                        ]
                    messages.append(assistant_msg)

                    # 批量执行工具
                    tool_results = await self._executor.execute_tool_calls(tool_calls)

                    # 记录轨迹
                    for tc in tool_calls:
                        name = tc["function"]["name"]
                        args = tc["function"]["arguments"][:100]
                        if name not in self._tools_used:
                            self._tools_used.append(name)
                        trace.append({
                            "step": self._step_count,
                            "action": "tool_call",
                            "tool": name,
                            "args": args,
                        })

                    # 注入工具结果到消息列表
                    for tr in tool_results:
                        messages.append(tr)
                        if memory:
                            memory.add_message("tool", tr["content"],
                                                tool_call_id=tr["tool_call_id"])

                # 超出最大步数
                if self._step_count >= self.max_steps:
                    answer = "抱歉，我尝试了多次步骤仍未能完整回答您的问题。请尝试简化或拆分您的查询。"
                    if memory:
                        memory.add_message("assistant", answer)
                    trace.append({
                        "step": self._step_count,
                        "action": "max_steps_reached",
                        "output": answer,
                    })

        except httpx.TimeoutException:
            answer = f"请求超时（{self.timeout}秒），请稍后重试。"
            trace.append({"step": 0, "action": "timeout", "output": answer})

        except Exception as exc:
            answer = f"Agent 执行异常: {exc}"
            logger.exception("Agent loop failed")
            trace.append({"step": 0, "action": "error", "output": str(exc)})

        return {
            "answer": answer,
            "tool_calls": self._tools_used,
            "steps": self._step_count,
            "trace": trace,
        }

    def reset(self) -> None:
        """重置内部状态。"""
        self._step_count = 0
        self._tools_used = []
