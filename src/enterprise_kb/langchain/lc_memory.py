"""LangChain 版对话记忆 — 三层记忆。

对照手写版：agent/memory.py → ConversationMemory + MemoryManager

注意：LangChain 1.x 移除了 langchain.memory 模块。
这里直接用原生 Python 实现三层记忆，功能对等于手写版。
"""

from collections import deque
from typing import Any


class LcMemory:
    """三层对话记忆（不依赖 langchain.memory）。

    功能对照：
      手写 ConversationMemory.messages  → 这里用 deque 做滑动窗口
      手写 ConversationMemory.key_facts → 保留 key_facts 列表
    """

    def __init__(self, max_turns: int = 20):
        self._messages: deque[dict[str, Any]] = deque()
        self._max_turns = max_turns
        self._key_facts: list[str] = []

    # ── 短期记忆 ─────────────────────────────────────

    def add_message(self, role: str, content: str, **extra) -> None:
        """添加一条消息。"""
        msg = {"role": role, "content": content, **extra}
        self._messages.append(msg)
        self._trim()

    def add_user_message(self, content: str) -> None:
        self.add_message("user", content)

    def add_ai_message(self, content: str) -> None:
        self.add_message("assistant", content)

    def add_tool_message(self, content: str, tool_call_id: str) -> None:
        self.add_message("tool", content, tool_call_id=tool_call_id)

    def get_history(self) -> list[dict[str, Any]]:
        """获取当前对话历史（浅拷贝，安全使用）。"""
        return list(self._messages)

    def get_history_for_llm(self) -> list[dict[str, Any]]:
        """获取适合传给 LLM API 的消息格式。"""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self._messages
        ]

    def clear(self) -> None:
        """清空短期记忆。"""
        self._messages.clear()

    def _trim(self) -> None:
        """滑动窗口裁剪——只保留最近 max_turns*2 条消息。

        每轮产生 user + assistant 两条消息，所以按 max_turns*2 裁剪。
        """
        max_len = self._max_turns * 2
        while len(self._messages) > max_len:
            self._messages.popleft()

    # ── 工作记忆（key_facts）─────────────────────────

    def add_fact(self, fact: str) -> None:
        """添加关键事实（不会被滑动窗口裁掉）。"""
        if fact not in self._key_facts:
            self._key_facts.append(fact)

    def get_facts(self) -> list[str]:
        return list(self._key_facts)

    def clear_facts(self) -> None:
        self._key_facts.clear()

    @property
    def fact_count(self) -> int:
        return len(self._key_facts)

    @property
    def message_count(self) -> int:
        return len(self._messages)


class LcMemoryManager:
    """记忆管理器（按 session_id 隔离）。"""

    def __init__(self):
        self._sessions: dict[str, LcMemory] = {}

    def get_or_create(self, session_id: str, max_turns: int = 20) -> LcMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = LcMemory(max_turns=max_turns)
        return self._sessions[session_id]

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def clear_all(self) -> None:
        self._sessions.clear()

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)
