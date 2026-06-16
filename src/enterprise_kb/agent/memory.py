"""对话记忆模块 — 管理 Agent 的多轮对话上下文。

三种存储层级：
1. 短期记忆：当前会话的原始消息列表（滑动窗口）
2. 工作记忆：关键事实提取（后续可持久化）
3. 长期记忆：持久化存储（后续可用向量数据库）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ConversationMemory:
    """对话记忆 — 管理消息历史与关键信息。

    使用滑动窗口策略：超过 max_turns 时丢弃最早的消息。
    """

    session_id: str
    max_turns: int = 20
    messages: list[dict[str, Any]] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)

    def add_message(self, role: str, content: str, **extra: Any) -> None:
        """添加一条消息到历史记录。

        Args:
            role: user / assistant / tool / system
            content: 消息内容
            extra: 额外字段（如 tool_call_id, name 等）
        """
        msg = {"role": role, "content": content, **extra}
        self.messages.append(msg)
        self._trim()

    def add_tool_message(self, tool_call_id: str, content: str) -> None:
        """添加工具响应消息。"""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })
        self._trim()

    def get_recent(self, n: int = 5) -> list[dict[str, Any]]:
        """获取最近 n 轮对话。"""
        return self.messages[-n * 2:] if self.messages else []

    def get_full_history(self) -> list[dict[str, Any]]:
        """获取完整的消息历史（已修剪过）。"""
        return list(self.messages)

    def add_fact(self, fact: str) -> None:
        """记录一个从对话中提取的关键事实。"""
        if fact not in self.key_facts:
            self.key_facts.append(fact)

    def _trim(self) -> None:
        """滑动窗口裁剪，保留最近 max_turns 轮。"""
        # 每轮可能有 user + assistant + 多个 tool，用 role 计数
        user_count = sum(1 for m in self.messages if m["role"] == "user")
        if user_count > self.max_turns:
            # 找到需要删除的 user 消息数量
            excess = user_count - self.max_turns
            count = 0
            cut_index = 0
            for i, m in enumerate(self.messages):
                if m["role"] == "user":
                    count += 1
                if count > excess:
                    cut_index = i
                    break
            self.messages = self.messages[cut_index:]

    def clear(self) -> None:
        """清空当前会话。"""
        self.messages.clear()
        self.key_facts.clear()


class MemoryManager:
    """全局记忆管理器 — 管理多个会话的记忆。"""

    _sessions: dict[str, ConversationMemory] = {}

    def get_or_create(self, session_id: str, max_turns: int = 20) -> ConversationMemory:
        """获取或创建会话记忆。"""
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(
                session_id=session_id,
                max_turns=max_turns,
            )
        return self._sessions[session_id]

    def delete(self, session_id: str) -> None:
        """删除会话记忆。"""
        self._sessions.pop(session_id, None)

    def clear_all(self) -> None:
        """清空所有记忆。"""
        self._sessions.clear()
