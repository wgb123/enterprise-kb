"""LangChain 并行实现 — 与手写版功能对等。

对比手写版：
  core/generator.py       → lc_generator.py  (ChatOpenAI + Pipeline)
  agent/tool_registry.py  → lc_tools.py      (@tool + StructuredTool)
  agent/agent_loop.py     → lc_agent.py      (AgentExecutor)
  agent/memory.py         → lc_memory.py     (ConversationBufferWindowMemory)
  core/retriever.py       → lc_retriever.py  (EnsembleRetriever)

从手写版迁移只需改 import 路径。
"""

from .lc_generator import LcGenerator
from .lc_tools import get_default_tools
from .lc_agent import LcAgent

__all__ = ["LcGenerator", "get_default_tools", "LcAgent"]
