"""LangChain 版 Agent — create_agent（LangChain 1.x API）。

对照手写版：agent/agent_loop.py → AgentLoop

API 变化（vs 0.2.x 的 create_openai_functions_agent + AgentExecutor）：
  - create_agent() 内置 AgentExecutor + langgraph，返回 CompiledStateGraph
  - 不用分别创建 agent 和 executor
  - 输入格式：{"messages": [{"role": "user", "content": query}]}
  - 输出格式：含 messages 字段，最后一条 assistant 消息是答案
"""

from typing import Optional

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from enterprise_kb.config import settings
from enterprise_kb.langchain.lc_tools import get_default_tools


AGENT_SYSTEM_PROMPT = """你是一个企业知识库助手。你可以调用工具来获取信息或执行操作。

决策流程：
1. 先理解用户问题需要什么信息
2. 选择最合适的工具去获取
3. 如果工具结果不够，可以继续调其他工具
4. 得到足够信息后组织最终答案

规则：
- 只在有足够信息时回答，不要编造
- 回答时引用来源
- 如果多个来源冲突，优先采信 Wiki 知识库
- 最多调用 10 次工具"""


class LcAgent:
    """LangChain 版 Agent（LangChain 1.x API）。

    使用 create_agent() 替代 create_openai_functions_agent + AgentExecutor。
    内部基于 langgraph，返回 CompiledStateGraph。
    ainvoke 一次 = 手写版 AgentLoop.run() 整个循环。
    """

    def __init__(
        self,
        api_url: str = "",
        api_key: str = "",
        model_name: str = "",
        temperature: float = 0.3,
        max_iterations: int = 10,
    ) -> None:
        self.api_url = (api_url or settings.vllm_api_url).rstrip("/")
        self.api_key = api_key or settings.vllm_api_key
        self.model_name = model_name or settings.vllm_model_name
        self.temperature = temperature or settings.vllm_temperature
        self.max_iterations = max_iterations

        # 1. 创建 LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            base_url=self.api_url,
            api_key=self.api_key,
        )

        # 2. 创建工具
        self.tools = get_default_tools()

        # 3. 创建 Agent（create_agent 返回 CompiledStateGraph）
        self.graph = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=AGENT_SYSTEM_PROMPT,
        )

    async def run(self, query: str) -> dict:
        """运行 Agent。

        输入：用户查询字符串
        输出：{answer, steps, trace}

        对照手写版 AgentLoop.run() 的 while 循环。
        LangChain 1.x 的 create_agent 内部用 langgraph 实现了同样的 Think→Act→Observe。
        """
        result = await self.graph.ainvoke(
            {"messages": [{"role": "user", "content": query}]}
        )

        # 从返回的 messages 中提取最终答案
        messages = result.get("messages", [])
        answer = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content and getattr(msg, "role", "") in ("assistant", "ai"):
                answer = msg.content
                break

        return {
            "answer": answer,
            "steps": len(messages),
            "trace": [
                {
                    "role": getattr(m, "role", str(type(m).__name__)),
                    "content": (m.content[:200] if hasattr(m, "content") and m.content else str(m))[:200],
                }
                for m in messages[-10:]
            ],
        }
