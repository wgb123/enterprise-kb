#!/usr/bin/env python3
"""LangChain vs 手写 Agent 对照 demo。

目的：亲眼看看 LangChain 内部生成的 messages 序列长什么样，
     和你手写的 AgentLoop 代码做对比。

运行方式：
  .venv/bin/python scripts/demo_langchain_agent.py

不需要配置 API Key，只展示消息结构和流程。
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# ═══════════════════════════════════════════════
# 第一部分：看看 LangChain Agent 内部的消息序列
# ═══════════════════════════════════════════════

def langchain_messages_demo():
    """模拟 LangChain AgentExecutor 内部生成的消息序列。

    这是在 LangChain 源码（agent_executor.py）里实际发生的流程。
    你可以打开 .venv 里看看源码：
    .venv/lib/python3.10/site-packages/langchain/agents/agent_executor.py
    """
    print("=" * 70)
    print("1. LangChain AgentExecutor 内部消息流")
    print("=" * 70)
    print()
    print("场景：用户问「现在几点？」，Agent 调用 get_current_time 工具")
    print()

    # Step 1: 初始消息
    messages = [
        {"role": "system", "content": "你是一个助手，可以调用工具来回答问题。"},
        {"role": "user", "content": "现在几点了？"},
    ]
    print("Step 1 - 初始消息列表:")
    for m in messages:
        print(f"  {m['role']:>10}: {m['content'][:60]}...")
    print()

    # Step 2: LLM 返回 tool_call（LangChain 内部叫 AgentAction）
    tool_call_response = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_time",
                    "arguments": "{}",
                },
            }
        ],
    }
    messages.append(tool_call_response)

    print("Step 2 - LLM 决定调用 get_time 工具:")
    print(f"  → AgentAction: tool=get_time, args={{}}")
    print(f"  （LangChain 内部把 tool_call 包装为 AgentAction 对象）")
    print()

    # Step 3: 执行工具，返回结果（LangChain 内部叫 AgentFinish 前的 Observation）
    tool_result = {
        "role": "tool",
        "tool_call_id": "call_abc123",
        "content": "当前时间: 2026-06-16 14:30:00",
    }
    messages.append(tool_result)

    print("Step 3 - 工具执行结果:")
    print(f"  → Observation: {tool_result['content']}")
    print(f"  （LangChain 内部包装为 AgentStep 对象）")
    print()

    # Step 4: LLM 给出最终答案
    final = {
        "role": "assistant",
        "content": "现在是 2026年6月16日 14:30。",
    }
    messages.append(final)

    print("Step 4 - LLM 最终回答:")
    print(f"  → AgentFinish: {final['content']}")
    print()

    print("←─ 完整消息序列（共 5 条消息）─→")
    for m in messages:
        role = m["role"]
        content = m.get("content", "") or ("[tool_call: " + m["tool_calls"][0]["function"]["name"] + "]" if "tool_calls" in m else "")
        print(f"  [{role:>10}] {content[:60]}")
    print()


# ═══════════════════════════════════════════════
# 第二部分：你手写的 AgentLoop 对比
# ═══════════════════════════════════════════════

def hand_written_agent_comparison():
    """拿出你项目里的 AgentLoop 代码做对比。

    不用跑起来，直接看代码逻辑的对应关系。
    """

    agent_loop_code = """
  ╔══════════════════════════════════════╗
  ║  enterprise_kb/agent/agent_loop.py  ║
  ╚══════════════════════════════════════╝

  while self._step_count < self.max_steps:
                              ←── LangChain 的 AgentExecutor.__call__()

      resp = await client.post(...)
      data = resp.json()
      msg = data["choices"][0]["message"]

      tool_calls = msg.get("tool_calls")
                              ←── LangChain 的 Agent.plan() → AgentAction

      if not tool_calls:
          break              ←── LangChain 的 AgentFinish

      tool_results = await self._executor.execute_tool_calls(tool_calls)
                              ←── LangChain 的 Tool.run()

      messages.append(tr)    ←── LangChain 的 intermediate_steps.append()
"""

    print("=" * 70)
    print("2. 手写 AgentLoop 与 LangChain 的逐行对照")
    print("=" * 70)
    print()
    print("下面是你项目里的代码，右边注释是对应的 LangChain 概念：")
    print(agent_loop_code)
    print()


# ═══════════════════════════════════════════════
# 第三部分：关键结论
# ═══════════════════════════════════════════════

def conclusion():
    print("=" * 70)
    print("3. 结论")
    print("=" * 70)
    print()
    print("  LangChain 的 AgentExecutor 内部做的三件事：")
    print()
    print("  1. plan()  → LLM 决定调哪个工具 = AgentAction")
    print("               ← 你代码里的 tool_calls 解析")
    print()
    print("  2. tool.run() → 执行工具 = Observation")
    print("               ← 你代码里的 _executor.execute_tool_calls()")
    print()
    print("  3. 检查返回 = AgentFinish 还是继续循环")
    print("               ← 你代码里的 if not tool_calls: break")
    print()
    print("  · 核心逻辑完全相同，LangChain 只是多包了一层抽象")
    print("  · 你吃透了本质，不需要学 API")
    print("  · 面试时可以说：我手写了一个等效实现,随后对比了 LangChain 源码")
    print()


# ═══════════════════════════════════════════════
# 第四部分：【可选】真的跑一下 LangChain
# ═══════════════════════════════════════════════

def run_langchain_live():
    """如果真的配了 API Key，跑这个看看真实的 LangChain 输出。

    需要 .env 中配置 VLLM_API_KEY
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("VLLM_API_KEY")
    api_url = os.getenv("VLLM_API_URL", "https://api.deepseek.com/v1")
    model = os.getenv("VLLM_MODEL_NAME", "deepseek-chat")

    if not api_key:
        print("  ⚠️  未配置 VLLM_API_KEY，跳过真实调用")
        print("  配置后运行: python scripts/demo_langchain_agent.py --live")
        return

    print()
    print("=" * 70)
    print("4.【可选】使用 LangChain 调用真实 LLM")
    print("=" * 70)
    print()

    from langchain_openai import ChatOpenAI
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain.tools import tool
    from langchain_core.prompts import ChatPromptTemplate
    import datetime

    # 定义一个工具
    @tool
    def get_current_time() -> str:
        """返回当前日期和时间"""
        now = datetime.datetime.now()
        return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"

    # 创建 LLM 和 Agent
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=api_url,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个助手，可以调用工具来回答问题。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, [get_current_time], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[get_current_time], verbose=True)

    try:
        print("  调用 LangChain AgentExecutor.verbose=True 模式")
        print("  ---------------------------------------------")
        print("  可以看到每步的 AgentAction 和 Observation：")
        print()
        result = agent_executor.invoke({"input": "现在几点了？"})
        print()
        print(f"  最终回答: {result['output']}")
    except Exception as exc:
        print(f"  调用失败: {exc}")


if __name__ == "__main__":
    langchain_messages_demo()
    hand_written_agent_comparison()
    conclusion()

    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        run_langchain_live()
    else:
        print("  提示: 加 --live 参数可实际调用 LLM (需配置 VLLM_API_KEY)")
        print()
