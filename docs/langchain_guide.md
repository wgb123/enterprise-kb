# LangChain 理解指南 — 与手写 Agent 对照

> **阅读方式**：左边的 LangChain 概念用白话解释，右边是你已经在当前项目里手写过的东西。等你理解了「原来这个类做的是我已经写过的那件事」，面试时就能说清楚「我理解 LangChain 的核心抽象，但根据场景选择手写」。

---

## 一、核心抽象速查表

| LangChain 概念 | 一句话白话 | 当前项目里的对应 | 你已掌握 |
|---|---|---|---|
| **Chain** | 把多个步骤串成一条线 | `AgentLoop.run()` 里的 while 循环 | ✅ |
| **Tool** | LLM 可调用的函数 | `ToolSpec` + `register()` | ✅ |
| **Agent / AgentExecutor** | 管理 Think-Act-Observe 循环 | `AgentLoop` 整类 | ✅ |
| **Memory** | 保存对话历史 | `ConversationMemory` + `MemoryManager` | ✅ |
| **Retriever** | 从知识库里搜东西 | `HybridRetriever.retrieve()` | ✅ |
| **VectorStore** | 存向量的地方 | `QdrantStore` | ✅ |
| **Prompt Template** | 把变量塞进固定模板 | `DEFAULT_SYSTEM_PROMPT` + `_build_user_prompt()` | ✅ |
| **Document Loader** | 把 PDF/Word 读成文本 | `DocumentProcessor.process_file()` | ✅ |
| **Text Splitter** | 把长文本切成小块 | `Chunker.chunk()` | ✅ |
| **Embeddings** | 把文本变成向量 | `BGEM3Embedder.embed()` | ✅ |
| **Runnable（LCEL）** | 管道操作符 `|` 串联步骤 | 无直接对应（手写顺序调用） | ⚠️ |

---

## 二、五大核心抽象深入

### 1. Chain（链）— 就是把步骤串起来

#### LangChain 做了什么

```python
# LangChain LCEL 写法
chain = (
    prompt
    | llm
    | StrOutputParser()
)
result = chain.invoke({"question": "什么是 RAG？"})
```

这里的 `|` 就是一个**管道操作符**，把上一步的输出自动传给下一步。本质上就是：

```python
# 等价的手写代码
prompt_text = prompt.format(question="什么是 RAG？")
llm_output = llm.invoke(prompt_text)
result = StrOutputParser().parse(llm_output)
```

#### 你的项目里对应的

你没有用管道操作符，而是直接写的顺序调用：

```python
# enterprise_kb/core/generator.py
context_text = self._format_context(context_chunks)
user_prompt = self._build_user_prompt(query, context_text)

payload = {
    "model": self.model_name,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
}
resp = await client.post(self.api_url, json=payload, headers=headers)
answer = data["choices"][0]["message"]["content"]
```

> **面试话术**：LangChain 的 Chain 本质上就是执行顺序把函数的输入输出串起来。我没用它的管道语法，因为手写的流程更透明——每一步的输入输出我都能直接看到和调试。但完全理解 LCEL 的设计思路。

---

### 2. Tool（工具）— 就是 LLM 能调用的函数

#### LangChain 做了什么

```python
# LangChain 定义一个工具
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """执行数学计算。"""
    return str(eval(expression))
```

LangChain 会从函数的参数类型推断 JSON Schema，从 docstring 生成 `description`。

#### 你的项目里对应的

你手写了完整的工具注册系统，比 LangChain 的 `@tool` 更显式、更可控：

```python
# enterprise_kb/agent/tool_registry.py
ToolSpec(
    name="calculator",
    description="执行数学计算。支持加减乘除、幂运算、括号。",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "数学表达式"}
        },
        "required": ["expression"],
    },
    handler=handler_calculator,
)

# enterprise_kb/agent/tool_registry.py 第 79 行
def to_openai_tools(self) -> list[dict]:
    """生成 OpenAI Function Calling 格式的工具列表。"""
```

> **面试话术**：LangChain 用装饰器自动推断工具 schema，我用的是显式注册的 ToolSpec。显式注册更适合复杂场景——比如你想给同一个函数注册两个不同的工具名，或者需要精确控制 JSON Schema 的 `required` 字段。

---

### 3. Agent / AgentExecutor — 就是 Think-Act-Observe 循环

#### LangChain 做了什么

```python
# LangChain Agent
from langchain.agents import create_openai_functions_agent, AgentExecutor

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, max_iterations=10)
result = agent_executor.invoke({"input": "现在几点？"})
```

LangChain 的 `AgentExecutor` 内部就是一个 `while` 循环：

```python
# LangChain 源码核心逻辑（简化）
while steps < max_iterations:
    output = self.agent.plan(intermediate_steps)
    if output is AgentFinish:
        return output
    # AgentAction → 执行工具 → 观察结果 → 继续循环
    observation = tool.run(action)
    intermediate_steps.append((action, observation))
```

#### 你的项目里对应的

你手写的 Agent Loop 逻辑几乎一模一样：

```python
# enterprise_kb/agent/agent_loop.py 第 144 行
while self._step_count < self.max_steps:
    self._step_count += 1

    # 调用 LLM（带 tools 参数）
    resp = await client.post(self.api_url, json=payload, headers=headers)
    data = resp.json()
    choice = data["choices"][0]
    msg = choice["message"]

    # 检查是否有 tool_calls
    tool_calls = msg.get("tool_calls")
    if not tool_calls:
        # LLM 直接回复 → 结束循环
        answer = msg.get("content", "")
        break

    # 执行工具 → 观察结果 → 注入消息列表 → 继续循环
    tool_results = await self._executor.execute_tool_calls(tool_calls)
    for tr in tool_results:
        messages.append(tr)
```

> **面试话术**：LangChain 的 AgentExecutor 和我手写的 AgentLoop 核心逻辑一致——都是 while 循环 + tool_call 检测 + 结果注入。我选择手写的原因是在极早期的开发阶段就能精确控制每一步——比如工具执行失败时是抛异常还是返回错误信息给 LLM，这些细节框架里不好控制。

---

### 4. Memory — 就是保存对话历史

#### LangChain 做了什么

```python
# LangChain Memory
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=5)  # 只保留最近 5 轮
memory.save_context({"input": "你好"}, {"output": "你好！"})
memory.load_memory_variables({})  # → {"history": "Human: 你好\nAI: 你好！"}
```

#### 你的项目里对应的

你手写的 `ConversationMemory` 功能更丰富：

```python
# enterprise_kb/agent/memory.py
class ConversationMemory:
    session_id: str
    max_turns: int = 20                    # ← 对应 ConversationBufferWindowMemory
    messages: list[dict]                   # ← 存储原始消息
    key_facts: list[str]                   # ← LangChain 没有的"关键事实提取"

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        self._trim()  # 滑动窗口裁剪

    def get_full_history(self) -> list[dict]:
        return list(self.messages)
```

> **面试话术**：LangChain 的 ConversationBufferWindowMemory 只能自动拼接 Human/AI 对话。我手写的版本保留了完整的 OpenAI 消息格式（包括 tool role），而且加了滑动窗口裁剪和关键事实提取。这对 Agent 多步推理时的上下文管理更友好。

---

### 5. Retriever — 就是检索知识库

#### LangChain 做了什么

```python
# LangChain Retriever
from langchain_community.vectorstores import Qdrant
from langchain.retrievers import BM25Retriever

# 向量检索
retriever = Qdrant(embeddings=embeddings, ...).as_retriever()

# 关键词检索
bm25_retriever = BM25Retriever.from_documents(docs)

# 混合检索
ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever, bm25_retriever],
    weights=[0.5, 0.5],
)
```

#### 你的项目里对应的

你手写的 `HybridRetriever` 用的是 **RRF 融合**而不是权重：

```python
# enterprise_kb/core/retriever.py
class HybridRetriever(BaseRetriever):
    async def retrieve(self, query: str):
        # 向量检索
        query_vec = await self.embedder.embed([query])
        semantic_results = await self.vector_store.search(query_vec[0])

        # BM25 关键词检索
        bm25_results = self.bm25_index.search(query)

        # RRF 融合（vs LangChain 用 weights）
        fused = self._rrf_fusion(semantic_results, bm25_results, k=60)

        # BGE-Reranker 精排
        reranked = await self.reranker.rerank(query, fused)

        return reranked
```

> **面试话术**：LangChain 的 EnsembleRetriever 用 weights 加权融合，我用的 RRF 不需要调参，k=60 在不同数据集上表现稳定。而且我在 RRF 之后又多接了一层 BGE-Reranker 交叉编码重排序。LangChain 也有 ContextualCompressionRetriever 可以接 reranker，但我手写的管道更清晰。

---

## 三、LangChain 中有、你还没有的东西

这些是 LangChain 的差异化能力，值得了解（虽然不一定要用）：

| LangChain 能力 | 白话解释 | 面试时可以说 |
|---|---|---|
| **LCEL（LangChain Expression Language）** | 用 `\|` 管道符串联步骤，自动并行 | "理解设计，但手写的顺序调用更透明" |
| **Hub（Prompt Hub）** | 社区 prompt 模板仓库 | "理解理念，但我们场景的 prompt 高度定制" |
| **Callback 系统** | 每一步执行前/后自动触发钩子 | "可以用装饰器自己实现，不需要框架" |
| **Multi-Agent** | 多个 Agent 相互对话/竞争 | "单 Agent + 多工具 更可控" |
| **Streaming 支持** | 流式输出、中间过程展示 | "Rust 项目里已实现 SSE" |
| **LangSmith** | 生产环境的 LLM 调用追踪 | "Prompt 回测体系等效于 LangSmith 的核心功能" |

---

## 四、面试黄金话术

面试官问："你对 LangChain 了解吗？"

> **标准回答**：
> "我理解 LangChain 的核心抽象——Chain、Tool、Agent、Memory、Retriever。
> 我的两个项目里，Agent Loop 和 RAG 管道是自己手写的，因为手写能精确控制每一步的输入输出和异常处理。
>
> **但我在开发过程中也深入对比了 LangChain 的实现**：
> - 它的 `AgentExecutor` 和我手写的 `AgentLoop` 核心逻辑一致
> - 我手写的 ToolRegistry 比它的 `@tool` 装饰器更显式
> - 我用的 RRF 融合比它的 EnsembleRetriever 的 weights 更稳定
>
> 我选择手写的原因是希望把 Agent 的核心机制吃透。如果有团队用 LangChain 做技术栈，我也能快速上手——我知道它的抽象层怎么映射到实际代码。"

**效果**：展示了理解 + 对比能力 + 务实态度，三连击。

---

## 五、快速上手指南

如果你需要真的跑一下 LangChain（理解而不是依赖），半小时足够：

```bash
# 1. 安装最小依赖（不要装全量）
pip install langchain-core langchain-openai

# 2. 写一个 10 行的 Agent（对照你已有的 AgentLoop）
# 3. 观察它内部生成的 messages 序列
# 4. 和你手写的 messages 列表对比
```

**重点不是"学会用 LangChain"，而是"能说出 LangChain 每一步内部做了什么"**——这就是面试里的理解深度。
