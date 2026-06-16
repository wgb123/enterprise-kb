# Agent 层升级计划 — 从知识库系统到 AI Agent 系统

> 目标：在当前 Enterprise KB 的基础上，叠加 Agent 核心能力，
> 使其从一个「RAG 知识库系统」升级为「Agent 驱动的企业智能知识库系统」。
> 覆盖 JD：Dify/LangChain 框架使用、Agent 工程化落地、企业级 RAG 交付。

---

## 当前状态

```
Enterprise KB (已完成)
├── 三层记忆架构 (A/B/C)
├── HybridRAG 检索管道
├── FastAPI REST API
├── Qdrant + BM25 + BGE-Reranker
├── 维基百科批量导入
└── 文档 (README + docs/)
```

**不足之处**：缺少 Agent 核心机制（工具调用、自主规划、执行循环），只能做「问答」不能做「执行」。

---

## 目标架构

```
用户查询
  │
  ▼
┌─────────────────────────────────┐
│         Agent Loop              │
│  Think → Act → Observe → 继续  │  
│  ┌─────────────────────────┐   │
│  │ LLM (Function Calling)  │   │
│  └──────┬──────────────────┘   │
└─────────┼──────────────────────┘
          │
    ┌─────┼─────┬──────────────────┐
    ▼     ▼     ▼       ▼
┌──────┐ ┌────┐ ┌────────┐ ┌──────────┐
│ Wiki  │ │RAG │ │SQL查询 │ │计算器/   │
│搜索   │ │检索 │ │对话日志│ │其他工具  │
└──────┘ └────┘ └────────┘ └──────────┘
    │       │       │           │
    └───────┴───┬───┴───────────┘
                ▼
        ┌──────────────┐
        │ ContextFusion│
        └──────┬───────┘
               ▼
        ┌──────────────┐
        │ LLM 生成答案  │
        └──────────────┘
```

---

## 新增模块结构

```
src/enterprise_kb/agent/          ← 新增
├── __init__.py
├── tool_registry.py   ✅ 已完成   工具注册与发现
├── tool_executor.py   ✅ 已完成   工具执行引擎
├── agent_loop.py      ⏳ 待实现   Agent 主循环
└── memory.py          ✅ 已完成   对话记忆管理
```

### 各模块说明

### 1. Tool Registry (`tool_registry.py`)
**状态：✅ 已完成**

集中管理 Agent 可调用的工具，每个工具包含：
- `name` — 工具名称（LLM 通过此名称调用）
- `description` — 自然语言描述（LLM 选择依据）
- `parameters` — JSON Schema 格式的参数定义
- `handler` — 实际执行的异步函数

支持导出为 OpenAI Function Calling 格式，直接传入 `chat/completions` 的 `tools` 参数。

```python
# 注册工具示例
registry.register(ToolSpec(
    name="wiki_search",
    description="在本地知识库中搜索相关内容",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"}
        },
        "required": ["query"],
    },
    handler=wiki_search_handler,
))
```

### 2. Tool Executor (`tool_executor.py`)
**状态：✅ 已完成**

接收 LLM 返回的 tool_call 请求，解析参数，执行 handler，返回结果字符串。
支持：
- 异步/同步 handler 自动适配
- 批量并行执行
- 异常捕获与工具级错误处理

### 3. Agent Loop (`agent_loop.py`)
**状态：⏳ 待实现**

核心循环：Think → Act → Observe → Continue or End

```
while steps < max_steps:
    1. 构建 messages（含历史 + 新观察）
    2. 调用 LLM（带 tools 参数）
    3. 判断响应:
       - 包含 tool_calls → 执行工具 → 结果反馈 → 下一轮
       - 纯文本回复 → 返回结果，结束循环
    4. 步数超限 → 强制结束
```

需要覆盖：
- 工具选择与执行
- 观察结果注入
- 最大步数保护（默认 10 步）
- 错误恢复
- 与对话记忆集成

### 4. Memory (`memory.py`)
**状态：✅ 已完成**

三层记忆架构：
- **短期记忆**：当前会话的消息列表（滑动窗口，默认 20 轮）
- **工作记忆**：关键事实提取（可持久化到文件/数据库）
- **长期记忆**：预留接口，后续可接入向量数据库

---

## 工具清单（计划注册）

| 工具名称 | 描述 | 参数 | 状态 |
|---|---|---|---|
| `wiki_search` | 搜索本地 A 类知识库 | query: str | ⏳ |
| `rag_search` | 在文档库中检索相关内容 | query: str | ⏳ |
| `calculator` | 执行数学计算 | expression: str | ⏳ |
| `get_time` | 获取当前日期和时间 | 无参数 | ⏳ |
| `translate_text` | 翻译文本 | text, target_lang | 🟡 可选 |

---

## API 端点变更

新增 `/api/v1/agent/chat` 端点：

```json
POST /api/v1/agent/chat
{
    "query": "查找系统架构文档，然后计算一个数字",
    "session_id": "user_001"
}

Response:
{
    "answer": "...",
    "tool_calls": ["wiki_search", "calculator"],
    "steps": 3,
    "session_id": "user_001"
}
```

同时保留现有的 `/api/v1/query` 端点（非 Agent 模式）。

---

## 对外面试价值

| 能力 | 项目体现 | 面试可说的点 |
|---|---|---|
| Function Calling | 自建 Tool Registry + Tool Executor | "我理解了 Function Calling 的本质是工具描述的 JSON Schema + 循环" |
| Agent Loop | `agent_loop.py` Think-Act-Observe | "我手写了 Agent 循环，相比之下理解了 LangChain 的设计取舍" |
| 记忆管理 | `memory.py` 三层记忆 | "我设计了滑动窗口 + 关键事实提取的记忆架构" |
| 工具链扩展 | 注册式架构 | "新增工具只需 3 行代码" |
| 框架理解 | 可与 LangChain/Dify 对比 | "我用原生 Python 实现后，去看了 LangChain 源码，理解了它的 ReAct 模式" |
| 企业级交付 | Docker + API + 文档 | "从 0 到 1 落地了完整的企业级 Agent 系统" |

---

## 后续计划

### 第一批次（当前已完成）
- [x] Tool Registry — 工具注册系统
- [x] Tool Executor — 工具执行引擎
- [x] Memory — 对话记忆管理

### 第二批次（当前已完成）
- [x] Agent Loop — 完整的 Think-Act-Observe 主循环
- [x] 注册内置工具（wiki_search, rag_search, calculator, get_time 共 4 个）
- [x] API 端点 `POST /api/v1/agent/chat`
- [ ] 验证测试（待配好 API Key 后跑通）
- [ ] .gitignore 排除 __pycache__

### 第三批次（可选增强）
- [ ] 自定义 Prompt 模板
- [ ] 流式输出（SSE）
- [ ] 与 LangChain Agent 的对比评测文档
