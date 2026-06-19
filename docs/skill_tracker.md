# 知识掌握度追踪表 — AI Agent 开发转型

> 使用方式：每完成一个等级，将 ❌ 改为 ✅。
>
> **考核规则（AI Coding 时代版）**：
> - **L1 理解**：能用白板/纸笔给别人讲清楚原理和流程
> - **L2 审查**：AI 写了一段代码，你能看懂它在做什么、指出潜在 bug、告诉 AI 在哪里改
> - **L3 指挥**：能给 AI 下达精准的优化/重构指令，独立定位瓶颈并让 AI 帮你解决

最后更新：2026-06-18

---

## 一、RAG 检索增强生成

| 知识点 | L1 理解 | L2 审查 | L3 指挥 | 验证方式 |
|---|---|---|---|---|
| **1.1 文档解析管道** | ✅ | ❌ | ❌ | 讲清楚 PDF/Word/Markdown → 纯文本的全链路 |
| PDF 解析（LlamaParse REST API） | ✅ | ❌ | ❌ | AI 写出 `_upload_pdf` + `_poll_job`，你指出缺了超时处理和错误状态码检查 |
| Word/MD 解析（python-docx/markdown） | ✅ | ❌ | ❌ | AI 写出 `DocumentProcessor.process_file()`，你指出 `try-except` 没区分文件类型 |
| **1.2 文本分块策略** | ✅ | ❌ | ❌ | 讲清楚固定大小 / 语义 / Markdown 标题三种策略的差异和适用场景 |
| 固定 token 分块 | ✅ | ❌ | ❌ | AI 写了 `split_by_tokens`，你指出它没考虑 `chunk_overlap` 会导致上下文断裂 |
| 语义句子分块 | ✅ | ❌ | ❌ | 给 AI 指令："用 spaCy 写一个按语义分割句子的 chunker，保留段落边界" |
| 重叠窗口（overlap） | ✅ | ❌ | ❌ | 给 AI 指令："计算 chunk_size=512, overlap=64 时每一段的起始位置" |
| **1.3 向量嵌入** | ✅ | ❌ | ❌ | 解释文本→向量的过程和余弦相似度的数学意义 |
| BGE-M3 模型加载与调用 | ✅ | ❌ | ❌ | AI 写 `FlagAutoModel.from_finetuned()` + `encode()`，你指出 `normalize_embeddings=True` 的必要性 |
| API 降级模式 | ✅ | ❌ | ❌ | 给 AI 指令："写一个嵌入函数，本地模型失败时自动降级到文本 embedding API" |
| 批处理优化 | ✅ | ❌ | ❌ | 给 AI 指令："当前是逐条调用，改成批量编码，batch_size 设为 32 并加进度条" |
| **1.4 向量数据库** | ✅ | ❌ | ❌ | 讲清楚 Qdrant 的 HNSW 索引和近邻搜索原理 |
| Qdrant 部署与管理 | ✅ | ❌ | ❌ | 独立下载二进制、配置 `config.yaml`、启动、验证 `curl :6333/collections` |
| collection 设计 | ✅ | ❌ | ❌ | 给 AI 指令："设计一个 1024 维 + Cosine 距离的 collection，解释为什么选这些参数" |
| 增删查操作 | ✅ | ❌ | ❌ | AI 写 `upsert` / `search` / `delete`，你指出 `wait=True` 在大量写入时的性能问题 |
| **1.5 关键词检索（BM25）** | ✅ | ❌ | ❌ | 讲清楚 TF-IDF → BM25 的演进，及 BM25 对 IDF 饱和度的改进 |
| rank-bm25 索引构建 | ✅ | ❌ | ❌ | AI 写 `BM25Okapi(corpus)`，你指出 `corpus` 需要先中文分词 |
| 增量更新问题 | ✅ | ❌ | ❌ | 给 AI 指令："当前全量重建太慢，设计一个增量更新方案" |
| **1.6 融合排序（RRF）** | ✅ | ❌ | ❌ | 写/解释 Reciprocal Rank Fusion 公式：$RRF(d) = \sum_{r \in R} \frac{1}{k + r(d)}$ |
| 向量 + BM25 两路融合 | ✅ | ❌ | ❌ | AI 写 `rrf_fusion()`，你指出 `k=60` 这个常数应该做成参数 |
| **1.7 重排序** | ✅ | ❌ | ❌ | 讲清楚交叉编码器（Cross-encoder）vs 双编码器（Bi-encoder）的区别 |
| BGE-Reranker-v2-m3 | ✅ | ❌ | ❌ | AI 写 `compute_score()`，你指出第一次调用要加载模型，应加懒加载缓存 |
| API 降级 | ✅ | ❌ | ❌ | 给 AI 指令："加一个 fallback：模型跑不动时直接按原始分数排序返回" |

---

## 二、LLM API 调用与 Prompt 工程

| 知识点 | L1 理解 | L2 审查 | L3 指挥 | 验证方式 |
|---|---|---|---|---|
| **2.1 OpenAI 兼容 API** | ✅ | ❌ | ❌ | 讲清楚 `/chat/completions` 的请求体结构和三种 role 的分工 |
| 多种 LLM 服务商接入 | ✅ | ❌ | ❌ | AI 写 `_build_payload()`，你指出 `max_tokens` 不同模型阈值不同，应做成配置 |
| 超时、重试、降级 | ✅ | ❌ | ❌ | 给 AI 指令："加一个重试装饰器：超时后等 1s 重试，最多 3 次，全部失败给降级回复" |
| **2.2 Prompt 工程** | ✅ | ❌ | ❌ | 讲清楚 system prompt / user prompt / assistant 的分工和各自负载 |
| 结构化 Prompt 模板 | ✅ | ❌ | ❌ | 给 AI 指令："写一个 RAG 问答的 prompt，标记出上下文注入位、指令区和输出格式" |
| Few-shot 设计 | ✅ | ❌ | ❌ | 给 AI 指令："LLM 总是忘了引用来源，写 3 个 few-shot 示例纠正" |
| **2.3 流式输出（SSE）** | ✅ | ❌ | ❌ | 讲清楚 `stream=True` 的工作原理，对比 SSE 和 WebSocket 的取舍 |

---

## 三、Agent 核心机制

| 知识点 | L1 理解 | L2 审查 | L3 指挥 | 验证方式 |
|---|---|---|---|---|
| **3.1 Function Calling** | ✅ | ❌ | ❌ | 讲清楚 LLM 如何通过 tools 参数「选择」并返回函数调用 |
| Tool JSON Schema 设计 | ✅ | ❌ | ❌ | AI 写了一个搜索工具的 function 定义，你指出 `description` 太短会影响 LLM 的选择准确率 |
| tool_call 响应解析 | ✅ | ❌ | ❌ | AI 写了解析逻辑，你指出 `tool_calls` 为空时应该直接走回复逻辑而不是抛异常 |
| tool 结果注入 | ✅ | ❌ | ❌ | AI 写 `{"role": "tool", "tool_call_id": "..."}`，你指出结果太长则应截断 |
| **3.2 Agent Loop** | ✅ | ❌ | ❌ | 讲清楚 Think-Act-Observe 循环的完整流程 |
| 消息列表管理 | ✅ | ❌ | ❌ | AI 写 Agent 循环，你指出 `system prompt` 重复注入会导致幻觉 |
| 步数保护与提前终止 | ✅ | ❌ | ❌ | 给 AI 指令："Agent 会死循环，加一个 max_steps=10 的保护" |
| 异常恢复 | ✅ | ❌ | ❌ | 给 AI 指令："工具执行失败时，把错误信息返回给 LLM，让 LLM 决定下一步" |
| **3.3 工具注册系统** | ✅ | ❌ | ❌ | 讲清楚注册表模式（Registry Pattern）的价值和适用场景 |
| 动态注册与发现 | ✅ | ❌ | ❌ | AI 写 `register()`，你指出缺少去重检查，重复注册时应该抛异常 |
| **3.4 对话记忆** | ✅ | ❌ | ❌ | 讲清楚短期记忆 / 工作记忆 / 长期记忆三层的区别 |
| 滑动窗口裁剪 | ✅ | ❌ | ❌ | AI 写 `_trim()`，你指出它只按消息数量裁剪，没考虑 token 长度 |
| 关键事实提取 | ✅ | ❌ | ❌ | 给 AI 指令："设计一个提取对话中关键事实的机制" |

---

## 四、企业级工程化

| 知识点 | L1 理解 | L2 审查 | L3 指挥 | 验证方式 |
|---|---|---|---|---|
| **4.1 FastAPI 后端** | ✅ | ❌ | ❌ | 讲清楚路由注册、Depends 注入、lifespan 生命周期的协作 |
| Pydantic v2 模型 | ✅ | ❌ | ❌ | AI 写了一个请求体模型，你指出 `min_length` 和 `max_length` validation 没加 |
| 异常处理（HTTPException） | ✅ | ❌ | ❌ | 给 AI 指令："给所有路由加统一的异常处理器，返回 JSON 格式的错误" |
| Swagger 文档 | ✅ | ❌ | ❌ | 启动应用，访问 `/docs`，验证所有端点 |
| **4.2 配置管理** | ✅ | ❌ | ❌ | 讲清楚 pydantic-settings 从 `.env` 加载配置的机制 |
| 多环境配置 | ✅ | ❌ | ❌ | 给 AI 指令："设计 dev / staging / prod 三套配置，适配不同的 .env 文件" |
| **4.3 日志与监控** | ✅ | ❌ | ❌ | 讲清楚结构化日志和不同日志级别的应用场景 |
| 标准 logging | ✅ | ❌ | ❌ | AI 写 log 格式，你指出少了 `request_id`，排查问题时无法串联日志 |
| **4.4 测试** | ✅ | ❌ | ❌ | 讲清楚 pytest fixture / parametrize / mock 的用途 |
| 单元测试 | ✅ | ❌ | ❌ | AI 写 `test_chunker`，你指出边界条件（空文本 / 单句）没覆盖 |
| Mock LLM 调用 | ✅ | ❌ | ❌ | 给 AI 指令："用 `unittest.mock.patch` 模拟 LLM API 返回" |
| **4.5 部署** | ✅ | ❌ | ❌ | 讲清楚 Dockerfile 的 FROM/COPY/RUN/CMD 每层的作用 |
| Docker 容器化 | ✅ | ❌ | ❌ | 给 AI 指令："写 FastAPI + Qdrant 的 docker-compose.yml" |
| 环境变量注入 | ✅ | ❌ | ❌ | AI 写 docker-compose，你指出敏感信息应走 `env_file` 而不是硬编码 |

---

## 五、框架理解（LangChain / LlamaIndex / Dify）

| 知识点 | L1 理解 | L2 审查 | L3 指挥 | 验证方式 |
|---|---|---|---|---|
| **5.1 LangChain** | ❌ | ❌ | ❌ | 讲清楚 Chain/Agent/Tool/Runnable 的核心抽象 |
| LCEL（LangChain Expression Language） | ❌ | ❌ | ❌ | AI 写 `chain = prompt \| llm \| parser`，你指出 `str_output_parser` 的局限性 |
| Runnable 接口 | ❌ | ❌ | ❌ | 讲清楚 `invoke` / `batch` / `stream` 三种调用方式的差异 |
| **5.2 LlamaIndex** | ❌ | ❌ | ❌ | 讲清楚 Node/Document/Index 三层抽象的关系 |
| RouterQueryEngine | ❌ | ❌ | ❌ | AI 用 LlamaIndex 写 RouterQueryEngine，你指出 `description` 质量决定了路由准确率 |
| 手写 vs 框架对比 | ❌ | ❌ | ❌ | 给 AI 指令："写一份对比表，列出手写 Router 和 LlamaIndex Router 的 5 个差异" |
| **5.3 Dify** | ❌ | ❌ | ❌ | 讲清楚 Dify 的工作流 / Agent / Chatflow 三种模式的差异 |
| 工作流 vs Agent vs Chatflow | ❌ | ❌ | ❌ | 给 AI 指令："分别给出三种模式的典型应用场景" |
| **5.4 框架 vs 手写** | ❌ | ❌ | ❌ | 能在面试中说清楚「什么时候用框架、什么时候自己写」的决策准则 |

---

## 六、软技能与面试准备

| 知识点 | L1 理解 | L2 审查 | L3 指挥 | 验证方式 |
|---|---|---|---|---|
| **6.1 项目叙事** | ❌ | ❌ | ❌ | 能用 5 分钟讲清楚项目的痛点、方案和成果 |
| 从 0 到 1 的落地过程 | ❌ | ❌ | ❌ | 准备一个 STAR 格式的项目故事（准备好，不是即兴） |
| 技术选型理由 | ❌ | ❌ | ❌ | 讲清楚「为什么用 Qdrant 而不是 Milvus」等选型理由 |
| **6.2 代码规范** | ❌ | ❌ | ❌ | 讲清楚 PEP 8 / ruff / mypy 各自解决什么问题 |
| 类型注解 | ❌ | ❌ | ❌ | AI 写了一段没类型注解的代码，你告诉它哪里该加注解 |
| **6.3 调试与排错** | ❌ | ❌ | ❌ | 给 AI 指令："Qdrant 提示 connection refused，列出排查步骤" |

---

## 七、Dota 2 AI Agent（Rust 工程化 + LLM 回测）

| 知识点 | L1 理解 | L2 审查 | L3 指挥 | 验证方式 |
|---|---|---|---|---|
| **7.1 Rust + Axum 高性能服务** | ❌ | ❌ | ❌ | 讲清楚 Rust 的所有权模型如何保证 Web 服务的线程安全 |
| Axum 路由与状态管理 | ❌ | ❌ | ❌ | AI 写 Axum router，你指出 `Arc<AppState>` 需要在多个 handler 间共享 |
| SSE 实时进度推送 | ❌ | ❌ | ❌ | AI 写 `EventSource` 推送，你指出断连重连时 `Last-Event-ID` 没处理 |
| 异步并发（Tokio） | ❌ | ❌ | ❌ | 给 AI 指令："用 tokio::spawn 并发请求 5 个 API，全部成功后合并结果" |
| **7.2 多数据源并发聚合** | ❌ | ❌ | ❌ | 讲清楚如何同时拉取 STRATZ / OpenDota / Liquipedia 并去重 |
| GraphQL 客户端 | ❌ | ❌ | ❌ | AI 写 GraphQL query，你指出变量绑定和 fragments 复用的优化点 |
| REST API 限流与错误处理 | ❌ | ❌ | ❌ | AI 写 OpenDota 调用，你指出 429 状态码没做退避重试 |
| 数据缓存策略 | ❌ | ❌ | ❌ | 给 AI 指令："设计一个 TTL 缓存，英雄数据 1h，选手数据 24h" |
| **7.3 LLM 多 Provider 切换** | ❌ | ❌ | ❌ | 讲清楚 MiMo / DeepSeek 等不同 Provider 的协议差异 |
| Provider 抽象层 | ❌ | ❌ | ❌ | AI 写 `LlmProvider` trait，你指出重试策略应该在各 provider 实现中而非外部 |
| 动态切换机制 | ❌ | ❌ | ❌ | AI 写前端下拉框切换 Provider，你指出热切换时游标管理未处理 |
| **7.4 LLM Prompt 回测体系** | ❌ | ❌ | ❌ | **讲清楚为什么需要 prompt 回测、回测的核心指标** |
| Prompt 归档存储 | ❌ | ❌ | ❌ | AI 写 prompt 持久化，你指出缺少 `prompt_tokens` / `completion_tokens` 统计 |
| 回测对比分析 | ❌ | ❌ | ❌ | 给 AI 指令："比较同一阵容在不同 LLM Provider 下的输出质量差异" |
| 评估指标设计（准确率/完整性/幻觉率） | ❌ | ❌ | ❌ | 给 AI 指令："设计一套 LLM 输出质量评估的打分标准" |
| **7.5 Agent 决策流程** | ❌ | ❌ | ❌ | 讲清楚游戏 Agent 和 LLM Agent 的观察-决策-反馈循环异同 |
| 多步骤 LLM 分析流水线 | ❌ | ❌ | ❌ | AI 写 `matchup → players → llm` 三步流程，你指出每步之间缺少状态传递 |
| SSE 进度跟踪 | ❌ | ❌ | ❌ | AI 写 SSE EventSource，你指出进度消息缺少 `step_id` 前端无法追踪具体哪一步 |
| **7.6 项目差异化价值** | ❌ | ❌ | ❌ | 能说清楚这个项目解决的是「真实世界的预测市场问题」而不是 toy project |
| Polymarket 赔率集成 | ❌ | ❌ | ❌ | 讲清楚如何将 LLM 分析结论映射到预测市场的赔率对比 |
| Rust vs Python 技术选型 | ❌ | ❌ | ❌ | 能说出为什么用 Rust 而不是 Python：GSI 实时流 + 并发数据聚合的性能要求 |

## 进度统计

| 模块 | 总数 | L1 ✅ | L2 ✅ | L3 ✅ | 完成率 |
|---|---|---|---|---|---|
| 一、RAG 检索增强生成 | 21 | 19 | 0 | 0 | 90% |
| 二、LLM API 与 Prompt | 7 | 7 | 0 | 0 | 100% |
| 三、Agent 核心机制 | 12 | 12 | 0 | 0 | 100% |
| 四、企业级工程化 | 14 | 14 | 0 | 0 | 100% |
| 五、框架理解 | 9 | 0 | 0 | 0 | 0% |
| 六、软技能与面试 | 7 | 0 | 0 | 0 | 0% |
| 七、Dota 2 AI Agent | 18 | 0 | 0 | 0 | 0% |
| **总计** | **88** | **52** | **0** | **0** | **59%** |

---

## 使用建议

1. **每天 2-3 个知识点**，从 L1 开始，不跳级
2. **L1 理解**：对着空气讲一遍，录下来自己听，看能不能 1 分钟把一件事讲清楚
3. **L2 审查**：打开项目对应代码，让 AI 重新写一遍，然后挑毛病——能指出问题的就算过关
4. **L3 指挥**：自然语言描述你的需求，看 AI 写出来的代码是否符合预期——不符合就说明你的「指挥」还不够精确
5. **每周日更新统计表**
