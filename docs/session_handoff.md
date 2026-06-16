# 会话手稿 — Enterprise KB Agent 转型项目

> 最后更新：2026-06-16 17:20
> 技术栈：Python 3.10 + FastAPI + Qdrant + BGE-M3 + DeepSeek API (deepseek-v4-pro)
> 参与项目：enterprise-kb（Python）、dota2-ai-agent（Rust）

---

## 项目真正目的

**这不是一个"交付项目"，而是一个学习/转岗平台。**

Enterprise KB 是你的动手实验室——通过手写 Agent 系统的每个模块，深入理解 AI Agent 开发，为转岗 AI Agent 开发（Python 后端）做准备。

### 项目→面试映射

```
你通过这个项目理解               面试时你能讲清楚
─────────────────────────────────────────────────────
Agent Loop (Think-Act-Observe) → 为什么这样设计，不依赖框架
Function Calling                → Tool JSON Schema 如何影响 LLM 选择
Tool Registry (单例模式)        → 注册表模式的价值
Memory 滑动窗口 + 三层记忆      → 短期/工作/长期记忆的分工
HybridRAG (向量+BM25+RRF+重排)  → 为什么双通道，RRF 为什么不用调参
手写 vs 框架                    → 什么时候用框架、什么时候手写
Dota 2 Agent (Rust)             → LLM 回测体系、多数据源并发聚合
```

### 已有背景

- Python RAG + Agent（enterprise-kb）
- Rust 高并发（dota2-ai-agent）
- Docker / Linux
- Java 后端背景

### 缺的

- MySQL/PostgreSQL 经验（JD 写了"了解即可"）

---

## 已完成

### 项目搭建
- [x] Python 虚拟环境 `.venv`（pip 安装）
- [x] 核心依赖：torch (CPU-only) + FlagEmbedding + qdrant-client + llama-cloud
- [x] Qdrant 1.18.2 二进制部署（musl 静态编译）
- [x] `.env.example` 配置模板
- [x] 维基百科批量导入脚本 `scripts/wiki_importer.py`（断点续传 + BM25 + 摘要向量）
- [x] LangChain / llama-index 已安装（`scripts/demo_*.py` 对比脚本就绪）

### Agent 层（完整实现）
- [x] `tool_registry.py` — 工具注册系统（单例 + OpenAI 格式导出）
- [x] `tool_executor.py` — 工具执行引擎（异步/同步 + 并行批量）
- [x] `agent_loop.py` — Agent 主循环（Think-Act-Observe + 步数保护）
- [x] `memory.py` — 对话记忆（滑动窗口 20 轮 + 关键事实提取 + 多会话管理）
- [x] `default_tools.py` — 4 个内置工具（get_time / calculator / wiki_search / rag_search）
- [x] API 端点 `POST /api/v1/agent/chat`（新增 `AgentChatRequest/Response` schema）
- [x] `main.py` 启动时自动注册工具

### 服务启动
- [x] Qdrant 服务启动（端口 6333）
- [x] FastAPI 应用启动（端口 8000）
- [x] 4 个 Agent 工具自动注册
- [x] Agent 端点验证：get_time ✅ / calculator ✅

### 文档
- [x] `docs/architecture.md` — 架构设计
- [x] `docs/development.md` — 开发指南
- [x] `docs/operations.md` — 运维手册
- [x] `docs/agent_upgrade_plan.md` — Agent 升级计划
- [x] `docs/skill_tracker.md` — 88 个知识点考核追踪表（7 个模块）
- [x] `docs/interview_prep.md` — 面试三道必考题话术
- [x] `docs/langchain_guide.md` — LangChain 对照理解
- [x] `docs/session_handoff.md` — 本文件
- [x] `docs/learning_notes.md` — **新增** 知识点逐项学习笔记
- [x] `wiki/SCHEMA.md` + `wiki/index.md` — A 类知识库规范

### Git
- [x] 初始提交 + 清理提交，已推送至 `github.com/wgb123/enterprise-kb`

---

## 本轮会话操作记录（2026-06-16）

| 操作 | 结果 |
|------|------|
| 恢复会话，检查仓库状态 | ✅ 代码/文档已就绪 |
| 更新 session_handoff.md 加入学习/转岗视角 | ✅ |
| README.md 添加指引 → 新会话自动读 handoff | ✅ |
| `.env.example` 复制为 `.env` | ✅ 配置了 DeepSeek API Key |
| 启动 Qdrant + FastAPI | ✅ 6333 / 8000 |
| 测试 Agent 端点 | ✅ get_time / calculator |
| 测试导入 wiki 数据 1,000 篇 | ✅ 17,813 块，0 错误 |
| **创建 `docs/learning_notes.md`** | ✅ 知识点讲解自动追加 |
| **讲解并记录知识点 1.1~1.7** | ✅ 模块一全部 7 个主知识点 |
| **讲解并记录知识点 2.1~2.2** | ✅ API 调用 + Prompt 工程 |

### 知识掌握进度

| 模块 | 知识点 | 状态 |
|------|-------|------|
| 一、RAG | 1.1 文档解析 → 1.7 重排序（21 项） | ✅ 全部讲解 |
| 二、LLM API | 2.1 OpenAI 兼容 API + 2.2 Prompt 工程（7 项） | ✅ 2/3 完成 |
| 三、Agent 核心机制 | （12 项） | ❌ 未开始 |
| 四、企业级工程化 | （14 项） | ❌ 未开始 |
| 五、框架理解 | （9 项） | ❌ 未开始 |
| 六、软技能与面试 | （7 项） | ❌ 未开始 |
| 七、Dota 2 AI Agent | （18 项） | ❌ 未开始 |

### 发现并修复的 Bug

**Bug 1: wiki_importer embedder=None**

文件：`scripts/wiki_importer.py` 第 187 行

```python
# 修复前
self.embedder = BGEM3Embedder() if not embed_only_summary else None  # ❌

# 修复后
self.embedder = BGEM3Embedder()  # ✅ 始终创建
```

当 `embed_only_summary=True` 时，嵌入器被设为 `None`，导致 `_process_article()` 累加到 `_vector_buffer` 后，`_flush_buffers()` 因 `self.embedder` 为 None 而跳过向量写入。Qdrant 永远收不到数据。

后果：BM25 正常构建，但 Qdrant 始终 points_count=0。

---

## 当前状态（2026-06-16 17:20）

### 代码层面 ✅
Agent 模块（6 文件）+ README + session_handoff + learning_notes + wiki_importer 修复

### 验证链路
- `.env` 已配置（DeepSeek API Key）
- Qdrant / FastAPI / Agent 端点已验证通过
- Wiki 导入：已测试 1,000 篇，全量导入已暂停

### 学习进度
- 模块一（RAG）：✅ 全部讲解
- 模块二（LLM API）：✅ 2/3
- 继续模块二 2.3 流式输出

---

## 关键决策记录

| 决策 | 选择 | 理由 |
|---|---|---|
| PyTorch | CPU-only | 无 GPU，省 ~1.5GB |
| LLM 推理 | DeepSeek API（非本地 vLLM） | 免 GPU，兼容 OpenAI 格式 |
| 融合排序 | RRF（非权重融合） | 无参，k=60 跨数据集稳定 |
| Agent 框架 | 手写（非 LangChain） | 透明可控，理解后再对比 |
| Qdrant 部署 | 本地二进制（非 Docker） | 单文件，低摩擦 |
| 项目定位 | 学习/转岗平台（非交付项目） | 每个模块都要能讲清楚"为什么" |
| 知识点讲解 | 自动追加到 learning_notes.md | 每讲一个知识点自动记录，不询问 |

---

## 待办

### 🔴 高
- [ ] 继续模块二 2.3 流式输出
- [ ] 完成后进入模块三 Agent 核心机制
- [ ] wiki 全量导入（BGE-M3 模型下载完成后）

### 🟡 中
- [ ] 配置 `.env.example` 恢复为模板（API Key 已脱敏）
- [ ] 每次 commit 前恢复 `.env.example` 模板

### 🟢 低
- [ ] 验证 wiki_search / rag_search Agent 工具（导入完成后）
- [ ] 练习面试第一题

---

## 环境配置

```bash
# 代理
export HTTP_PROXY=http://192.168.1.49:7890
export HTTPS_PROXY=http://192.168.1.49:7890

# 启动 Qdrant
cd qdrant_data && ./qdrant --config-path config.yaml

# 启动应用
.venv/bin/uvicorn enterprise_kb.main:app --reload --port 8000

# 启动 wiki 导入
cd /home/wgb/projects/enterprise-kb && export HTTP_PROXY=http://192.168.1.49:7890 && export HTTPS_PROXY=http://192.168.1.49:7890 && .venv/bin/python scripts/wiki_importer.py /mnt/d/zhwiki-20260601-pages-articles-multistream.xml.bz2 --batch-size 500 --embed-mode summary_only
```

## 路径

```
项目: /home/wgb/projects/enterprise-kb
虚拟环境: .venv
Qdrant: qdrant_data/qdrant
Wiki 数据: /mnt/d/zhwiki-20260601-pages-articles-multistream.xml.bz2 (3.27 GB)
Dota2: /mnt/e/IdeaProjects/dota2-ai-agent
```

---

## 新会话恢复流程

1. README.md 顶部有指引 → 自动读取本文件
2. 检查 `.env` 是否存在（API Key 配置）
3. 检查 Qdrant 服务是否运行（`curl localhost:6333/collections`）
4. 检查 FastAPI 应用是否运行（`curl localhost:8000/api/v1/health`）
5. 检查 wiki 导入进度（`cat data/wiki_import_checkpoint.json`）
6. 继续知识点讲解：从 `docs/learning_notes.md` 断点续学，每次讲完将新的知识点追加到该文档末尾
