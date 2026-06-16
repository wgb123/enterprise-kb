# Enterprise KB — 企业级分层知识库系统

> 三层记忆架构：编译式记忆（A类） + HybridRAG 检索式记忆（B类） + GraphRAG 预留（C类）

## 架构概览

```
用户
 │
 ▼
┌──────────────┐
│ SmartRouter  │  智能路由 → 意图分类
└──────┬───────┘
       │
       ├──→ A 类：WikiNavigator（本地 Markdown 知识库，最高优先级）
       │
       ├──→ B 类：HybridRetriever（向量+BM25+RRF+Reranker）
       │        ├─ Qdrant 向量检索（BGE-M3 嵌入）
       │        ├─ BM25 关键词检索（rank-bm25）
       │        ├─ RRF 融合排序
       │        └─ BGE-Reranker 交叉编码器重排
       │
       └──→ C 类：GraphRAG（预留接口，未实现）
       │
       ▼
┌──────────────┐
│ ContextFusion│  多引擎结果合并、去重、冲突解决
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ VLLMGenerator│  LLM API 生成最终答案（OpenAI 兼容）
└──────────────┘
```

### 三层记忆

| 层级 | 名称 | 存储 | 特点 |
|---|---|---|---|
| A 类 | 编译式记忆 (DeepWiki) | `wiki/` Markdown 文件 | 精选、权威、优先级最高 |
| B 类 | HybridRAG 检索式记忆 | Qdrant + BM25 Index | 海量、自动检索、语义+关键词 |
| C 类 | GraphRAG（预留） | — | 知识图谱驱动，待实现 |

---

## 快速开始

### 前置条件

- Python 3.10+
- Qdrant 服务（本地二进制或 Docker）
- LLM API 服务（OpenAI 兼容接口）

### 1. 安装

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. 配置

```bash
cp .env.example .env
# 编辑 .env 修改 LLM_API_URL、LLM_API_KEY 等
```

### 3. 启动依赖服务

**Qdrant（本地二进制，推荐）：**

```bash
cd qdrant_data && ./qdrant --config-path config.yaml
```

详见 [运维手册](docs/operations.md)。

### 4. 启动应用

```bash
uvicorn enterprise_kb.main:app --reload --port 8000
```

### 5. 验证

```bash
curl http://localhost:8000/api/v1/health
```

---

## API 文档

启动后自动生成：`http://localhost:8000/docs`（Swagger UI）

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/ingest` | POST | 文档入库（PDF/Word/Markdown） |
| `/api/v1/query` | POST | 知识库查询（含 LLM 生成） |
| `/api/v1/wiki` | GET | 搜索 Wiki 知识库 |
| `/api/v1/wiki/{path}` | GET | 获取指定 Wiki 页面 |

---

## 项目结构

```
enterprise-kb/
├── pyproject.toml          # 项目配置与依赖
├── README.md               # ← 本文件
├── .env                    # 环境变量（敏感信息，gitignore）
├── .env.example            # 环境变量模板
│
├── docs/                   # 项目文档
│   ├── architecture.md     # 系统架构设计
│   ├── development.md      # 开发指南
│   └── operations.md       # 运维手册
│
├── src/enterprise_kb/      # 源代码
│   ├── main.py             # FastAPI 应用入口
│   ├── config.py           # pydantic-settings 配置
│   ├── api/                # REST API 层
│   ├── core/               # 核心业务逻辑
│   ├── storage/            # 数据存储
│   ├── interfaces/         # 抽象接口
│   └── utils/              # 工具类
│
├── scripts/                # 工具脚本
├── tests/                  # pytest 单元测试
│
├── wiki/                   # A 类编译式记忆（DeepWiki）数据
├── qdrant_data/            # Qdrant 服务端二进制 + 数据
└── data/                   # 持久化数据
```

---

## 核心模块速览

| 模块 | 功能 | 文件 |
|---|---|---|
| DocumentProcessor | PDF/Word/Markdown → 纯文本 | `core/document_processor.py` |
| Chunker | 固定token / Markdown标题 / 语义句子 分块 | `core/chunker.py` |
| BGEM3Embedder | BGE-M3 文本→向量（本地/API双模式） | `core/embedder.py` |
| HybridRetriever | 向量+BM25+RRF+Reranker 完整管道 | `core/retriever.py` |
| BGEReranker | BGE-Reranker 交叉编码器重排序 | `core/reranker.py` |
| SmartRouter | 关键词意图识别 → 引擎分发 | `core/router.py` |
| ContextFusion | 多引擎结果合并、优先级加权、去重 | `core/fusion.py` |
| VLLMGenerator | OpenAI 兼容 API 答案生成+异常降级 | `core/generator.py` |
| WikiNavigator | Markdown 解析、内部链接、标签过滤、搜索 | `core/wiki_navigator.py` |
| QdrantStore | 异步 Qdrant 客户端封装 | `storage/qdrant_store.py` |
| BM25Index | rank-bm25 关键词索引+持久化 | `storage/bm25_index.py` |

---

## 文档

- [架构设计](docs/architecture.md)
- [开发指南](docs/development.md)
- [运维手册](docs/operations.md)
- [A 类知识库规范](wiki/SCHEMA.md)

---

## 许可证

MIT
