# 开发指南

## 环境配置

### 前置条件

- Python 3.10+
- Qdrant 向量数据库（启动方式见 [运维手册](operations.md)）
- LLM API 服务（OpenAI 兼容接口，如 DeepSeek / OpenAI / 通义千问）

### 1. 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安装项目依赖

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

> **注意**：`FlagEmbedding` 依赖 PyTorch。如果当前环境没有 NVIDIA GPU，安装时会自动下载 CPU-only 版本。
> 如需手动指定 CPU 版：`pip install torch --index-url https://download.pytorch.org/whl/cpu`

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 中的 API Key 和连接地址
```

关键配置项：

| 变量 | 说明 | 示例值 |
|---|---|---|
| `VLLM_API_URL` | LLM API 地址 | `https://api.deepseek.com/v1` |
| `VLLM_API_KEY` | API Key | `sk-xxx` |
| `VLLM_MODEL_NAME` | 模型名称 | `deepseek-chat` |
| `QDRANT_HOST` | Qdrant 主机 | `localhost` |
| `EMBEDDING_MODEL_DEVICE` | 嵌入模型设备 | `cpu`（无 GPU） |

### 4. 启动 Qdrant

```bash
cd qdrant_data && ./qdrant --config-path config.yaml
```

详见 [运维手册](operations.md)。

### 5. 启动开发服务器

```bash
uvicorn enterprise_kb.main:app --reload --port 8000
```

### 6. 验证

```bash
curl http://localhost:8000/api/v1/health
# {"status":"ok","version":"0.1.0","service":"Enterprise KB"}
```

---

## 项目结构

```
├── pyproject.toml          # 项目配置与依赖
├── README.md               # 项目说明
├── .env.example            # 环境变量模板
│
├── src/enterprise_kb/
│   ├── main.py             # FastAPI 应用入口
│   ├── config.py           # pydantic-settings 配置
│   ├── api/                # REST API 层
│   │   ├── routes.py       # 路由
│   │   ├── schemas.py      # 请求/响应模型
│   │   └── dependencies.py # 依赖注入
│   ├── core/               # 核心业务逻辑
│   │   ├── document_processor.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   ├── router.py
│   │   ├── fusion.py
│   │   ├── generator.py
│   │   └── wiki_navigator.py
│   ├── storage/            # 数据存储
│   │   ├── qdrant_store.py
│   │   └── bm25_index.py
│   ├── interfaces/         # 抽象接口
│   │   ├── base.py
│   │   └── graph_rag.py
│   └── utils/              # 工具类
│       ├── logger.py
│       └── text_utils.py
│
├── scripts/                # 工具脚本
│   ├── wiki_importer.py    # 维基百科 XML dump 批量导入
│   └── demo_router_comparison.py  # 路由对比 demo
│
├── tests/                  # pytest 单元测试
│
├── wiki/                   # A 类编译式记忆（DeepWiki）数据目录
│
├── qdrant_data/            # Qdrant 服务端二进制 + 数据
│   ├── qdrant
│   ├── config.yaml
│   └── storage/
│
└── data/                   # 持久化数据
    ├── wiki_bm25.pkl       # BM25 索引
    └── wiki_import_checkpoint.json
```

---

## 核心模块说明

### 查询流水线

```
用户 → SmartRouter(意图分类)
        ├─→ WikiNavigator (A 类)
        ├─→ HybridRetriever (B 类)
        │     ├─ Qdrant 向量检索 (BGE-M3 嵌入)
        │     ├─ BM25 关键词检索 (rank-bm25)
        │     ├─ RRF 融合排序
        │     └─ BGE-Reranker 重排序
        └─→ [GraphRAG (预留)]
        ↓
    ContextFusion (融合+去重)
        ↓
    VLLMGenerator (LLM API 生成答案)
```

### 各模块职责

| 模块 | 职责 | 关键类/方法 |
|---|---|---|
| `document_processor.py` | 解析 PDF/Word/Markdown 为纯文本 | `DocumentProcessor.process_file()` |
| `chunker.py` | 3 种分块策略 | `Chunker.chunk()` |
| `embedder.py` | BGE-M3 嵌入，本地/API双模式 | `BGEM3Embedder.embed()` |
| `retriever.py` | HybridRAG 检索管道 | `HybridRetriever.retrieve()` |
| `reranker.py` | BGE-Reranker 交叉编码器重排序 | `BGEReranker.rerank()` |
| `router.py` | 关键词意图分类 → 引擎分发 | `SmartRouter.classify()` |
| `fusion.py` | 多引擎结果合并、去重 | `ContextFusion.fuse()` |
| `generator.py` | OpenAI 兼容 API + 降级 | `VLLMGenerator.generate()` |
| `wiki_navigator.py` | Markdown 解析、搜索 | `WikiNavigator.query()` |

---

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 带覆盖率
pytest tests/ -v --cov=enterprise_kb --cov-report=term-missing

# 代码检查
ruff check src/ tests/
mypy src/
```

---

## 常见问题

### Q: `pip install -e .` 失败
确保 `.venv` 已激活，`pip` 已升级。

### Q: Qdrant 连接拒绝
Qdrant 服务未启动，详见 [运维手册](operations.md)。

### Q: FlagEmbedding 导入慢 / OOM
首次导入会下载模型（BGE-M3 约 2.2GB）。CPU 环境下减小 `EMBEDDING_BATCH_SIZE`。

### Q: 如何切换 LLM 服务商？
修改 `.env` 中的 `VLLM_API_URL`，兼容 OpenAI `/chat/completions` 接口即可。
