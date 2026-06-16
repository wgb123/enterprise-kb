# 运维手册

## 依赖服务

Enterprise KB 依赖两个外部服务：

1. **Qdrant** — 向量数据库（必须）
2. **LLM API** — OpenAI 兼容接口（必须，用于答案生成）

---

## Qdrant 部署

### 方式一：本地二进制（推荐）

项目已预置 Qdrant 1.18.2 静态编译二进制：

```bash
cd qdrant_data

# 启动
./qdrant --config-path config.yaml
```

默认监听：
- HTTP API：`http://localhost:6333`
- gRPC API：`localhost:6334`

配置文件 `qdrant_data/config.yaml`：

```yaml
log_level: INFO
storage:
  storage_path: ./storage
  optimizers:
    default_segment_number: 2
service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334
```

### 方式二：Docker

```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### 验证 Qdrant

```bash
curl http://localhost:6333/collections
# {"result":{"collections":[]},"status":"ok","time":...}
```

---

## LLM API 配置

项目通过 OpenAI 兼容接口调用 LLM，支持以下服务商：

| 服务商 | API 地址 | 模型 |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` / `gpt-4o-mini` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| Ollama（本地） | `http://localhost:11434/v1` | `llama3` |

配置方式（在 `.env` 中）：

```ini
VLLM_API_URL=https://api.deepseek.com/v1
VLLM_API_KEY=sk-your-api-key-here
VLLM_MODEL_NAME=deepseek-chat
VLLM_MAX_TOKENS=2048
VLLM_TEMPERATURE=0.3
VLLM_TIMEOUT=60
```

验证连接：

```bash
curl https://api.deepseek.com/v1/models \
  -H "Authorization: Bearer sk-your-api-key"
```

---

## 应用部署

```bash
# 开发模式（热重载）
uvicorn enterprise_kb.main:app --reload --port 8000

# 生产模式
uvicorn enterprise_kb.main:app --host 0.0.0.0 --port 8000 --workers 4
```

验证：

```bash
curl http://localhost:8000/api/v1/health
# {"status":"ok","version":"0.1.0","service":"Enterprise KB"}
```

---

## 维基百科导入

当维基百科 XML dump 下载完成后：

```bash
.venv/bin/python scripts/wiki_importer.py wiki.xml.bz2
```

导入策略：
- **BM25 全量索引**：所有文章按章节分块入库
- **Qdrant 选择性向量化**：仅标题+前 500 字符
- **断点续传**：中断后重跑自动续传

---

## 故障排除

### Qdrant 连接失败
```bash
# 检查进程
ps aux | grep qdrant
# 检查端口
curl http://localhost:6333/healthz
# 查看日志
cat qdrant_data/qdrant.log | tail -20
```

### LLM API 调用失败
```bash
# 验证 API Key
curl -s -o /dev/null -w "%{http_code}" \
  https://api.deepseek.com/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"
```
检查 `.env` 中 `VLLM_API_URL` 和 `VLLM_API_KEY`。

### 嵌入模型 OOM
在 `.env` 中降低批处理大小：
```ini
EMBEDDING_BATCH_SIZE=8
EMBEDDING_MODEL_DEVICE=cpu
```

---

## 数据备份

```text
qdrant_data/storage/       # Qdrant 向量数据
data/wiki_bm25.pkl         # BM25 关键词索引
wiki/                       # A 类知识库
.env                        # 配置
```

---

## 性能参考

| 配置 | 性能 |
|---|---|
| CPU-only BGE-M3 嵌入 | ~10-20 篇/秒（摘要嵌入） |
| CPU-only BM25 索引 | ~1000-3000 篇/秒 |
| 查询响应（含 LLM API） | ~1-5 秒 |
| Qdrant 向量搜索（<10万条） | <50ms |
