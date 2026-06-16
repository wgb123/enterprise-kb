# 学习笔记 — Enterprise KB 知识点逐项梳理

> 按 `docs/skill_tracker.md` 的顺序，逐个知识点讲解。
> 每次讲完自动追加到本文档末尾。

---

## 模块一：RAG 检索增强生成

### 1.1 文档解析管道

**文件：** `src/enterprise_kb/core/document_processor.py`

#### 职责

把 PDF、Word、Markdown 等文件转成纯文本，让后续的分块和检索能用。

#### 核心流程

```
process_file(path)
   │
   ├── .pdf    → LlamaParse（云端 API）→ 降级 PyMuPDF
   ├── .docx   → python-docx 库
   ├── .md     → 内置 Markdown 解析
   └── 其他    → 直接按文本读
```

#### 关键设计

**1. 三档降级策略（PDF 为例）**

```
LlamaParse（云端）→ 精度最高，要 API Key
       ↓ 没有 Key
PyMuPDF（本地）  → 够用，不支持复杂排版
       ↓ 也没有
抛异常告知用户
```

项目里 embedder、reranker 也是同样的"有好的用好的，没有就降级"模式。

**2. 并发控制**

```python
sem = asyncio.Semaphore(concurrency)  # 默认并发 4
```

信号量限制同时最多处理 4 个文件，防止 LlamaParse HTTP 调用打满。

#### L1 理解（能说清楚）

- 用户上传 PDF → 系统怎么变成文本的？
- PDF 解析失败怎么办（降级路径）？
- 为什么没用 OCR？（假设文字版 PDF，不是扫描件）

#### L2 审查（能指出 AI 代码的问题）

如果 AI 写出 `_upload_pdf()` + `_poll_job()`，你要能指出：
- 缺少超时处理
- 没检查错误状态码
- 如果 LlamaParse 不可用，没走降级路径

#### 拓展思考：多模态资料（图片/音频/视频）

**当前系统的能力边界：** 只处理文本，图片/音频/视频直接忽略或抛异常。

**真实场景中图片的实际情况：** 90% 的企业文档图片（架构图、截图、图表、产品照片）的信息已在正文中用文字描述过，不需要单独处理。唯一必须处理的是**扫描件 PDF**（文字只在图片里）。

**如果要支持，分三个层级：**

| 层级 | 方案 | 适用场景 | 代价 |
|------|------|---------|------|
| L1 | OCR（Tesseract / PaddleOCR） | 扫描件 PDF，提取图片中的文字 | 慢，每张图多一次 OCR |
| L2 | 多模态嵌入（CLIP / ImageBind） | 跨模态检索（"找一张架构图"） | 需要多模态模型 |
| L3 | 多模态 LLM 理解（GPT-4V） | 理解图片语义（"这张图用了什么协议"） | 贵，每张图多一次 LLM 调用 |

**结论：** 对当前项目来说，多模态不是瓶颈。真要补的话，加 OCR 处理扫描件就够了。

---

### 1.2 文本分块策略

**文件：** `src/enterprise_kb/core/chunker.py`

#### 为什么要分块

LLM 有上下文窗口限制（4K~8K token），不能把整本书直接塞进去。需要把长文档切成小块，检索时只找最相关的几块。

#### 三种策略

**1. 固定大小分块（FixedSizeChunker）**

```
文本 → 按 token 数硬切 → 带重叠窗口
```

- 用 `tiktoken` 精确计 token（不是按字符数）
- 默认 chunk_size=512，overlap=128
- 优点：速度快、token 精确可控
- 缺点：可能从一句话中间切断，丢失语义

**重叠的作用**：chunk1 [0-512]，chunk2 [384-896]，中间 128 token 在两个块里都出现，防止边界信息丢失。

**2. Markdown 标题分块（MarkdownHeaderChunker）**

```
Markdown → 按 ## 标题分段 → 超长段再裁
```

- 按 `##` / `###` 标题切分，保留标题层级
- 超长段落内部用 FixedSizeChunker 二次切分
- 每个 chunk 带上 `heading` 和 `heading_level` 元数据
- 优点：语义完整、自包含
- 适用：技术文档、Wiki 等有层级结构的内容

**3. 语义句子分块（SemanticSentenceChunker）**

```
文本 → 按句号拆句子 → 滑动窗口合并
```

- 保证不在一句话中间切断
- 默认 8 句一段，overlap 2 句
- 优点：语义完整性最高
- 缺点：句子长短不一，token 不可控

#### 三种策略对比

| 策略 | 语义完整性 | token 可控 | 速度 | 适用场景 |
|------|-----------|-----------|------|---------|
| 固定大小 | ❌ 差 | ✅ 精确 | ✅ 快 | 通用兜底 |
| Markdown 标题 | ✅ 好 | ✅ 较好 | ✅ 快 | 技术文档/Wiki |
| 语义句子 | ✅ 最好 | ❌ 不可控 | 🟡 中 | 注重语义的场景 |

#### 为什么三种都要有

按文档类型选策略，不是"用哪个最好"：
- Markdown 文档 → MarkdownHeaderChunker
- 纯文本、无结构 → FixedSizeChunker 兜底
- 对话、文章 → SemanticSentenceChunker

当前项目默认用 Markdown 标题分块（`CHUNK_MARKDOWN_BY_HEADER=true`）。

#### 维基数据导入实际用的策略

`scripts/wiki_importer.py` **没有用 chunker.py 里的类**，而是自己实现了一个 `split_by_section()`。效果和 MarkdownHeaderChunker 类似——按 `== 标题 ==` 分段，每段一个 chunk。只是针对维基百科的 Wiki 标记语法做了适配。

```
BM25 索引  → 全量文本，每节一个文档
Qdrant 向量 → 仅标题 + 引言前 500 字符（embed_only_summary 模式）
```

原因：全量向量化 CPU 太慢，标题+摘要足够语义检索。

#### L1 理解（能说清楚）

- 三种策略分别怎么切、各有什么优劣？
- 为什么需要 overlap？不设置会有什么问题？
- 维基百科导入用的什么策略，为什么这么选？

#### L2 审查（能指出 AI 代码的问题）

- AI 写 `FixedSizeChunker` 时，只按字符数切没用 tiktoken → 中英文混排 token 数不准
- `MarkdownHeaderChunker` 没处理超长段落 → 超过 max_chunk_tokens 的段应该二次切分
- `split_by_section` 只分了一层 `==`，没递归处理子标题 → 嵌套标题的粒度太粗

#### 拓展思考：为什么 BM25 存全文，向量只存摘要

**原因：CPU 算不动向量。**
- BM25 索引 = 数词频，纯 CPU 操作，微秒级
- BGE-M3 向量化 = 神经网络推理，CPU 上几百毫秒/段
- 如果全文向量化：450 万段 × 0.3 秒 ≈ 375 小时（15 天）

**为什么还要保留 BM25？**

向量检索有盲区——对专有名词不敏感。例：

```
搜"Qdrant HNSW 索引参数调优"
- 向量 → 可能召回"数据库优化方案"（语义接近但不精确）
- BM25 → 精准命中"Qdrant"+"HNSW"
```

BM25 覆盖精度，向量覆盖语义，两者通过 RRF 融合互补。

**对效果的实际影响：**

| 场景 | BM25（全文） | 向量（摘要） | RRF 结果 |
|------|-------------|-------------|---------|
| 关键词精确匹配 | ✅ 精准 | ❌ 可能没有 | ✅ BM25 拉回 |
| 语义相似搜索 | ⚠️ 可能缺词 | ✅ 摘要有核心概念 | ✅ 向量拉回 |
| 正文深处语义 | ✅ BM25 搜到关键词 | ❌ 摘要没有 | ✅ BM25 仍然兜底 |

真正漏掉的只有一种情况：纯语义、无关键词可匹配、且概念在正文深处。但这种场景在企业知识库中极少见。

**结论：用微不足道的精度损失，换了几百倍的导入速度，非常值。**

#### 拓展思考：什么是 HybridRAG

HybridRAG ≠ 简单的 "BM25 + 向量"。它分两个层面理解：

**狭义：检索层面的混合**

```
查询 → BM25（关键词）+ 向量（语义）→ RRF 融合 → 生成
```

两个独立通道，用 RRF 合并排序。这是项目的当前做法。

**广义：整个管道的混合**

```
文档输入层  → PDF + MD + Word 混合解析
检索策略层  → BM25 + 向量 + Reranker 混合  ← BM25+向量只是这一环
融合排序层  → RRF / 加权融合
生成层      → 多轮对话 + 引用标注 + 降级策略
```

**HybridRAG 在 RAG 演进中的位置：**

```
Naive RAG:     查询 → 向量检索 → 生成
                    ↑
Hybrid RAG:    查询 → 向量检索 + BM25 → RRF → Reranker → 生成   ← 项目定位
                    ↑
Advanced RAG:  多了查询改写、HyDE、GraphRAG、Agent RAG 等
```

当前项目从 BM25 到 Reranker 全链路都有，是完整的 HybridRAG 实现。

#### 拓展思考：Advanced RAG 为什么项目没全部用上

从 HybridRAG 再往前一步就是 Advanced RAG，多了四种技术：

**1. 查询改写（Query Rewriting）**

做什么：把用户模糊问题改写成多个精确查询，分别检索。
```
"那个 Qdrant 的东西怎么调"
  → "Qdrant HNSW 索引参数调优"
  → "Qdrant 搜索性能优化配置"
  → "Qdrant ef_construction 和 M 参数说明"
```

为什么没加：已经被 Agent Loop 覆盖了——Agent 拿到模糊查询，自己决定调哪个工具。单独加查询改写等于在 RAG 管道里多一次 LLM 调用，延迟增加 1-2 秒，收益不大。

**2. HyDE（Hypothetical Document Embeddings）**

做什么：先让 LLM 假设性生成一篇"完美回答"，用生成的内容做向量检索。
```
用户问 → LLM 先生成假设答案 → 用这个答案搜 → 更接近真实文档
```

为什么没加：生成方向不对就全歪，不稳定。有 GPU + 全量向量后可考虑。

**3. GraphRAG**

做什么：从文档中提取实体和关系，存成知识图谱。擅长回答"全局性问题"。
```
Naive RAG: "Qdrant 用什么算法？" → 搜段落 → 回答
GraphRAG:  "所有向量数据库的共同设计模式？" → 图遍历 → 社区摘要 → 回答
```

为什么没加：接口已预留（`interfaces/graph_rag.py`），但实现极复杂——实体抽取→关系抽取→图存储→图遍历→社区摘要。适用场景窄，放在 C 类记忆预留。

**4. Agent RAG**

做什么：用 Agent 控制 RAG 流程，LLM 自己决定何时检索、检索什么。
```
Agent RAG = 用户 → Agent Loop（带 tools）→ 工具（含检索）→ 生成
```

项目现状：✅ **已实现**。整个 Agent 层就是 Agent RAG。

**总结：**

| 技术 | 项目状态 | 原因 |
|------|---------|------|
| 查询改写 | ❌ 没单独做 | 被 Agent Loop 覆盖 |
| HyDE | ❌ 没做 | 收益小风险大 |
| GraphRAG | ❌ 预留接口 | 复杂度高、场景窄 |
| Agent RAG | ✅ 已实现 | 就是 agent 层 |

设计哲学：先把基础打扎实，再从实际需求出发按需加高级技术。

---

## 1.3 向量嵌入

**文件：** `src/enterprise_kb/core/embedder.py`

#### 职责

把文本转成向量（浮点数数组），让 Qdrant 能算语义相似度做检索。

```
"Transformer 架构" → [0.023, -0.145, 0.672, ...] (1024 维)
"CNN 网络"        → [0.189, -0.032, 0.411, ...] (1024 维)
                  ↑ 余弦相似度高（语义接近）
```

#### 使用的模型

**BAAI/bge-m3** — 1024 维，FlagEmbedding 库加载。

特点：
- 多语言（中英文不换模型）
- M3 = Multi-Lingual + Multi-Function + Multi-Granularity
- 1024 维（比 ada-002 的 1536 小，算得快）

#### 双模式架构

```
embed(texts)
  │
  ├── 本地模式（FlagEmbedding）
  │     模型加载到 CPU → encode()
  │     优点：免费、不限速、离线
  │     缺点：CPU 慢（几百毫秒/段）
  │
  └── API 模式（OpenAI 兼容接口）
       POST /v1/embeddings
       优点：快（GPU 服务端）
       缺点：要网络、要钱、有速率限制
```

加载本地模型失败自动降级到 API 模式。

#### 关键设计

**1. 懒加载**

```python
async def _ensure_model(self):
    if self._model is not None:
        return  # 只加载一次
```

启动时不占内存，第一次调用 embed() 才加载模型。

**2. L2 归一化**

```python
vector = vector / ||vector||
```

如果不归一化：`[100, 0]` 和 `[50, 0]` 方向相同但长度不同 → 余弦相似度=1.0（算出来一样！）
归一化后长度都是 1，点积 = 真正的余弦相似度。

**3. 并发控制**

```python
self._semaphore = asyncio.Semaphore(2)  # 最多 2 个并发
```

CPU encode 本身已多线程，再加太多异步并发会导致线程爆炸。

**4. 批处理**

一次 32 段批量编码，摊薄模型推理的 overhead。

#### 常见混淆：1.2 分块 vs 1.3 嵌入

这两道工序是流水线关系，不是二选一：

```
原始文档
  │
  ▼
1.2 文本分块  → 把长文本切成小段（文本 → 文本，不需要 AI 模型）
  │
  ▼
1.3 向量嵌入  → 把每段文本转成向量（文本 → [0.1, ...]，需要 BGE-M3 推理）
```

**1.2 的输出是 1.3 的输入。** 没分块就不能嵌入，没嵌入 Qdrant 就没法搜。

wiki 导入脚本里两件事写在一个函数里（`_process_article()`），但逻辑上是分开的：
- `split_by_section()` → 分块（1.2 的职责）
- BM25 直接收文本（不需要向量）
- `embedder.embed(summary)` → 向量化（1.3 的职责）
- Qdrant upsert → 存入

#### L1 理解

- 为什么文本要转成向量？余弦相似度怎么算？
- BGE-M3 有什么特点（1024 维、多语言）？
- 为什么每次 embedding 后要 L2 归一化？
- 本地模型加载失败怎么办？
- 1.2 分块和 1.3 嵌入是什么关系？

#### L2 审查

- AI 每次 encode() 都重新加载模型 → 应该懒加载
- 没做 normalize=True → 余弦相似度不准
- 逐条 encode 不批量 → 要加 batch_size
- 没并发控制 → CPU + 异步并发导致线程爆炸

---

### 1.4 向量数据库（Qdrant）

**文件：** `src/enterprise_kb/storage/qdrant_store.py` — 237 行

#### 向量数据库 vs 传统数据库

| | 传统数据库（MySQL） | 向量数据库（Qdrant） |
|------|-------------------|--------------------|
| 存什么 | 行 + 列（结构化数据） | 向量（浮点数数组） |
| 怎么查 | 精确匹配（WHERE title LIKE '%Qdrant%'） | 近似最近邻（ANN） |
| 排序 | ORDER BY 某种字段 | 按向量距离排序 |
| 典型查询 | "价格大于100的商品" | "和这个句子意思最接近的段落" |

向量数据库**不是替代**传统数据库，而是互补——文档的元数据（标题、来源、时间）仍然需要传统存储来过滤。

#### Qdrant 核心概念

```
Collection（集合）— 相当于 MySQL 的表
  └── Point（数据点）— 相当于一行记录
        ├── id: 唯一标识（数字或 UUID）
        ├── vector: [0.1, -0.2, ...] — BGE-M3 输出的 1024 维向量
        └── payload: {"title": "..." , "source": "wikipedia"} — 元数据
```

#### 项目配置

```python
vectors_config=VectorParams(
    size=1024,               # 必须和 BGE-M3 输出维度一致
    distance=Distance.COSINE, # 余弦相似度
)
```

**为什么用 Cosine 距离？**
- 向量已经 L2 归一化了，cosine = 点积（最快）
- 欧氏距离多了一步开平方，效果一样但更慢
- Qdrant 也支持 Euclid / Dot 距离

#### 索引算法：HNSW

```python
hnsw_config={
    "m": 16,               # 每个节点连几个邻居
    "ef_construct": 200,   # 构建索引时搜多深
}
```

HNSW（Hierarchical Navigable Small World）是目前最主流的 ANN 算法：

- 构建时：每个点连接 m 个最近邻居，形成导航图
- 搜索时：从顶层随机点开始，逐层往下找邻居
- m 越大 → 召回率越高，但索引越大、写入越慢
- ef_construct 越大 → 索引质量越好，但构建越慢

这套参数适合 100 万以内的向量。千万级需要调大。

#### QdrantStore 设计

**1. 懒加载客户端**

```python
async def _get_client(self) -> AsyncQdrantClient:
    if self._client is None:  # 只创建一次
        self._client = AsyncQdrantClient(url=url, ...)
    return self._client
```

不是每个请求都 new 一个 client，而是复用。

**2. 自动建表**

```python
async def ensure_collection(self):
    # 检查 collection 是否存在，不存在则创建
```

第一次写入数据时自动调用，不用手动建表。

**3. 搜索加过滤**

```python
FieldCondition(key="source", match=MatchValue(value="wikipedia"))
```

可以在搜向量时只搜某个来源的数据。

**4. `wait=True` 的代价**

```python
await client.upsert(points=points, wait=True)
```

`wait=True` 等价于数据库的 `COMMIT`——确认写入完成后才返回。保证数据一定落盘，但慢。
大量导入时应该批量 upsert（攒够 N 篇 flush 一次），而不是每篇都 wait。

#### Qdrant 部署模式

```
项目用：本地二进制
  cd qdrant_data && ./qdrant --config-path config.yaml

备选：Docker
  docker run -p 6333:6333 qdrant/qdrant

不选 Docker 的原因：单文件就够，低摩擦启动
```

#### 怎么看原始数据

Qdrant 支持通过 API 查看已存数据：

```bash
# 查看集合统计
curl localhost:6333/collections/enterprise_kb

# 滚动查看数据（带 payload）
curl 'localhost:6333/collections/enterprise_kb/points/scroll?limit=5&with_payload=true'
```

项目里封装了 `scroll_all()` 方法，直接取所有数据。

#### L1 理解

- 向量数据库和传统数据库的核心区别是什么？
- Qdrant 的三个核心概念（Collection, Point, Payload）
- HNSW 是什么？m 和 ef_construct 各控制什么？
- 为什么用 Cosine 不用欧氏距离？
- `wait=True` 有什么问题？

#### L2 审查

- AI 搜 Qdrant 没传 `with_payload=True` → 搜出向量但拿不到元数据
- 每次搜都 new client → 应该复用，懒加载
- Collection vector_size 和模型维度不匹配 → 写入失败（实测报错）
- 大量写入每条都 `wait=True` → 速度慢到无法接受

---

### 1.5 BM25 关键词索引

**文件：** `src/enterprise_kb/storage/bm25_index.py` — 约 170 行

#### 什么是 BM25

BM25（Best Matching 25）是信息检索领域最经典的排序算法，用来算"一段文本和查询有多匹配"。

```
查询：Qdrant HNSW 参数

文档1："Qdrant的HNSW索引有两个关键参数m和ef_construct"
文档2："向量数据库的索引算法有很多种"

BM25 分数：
  文档1 → 高（Qdrant+HNSW 命中）
  文档2 → 低（没有匹配词）
```

#### BM25 的核心公式（理解思路即可）

```
BM25 分数 = 词频(TF) × 逆文档频率(IDF) × 长度归一化
```

三部分各干各的：

| 部分 | 作用 | 直觉 |
|------|------|------|
| 词频（TF） | 查询词在文档中出现越多次，分数越高 | "Qdrant"在一段里出现 3 次比 1 次重要 |
| 逆文档频率（IDF） | 越稀有的词权重越高 | "Qdrant"比"的"重要得多 |
| 长度归一化 | 长文档（全文）比短文档（摘要）容易命中，需要惩罚 | 一段 50 字的出处比一段 5000 字的全文更有价值 |

项目用的是 `rank_bm25` 库里的 `BM25Okapi` 实现，用默认参数（k1=1.5, b=0.75）。

#### 为什么需要 BM25，有向量检索还不够吗？

**向量检索的盲区——专有名词：**

```
搜"Qdrant HNSW ef_construct 设置多大合适"
  → 向量检索可能找回："向量索引参数优化建议"（语义接近但没说到具体参数）
  → BM25 精准命中："Qdrant HNSW ef_construct 设为 200"

两个结果 RRF 融合 → 用户得到既有语义相关又有精确匹配的回答
```

简单说：**BM25 管精确命中，向量管语义相似，两者互补不互斥。**

#### BM25Index 设计

**1. rank_bm25 封装**

```python
tokenized = [self._tokenize(t) for t in self._corpus]
self._index = BM25Okapi(corpus=tokenized)
```

tokenize 的逻辑：按非字母数字切分 + 小写 + 过滤单字符。

**2. 持久化（pickle）**

```python
def save(self, path):
    data = {
        "corpus": self._corpus,      # 原文列表
        "metadata": self._metadata,   # 元数据
        "index_state": {...},         # BM25 内部状态
    }
    pickle.dump(data, open(path, "wb"))
```

把整个索引序列化成文件，下次启动 `load()` 恢复。不需要从头重新构建。

**3. 增量构建（全量重建）**

```python
def add_documents(self, documents):
    self.build(self._corpus_texts() + documents)  # 全量重建
```

BM25 不支持真正的增量添加，所以 `add_documents` 是"老数据 + 新数据 → 全量重跑"。文章少的时候可以接受，百万篇只能一次性 build。

#### BM25 vs 向量：怎么选，什么时候组合

| 场景 | BM25 | 向量 | 组合（RRF） |
|------|------|------|------------|
| 搜专有名词（"Qdrant ef_construct"） | ✅ 精准命中 | ❌ 可能跑偏 | ✅ BM25 拉回来 |
| 搜概念（"语义相似度怎么算"） | ❌ 没共同关键词 | ✅ 语义匹配合适 | ✅ 向量主导 |
| 搜长尾低频词 | ✅ 精准 | ❌ 没训练好 | ✅ BM25 兜底 |
| 跨语言搜索 | ❌ 词不匹配 | ✅ 语义向量跨语言 | ✅ 向量主导 |

#### L1 理解

- BM25 的三要素（TF、IDF、长度归一化）各管什么？
- 为什么有了向量检索还需要 BM25？
- BM25 的持久化怎么做的？（pickle）
- BM25 能增量添加吗？不能的话怎么办？

#### L2 审查

- AI 只做向量检索不做 BM25 → 专有名词查不到
- BM25 tokenizer 太简单，没处理中文分词 → "QdrantHNSW" 搜不到 "Qdrant HNSW"
- `add_documents` 每次全量重建 → 百万级数据要几分钟，应该用 lazy rebuild
- 没持久化 → 每次启动重新构建，浪费几十秒

#### 为什么向量搜不了专有名词（原因：有损压缩）

向量模型把句子压缩成 1024 维向量，**没有一个维度专门代表 `ef_construct`**。模型只能猜"这大概是个技术参数"，向量落在"参数调优"附近。BM25 是精确抄写，`Qdrant` 出现了就命中。所以两者必须互补。

**BM25 性能不会随数据量变差**——底层有倒排索引，查询时直接跳到匹配文档算分，不是全表扫描。百万级 < 30ms。真正慢的是 Reranker。

**ES 从 5.0 开始默认就是 BM25**，参数 k1/b 和项目一致。再往上量级可以换 ES（亿级）或 MeiliSearch/Typesense（千万级），算法不变。

---

### 1.6 RRF 融合排序

**文件：** `src/enterprise_kb/core/retriever.py`（`_rrf_fuse` 方法）

#### RRF 解决了什么问题

BM25 和向量检索各自返回一列结果。两个列表的**顺序不同、分数不可比**——BM25 的分是 0~10 的小数，向量距离是 0~1 的浮点数，不能直接相加。

RRF（Reciprocal Rank Fusion）的作用：**把两个不可比的排序列表融合成一个新排序**。

#### RRF 公式

```
RRF 分数 = 1 / (k + rank)

k=60 是经验值
  
文档A：向量排第 3 → 1/(60+3) = 0.0159
       BM25 排第 7 → 1/(60+7) = 0.0149
       RRF 总分 = 0.0307 ← 两个引擎都命中，分最高

文档B：向量排第 1 → 1/(60+1) = 0.0164
       BM25 没命中 → 0
       RRF 总分 = 0.0164 ← 只有一个引擎命中

文档C：向量没命中 → 0
       BM25 排第 2 → 1/(60+2) = 0.0161
       RRF 总分 = 0.0161 ← 只有 BM25 命中
```

**两个引擎都命中的文档 > 只有一个引擎命中的文档。** 这就是 RRF 的融合效果。

#### 为什么 k=60

k 越大，排名的差异越小（所有分数越接近）。k 越小，排名越前面的文档权重越高。

```
k=1:    排第 1 的 rrf=1.0, 排第 100 的 rrf=0.0099  → 差距巨大
k=60:   排第 1 的 rrf=0.016, 排第 100 的 rrf=0.006  → 差距缩小
k=200:  排第 1 的 rrf=0.005, 排第 100 的 rrf=0.0033 → 差距很小
```

**k=60 是学界经验值，在多个数据集上表现稳定，不需要调参。**

#### 项目中的 RRF 位置

```
用户查询
  │
  ├── BM25 检索  ─→ 按 rank 给分
  ├── 向量检索  ─→ 按 rank 给分
  │
  ▼
  1. RRF 融合  ← 合并两个排名的分数
  │
  ▼
  2. Reranker  ← 交叉编码器精排 top-N
  │
  ▼
  3. ContextFusion ← 多引擎（Wiki+HybridRAG）优先级去重融合
  │
  ▼
  生成答案
```

**RRF 融合的是两个检索引擎的结果。ContextFusion 融合的是多个引擎类型（A类 Wiki、B类 HybridRAG）的结果。** 两者不在同一层。

#### RRF vs 其他融合方法

| 方法 | 原理 | 需要调参 | 推荐 |
|------|-----|---------|------|
| RRF | 倒数排名融合 | ❌ k=60 经验值 | ✅ 项目用的 |
| 加权融合 | score_a × w + score_b × (1-w) | ✅ 要调 w | ❌ 每换场景就要调 |
| 学习排序 | 用训练数据学权重 | ✅ 要训练 | ❌ 没有训练数据 |

**RRF 最大的优势：无参。** 不需要知道向量和 BM25 的分数范围，不需要为每个数据集调整权重。

#### L1 理解

- RRF 解决了什么问题？（两个排序列表不可比）
- RRF 公式怎么算？为什么 k=60？
- RRF 在管道中的位置：BM25→向量→RRF→Reranker→ContextFusion
- RRF 相比加权融合的优势是什么？

#### L2 审查

- AI 直接把 BM25 分数和向量分数相加 → 量纲不同，结果不可用
- Reranker 排完序忘了恢复 RRF 分数 → 重排后 RRF 白做了
- ContextFusion 的优先级加分和 RRF 分数叠加 → Wiki 结果永远第一（预期行为）

#### 拓展：RRF 的原理直觉

RRF 不关心原始分数，只关心**顺序**（排名）：

```
引擎A排第1的文档 → 不管打几分 → RRF = 1/(60+1)
引擎B排第1的文档 → 不管打几分 → RRF = 1/(60+1)
```

为什么用**倒数**？倒数函数的特点是：排名高的文档权重差距大（1→2 差 0.0003），排名低的差距小（99→100 几乎没差）。前几名更相关但不垄断。

#### 拓展：加权融合为什么不好

加权融合：`final = norm(bm25_score) × w + norm(vector_score) × (1-w)`

| 问题 | 后果 |
|------|------|
| 归一化方法不唯一 | min-max / z-score / rank 选哪个？ |
| 权重 w 依赖场景 | 文档搜索 w=0.3，代码搜索 w=0.7，每场景调 |
| 异常值压制 | BM25 某文档得分特高，归一化后其他分被压到 0.1 |
| 分数意义不同 | BM25 8.5 ≠ 向量 0.92，不代表同一件事 |

**RRF 不需要知道分数含义，不需要归一化，不需要调参。这是它相对加权融合的核心优势。**

---

### 1.7 Reranker 重排序

**文件：** `src/enterprise_kb/core/reranker.py` — 约 170 行

#### Reranker 和 RRF 的区别

这是面试里容易混淆的点：

```
RRF（1.6）→ 两个检索引擎的结果融合，只看排名不看分数，无参
Reranker → 对候选列表精排，用交叉编码器重新打分，有模型
```

- RRF 是**规则**（倒数排名融合公式），不训练、不加载模型
- Reranker 是**模型**（BGE-Reranker-v2-m3），需要加载交叉编码器推理

#### Reranker 到底在做什么

用个完整例子说明检索管道的每一步：

```
查询："怎么用 Docker 部署 Qdrant"
```

**Step 1: 向量检索** — 各自编码，算余弦相似度，只看"像不像"：

```
① "Qdrant 可以用 Docker Compose 部署" → 0.88  ← 命中
② "Docker 是一个容器化平台"           → 0.82  ← 含 Docker 但没说 Qdrant
③ "向量数据库架构设计指南"            → 0.79  ← 向量数据库相关
④ "Qdrant 使用 HNSW 做近似搜索"       → 0.76  ← 含 Qdrant 但无关
```

②、③、④ 在某些词上"相似"（Docker / 向量数据库 / Qdrant），但都不是真正回答了查询。

**Step 2: RRF 融合** — 合并 BM25 和向量的排序。

**Step 3: Reranker 重排** — 查询+文档拼接，过 Transformer，看"答没答到点上"：

```
查询+① 交叉编码 → 0.95 ← 完美匹配，排第一
查询+② 交叉编码 → 0.30 ← 跑题，降到后面
查询+③ 交叉编码 → 0.55 ← 相关但没说 Docker
查询+④ 交叉编码 → 0.35 ← 明确不相关
```

**核心区别：**
- 向量检索（双编码器）："Docker" 和 "Docker" 长得像 → 高相似度。**不管两个放一起是什么意思。**
- Reranker（交叉编码器）：查询和文档拼接后一起看 → 能判断"后者到底有没有回答前者的问题"。

**Reranker 是生成前的最后一道质量把关。** 所以管道里它总是最后一个检索环节。

```
RRF 融合后的 top-10：                Reranker 重排后：
① "Qdrant HNSW 参数"    ← 相关       ① "Qdrant HNSW 参数"
② "向量数据库概述"      ← 一般       ③ "HNSW 原理与实现"
③ "HNSW 原理与实现"     ← 相关       ② "向量数据库概述"   ← 被降到第2
④ "什么是神经网络"      ← 不相关     ⑤ "MySQL 索引优化"   ← 不相关降到最后
⑤ "MySQL 索引优化"      ← 不相关     ④ "什么是神经网络"
```

Reranker（交叉编码器）比向量检索（双编码器）精度高得多，因为查询和文档同时过一遍 Transformer，能看到交互信息。

#### 模型：BGE-Reranker-v2-m3

```
向量检索用的模型：BGE-M3（双编码器 Bi-Encoder）
                 查询和文档各自编码，余弦相似度计算
                 速度快，精度一般

重排序用的模型：BGE-Reranker-v2-m3（交叉编码器 Cross-Encoder）
               查询和文档拼接在一起过 Transformer
               速度慢，精度高
```

**双编码器 vs 交叉编码器：**

```
双编码器：       "Qdrant" → [0.01, ...]       ← 各自独立编码
                 "HNSW"   → [0.02, ...]       

交叉编码器：     "[CLS] Qdrant [SEP] HNSW 是..."  ← 拼接编码
                 能看到查询和文档的关系
```

交叉编码器精度高但慢（每条都要过一遍 Transformer），所以**只能重排 top-N（如 20 条），不能搜全量**。

#### 双模式（和 embedder 一样）

```python
# 本地模式（FlagReranking，加载模型到 CPU）
self._model = FlagReranker(model_name_or_path="BAAI/bge-reranker-v2-m3")

# API 模式（调用远程 rerank 接口）
resp = await client.post("/rerank", json={"query": ..., "documents": [...]})
```

本地模型加载失败自动降级到 API。

#### 重排序在管道中的位置

```
BM25 → 向量 → RRF 融合 → Reranker 重排 → ContextFusion → 生成
                          ↑
                    这里是最慢的一环
                    100 条 × 50ms ≈ 5 秒
```

所以：
- BM25、向量各自返回 N 条
- RRF 融合成一份候选列表
- Reranker **只重排 top-20**（避免全覆盖）
- ContextFusion 处理多引擎优先级

#### L1 理解

- Reranker 和 RRF 有什么区别？
- 为什么有了向量检索还需要 Reranker？
- 双编码器和交叉编码器的区别（速度 vs 精度）？
- 为什么 Reranker 不能搜全量，只能重排 top-N？

#### L2 审查

- AI 把 Reranker 当向量检索用 → 1000 条×50ms=50 秒，不可接受
- Reranker 的 top_k 设得比 RRF 还大 → 重排的候选太多，白费工夫
- 每次 rerank 都重新加载模型 → 应该懒加载
- 忘记降级 → API 不可用时应该退回到 RRF 排序结果

---

## 模块二：LLM API 与 Prompt 工程

### 2.1 OpenAI 兼容 API

**文件：** `src/enterprise_kb/core/generator.py` — 167 行

#### 项目用的什么

DeepSeek API（deepseek-v4-pro），兼容 OpenAI 的 `/chat/completions` 格式。

#### 请求结构

```python
payload = {
    "model": "deepseek-v4-pro",
    "messages": [
        {"role": "system", "content": "你是一个企业知识库助手..."},
        {"role": "user", "content": user_prompt},
    ],
    "temperature": 0.3,
    "max_tokens": 2048,
}
resp = httpx.post(url, json=payload, headers=headers)
answer = resp.json()["choices"][0]["message"]["content"]
```

#### 三种 role 的分工

| role | 职责 | 项目实际内容 |
|------|------|------------|
| **system** | 行为约束、全局规则 | 6 条规则：只使用上下文、引用来源、冲突时优先 Wiki、保持代码格式... |
| **user** | 用户查询 + 检索到的上下文 | 拼接后的上下文文本 + 用户问题 |
| **assistant** | 模型生成的回复 | API 返回的 content |

system 定规则，user 给材料，assistant 收答案。

#### 异常降级

```python
try:
    resp = await client.post(...)
except TimeoutException:    → "请求超时" + 返回上下文原文
except HTTPStatusError:     → "服务异常(4xx/5xx)" + 返回上下文原文
except Exception:           → "生成失败" + 返回上下文原文
```

任何时候都不让用户看到空白——至少给上下文原文。

#### L1 理解

- OpenAI 兼容 API 的三种 role 分别负责什么？
- 项目中 LLM 调用发生在管道的什么位置？
- 异常降级的三种场景分别怎么处理？

### 2.2 Prompt 工程

**文件：** `src/enterprise_kb/core/generator.py` — `DEFAULT_SYSTEM_PROMPT`

#### 项目用的 System Prompt

```text
你是一个企业知识库助手。请根据提供的上下文信息回答用户问题。

规则：
1. 只使用上下文中的信息回答，不要编造事实
2. 如果上下文不足以回答问题，请说"根据现有知识库无法回答这个问题"
3. 在回答中引用来源，使用 [来源: 文档名称] 格式
4. 如果多个来源的信息有冲突，优先采信 Wiki 知识库的内容
5. 用清晰、结构化的语言组织答案
6. 如果上下文包含代码块，在答案中保持代码格式
```

#### 每条规则对应的实际问题

| 规则 | 解决 | 面试关键词 |
|------|------|-----------|
| ① 只使用上下文 | **幻觉**——不限制 LLM 就编 | "我们通过显式约束上下文来减少幻觉" |
| ② 无法回答时直说 | **过度自信**——不说不知道就编 | "我们处理了模型对不确定性的表达" |
| ③ 引用来源 | **可信度**——用户想知道来源 | "我们强制输出引用标记，可溯源" |
| ④ Wiki 优先 | **冲突解决**——不同来源数据打架 | "不同层级的记忆有明确的优先级规则" |
| ⑤ 结构化 | **可读性**——一段话糊脸上没人看 | "我们要求结构化输出" |
| ⑥ 保持代码格式 | **格式损坏**——代码被 markdown 吃 | "我们保留了代码块结构" |

6 条规则 = RAG Prompt 的**最佳实践清单**。

#### L1 理解

- System Prompt 的 6 条规则各自解决什么问题？
- 没有规则 ① 会有什么问题？（幻觉）
- 没有规则 ② 会有什么问题？（瞎编答案）
- 规则 ④ 为什么是 Wiki 优先？

---
