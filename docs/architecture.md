# 系统架构概览

## 分层记忆架构

Enterprise KB 采用三层记忆架构：

### A 类：编译式记忆（DeepWiki）
- 存储在 `wiki/` 目录下的 Markdown 文件
- 结构为 `SCHEMA.md` → `index.md` → 分类页面
- 支持自动导航和内部链接跳转
- 优先级最高，作为单一真相来源

### B 类：HybridRAG 检索式记忆
- Qdrant 向量数据库存储语义向量
- BM25 关键词检索作为互补
- RRF（Reciprocal Rank Fusion）融合两种检索结果
- BGE-Reranker 进行最终重排序

### C 类：GraphRAG（预留）
- 知识图谱驱动的检索增强生成
- 接口已设计，实现待后续扩展

## 查询流程

1. 用户输入查询
2. 智能路由模块判断查询意图
3. 根据意图分发到 Wiki / HybridRAG / 两者并行
4. 上下文融合层合并结果
5. LLM API 生成最终答案

## 技术栈

- Python 3.10+
- FastAPI + Pydantic v2
- LlamaParse / llama-cloud 文档解析
- Qdrant 向量数据库
- BGE-M3 / BGE-Reranker-v2-m3
- OpenAI 兼容 LLM API（DeepSeek / 通义千问 等）
