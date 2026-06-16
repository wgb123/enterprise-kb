---
title: A 类知识库格式规范
category: schema
tags: [schema, specification]
last_updated: 2026-06-16
author: system
---

# Enterprise KB — 编译式记忆 (DeepWiki) Schema

> 单一真相来源 (Single Source of Truth)
> 优先级最高，覆盖其他所有记忆层。

## 定位

`wiki/` 目录是 **A 类编译式记忆的数据存储**，存放精选的、权威的知识条目。
与之不同的是，`docs/` 目录存放的是**项目文档**（架构、开发、运维），两者不重叠。

## 目录结构

```
wiki/
├── SCHEMA.md          ← 本文件：定义知识库规范和结构
├── index.md           ← 全局入口页面，管理导航
└── categories/        ← 按领域分类的知识条目
    └── ...            ← 实际的知识内容（Markdown 文件）
```

## 页面规范

### 1. 元数据头（每个页面必填）

每个 `.md` 文件必须以 YAML front-matter 开头：

```yaml
---
title: 页面标题
category: 所属分类 (e.g., architecture, development)
tags: [tag1, tag2, tag3]
last_updated: YYYY-MM-DD
author: maintainer_name
---
```

### 2. 内部链接语法

使用标准 Markdown 链接格式，路径相对于 `wiki/` 根目录：

```markdown
- [相关概念](categories/some-topic.md)
- [具体章节](categories/some-topic.md#section-anchor)
```

### 3. 导航层级

- `index.md` 是知识库默认路由入口
- 每个分类页面应在顶部包含 `[[TOC]]` 占位符（由 WikiNavigator 自动解析）
- 分类页面通过 `##` 标题实现锚点跳转

### 4. 标签系统

每页的 `tags` 元数据用于自动生成标签索引。
WikiNavigator 支持按标签过滤。

### 5. 版本管理

当知识库内容与 HybridRAG 检索结果冲突时，
**Wiki 编译式记忆的权威性最高**，始终以 Wiki 为准。
