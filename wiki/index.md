---
title: Enterprise KB 知识库入口
category: root
tags: [index, navigation]
last_updated: 2026-06-16
author: system
---

# Enterprise KB 知识库

> 这是 A 类编译式记忆（DeepWiki）的总入口。
> 优先级最高，作为系统的单一真相来源。

## 目录结构

```
wiki/
├── SCHEMA.md           ← 知识库格式规范
├── index.md            ← 本文件：入口导航
└── categories/         ← 按分类的知识条目
    └── ...
```

## 使用方式

- 知识库通过 `WikiNavigator` 模块自动加载
- 通过 API `GET /api/v1/wiki?q=关键词` 搜索
- 项目文档位于 `docs/` 目录，不在此知识库中
