#!/usr/bin/env python3
"""对比 demo：手写 SmartRouter（关键词路由） vs LlamaIndex RouterQueryEngine（LLM语义路由）。

运行方式：
  .venv/bin/python scripts/demo_router_comparison.py

不依赖外部服务，可独立运行。
需要 DeepSeek/OpenAI API Key 的场景会注明。
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# ── 测试查询集 ──

TEST_QUERIES = [
    # (查询, 期望的引擎, 说明)
    ("系统架构是什么样的？",      "wiki",     "A 类：架构文档"),
    ("DeepWiki 和编译式记忆",    "wiki",     "A 类：Wiki 关键词"),
    ("开发环境和依赖怎么安装",    "wiki",     "A 类：开发指南"),
    ("运维怎么部署 Qdrant",      "wiki",     "A 类：运维手册"),
    ("查找关于 Python 的文档",   "hybrid",   "B 类：文档检索"),
    ("谁发明了 Python 语言",     "hybrid",   "B 类：通用知识"),
    ("数据库有哪些种类",         "hybrid",   "B 类：需要混合检索"),
    ("今天天气怎么样",           "hybrid",   "通用：需要 LLM 生成"),
    ("计算 12345 * 6789",        "hybrid",   "通用：数学计算"),
    ("LlamaParse 和 Qdrant",     "hybrid",   "技术术语混合"),
]

# ============================================================
# 第一部分：手写 SmartRouter（关键词路由）
# ============================================================

def demo_smartrouter():
    """演示当前手写的 SmartRouter 关键词路由。"""
    print("=" * 70)
    print("🔧 【方法一】手写 SmartRouter — 关键词匹配路由")
    print("=" * 70)

    from enterprise_kb.core.router import SmartRouter
    from enterprise_kb.interfaces.router import QueryIntent

    router = SmartRouter()

    print(f"\n{'查询':<28} {'意图':<12} {'→Wiki':<8} {'→HybridRAG':<10} {'期望':<10} {'✅/❌'}")
    print("-" * 80)

    correct = 0
    for query, expected, note in TEST_QUERIES:
        intent = router.classify(query)
        use_wiki = router.should_use_wiki(query)
        use_rag = router.should_use_hybrid_rag(query)

        # 判断是否匹配期望
        if expected == "wiki":
            ok = use_wiki and not use_rag
        else:
            ok = use_wiki and use_rag  # Hybrid 模式两者都走

        if ok:
            correct += 1
        mark = "✅" if ok else "❌"

        short_q = query[:26] + ".." if len(query) > 26 else query
        print(f"  {short_q:<26} {intent.value:<12} {str(use_wiki):<8} {str(use_rag):<10} {expected:<10} {mark}")

    total = len(TEST_QUERIES)
    print(f"\n📊 准确率: {correct}/{total} = {correct/total*100:.0f}%")
    print(f"   问题: 关键词路由无法理解语义，\"谁发明了 Python\" 无法命中 Wiki 关键词")
    return router


# ============================================================
# 第二部分：LlamaIndex RouterQueryEngine（LLM 语义路由）
# ============================================================

def build_llamaindex_router(api_key: str = ""):
    """构建 LlamaIndex 的 RouterQueryEngine。

    用 LLM 做语义判断来决定将查询路由到哪个引擎。
    """
    from llama_index.core.query_engine import RouterQueryEngine
    from llama_index.core.tools import QueryEngineTool
    from llama_index.core import Document, SummaryIndex
    from llama_index.core.selectors import LLMSingleSelector
    from llama_index.llms.openai import OpenAI

    # ── 创建 LLM（OpenAI 兼容接口，支持 DeepSeek / OpenAI 等） ──
    llm = OpenAI(
        model="deepseek-chat",
        api_key=api_key,
        api_base="https://api.deepseek.com/v1",
        temperature=0,
    )

    # ── 创建两个虚拟的 QueryEngine ──
    # 在实际项目中，这里就是真实的 WikiNavigator 和 HybridRetriever

    # Agent A：Wiki 引擎
    wiki_docs = [
        Document(
            text="Enterprise KB 采用三层记忆架构：A类编译式记忆(DeepWiki)、B类HybridRAG、C类GraphRAG(预留)。"
                 "Wiki 存储在本地 wiki/ 目录下的 Markdown 文件中。",
            metadata={"source": "wiki", "topic": "architecture"},
        ),
        Document(
            text="开发环境配置：Python 3.10+，创建虚拟环境 .venv，pip install -e '.[dev]'，配置 .env 文件。",
            metadata={"source": "wiki", "topic": "development"},
        ),
        Document(
            text="Qdrant 部署方式：本地二进制 qdrant_data/qdrant --config-path config.yaml，或 Docker。",
            metadata={"source": "wiki", "topic": "operations"},
        ),
    ]
    wiki_index = SummaryIndex.from_documents(wiki_docs)
    wiki_engine = wiki_index.as_query_engine(llm=llm)

    # Agent B：HybridRAG 引擎
    rag_docs = [
        Document(
            text="Python 是一种广泛使用的解释型高级编程语言，由 Guido van Rossum 于 1991 年发布。"
                 "支持面向对象、函数式和过程式编程范式。",
            metadata={"source": "hybrid_rag", "topic": "python"},
        ),
        Document(
            text="Qdrant 是一个高性能向量数据库，专为向量相似度搜索设计。"
                 "支持余弦相似度、点积，支持 HNSW 索引和过滤搜索。",
            metadata={"source": "hybrid_rag", "topic": "database"},
        ),
    ]
    rag_index = SummaryIndex.from_documents(rag_docs)
    rag_engine = rag_index.as_query_engine(llm=llm)

    # ── 创建 RouterQueryEngine ──
    wiki_tool = QueryEngineTool.from_defaults(
        query_engine=wiki_engine,
        description=(
            "用于回答关于系统架构、开发指南、运维部署等内部知识的问题。"
            "包括架构设计、三层记忆、环境配置、Qdrant 部署等。"
            "如果查询与系统架构、技术决策、开发配置相关，请选择此工具。"
        ),
    )

    rag_tool = QueryEngineTool.from_defaults(
        query_engine=rag_engine,
        description=(
            "用于回答关于外部知识、通用技术问题、具体技术文档内容的问题。"
            "包括编程语言、框架、数据库、算法等的详细信息。"
            "如果查询需要外部知识或具体技术名词的解释，请选择此工具。"
        ),
    )

    router_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=[wiki_tool, rag_tool],
        verbose=True,
    )

    return router_engine


def demo_llamaindex():
    """演示 LlamaIndex RouterQueryEngine（需要 API Key）。"""
    print("\n" + "=" * 70)
    print("🤖 【方法二】LlamaIndex RouterQueryEngine — LLM 语义路由")
    print("=" * 70)
    print()
    print("  原理：LLM 读取查询 + 每个引擎的描述 → 判断走哪个引擎")
    print()

    # 尝试读取 API Key
    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("VLLM_API_KEY", "")
    api_url = os.getenv("VLLM_API_URL", "")

    if not api_key:
        print("  ⚠️  .env 中未配置 VLLM_API_KEY，无法调用 LLM")
        print("  下面展示的是配置代码和路由原理，无法实际执行。")
        print()
        _show_code_pattern()
        return

    print(f"  使用 API: {api_url}")
    print(f"  使用模型: {os.getenv('VLLM_MODEL_NAME', 'deepseek-chat')}")
    print()

    try:
        router_engine = build_llamaindex_router(api_key)

        print(f"\n{'查询':<28} {'→引擎':<20} {'说明'}")
        print("-" * 70)

        for query, expected, note in TEST_QUERIES:
            response = router_engine.query(query)
            # 从响应中提取引擎选择信息
            chosen = getattr(response, "metadata", {}).get("selector_result", str(response))

            short_q = query[:26] + ".." if len(query) > 26 else query
            print(f"  {short_q:<26} → 选择 Wiki 或 RAG (LLM 判断)")

    except Exception as exc:
        print(f"  ❌ 调用失败: {exc}")
        print("  可能是 API Key 问题或网络问题。")
        _show_code_pattern()


def _show_code_pattern():
    """展示 LlamaIndex RouterQueryEngine 的核心代码模式。"""
    print("  ┌────────────────────────────────────────────────────────────────┐")
    print("  │ LlamaIndex RouterQueryEngine 核心配置代码:                      │")
    print("  └────────────────────────────────────────────────────────────────┘")
    print("""
  from llama_index.core.query_engine import RouterQueryEngine
  from llama_index.core.tools import QueryEngineTool
  from llama_index.core.selectors import LLMSingleSelector
  from llama_index.llms.openai import OpenAI

  llm = OpenAI(model="deepseek-chat", api_key="sk-xxx")

  # 每个工具有一段自然语言描述
  wiki_tool = QueryEngineTool.from_defaults(
      query_engine=wiki_engine,
      description="用于回答系统架构、开发、运维等内部知识...",
  )
  rag_tool = QueryEngineTool.from_defaults(
      query_engine=rag_engine,
      description="用于回答外部知识、通用技术问题...",
  )

  # LLM 根据查询 + 描述，自动选择引擎
  engine = RouterQueryEngine(
      selector=LLMSingleSelector.from_defaults(llm=llm),
      query_engine_tools=[wiki_tool, rag_tool],
  )

  result = engine.query("谁发明了 Python")
  # LLM 会理解"Python"是通用知识 → 自动走 RAG 引擎
  """)


# ============================================================
# 第三部分：对比总结
# ============================================================

def comparison_summary():
    print("\n" + "=" * 70)
    print("📊 对比总结")
    print("=" * 70)
    print()
    print("  ┌──────────────────────────────┬──────────────────────────────┐")
    print("  │      手写 SmartRouter        │    LlamaIndex 语义路由       │")
    print("  ├──────────────────────────────┼──────────────────────────────┤")
    print("  │ 路由依据：关键词匹配         │ 路由依据：LLM 语义理解       │")
    print("  │ 示例：\"wiki\" → Wiki 引擎     │ 示例：\"谁发明的 Python\"     │")
    print("  │       \"文档\" → RAG 引擎      │       → LLM 判断：RAG 引擎   │")
    print("  ├──────────────────────────────┼──────────────────────────────┤")
    print("  │ ✅ 零依赖，轻量              │ ✅ 语义准确，误判少         │")
    print("  │ ✅ 速度快（μs级）            │ ❌ 慢（需调 LLM，秒级）     │")
    print("  │ ✅ 完全可控                  │ ❌ 依赖外部 LLM 服务        │")
    print("  │ ❌ 语义盲区                  │ ❌ 每次路由花 token 费     │")
    print("  │   \"谁发明了 Python\"         │ ❌ LLM 可能选错             │")
    print("  │   关键词不匹配 → 走 general  │                              │")
    print("  ├──────────────────────────────┼──────────────────────────────┤")
    print("  │ 代码量：~60行                │ 代码量：~20行 + 框架        │")
    print("  │ 维护成本：低                 │ 维护成本：需跟踪框架版本    │")
    print("  │ 扩展方式：加关键词列表       │ 扩展方式：加工具描述文本    │")
    print("  └──────────────────────────────┴──────────────────────────────┘")
    print()
    print("  💡 最佳实践：两者结合")
    print("    先用 SmartRouter 做快速过滤（成本几乎为零）")
    print("    边界模糊的查询再升级到 LLM 判断（精准但慢）")
    print("""
    伪代码：
      quick = smart_router.classify(query)
      if quick == QueryIntent.GENERAL:
          # 关键词无法判断 → 交给 LLM
          final = llm_router.route(query)
      else:
          # 关键词明确 → 直接用
          final = quick
    """)


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   SmartRouter  vs  LlamaIndex RouterQueryEngine         ║")
    print("║   手写关键词路由 vs LLM 语义路由对比                    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_smartrouter()
    demo_llamaindex()
    comparison_summary()
