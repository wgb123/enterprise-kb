"""FastAPI 应用主入口 — 生命周期管理 + 路由注册。

启动命令：
    uvicorn enterprise_kb.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from enterprise_kb.agent import ToolRegistry
from enterprise_kb.agent.default_tools import register_default_tools
from enterprise_kb.api.dependencies import get_hybrid_retriever, get_wiki_navigator
from enterprise_kb.api.routes import router
from enterprise_kb.langchain.lc_routes import router as lc_router
from enterprise_kb.config import settings
from enterprise_kb.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。

    启动时：
    1. 预热 Wiki 导航器
    2. 初始化 Qdrant 集合
    3. 注册 Agent 内置工具
    关闭时：释放资源。
    """
    logger.info("Starting %s ...", settings.app_name)

    wiki_nav = None
    hybrid = None

    # 启动时预热模块
    try:
        wiki_nav = get_wiki_navigator()
        await wiki_nav.load()
        logger.info("Wiki navigator initialized: %d pages", len(wiki_nav._pages))
    except Exception as exc:
        logger.warning("Wiki navigator init skipped: %s", exc)

    try:
        hybrid = get_hybrid_retriever()
        await hybrid.vector_store.ensure_collection()
        logger.info("Qdrant collection ready")
    except Exception as exc:
        logger.warning("Qdrant init skipped (offline mode): %s", exc)

    # 注册 Agent 工具
    try:
        tool_count = register_default_tools(
            wiki_nav=wiki_nav,
            hybrid_retriever=hybrid,
        )
        logger.info("Agent tools registered: %d tools", tool_count)
    except Exception as exc:
        logger.warning("Agent tool registration skipped: %s", exc)

    yield

    # 关闭时释放资源
    try:
        if hybrid:
            await hybrid.vector_store.close()
            logger.info("Qdrant client closed")
    except Exception:
        pass

    logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Enterprise Layered Knowledge Base API\n\n"
    "分层记忆架构：\n"
    "- A类: 编译式记忆 (DeepWiki)\n"
    "- B类: HybridRAG 检索式记忆\n"
    "- C类: GraphRAG (预留)\n",
    lifespan=lifespan,
)

# 挂载静态文件（Web 查询页面）
static_dir = Path(__file__).resolve().parent.parent.parent / "static"
if static_dir.is_dir():
    app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="static")


@app.get("/")
async def root_redirect():
    """根路径重定向到 Web UI。"""
    return RedirectResponse(url="/ui/")

# 注册路由
app.include_router(router)
app.include_router(lc_router)
