"""Wiki 导航器单元测试。"""

from pathlib import Path

import pytest

from enterprise_kb.core.wiki_navigator import WikiNavigator, WikiPage


@pytest.mark.asyncio
class TestWikiNavigator:
    """WikiNavigator 核心功能测试。"""

    async def test_load(self, wiki_root: Path):
        """应正确加载 Wiki 目录中的所有 .md 文件。"""
        nav = WikiNavigator(wiki_root)
        await nav.load()
        assert len(nav._pages) >= 4  # SCHEMA.md + index.md + 3 categories

    async def test_get_index(self, wiki_root: Path):
        """应返回 index.md 页面。"""
        nav = WikiNavigator(wiki_root)
        page = await nav.get_index()
        assert page is not None
        assert page.title == "Enterprise KB 知识库入口"
        assert page.category == "root"

    async def test_get_page(self, wiki_root: Path):
        """应返回指定路径的页面。"""
        nav = WikiNavigator(wiki_root)
        page = await nav.get_page("categories/architecture.md")
        assert page is not None
        assert "系统架构" in page.title

    async def test_get_page_without_extension(self, wiki_root: Path):
        """自动补全 .md 后缀。"""
        nav = WikiNavigator(wiki_root)
        page = await nav.get_page("index")
        assert page is not None
        assert page.title == "Enterprise KB 知识库入口"

    async def test_query(self, wiki_root: Path):
        """全文搜索应返回匹配结果。"""
        nav = WikiNavigator(wiki_root)
        await nav.load()
        result = await nav.query("架构")
        assert len(result.chunks) > 0
        assert any("架构" in c.content for c in result.chunks)

    async def test_get_tags(self, wiki_root: Path):
        """应返回所有唯一标签。"""
        nav = WikiNavigator(wiki_root)
        await nav.load()
        tags = await nav.get_tags()
        assert isinstance(tags, list)
        assert "architecture" in tags
        assert "development" in tags

    async def test_get_by_tag(self, wiki_root: Path):
        """应按标签过滤页面。"""
        nav = WikiNavigator(wiki_root)
        await nav.load()
        pages = await nav.get_by_tag("operations")
        assert len(pages) > 0
        assert all("operations" in p.tags for p in pages)

    async def test_resolve_link(self, wiki_root: Path):
        """内部链接应正确解析。"""
        nav = WikiNavigator(wiki_root)
        page = await nav.resolve_link("categories/architecture")
        assert page is not None
        assert "架构" in page.title

    async def test_missing_page(self, wiki_root: Path):
        """不存在的页面应返回 None。"""
        nav = WikiNavigator(wiki_root)
        page = await nav.get_page("nonexistent.md")
        assert page is None

    def test_generate_toc(self):
        """TOC 应正确解析 Markdown 标题。"""
        md = "# H1\nContent\n## H2\nMore\n### H3\nDetail"
        toc = WikiNavigator._generate_toc(md)
        assert "H1" in toc or "H2" in toc or "H3" in toc

    def test_page_dataclass(self):
        """WikiPage 应正确初始化。"""
        page = WikiPage(
            path=Path("/test.md"),
            title="Test",
            category="test",
            tags=["tag1", "tag2"],
            content="Hello",
            html_toc="- [Hello](#hello)",
        )
        assert page.title == "Test"
        assert "tag1" in page.tags
