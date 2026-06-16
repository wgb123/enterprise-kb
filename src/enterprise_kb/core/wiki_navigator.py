"""Wiki Navigator — A类编译式记忆 (DeepWiki) 的导航与检索模块。

支持:
- 解析 wiki/ 目录下的 Markdown 页面结构
- YAML front-matter 元数据提取
- 内部链接跳转（[[wikilink]] 和 Markdown 链接）
- 页面搜索和标签过滤
- [[TOC]] 占位符自动替换为导航目录
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from enterprise_kb.interfaces.base import BaseEngine, QueryResult, RetrievedChunk


@dataclass
class WikiPage:
    """Representation of a single Wiki page."""

    path: Path  # absolute filesystem path
    title: str
    category: str
    tags: list[str] = field(default_factory=list)
    content: str = ""
    html_toc: str = ""


@dataclass
class WikiLink:
    """An internal Wiki link (either [[wikilink]] or [text](path))."""

    text: str
    target: str  # relative path from wiki root
    anchor: str = ""


class WikiNavigator(BaseEngine):
    """Navigates the local Markdown knowledge base (A-class memory).

    The Wiki root directory structure must follow the SCHEMA.md convention:
        wiki/
        ├── SCHEMA.md
        ├── index.md
        └── categories/ ...
    """

    def __init__(self, wiki_root: str | Path) -> None:
        self._root = Path(wiki_root).resolve()
        self._pages: dict[str, WikiPage] = {}  # keyed by relative path
        self._loaded = False

    # ── Public API ──

    async def load(self) -> None:
        """Scan the wiki directory and parse all Markdown pages."""
        if not self._root.is_dir():
            msg = f"Wiki root not found: {self._root}"
            raise FileNotFoundError(msg)

        self._pages.clear()
        md_files = sorted(self._root.rglob("*.md"))
        for filepath in md_files:
            rel = filepath.relative_to(self._root).as_posix()
            raw = filepath.read_text(encoding="utf-8")
            page = self._parse_page(filepath, rel, raw)
            self._pages[rel] = page

        self._loaded = True

    async def query(self, query: str, top_k: int = 10) -> QueryResult:
        """Retrieve Wiki pages by full-text search over title + content.

        Args:
            query: User query string.
            top_k: Maximum number of matching pages.

        Returns:
            :class:`QueryResult` with matched chunks.
        """
        if not self._loaded:
            await self.load()

        q_lower = query.lower()
        scored: list[tuple[float, RetrievedChunk]] = []

        for rel, page in self._pages.items():
            score = self._score_page(page, q_lower)
            if score <= 0:
                continue
            chunk = RetrievedChunk(
                content=page.content[:500],  # preview length
                source="wiki",
                document_id=rel,
                chunk_id=rel,
                score=score,
                metadata={
                    "title": page.title,
                    "category": page.category,
                    "tags": page.tags,
                    "path": rel,
                },
            )
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        chunks = [c for _, c in scored[:top_k]]

        return QueryResult(query=query, chunks=chunks, engine="wiki")

    async def get_page(self, path: str) -> WikiPage | None:
        """Get a parsed page by its relative path from wiki root.

        Args:
            path: Relative path e.g. ``categories/architecture.md``.

        Returns:
            The :class:`WikiPage` if found, else ``None``.
        """
        if not self._loaded:
            await self.load()
        return self._pages.get(path)

    async def get_index(self) -> WikiPage | None:
        """Get the index (root) page.

        Returns:
            The index :class:`WikiPage`, or ``None`` if missing.
        """
        return await self.get_page("index.md")

    async def search(self, query: str, top_k: int = 10) -> list[WikiPage]:
        """Full-text search across Wiki pages (returns parsed objects).

        Args:
            query: Search query.
            top_k: Max results.

        Returns:
            List of matching :class:`WikiPage` objects.
        """
        result = await self.query(query, top_k)
        matched: list[WikiPage] = []
        for chunk in result.chunks:
            page = await self.get_page(chunk.document_id)
            if page:
                matched.append(page)
        return matched

    async def resolve_link(self, link_text: str) -> WikiPage | None:
        """Resolve an internal wiki link to a page.

        Supports:
        - ``[[target]]`` → ``target.md``
        - ``[[target|display]]`` → ``target.md``
        - ``[text](target.md)`` → ``target.md``

        Args:
            link_text: The link target (without file extension if markdown).

        Returns:
            The linked :class:`WikiPage`, or ``None`` if unresolvable.
        """
        target = link_text.strip()
        # Strip display text from wikilink
        if "|" in target:
            target = target.split("|")[0]
        if "#" in target:
            target, _ = target.split("#", 1)

        target = target.strip()
        if not target.endswith(".md"):
            target = target + ".md"

        return await self.get_page(target)

    async def get_tags(self) -> list[str]:
        """Get all unique tags across Wiki pages.

        Returns:
            Sorted list of tag strings.
        """
        if not self._loaded:
            await self.load()
        tags: set[str] = set()
        for page in self._pages.values():
            tags.update(page.tags)
        return sorted(tags)

    async def get_by_tag(self, tag: str) -> list[WikiPage]:
        """Get all pages that have a specific tag.

        Args:
            tag: Tag to filter by.

        Returns:
            List of matching :class:`WikiPage` objects.
        """
        if not self._loaded:
            await self.load()
        return [p for p in self._pages.values() if tag in p.tags]

    # ── Internal helpers ──

    @staticmethod
    def _parse_page(filepath: Path, rel: str, raw: str) -> WikiPage:
        """Parse a Markdown file into a WikiPage dataclass."""
        content = raw
        front_matter: dict = {}
        page_title = rel

        # Extract YAML front-matter
        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
        if fm_match:
            try:
                front_matter = yaml.safe_load(fm_match.group(1)) or {}
            except yaml.YAMLError:
                front_matter = {}
            content = raw[fm_match.end() :]

        title = str(front_matter.get("title", page_title))
        category = str(front_matter.get("category", "uncategorized"))
        tags_raw = front_matter.get("tags", [])
        tags = [str(t).strip() for t in tags_raw] if isinstance(tags_raw, list) else []

        # Generate TOC
        toc = WikiNavigator._generate_toc(content)

        # Remove [[TOC]] placeholder (replaced by generated TOC during display)
        content = content.replace("[[TOC]]", toc)

        return WikiPage(
            path=filepath,
            title=title,
            category=category,
            tags=tags,
            content=content,
            html_toc=toc,
        )

    @staticmethod
    def _generate_toc(markdown: str) -> str:
        """Generate a simple text-based table of contents from Markdown headers."""
        lines = []
        for line in markdown.splitlines():
            m = re.match(r"^(#{1,4})\s+(.+)$", line.strip())
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                indent = "  " * (level - 1)
                anchor = title.lower().replace(" ", "-").replace("/", "-")
                lines.append(f"{indent}- [{title}](#{anchor})")
        return "\n".join(lines)

    @staticmethod
    def _score_page(page: WikiPage, query_lower: str) -> float:
        """Simple scoring: title match > content match."""
        score = 0.0
        if query_lower in page.title.lower():
            score += 10.0
        if query_lower in page.content.lower():
            score += 1.0 * (page.content.lower().count(query_lower))
        for tag in page.tags:
            if query_lower in tag.lower():
                score += 3.0
        return score
