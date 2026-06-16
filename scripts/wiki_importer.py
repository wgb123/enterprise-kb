#!/usr/bin/env python3
"""维基百科 XML dump 批量导入工具。

策略：BM25 全量索引 + 标题/摘要选择性向量化
- BM25 (rank-bm25)：全文关键词检索，对专有名词友好
- Qdrant 向量：仅对标题+前 200 tokens 做嵌入，作为语义补充
- 断点续传：每批处理后记录已处理的 page ID
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import bz2
from lxml import etree

# 项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from enterprise_kb.config import settings
from enterprise_kb.core.embedder import BGEM3Embedder
from enterprise_kb.storage.bm25_index import BM25Index
from enterprise_kb.storage.qdrant_store import QdrantStore
from enterprise_kb.utils.logger import logger


# ── Wiki Markup 清洗 ──

WIKI_TEMPLATE_RE = re.compile(r"\{\{[^}]*?\}\}|\{\{[^}]*?\n(?:[^{]|\{[^{])*?\}\}", re.DOTALL)
WIKI_REF_RE = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.DOTALL)
WIKI_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
WIKI_BOLD_ITALIC_RE = re.compile(r"'''''(.*?)'''''|'''(.*?)'''|''(.*?)''")
WIKI_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]|]+))?\]\]")
WIKI_HEADING_RE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.MULTILINE)
WIKI_LIST_RE = re.compile(r"^[*#:;]\s?", re.MULTILINE)
WIKI_TABLE_RE = re.compile(r"^\{|\|[\+\-](?:[^|]|\|[^}])*?\|\}", re.MULTILINE | re.DOTALL)
WIKI_CATEGORY_RE = re.compile(r"\[\[Category:[^\]]*\]\]", re.IGNORECASE)
WIKI_FILE_RE = re.compile(r"\[\[(?:File|Image|ファイル):[^\]]*\]\]", re.IGNORECASE)
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def clean_wiki_text(raw: str) -> str:
    """清洗维基百科 Wiki 标记，保留段落结构。"""
    text = raw

    # 移除注释、引用、模板、表格
    text = WIKI_COMMENT_RE.sub("", text)
    text = WIKI_REF_RE.sub("", text)
    text = WIKI_TEMPLATE_RE.sub("", text)
    text = WIKI_TABLE_RE.sub("", text)
    text = WIKI_CATEGORY_RE.sub("", text)
    text = WIKI_FILE_RE.sub("", text)

    # 转换内部链接 [[目标|显示]] → 显示文本，或 [[目标]] → 目标
    def _replace_link(m: re.Match) -> str:
        return m.group(2) if m.group(2) else m.group(1)

    text = WIKI_LINK_RE.sub(_replace_link, text)

    # 清洗加粗/斜体
    def _replace_bold(m: re.Match) -> str:
        return m.group(1) or m.group(2) or m.group(3) or ""

    text = WIKI_BOLD_ITALIC_RE.sub(_replace_bold, text)

    # 列表标记
    text = WIKI_LIST_RE.sub("", text)

    # 规范化空白
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    text = MULTI_SPACE_RE.sub(" ", text)

    return text.strip()


def split_by_section(text: str) -> list[tuple[str, str]]:
    """按 == 标题 == 将文章分节。

    Returns:
        [(heading, body), ...]  第一条 heading 为空（引言部分）
    """
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_body: list[str] = []

    for line in lines:
        m = WIKI_HEADING_RE.match(line)
        if m:
            if current_body or current_heading:
                sections.append((current_heading, "\n".join(current_body).strip()))
                current_body = []
            current_heading = line.strip()
        else:
            current_body.append(line)

    if current_body or current_heading:
        sections.append((current_heading, "\n".join(current_body).strip()))

    return sections


# ── XML 解析 ──

NS = "{http://www.mediawiki.org/xml/export-0.11/}"
# 兼容无命名空间和 0.10 版本
NS_ALT = "{http://www.mediawiki.org/xml/export-0.10/}"


def _tag(name: str, ns: str = NS) -> str:
    return f"{ns}{name}"


def extract_article(page_elem: etree.Element) -> Optional[dict[str, Any]]:
    """从 XML <page> 元素提取文章数据。"""
    # 尝试识别命名空间
    tag_raw = page_elem.tag
    ns = tag_raw[: tag_raw.rindex("}") + 1] if "}" in tag_raw else ""

    def t(name: str) -> str:
        return f"{ns}{name}" if ns else name

    title_el = page_elem.find(t("title"))
    id_el = page_elem.find(t("id"))
    revision = page_elem.find(t("revision"))
    if revision is None or title_el is None:
        return None

    text_el = revision.find(t("text"))
    if text_el is None or not text_el.text:
        return None

    # 跳过重定向
    redirect = page_elem.find(t("redirect"))
    if redirect is not None:
        return None

    raw_text = text_el.text
    # 跳过极小页面（低于 50 字符）
    if len(raw_text.strip()) < 50:
        return None

    return {
        "id": str(id_el.text) if id_el is not None else "",
        "title": title_el.text or "",
        "text": raw_text,
    }


# ── 导入器 ──


class WikipediaImporter:
    """维基百科导入器。

    流程:
    1. 流式读取 bz2 XML
    2. 提取文章 → 清洗 → 分节
    3. 全文→ BM25，标题+摘要 → Qdrant 向量
    4. 每 N 篇提交一批 + 保存断点
    """

    def __init__(
        self,
        batch_size: int = 200,
        embed_summary_chars: int = 500,
        max_chunk_tokens: int = 512,
        checkpoint_path: str = "",
        embed_only_summary: bool = True,
        skip_vectors: bool = False,
        vectors_only: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.embed_summary_chars = embed_summary_chars
        self.max_chunk_tokens = max_chunk_tokens
        self.embed_only_summary = embed_only_summary
        self.skip_vectors = skip_vectors
        self.vectors_only = vectors_only

        # 存储
        self.vector_store = QdrantStore()
        self.bm25_index = BM25Index()
        self.embedder = None if skip_vectors else BGEM3Embedder()

        # 检查点（vectors-only 用独立的 checkpoint 文件）
        if vectors_only:
            default_ckpt = "data/wiki_import_checkpoint_vectors.json"
        else:
            default_ckpt = "data/wiki_import_checkpoint.json"
        self.checkpoint_path = Path(checkpoint_path or project_root / default_ckpt)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # 批处理缓冲区
        self._bm25_buffer: list[dict] = []
        self._vector_buffer: list[dict] = []
        self._processed_ids: set[str] = set()
        self._page_count = 0
        self._chunk_count = 0
        self._start_time = time.time()

        # 全量 BM25 文档累积（_flush_buffers 中 build 只保留最后一批，
        # 所以我们独立累积所有文档，在 _finalize 中一次性构建完整索引）
        self._all_bm25_docs: list[dict] = []
        self._bm25_persist_path = project_root / "data" / "wiki_bm25.pkl"

    # ── 公开 API ──

    def run(self, xml_bz2_path: str, max_pages: int = 0) -> None:
        """运行导入流程。"""
        xml_path = Path(xml_bz2_path)
        if not xml_path.is_file():
            print(f"❌ 文件不存在: {xml_path}")
            sys.exit(1)

        self._load_checkpoint()
        stats = self._parse_and_ingest(xml_path, max_pages)
        self._flush_buffers()
        self._finalize(stats)

    # ── 检查点 ──

    def _load_checkpoint(self) -> None:
        """加载断点，恢复已处理的 page ID。"""
        if self.checkpoint_path.is_file():
            try:
                data = json.loads(self.checkpoint_path.read_text())
                # 兼容新旧格式: processed_ids (旧) 或 recent_ids (新)
                ids = data.get("processed_ids") or data.get("recent_ids") or []
                self._processed_ids = set(ids)
                print(f"📌 发现断点，已处理 {len(self._processed_ids)} 篇")
            except Exception as exc:
                print(f"⚠️ 断点文件损坏，重新开始: {exc}")
                self._processed_ids = set()
        else:
            print("🆕 全新导入（未发现断点）")
            self._processed_ids = set()

    def _save_checkpoint(self) -> None:
        """保存当前处理进度。"""
        ids_list = list(self._processed_ids)
        # 只保留最近的 ID（避免文件过大）
        if len(ids_list) > 10000:
            ids_list = ids_list[-10000:]

        data = {
            "processed_count": len(self._processed_ids),
            "recent_ids": ids_list,
            "page_count": self._page_count,
            "chunk_count": self._chunk_count,
            "timestamp": time.time(),
        }
        self.checkpoint_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    # ── 解析与入库 ──

    def _parse_and_ingest(self, xml_path: Path, max_pages: int) -> dict[str, Any]:
        """流式解析 bz2 XML 并逐篇处理。"""
        file_size = xml_path.stat().st_size
        print(f"\n📂 文件: {xml_path.name} ({file_size / 1024**3:.2f} GB)")
        print(f"📐 批次大小: {self.batch_size} 篇/批")
        print(f"🧠 BM25: 全量文本索引")
        if self.skip_vectors:
            print(f"⏭️  Qdrant: 已跳过（--skip-vectors）")
        else:
            print(f"🔤 Qdrant: 仅标题+前{self.embed_summary_chars}字符")
        print(f"📌 断点: {self.checkpoint_path}")
        print()

        skipped = 0
        errors = 0

        with bz2.BZ2File(str(xml_path), "r") as bz2_file:
            context = etree.iterparse(bz2_file, events=("end",), tag=f"{NS}page")

            for event, elem in context:
                # 如果找不到带命名空间的 page，尝试无命名空间
                if elem.tag != f"{NS}page":
                    # Fallback: 检查其他命名空间或空命名空间
                    pass

                try:
                    article = extract_article(elem)
                except Exception:
                    elem.clear()
                    continue

                if article is None:
                    skipped += 1
                    elem.clear()
                    continue

                # 断点跳过
                if article["id"] in self._processed_ids:
                    skipped += 1
                    elem.clear()
                    continue

                try:
                    self._process_article(article)
                except Exception as exc:
                    errors += 1
                    if errors <= 3:
                        print(f"  ⚠️  处理失败 [{article['title']}]: {exc}")
                    elem.clear()
                    continue

                self._page_count += 1
                self._processed_ids.add(article["id"])

                # 每批 flush + checkpoint
                if self._page_count % self.batch_size == 0:
                    self._flush_buffers()
                    self._save_checkpoint()

                # 每 50 篇打印进度
                if self._page_count % 50 == 0:
                    self._print_progress()

                if max_pages and self._page_count >= max_pages:
                    print(f"\n⏹️  达到限制页面数 ({max_pages})，停止")
                    break

                elem.clear()
                # 清除前驱元素释放内存
                while elem.getprevious() is not None:
                    del elem.getparent()[0]

        stats = {
            "total_pages": self._page_count,
            "total_chunks": self._chunk_count,
            "skipped": skipped,
            "errors": errors,
            "elapsed": time.time() - self._start_time,
        }
        return stats

    def _process_article(self, article: dict[str, Any]) -> None:
        """处理一篇文章：清洗 → 分节 → 准备 BM25 + 向量数据。"""
        title = article["title"]
        text = clean_wiki_text(article["text"])
        if not text:
            return

        sections = split_by_section(text)

        # --- BM25 入索引（vectors-only 模式跳过）---
        if not self.vectors_only:
            for i, (heading, body) in enumerate(sections):
                if not body:
                    continue

                section_text = body
                if heading:
                    section_text = f"{heading}\n{body}"

                chunk_id = hashlib.md5(section_text.encode()).hexdigest()[:12]
                self._bm25_buffer.append({
                    "text": section_text,
                    "metadata": {
                        "title": title,
                        "section_heading": heading.replace("=", "").strip() if heading else "introduction",
                        "source": "wikipedia",
                    },
                    "chunk_id": f"wiki_{article['id']}_{i}_{chunk_id}",
                })

        self._chunk_count += len(sections)

        # --- Qdrant 向量（仅标题+摘要）---
        if not self.skip_vectors and self.embed_only_summary:
            summary = title
            if sections and sections[0][1]:
                summary += "\n" + sections[0][1][:self.embed_summary_chars]

            self._vector_buffer.append({
                "text": summary,
                "metadata": {
                    "title": title,
                    "source": "wikipedia",
                    "section_count": len(sections),
                },
                "chunk_id": f"vec_{article['id']}",
            })

    # ── 提交入库 ──

    def _flush_buffers(self) -> None:
        """将当前缓冲区中的数据写入存储。"""
        if self._bm25_buffer:
            # 累积全量文档（用于最后构建完整 BM25 索引）
            self._all_bm25_docs.extend(self._bm25_buffer)
            # 批次级 build 仅用于统计当前批次的文档数
            self.bm25_index.build(self._bm25_buffer)
            self._bm25_buffer.clear()
            logger.debug("BM25 索引刷新: %d chunks", self.bm25_index.size)

            # 每次 flush 都持久化 BM25（确保中断不丢数据）
            self._save_bm25_checkpoint()

        if self._vector_buffer and self.embedder:
            self._flush_vectors()
        elif self._vector_buffer:
            # skip_vectors 模式: 清空缓冲区
            self._vector_buffer.clear()

    def _flush_vectors(self) -> None:
        """写入向量缓冲区到 Qdrant。"""
        import asyncio

        async def _flush():
            texts = [v["text"] for v in self._vector_buffer]
            if not texts:
                return

            # 确保集合存在
            await self.vector_store.ensure_collection()

            # 嵌入
            vectors = await self.embedder.embed(texts)

            # 构建 PointStruct
            from qdrant_client.models import PointStruct

            points = [
                PointStruct(
                    id=abs(hash(v["chunk_id"])) % (2**63),
                    vector=vec,
                    payload={
                        "text": v["text"][:500],
                        "chunk_id": v["chunk_id"],
                        **v["metadata"],
                    },
                )
                for v, vec in zip(self._vector_buffer, vectors)
            ]

            await self.vector_store.upsert(points)

        try:
            asyncio.run(_flush())
            logger.debug("向量写入完成: %d 篇", len(self._vector_buffer))
        except Exception as exc:
            logger.warning("向量写入失败 (跳过): %s", exc)

        self._vector_buffer.clear()

    # ── BM25 增量持久化 ──

    def _save_bm25_checkpoint(self) -> None:
        """每 10 次 flush 增量持久化当前累积的 BM25 索引。"""
        if not self._all_bm25_docs:
            return
        self._bm25_persist_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  💾 BM25 增量保存: {len(self._all_bm25_docs):,} 篇文档...")
        self.bm25_index.build(self._all_bm25_docs)
        self.bm25_index.save(str(self._bm25_persist_path))

    # ── 最终化 ──

    def _finalize(self, stats: dict[str, Any]) -> None:
        """导入完成后的清理和持久化。"""
        # 持久化 BM25 索引（vectors-only 模式跳过，Phase 1 已建好）
        if not self.vectors_only:
            bm25_path = project_root / "data" / "wiki_bm25.pkl"
            bm25_path.parent.mkdir(parents=True, exist_ok=True)
            if self._all_bm25_docs:
                print(f"\n🔨 构建完整 BM25 索引: {len(self._all_bm25_docs):,} 文档...")
                self.bm25_index.build(self._all_bm25_docs)
                self.bm25_index.save(str(bm25_path))
                print(f"💾 BM25 索引已持久化: {bm25_path} ({self.bm25_index.size} docs)")
            elif self.bm25_index.is_built:
                self.bm25_index.save(str(bm25_path))
                print(f"\n💾 BM25 索引已持久化: {bm25_path} ({self.bm25_index.size} docs)")

        # 保存最终检查点
        self._save_checkpoint()

        elapsed = stats.get("elapsed", 0)
        pages = stats.get("total_pages", 0)
        chunks = stats.get("total_chunks", 0)
        rate = pages / elapsed if elapsed > 0 else 0

        print("\n" + "=" * 60)
        print(f"✅ 导入完成!")
        print(f"   ├─ 文章: {pages:,} 篇")
        print(f"   ├─ 分块: {chunks:,} 块")
        print(f"   ├─ BM25: {self.bm25_index.size:,} 文档索引")
        print(f"   ├─ 跳过: {stats.get('skipped', 0):,} 篇")
        print(f"   ├─ 错误: {stats.get('errors', 0):,} 篇")
        print(f"   ├─ 耗时: {elapsed:.1f}秒 ({rate:.1f} 篇/秒)")
        print(f"   └─ 检查点: {self.checkpoint_path}")
        print("=" * 60)

    def _print_progress(self) -> None:
        """打印当前进度。"""
        elapsed = time.time() - self._start_time
        rate = self._page_count / elapsed if elapsed > 0 else 0
        print(
            f"  📊 {self._page_count:,} 篇 | "
            f"{self._chunk_count:,} 块 | "
            f"{rate:.0f} 篇/秒 | "
            f"BM25: {self.bm25_index.size:,} docs | "
            f"⏱ {elapsed:.0f}s"
        )


# ── CLI ──


def main() -> None:
    parser = argparse.ArgumentParser(
        description="维基百科 XML dump 批量导入工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  # 完整导入\n"
            "  python scripts/wiki_importer.py wiki.xml.bz2\n\n"
            "  # 测试导入前 1000 篇\n"
            "  python scripts/wiki_importer.py wiki.xml.bz2 --max-pages 1000\n\n"
            "  # 指定批次大小\n"
            "  python scripts/wiki_importer.py wiki.xml.bz2 --batch-size 500\n\n"
            "  # 从头开始（忽略断点）\n"
            "  python scripts/wiki_importer.py wiki.xml.bz2 --force\n"
        ),
    )
    parser.add_argument("input", help="维基百科 XML bz2 dump 文件路径")
    parser.add_argument("--batch-size", type=int, default=200, help="每批处理篇数 (默认: 200)")
    parser.add_argument("--max-pages", type=int, default=0, help="最大处理篇数 (0=全部)")
    parser.add_argument(
        "--embed-summary-chars", type=int, default=500, help="摘要向量化的最大字符数 (默认: 500)"
    )
    parser.add_argument(
        "--embed-mode",
        choices=["summary_only", "full"],
        default="summary_only",
        help="向量化模式: summary_only (仅标题+摘要) 或 full (全部文本，CPU 耗时) (默认: summary_only)",
    )
    parser.add_argument("--force", action="store_true", help="忽略断点，从头开始")
    parser.add_argument(
        "--skip-vectors", action="store_true", help="跳过 Qdrant 向量嵌入（CPU 极慢时使用，仅构建 BM25）"
    )
    parser.add_argument(
        "--vectors-only", action="store_true", help="仅嵌入向量（跳过 BM25，Phase 2 使用，使用独立的 checkpoint）"
    )

    args = parser.parse_args()

    if args.force:
        # 删除所有断点文件
        for ckpt_name in ["wiki_import_checkpoint.json", "wiki_import_checkpoint_vectors.json"]:
            ckpt = project_root / "data" / ckpt_name
            if ckpt.is_file():
                ckpt.unlink()
        print("🗑️  已删除断点文件，将从第一篇开始")

    importer = WikipediaImporter(
        batch_size=args.batch_size,
        embed_summary_chars=args.embed_summary_chars,
        embed_only_summary=(args.embed_mode == "summary_only"),
        skip_vectors=args.skip_vectors,
        vectors_only=args.vectors_only,
    )
    importer.run(args.input, max_pages=args.max_pages)


if __name__ == "__main__":
    main()
