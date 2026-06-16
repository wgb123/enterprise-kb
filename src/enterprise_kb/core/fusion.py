"""上下文融合层 — 合并多个引擎的结果，去重和解决冲突。

当 Wiki 引擎和 HybridRAG 引擎都返回结果时，需要：
1. 按权威性排序（Wiki > HybridRAG > GraphRAG）
2. 内容级去重（基于 shingle Jaccard 相似度）
3. 冲突解决（高优先级引擎的内容覆盖低优先级）
"""

from typing import Optional

from enterprise_kb.interfaces.base import QueryResult, RetrievedChunk
from enterprise_kb.utils.logger import logger
from enterprise_kb.utils.text_utils import deduplicate_chunks


class ContextFusion:
    """多引擎结果融合器。

    融合规则：
    - **A 类 Wiki 记忆** 优先级最高，内容不可被覆盖
    - **B 类 HybridRAG** 结果用于补充 Wiki 不足的部分
    - **C 类 GraphRAG**（预留）最低优先级
    - 内容级去重：同一语义片段只保留一份
    """

    ENGINE_PRIORITY = {
        "wiki": 3,  # 最高
        "hybrid_rag": 2,
        "graph_rag": 1,  # 最低（预留）
    }

    def fuse(
        self,
        results: list[QueryResult],
        max_chunks: int = 15,
    ) -> list[RetrievedChunk]:
        """融合多个查询引擎的结果。

        Args:
            results: 各引擎的查询结果列表。
            max_chunks: 融合后最多保留的文档块数。

        Returns:
            融合后的文档块列表，按权威性 + 分数排序。
        """
        if not results:
            return []

        # 1. 收集所有 chunk
        all_chunks: list[RetrievedChunk] = []
        source_engines: set[str] = set()

        for result in results:
            source_engines.add(result.engine)
            for chunk in result.chunks:
                chunk.score += self._priority_bonus(chunk.source)
                all_chunks.append(chunk)

        # 2. 按总分降序排列
        all_chunks.sort(key=lambda c: c.score, reverse=True)

        # 3. 内容级去重
        texts = [c.content for c in all_chunks]
        unique_texts = deduplicate_chunks(texts, similarity_threshold=0.80)

        text_to_chunks: dict[str, RetrievedChunk] = {}
        for chunk in all_chunks:
            text_to_chunks.setdefault(chunk.content, chunk)

        fused = [text_to_chunks[t] for t in unique_texts if t in text_to_chunks]

        logger.info(
            "Context fusion: %d inputs (%s) → %d unique chunks",
            len(all_chunks),
            ", ".join(sorted(source_engines)),
            len(fused),
        )

        return fused[:max_chunks]

    @classmethod
    def _priority_bonus(cls, source: str) -> float:
        """根据来源引擎赋予优先级加分。

        Wiki 引擎获得最高加分，确保 Wiki 结果始终排在前面。
        """
        pri = cls.ENGINE_PRIORITY.get(source, 0)
        # 将优先级映射为分数加成
        bonus_map = {3: 10.0, 2: 0.0, 1: -5.0}
        return bonus_map.get(pri, 0.0)
