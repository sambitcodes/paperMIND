"""
Optional reranking module.

This is a placeholder implementation that does not introduce extra heavy
dependencies. It can rerank retrieved chunks using simple heuristics
(e.g., query-term overlap). Replace with cross-encoder rerankers later.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


@dataclass
class RerankItem:
    text: str
    score: float
    index: int  # original position


class SimpleReranker:
    """
    Heuristic reranker based on token overlap between query and chunk.
    """

    def rerank(self, query: str, texts: List[str], top_k: int | None = None) -> List[str]:
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return texts[:top_k] if top_k else texts

        scored: List[RerankItem] = []
        for idx, t in enumerate(texts):
            t_tokens = set(_tokenize(t))
            overlap = len(q_tokens.intersection(t_tokens))
            denom = max(len(q_tokens), 1)
            score = overlap / denom
            scored.append(RerankItem(text=t, score=score, index=idx))

        scored.sort(key=lambda x: (x.score, -x.index), reverse=True)
        reranked = [x.text for x in scored]
        return reranked[:top_k] if top_k else reranked
