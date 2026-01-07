"""
Citation tracking utilities.

Retrieval already returns Citation objects and formatted strings.
This module provides a stable place to:
- deduplicate citations
- format citations consistently
- group citations by document/section/page (future-ready)
"""
from __future__ import annotations

from collections import OrderedDict
from typing import List, Sequence


def dedupe_citations(citations: Sequence[str]) -> List[str]:
    """
    Deduplicate citations while preserving order.
    """
    seen = OrderedDict()
    for c in citations or []:
        key = (c or "").strip()
        if key and key not in seen:
            seen[key] = True
    return list(seen.keys())


def format_citations_panel(citations: Sequence[str]) -> str:
    """
    Create a compact markdown block for citations.
    """
    cits = dedupe_citations(citations)
    if not cits:
        return "*No sources available.*"

    lines = ["**📚 Sources:**"]
    for i, c in enumerate(cits, 1):
        lines.append(f"{i}. {c}")
    return "\n".join(lines)
