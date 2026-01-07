"""
Helpers for normalizing and preserving document metadata.

This module is intentionally lightweight: PDF parsing + chunking already
extract metadata (page_number, section, subsection). This file provides
a single, consistent metadata schema used across indexing/retrieval/UI.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class DocumentChunkMetadata:
    """Metadata stored alongside each chunk in the vector store."""
    document: str
    page_number: int
    section: str = "Unknown"
    subsection: str = "Unknown"
    element_type: str = "Text"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a ChromaDB-storable dict (JSON-serializable)."""
        d = asdict(self)
        # Ensure page_number is int for consistent downstream handling.
        d["page_number"] = int(d.get("page_number") or 0)
        return d


def normalize_metadata(raw: Dict[str, Any]) -> DocumentChunkMetadata:
    """
    Normalize metadata coming from different upstream sources.

    Args:
        raw: Raw metadata dict (possibly missing keys).

    Returns:
        DocumentChunkMetadata with defaults applied.
    """
    return DocumentChunkMetadata(
        document=str(raw.get("document", "Unknown")),
        page_number=int(raw.get("page_number", 0) or 0),
        section=str(raw.get("section", "Unknown") or "Unknown"),
        subsection=str(raw.get("subsection", "Unknown") or "Unknown"),
        element_type=str(raw.get("element_type", "Text") or "Text"),
    )


def build_metadata(
    document: str,
    page_number: int,
    section: Optional[str] = None,
    subsection: Optional[str] = None,
    element_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience builder used by indexer to ensure consistent metadata dicts.
    """
    meta = DocumentChunkMetadata(
        document=document,
        page_number=int(page_number or 0),
        section=section or "Unknown",
        subsection=subsection or "Unknown",
        element_type=element_type or "Text",
    )
    return meta.to_dict()
