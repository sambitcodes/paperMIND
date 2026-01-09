"""
Chunker module for intelligent text chunking with metadata preservation.

Updated:
- No dependency on Unstructured Element classes.
- Works with SimpleElement from pdf_processor.py (needs .text and .metadata.page_number).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represent a text chunk with metadata."""
    text: str
    page_number: int
    section: Optional[str] = None
    subsection: Optional[str] = None
    element_type: Optional[str] = None
    chunk_id: Optional[str] = None


class SmartChunker:
    """Chunk text while preserving minimal document structure."""

    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)

    def chunk_elements(self, elements: List[Any], document_name: str) -> List[Chunk]:
        """
        Chunk elements into ~chunk_size words, keeping ~overlap words.

        elements: list of objects with .text and .metadata.page_number
        """
        chunks: List[Chunk] = []
        buffer_words: List[str] = []
        chunk_id = 0

        current_section = None
        current_subsection = None

        last_page_num = 1

        for element in elements:
            text = (getattr(element, "text", "") or "").strip()
            if not text:
                continue

            # page_number from our SimpleElement.metadata.page_number
            meta = getattr(element, "metadata", None)
            page_num = getattr(meta, "page_number", 1) if meta is not None else 1
            last_page_num = page_num

            # With pypdf we don't have true element types; keep a stable label.
            element_type = element.__class__.__name__

            # Naive “section” heuristic: if a line is short and uppercase-ish, treat as heading.
            # (Optional: can be removed; doesn’t affect functionality.)
            if len(text) < 80 and sum(c.isupper() for c in text) > (0.6 * max(1, len(text))):
                current_subsection = text

            words = text.split()
            if not words:
                continue

            buffer_words.extend(words)

            # Emit chunks while we have enough words
            while len(buffer_words) >= self.chunk_size:
                chunk_words = buffer_words[: self.chunk_size]
                chunk_text = " ".join(chunk_words)

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        page_number=page_num,
                        section=current_section,
                        subsection=current_subsection,
                        element_type=element_type,
                        chunk_id=f"{document_name}_chunk_{chunk_id}",
                    )
                )
                chunk_id += 1

                # Keep overlap words
                if self.overlap > 0:
                    buffer_words = buffer_words[self.chunk_size - self.overlap :]
                else:
                    buffer_words = buffer_words[self.chunk_size :]

        # Final chunk
        if buffer_words:
            chunks.append(
                Chunk(
                    text=" ".join(buffer_words),
                    page_number=last_page_num,
                    section=current_section,
                    subsection=current_subsection,
                    element_type=elements[-1].__class__.__name__ if elements else None,
                    chunk_id=f"{document_name}_chunk_{chunk_id}",
                )
            )

        logger.info(f"Created {len(chunks)} chunks from {len(elements)} elements")
        return chunks
