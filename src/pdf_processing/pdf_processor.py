"""
Lightweight PDF processing using PyPDF (pure Python).

Goal:
- Avoid unstructured/unstructured_inference/cv2/libGL conflicts on Streamlit Cloud.
- Provide a minimal "element" object with .text and .metadata.page_number
  so the rest of the pipeline (chunker -> embeddings -> vector store) stays intact.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class ElementMetadata:
    page_number: int = 1


@dataclass
class SimpleElement:
    text: str
    metadata: ElementMetadata


class PDFProcessor:
    """Process research papers and extract text by page (fast, embedded-text PDFs)."""

    def __init__(self, extract_images: bool = False, extract_tables: bool = False):
        # Kept for API compatibility with your existing IndexManager usage.
        self.extract_images = extract_images
        self.extract_tables = extract_tables

    def process_pdf(self, pdf_path: str) -> Tuple[List[SimpleElement], dict]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        reader = PdfReader(str(pdf_path))

        elements: List[SimpleElement] = []
        for page_idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            elements.append(SimpleElement(text=text, metadata=ElementMetadata(page_number=page_idx)))

        metadata = {
            "filename": pdf_path.name,
            "filepath": str(pdf_path),
            "strategy_used": "pypdf_text",
            "total_pages": len(reader.pages),
            "total_elements": len(elements),
        }

        logger.info(f"Extracted {len(elements)} page-elements from {pdf_path.name}")
        return elements, metadata

    def extract_text_by_page(self, pdf_path: str) -> dict:
        elements, _ = self.process_pdf(pdf_path)
        page_text = {}
        for el in elements:
            page_num = getattr(el.metadata, "page_number", 1)
            page_text.setdefault(page_num, []).append(el.text)
        return page_text

    def extract_structured(self, pdf_path: str) -> dict:
        # With pypdf we don’t get semantic element types; we return everything as narrative.
        elements, metadata = self.process_pdf(pdf_path)
        return {
            "metadata": metadata,
            "titles": [],
            "headings": [],
            "sections": [],
            "tables": [],
            "narrative": [el.text for el in elements],
        }
