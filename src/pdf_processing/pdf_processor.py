"""
Advanced PDF processing using Unstructured.
Handles multi-column layouts, tables, and references.
"""
import logging
from pathlib import Path
from typing import List, Tuple
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.auto import partition
from unstructured.documents.elements import (
    Element,
    Title,
    NarrativeText,
    ListItem,
    Table,
)


logger = logging.getLogger(__name__)

class PDFProcessor:
    """Process research papers and extract structured content."""
    
    def __init__(self, extract_images: bool = True, extract_tables: bool = True):
        self.extract_images = extract_images
        self.extract_tables = extract_tables
    
    def process_pdf(self, pdf_path: str) -> Tuple[List[Element], dict]:
        """
        Process PDF and return elements with metadata.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Tuple of (elements, metadata)
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        try:
            # Use partition_pdf for better handling of complex layouts
            elements = partition_pdf(
                str(pdf_path),
                infer_table_structure=self.extract_tables,
                strategy="fast",  # High-resolution strategy
                extract_image_block_types=["Image"] if self.extract_images else [],
            )
            
            logger.info(f"Extracted {len(elements)} elements from {pdf_path.name}")
            
            metadata = {
                "filename": pdf_path.name,
                "filepath": str(pdf_path),
                "total_elements": len(elements),
            }
            
            return elements, metadata
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def extract_text_by_page(self, pdf_path: str) -> dict:
        """Extract text organized by page number."""
        elements, _ = self.process_pdf(pdf_path)
        
        page_text = {}
        current_page = 1
        
        for element in elements:
            page_num = getattr(element, "metadata", {}).get("page_number", 1)
            
            if page_num not in page_text:
                page_text[page_num] = []
            
            page_text[page_num].append(element.text)
        
        return page_text
    
    def extract_structured(self, pdf_path: str) -> dict:
        """Extract structured content with element types."""
        elements, metadata = self.process_pdf(pdf_path)
        
        structured = {
            "metadata": metadata,
            "titles": [],
            "headings": [],
            "sections": [],
            "tables": [],
            "narrative": [],
        }
        
        for element in elements:
            if isinstance(element, Title):
                structured["titles"].append(element.text)
            elif getattr(element, "category", None) in ("Header", "Heading"):
                structured["headings"].append(element.text)
            elif isinstance(element, Table):
                structured["tables"].append({
                    "content": element.text,
                    "html": getattr(element, "metadata", {}).get("html_table", ""),
                })
            elif isinstance(element, NarrativeText):
                structured["narrative"].append(element.text)
            elif isinstance(element, ListItem):
                structured["narrative"].append(f"• {element.text}")
        
        return structured
