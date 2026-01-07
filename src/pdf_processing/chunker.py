"""
Chunker module for intelligent text chunking with metadata preservation.
Maintains context about sections, topics, and page numbers.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional
from unstructured.documents.elements import Element

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
    """Chunk text while preserving document structure."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_elements(
        self, 
        elements: List[Element], 
        document_name: str
    ) -> List[Chunk]:
        """
        Chunk unstructured elements intelligently.
        
        Args:
            elements: Unstructured elements from PDF
            document_name: Name of source document
        
        Returns:
            List of Chunk objects with metadata
        """
        chunks = []
        current_section = None
        current_subsection = None
        buffer = []
        buffer_size = 0
        chunk_id = 0
        
        for element in elements:
            element_type = element.__class__.__name__
            text = element.text.strip()
            page_num = getattr(element.metadata, "page_number", 1) if hasattr(element, "metadata") else 1
            
            # Update section tracking
            if element_type == "Title":
                current_section = text
                current_subsection = None
            elif element_type in ("Heading", "Header"):
                current_subsection = text

            
            # Skip empty elements
            if not text:
                continue
            
            # Add to buffer
            buffer.append(text)
            buffer_size += len(text.split())
            
            # Create chunk when buffer is full
            if buffer_size >= self.chunk_size:
                chunk_text = " ".join(buffer)
                chunk = Chunk(
                    text=chunk_text,
                    page_number=page_num,
                    section=current_section,
                    subsection=current_subsection,
                    element_type=element_type,
                    chunk_id=f"{document_name}_chunk_{chunk_id}",
                )
                chunks.append(chunk)
                chunk_id += 1
                
                # Keep overlap
                if self.overlap > 0 and len(buffer) > 1:
                    words = chunk_text.split()
                    overlap_words = int(self.overlap / self.chunk_size * len(words))
                    buffer = words[-overlap_words:] if overlap_words > 0 else []
                    buffer_size = sum(len(word) for word in buffer)
                else:
                    buffer = []
                    buffer_size = 0
        
        # Final chunk
        if buffer:
            chunk_text = " ".join(buffer)
            chunk = Chunk(
                text=chunk_text,
                page_number=getattr(elements[-1].metadata, "page_number", 1) if elements and hasattr(elements[-1], "metadata") else 1,
                section=current_section,
                subsection=current_subsection,
                element_type=elements[-1].__class__.__name__ if elements else None,
                chunk_id=f"{document_name}_chunk_{chunk_id}",
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {len(elements)} elements")
        return chunks
