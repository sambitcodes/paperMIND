"""
Manage document indexing pipeline.
Coordinates PDF processing → Chunking → Embedding → Storage.
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional
from src.pdf_processing.pdf_processor import PDFProcessor
from src.pdf_processing.chunker import SmartChunker, Chunk
from src.indexing.embedding_models import CachedEmbedder
from src.indexing.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class IndexManager:
    """End-to-end document indexing."""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        persist_dir: Path = None,
    ):
        self.pdf_processor = PDFProcessor()
        self.chunker = SmartChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.embedder = CachedEmbedder(embedding_model, persist_dir)
        self.vector_store = VectorStoreManager(persist_dir)
    
    def index_pdf(self, pdf_path: str, collection_name: str) -> None:
        """
        Index a single PDF document.
        
        Steps:
        1. Process PDF
        2. Extract and chunk content
        3. Generate embeddings
        4. Store in vector DB
        """
        logger.info(f"Starting indexing for {pdf_path}")
        
        # Step 1: Process PDF
        elements, metadata = self.pdf_processor.process_pdf(pdf_path)
        logger.info(f"Processed PDF: {len(elements)} elements")
        
        # Step 2: Chunk content
        chunks = self.chunker.chunk_elements(elements, Path(pdf_path).stem)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 3: Create collection if needed
        self.vector_store.create_collection(collection_name)
        
        # Step 4: Generate embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_documents(chunk_texts)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Step 5: Prepare metadata
        chunk_metadatas = []
        for chunk in chunks:
            chunk_metadatas.append({
                "page_number": chunk.page_number,
                "section": chunk.section or "Unknown",
                "subsection": chunk.subsection or "Unknown",
                "element_type": chunk.element_type or "Text",
                "document": Path(pdf_path).name,
            })
        
        # Step 6: Store in vector DB
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.vector_store.add_documents(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunk_texts,
            metadata=chunk_metadatas,
        )
        
        logger.info(f"✓ Successfully indexed {Path(pdf_path).name}")
    
    def index_multiple_pdfs(self, pdf_paths: List[str], collection_name: str) -> None:
        """Index multiple PDFs into same collection."""
        for pdf_path in pdf_paths:
            try:
                self.index_pdf(pdf_path, collection_name)
            except Exception as e:
                logger.error(f"Failed to index {pdf_path}: {e}")
    
    def get_vector_store(self) -> VectorStoreManager:
        """Return vector store for retrieval."""
        return self.vector_store
    
    def get_embedder(self):
        """Return embedder for query encoding."""
        return self.embedder
