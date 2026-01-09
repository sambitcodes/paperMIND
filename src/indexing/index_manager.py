"""
Manage document indexing pipeline.

Coordinates PDF processing → Chunking → Embedding → Storage.
"""

import logging
from pathlib import Path
from typing import List

from src.pdf_processing.pdf_processor import PDFProcessor
from src.pdf_processing.chunker import SmartChunker
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
        2. Chunk content
        3. Create collection
        4. Generate embeddings
        5. Store in vector DB
        """
        # Hard guard: never pass None/empty to Chroma collection creation
        collection_name = (collection_name or "").strip()
        if not collection_name:
            raise ValueError("collection_name cannot be empty/None.")

        logger.info(f"Starting indexing for {pdf_path} into '{collection_name}'")

        # Step 1: Process PDF
        elements, metadata = self.pdf_processor.process_pdf(pdf_path)
        logger.info(f"Processed PDF: {len(elements)} elements | meta={metadata.get('strategy_used')}")

        # Step 2: Chunk content
        doc_stem = Path(pdf_path).stem
        chunks = self.chunker.chunk_elements(elements, doc_stem)
        logger.info(f"Created {len(chunks)} chunks")

        if not chunks:
            raise ValueError(
                "No chunks were produced from this PDF. "
                "This usually means the PDF has no extractable text (scanned/image-only) "
                "or extraction returned empty content."
            )

        # Step 3: Create collection if needed
        self.vector_store.create_collection(collection_name)

        # Step 4: Generate embeddings
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_documents(chunk_texts)
        logger.info(f"Generated {len(embeddings)} embeddings")

        # Step 5: Prepare metadata
        pdf_name = Path(pdf_path).name
        chunk_metadatas = []
        for c in chunks:
            chunk_metadatas.append(
                {
                    "page_number": int(c.page_number or 1),
                    "section": c.section or "Unknown",
                    "subsection": c.subsection or "Unknown",
                    "element_type": c.element_type or "Text",
                    "document": pdf_name,
                }
            )

        # Step 6: Store in vector DB
        chunk_ids = [c.chunk_id or f"{doc_stem}_chunk_{i}" for i, c in enumerate(chunks)]

        self.vector_store.add_documents(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunk_texts,
            metadata=chunk_metadatas,
        )

        logger.info(f"✓ Successfully indexed {pdf_name} into '{collection_name}'")

    def index_multiple_pdfs(self, pdf_paths: List[str], collection_name: str) -> None:
        """Index multiple PDFs into the same collection."""
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
