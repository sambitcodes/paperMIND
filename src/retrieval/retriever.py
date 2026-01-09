"""
RAG retriever with citation tracking.

Retrieves relevant chunks and tracks their sources.
"""

import logging
from typing import List, Tuple, Any

from src.indexing.vector_store import VectorStoreManager
from src.indexing.embedding_models import CachedEmbedder

logger = logging.getLogger(__name__)


class Citation:
    """Represents a source citation."""

    def __init__(
        self,
        document: str,
        page_number: int,
        section: str,
        subsection: str,
        chunk_id: str,
    ):
        self.document = document
        self.page_number = page_number
        self.section = section
        self.subsection = subsection
        self.chunk_id = chunk_id

    def to_string(self) -> str:
        """Format citation as readable string."""
        citation = f"{self.document} - {self.section}"
        if self.subsection and self.subsection != "Unknown":
            citation += f" > {self.subsection}"
        citation += f" (page {self.page_number})"
        return citation


class RetrievalResult:
    """Result from retrieval with context."""

    def __init__(
        self,
        documents: List[str],
        citations: List[Citation],
        distances: List[float],
    ):
        self.documents = documents
        self.citations = citations
        self.distances = distances

    def get_context(self) -> str:
        """Combine documents into single context string."""
        return "\n\n---\n\n".join(self.documents)

    def get_citations(self) -> List[str]:
        """Get formatted citations."""
        return [c.to_string() for c in self.citations]


class RAGRetriever:
    """Retrieval Augmented Generation retriever."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        embedder: CachedEmbedder,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = int(top_k)

        # Kept for compatibility, but not used for hard filtering currently.
        self.similarity_threshold = float(similarity_threshold)

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Retrieve relevant documents for query.
        """
        query_embedding = self.embedder.embed_query(query)
        logger.info(f"Embedded query: {query[:100]}...")

        # Expecting: (doc_ids, metadatas, distances)
        doc_ids, metadatas, distances = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=self.top_k,
        )

        if not doc_ids:
            logger.warning("Vector store returned no ids for query.")
            return RetrievalResult(documents=[], citations=[], distances=[])

        aligned = list(zip(doc_ids, metadatas, distances))

        kept_ids: List[str] = []
        kept_distances: List[float] = []
        citations: List[Citation] = []

        # Keep all top_k results (stable behavior across Chroma distance metrics).
        for doc_id, metadata, distance in aligned:
            kept_ids.append(doc_id)
            kept_distances.append(float(distance))

            citations.append(
                Citation(
                    document=str(metadata.get("document", "Unknown")),
                    page_number=int(metadata.get("page_number", 0) or 0),
                    section=str(metadata.get("section", "Unknown")),
                    subsection=str(metadata.get("subsection", "Unknown")),
                    chunk_id=str(doc_id),
                )
            )

        # Fetch documents by ids
        documents: List[str] = []
        if kept_ids and getattr(self.vector_store, "collection", None):
            results = self.vector_store.collection.get(ids=kept_ids)
            documents = results.get("documents", []) or []

            # If order differs, re-map by id when possible.
            result_ids = results.get("ids", []) or []
            if len(result_ids) == len(documents) and result_ids != kept_ids:
                id_to_doc = {rid: doc for rid, doc in zip(result_ids, documents)}
                documents = [id_to_doc.get(i, "") for i in kept_ids]

        logger.info(f"Retrieved {len(documents)} documents (top_k={self.top_k})")
        return RetrievalResult(documents=documents, citations=citations, distances=kept_distances)
