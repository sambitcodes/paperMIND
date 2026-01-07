"""
RAG retriever with citation tracking.

Retrieves relevant chunks and tracks their sources.
"""

import logging
from typing import List

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
        self.top_k = top_k
        # Threshold kept but not used for hard filtering now (to avoid empty results).
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Retrieve relevant documents for query.

        Args:
            query: User query

        Returns:
            RetrievalResult with documents and citations
        """
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        logger.info(f"Embedded query: {query[:100]}...")

        # Query vector store
        doc_ids, metadatas, distances = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=self.top_k,
        )

        if not doc_ids:
            logger.warning("Vector store returned no ids for query.")
            return RetrievalResult(documents=[], citations=[], distances=[])

        # Keep results aligned as tuples
        aligned = list(zip(doc_ids, metadatas, distances))

        kept_ids = []
        citations: List[Citation] = []
        kept_distances: List[float] = []

        # For now: keep all top_k results (no distance-based filtering).
        # This avoids misinterpreting Chroma distances and ending up with 0 docs. [web:102]
        for doc_id, metadata, distance in aligned:
            kept_ids.append(doc_id)
            kept_distances.append(distance)

            citation = Citation(
                document=metadata.get("document", "Unknown"),
                page_number=int(metadata.get("page_number", 0)),
                section=metadata.get("section", "Unknown"),
                subsection=metadata.get("subsection", "Unknown"),
                chunk_id=doc_id,
            )
            citations.append(citation)

        logger.info(f"Retrieved {len(kept_ids)} documents (top_k={self.top_k})")

        # Fetch texts for kept ids
        documents: List[str] = []
        if kept_ids and self.vector_store.collection:
            results = self.vector_store.collection.get(ids=kept_ids)
            documents = results.get("documents", []) or []

            # Safety: if order differs, re-map by id when possible.
            result_ids = results.get("ids", []) or []
            if len(result_ids) == len(documents) and result_ids != kept_ids:
                id_to_doc = {rid: doc for rid, doc in zip(result_ids, documents)}
                documents = [id_to_doc.get(i, "") for i in kept_ids]

        return RetrievalResult(
            documents=documents,
            citations=citations,
            distances=kept_distances,
        )
