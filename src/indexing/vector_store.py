"""
Vector store abstraction using ChromaDB.

Persists and retrieves embeddings with metadata.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage vector database with ChromaDB."""

    def __init__(self, persist_dir: Path = None):
        self.persist_dir = persist_dir or Path("./data/chroma_db")
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Persistent client (recommended).
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection = None
        logger.info(f"Initialized ChromaDB PersistentClient at {self.persist_dir}")

    def create_collection(self, name: str, metadata: Optional[Dict] = None) -> None:
        """Create or get a collection."""
        try:
            self.collection = self.client.get_or_create_collection(
                name=name,
                metadata=metadata or {"source": "research_papers"},
            )
            logger.info(f"Created/loaded collection: {name}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadata: List[Dict],
    ) -> None:
        """Add documents with embeddings to collection."""
        if not self.collection:
            raise ValueError("Collection not created. Call create_collection first.")

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadata,
            )
            logger.info(f"Added {len(ids)} documents to collection")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Query the vector store.

        Returns:
            Tuple of (document_ids, metadatas, distances) for the single query.
        """
        if not self.collection:
            raise ValueError("Collection not created.")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=int(n_results),
        )

        ids = results["ids"][0] if results.get("ids") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        return ids, metadatas, distances

    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        try:
            self.client.delete_collection(name)
            logger.info(f"Deleted collection: {name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise

    def list_collections(self) -> List[str]:
        """List all collections."""
        return [col.name for col in self.client.list_collections()]
