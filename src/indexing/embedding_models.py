"""
Embedding models for document indexing and retrieval.

Updated:
- Removed langchain_community dependency entirely.
- Uses sentence-transformers directly.
"""

import logging
from pathlib import Path
from typing import List, Optional

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class CachedEmbedder:
    """Embedder wrapper around SentenceTransformer."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_dir: Optional[Path] = None):
        self.model_name = model_name
        self.persist_dir = persist_dir
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded SentenceTransformer model: {model_name}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        embedding = self.model.encode([text], normalize_embeddings=True)[0]
        return embedding.tolist()
