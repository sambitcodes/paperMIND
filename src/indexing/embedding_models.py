"""
Embedding model abstraction layer.
Supports HuggingFace, OpenAI, and other providers.
"""
import logging
import hashlib
import pickle
from typing import List
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Abstract embedding interface."""
    
    def __init__(self, model_name: str, cache_dir: Path = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        raise NotImplementedError
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        raise NotImplementedError

class HuggingFaceEmbedder(EmbeddingModel):
    """HuggingFace sentence transformer embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Path = None):
        super().__init__(model_name, cache_dir)
        self.model_name = f"sentence-transformers/{model_name}" if "/" not in model_name else model_name
        self._model = SentenceTransformer(self.model_name)
        logger.info(f"Loaded embedding model: {self.model_name}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query."""
        embedding = self._model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents efficiently."""
        embeddings = self._model.encode(texts, convert_to_tensor=False, batch_size=32)
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self._model.get_sentence_embedding_dimension()

class CachedEmbedder(HuggingFaceEmbedder):
    """Caching layer for embeddings to avoid recomputation."""
    
    def __init__(self, model_name: str, cache_dir: Path = None):
        super().__init__(model_name, cache_dir)
        self.cache_dir = cache_dir or Path("./data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed with caching."""
        cache_key = self._get_cache_key(text)
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        embedding = super().embed_query(text)
        
        with open(cache_path, "wb") as f:
            pickle.dump(embedding, f)
        
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with caching."""
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    embeddings.append(pickle.load(f))
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        if texts_to_embed:
            new_embeddings = super().embed_documents(texts_to_embed)
            for text, embedding, idx in zip(texts_to_embed, new_embeddings, indices_to_embed):
                cache_key = self._get_cache_key(text)
                cache_path = self.cache_dir / f"{cache_key}.pkl"
                with open(cache_path, "wb") as f:
                    pickle.dump(embedding, f)
                embeddings.insert(idx, embedding)
        
        return embeddings
