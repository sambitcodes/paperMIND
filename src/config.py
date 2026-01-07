"""
Centralized configuration for the QA bot.
Loads from .env and provides sensible defaults.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application configuration."""
    
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    HF_API_KEY: str = os.getenv("HF_API_KEY", "")
    
    # Models
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_GENERATIVE_MODEL: str = "llama-3.3-70b-versatile"
    EXTRACTIVE_MODEL: str = "allenai/scibert_scivocab_uncased"
    
    # Groq Models
    GROQ_MODELS: dict = {
        "llama-3.3-70b-versatile": {"tokens": 131072, "category": "generalist"},
        "groq/compound": {"tokens": 131072, "category": "agentic system"},
        "openai/gpt-oss-120b": {"tokens": 131072, "category": "moe reasoning"},
    }
    
    # RAG Settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 128
    TOP_K_RETRIEVAL: int = 5
    SIMILARITY_THRESHOLD: float = 0.5
    
    # Vector Store
    CHROMA_PERSIST_DIR: Path = Path("./data/chroma_db")
    CACHE_DIR: Path = Path("./data/cache")
    
    # PDF Processing
    PDF_MAX_SIZE_MB: int = 50
    EXTRACT_IMAGES: bool = True
    EXTRACT_TABLES: bool = True
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
