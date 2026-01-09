"""
Abstract base class for LLM integration.
"""
from abc import ABC, abstractmethod
from typing import Optional, Generator

class BaseLLM(ABC):
    """Abstract LLM interface."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """
        Generate text response.
        
        Args:
            prompt: Main prompt
            context: Optional retrieved context
            temperature: Sampling temperature
            max_tokens: Max response length
            stream: Whether to stream response
        
        Returns:
            Generated text or stream of chunks
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """Return model configuration."""
        pass
