"""
GROQ API integration for multiple LLMs.
Supports Llama 3.3, Mixtral, Gemma, and more.
"""
import logging
from typing import Optional, Generator
from groq import Groq
from src.llm.base_model import BaseLLM
from src.llm.prompts import get_groq_system_prompt

logger = logging.getLogger(__name__)

class GroqModel(BaseLLM):
    """GROQ-based LLM wrapper."""
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.client = Groq(api_key=api_key)
        logger.info(f"Initialized GroqModel: {model_name}")
    
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = 1024,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """Generate response via GROQ API."""
        
        temp = temperature or self.temperature
        
        # Build system prompt
        system_prompt = get_groq_system_prompt(self.model_name, context is not None)
        
        # Build user message with context
        if context:
            user_message = f"""Context from research papers:
{context}

---

User Question: {prompt}

Please answer based on the provided context. If information is not available in the context, state that clearly."""
        else:
            user_message = prompt
        
        try:
            if stream:
                return self._stream_generate(
                    system_prompt, user_message, temp, max_tokens
                )
            else:
                return self._non_stream_generate(
                    system_prompt, user_message, temp, max_tokens
                )
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def _non_stream_generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Non-streaming generation."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        
        return response.choices[0].message.content
    
    def _stream_generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> Generator[str, None, None]:
        """Streaming generation."""
        with self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        ) as response:
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
    
    def get_model_info(self) -> dict:
        """Return model info."""
        return {
            "name": self.model_name,
            "provider": "GROQ",
            "type": "generative",
            "temperature": self.temperature,
        }
