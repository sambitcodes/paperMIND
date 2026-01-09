"""
LLM orchestrator for Groq models.

Provides:
- Model registry
- generate_response() for non-streaming
- stream_response() for streaming
"""

import logging
import os
from typing import Dict, List, Optional, Iterable

from groq import Groq

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")

        self.client = Groq(api_key=api_key)

        # Register available models here
        self._models: Dict[str, Dict] = {
            "llama-3.3-70b-versatile": {
                "provider": "groq",
                "type": "chat",
                "id": "llama-3.3-70b-versatile",
            },
            "groq/compound": {
                "provider": "groq",
                "type": "agentic_system",
                "id": "groq/compound",
            },
            "openai/gpt-oss-120b": {
                "provider": "groq",
                "type": "moe",
                "id": "openai/gpt-oss-120b",
            },
        }

        # Default model
        self.current_model_name = "llama-3.3-70b-versatile"

    # ---------- Model metadata / selection ----------

    def get_available_models(self) -> List[str]:
        return list(self._models.keys())

    def get_model(self, name: str) -> "LLMOrchestrator":
        if name not in self._models:
            raise ValueError(f"Unknown model: {name}")
        self.current_model_name = name
        return self

    def get_model_info(self) -> Dict:
        return self._models.get(self.current_model_name, {})

    # ---------- Core generation API used by app ----------

    def generate_response(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        context: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        """
        Non-streaming response.
        """
        model_name = model_name or self.current_model_name
        if model_name not in self._models:
            raise ValueError(f"Unknown model: {model_name}")

        system_msg = "You are a helpful research assistant."
        if context:
            system_msg += (
                " Use the provided context to answer the question accurately."
                " If the answer cannot be found in the context, say you are not sure."
            )

        messages = [{"role": "system", "content": system_msg}]
        if context:
            messages.append({"role": "system", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = self.client.chat.completions.create(
                model=self._models[model_name]["id"],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = (
                resp.choices[0].message.content
                if resp and resp.choices and resp.choices[0].message
                else ""
            )
            return content or ""
        except Exception as e:
            logger.error(f"Groq chat completion error: {e}")
            raise

    def stream_response(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        context: Optional[str] = None,
    ) -> Iterable[str]:
        """
        Streaming response generator (yields deltas).

        Uses Groq streaming chat completions. [web:291]
        """
        model_name = model_name or self.current_model_name
        if model_name not in self._models:
            raise ValueError(f"Unknown model: {model_name}")

        system_msg = "You are a helpful research assistant."
        if context:
            system_msg += (
                " Use the provided context to answer the question accurately."
                " If the answer cannot be found in the context, say you are not sure."
            )

        messages = [{"role": "system", "content": system_msg}]
        if context:
            messages.append({"role": "system", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self._models[model_name]["id"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content or ""
            except Exception:
                delta = ""
            if delta:
                yield delta
