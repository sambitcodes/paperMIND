"""
LLM orchestrator for Groq models + extractive QA (SciBERTModel).

Keeps the existing API expected by the app:
- get_available_models()
- get_model()
- get_model_info()
- generate_response()
- stream_response()  (Groq streams; SciBERT yields once)
"""

import logging
import os
from typing import Dict, List, Optional, Iterable

from groq import Groq

from src.llm.scibert_model import SciBERTModel

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")

        self.client = Groq(api_key=api_key)

        self._models: Dict[str, Dict] = {
            "llama-3.3-70b-versatile": {"provider": "groq", "type": "chat", "id": "llama-3.3-70b-versatile"},
            "groq/compound": {"provider": "groq", "type": "agentic_system", "id": "groq/compound"},
            "openai/gpt-oss-120b": {"provider": "groq", "type": "moe", "id": "openai/gpt-oss-120b"},
            # New: extractive QA option (uses your existing scibert_model.py)
            "scibert-extractive": {"provider": "hf", "type": "extractive", "id": "scibert-extractive"},
        }

        self.current_model_name = "llama-3.3-70b-versatile"
        self._scibert: Optional[SciBERTModel] = None

    def get_available_models(self) -> List[str]:
        return list(self._models.keys())

    def get_model(self, name: str) -> "LLMOrchestrator":
        if name not in self._models:
            raise ValueError(f"Unknown model: {name}")
        self.current_model_name = name
        return self

    def get_model_info(self) -> Dict:
        name = self.current_model_name
        if name == "scibert-extractive":
            if self._scibert is None:
                self._scibert = SciBERTModel()
            return self._scibert.get_model_info()
        return self._models.get(name, {})

    def _build_messages(self, prompt: str, context: Optional[str] = None) -> List[Dict[str, str]]:
        system_msg = "You are a helpful research assistant."
        if context:
            system_msg += (
                " Use the provided context to answer the question accurately."
                " If the answer cannot be found in the context, say you are not sure."
            )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_msg}]
        if context:
            messages.append({"role": "system", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate_response(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        context: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        model_name = (model_name or self.current_model_name).strip()

        if model_name == "scibert-extractive":
            if self._scibert is None:
                self._scibert = SciBERTModel()
            out = self._scibert.generate(
                prompt=prompt,
                context=context,
                temperature=0.0,
                max_tokens=int(max_tokens),
                stream=False,
            )
            # BaseLLM allows generator, but we asked stream=False so it's str.
            return str(out)

        if model_name not in self._models:
            raise ValueError(f"Unknown model: {model_name}")

        messages = self._build_messages(prompt=prompt, context=context)

        resp = self.client.chat.completions.create(
            model=self._models[model_name]["id"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            return resp.choices[0].message.content or ""
        except Exception:
            return ""

    def stream_response(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        context: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> Iterable[str]:
        model_name = (model_name or self.current_model_name).strip()

        # SciBERT extractive: yield once (still compatible with streaming UI)
        if model_name == "scibert-extractive":
            if self._scibert is None:
                self._scibert = SciBERTModel()
            out = self._scibert.generate(
                prompt=prompt,
                context=context,
                temperature=0.0,
                max_tokens=int(max_tokens),
                stream=True,
            )
            # out can be a generator (per BaseLLM)
            if hasattr(out, "__iter__") and not isinstance(out, str):
                for x in out:
                    yield str(x)
            else:
                yield str(out)
            return

        if model_name not in self._models:
            raise ValueError(f"Unknown model: {model_name}")

        messages = self._build_messages(prompt=prompt, context=context)

        stream = self.client.chat.completions.create(
            model=self._models[model_name]["id"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            delta = ""
            try:
                delta = chunk.choices[0].delta.content or ""
            except Exception:
                delta = ""
            if delta:
                yield delta
