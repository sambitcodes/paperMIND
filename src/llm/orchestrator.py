"""
LLM orchestrator for Groq models + local extractive models.

Models supported:
- Groq chat models (streaming supported)
- Local extractive QA (RoBERTa SQuAD2) via SciBERTModel(mode="qa")
- Local SciBERT-weights extractive selection via SciBERTModel(mode="scibert_rank")
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
            # Local extractive models
            "roberta-extractive": {
                "provider": "hf",
                "type": "extractive_qa",
                "id": "roberta-extractive",
            },
            "scibert-extractive": {
                "provider": "hf",
                "type": "extractive_scibert",
                "id": "scibert-extractive",
            },
        }

        self.current_model_name = "llama-3.3-70b-versatile"

        # Lazy-loaded local models
        self._roberta_qa: Optional[SciBERTModel] = None
        self._scibert_rank: Optional[SciBERTModel] = None

    # ---------- Model metadata / selection ----------

    def get_available_models(self) -> List[str]:
        return list(self._models.keys())

    def get_model(self, name: str) -> "LLMOrchestrator":
        if name not in self._models:
            raise ValueError(f"Unknown model: {name}")
        self.current_model_name = name
        return self

    def get_model_info(self) -> Dict:
        name = self.current_model_name

        if name == "roberta-extractive":
            if self._roberta_qa is None:
                self._roberta_qa = SciBERTModel(
                    mode="qa",
                    model_name="deepset/roberta-large-squad2",
                    device=-1,
                )
            return self._roberta_qa.get_model_info()

        if name == "scibert-extractive":
            if self._scibert_rank is None:
                self._scibert_rank = SciBERTModel(
                    mode="scibert_rank",
                    model_name="allenai/scibert_scivocab_uncased",
                    device=-1,
                )
            return self._scibert_rank.get_model_info()

        return self._models.get(name, {})

    # ---------- Internal: build Groq messages ----------

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

    # ---------- Core generation API used by app ----------

    def generate_response(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        context: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        model_name = (model_name or self.current_model_name).strip()

        # Local: RoBERTa extractive QA (true QA head)
        if model_name == "roberta-extractive":
            if self._roberta_qa is None:
                self._roberta_qa = SciBERTModel(
                    mode="qa",
                    model_name="deepset/roberta-large-squad2",
                    device=-1,
                )
            out = self._roberta_qa.generate(
                prompt=prompt,
                context=context,
                temperature=0.0,
                max_tokens=min(int(max_tokens), 512),
                stream=False,
            )
            return str(out)

        # Local: SciBERT weights extractive selection (no QA head)
        if model_name == "scibert-extractive":
            if self._scibert_rank is None:
                self._scibert_rank = SciBERTModel(
                    mode="scibert_rank",
                    model_name="allenai/scibert_scivocab_uncased",
                    device=-1,
                )
            out = self._scibert_rank.generate(
                prompt=prompt,
                context=context,
                temperature=0.0,
                max_tokens=min(int(max_tokens), 512),
                stream=False,
            )
            return str(out)

        # Groq models
        if model_name not in self._models:
            raise ValueError(f"Unknown model: {model_name}")

        messages = self._build_messages(prompt=prompt, context=context)

        try:
            resp = self.client.chat.completions.create(
                model=self._models[model_name]["id"],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Groq chat completion error: {e}")
            raise

    def stream_response(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        context: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> Iterable[str]:
        model_name = (model_name or self.current_model_name).strip()

        # Local models: yield once so your streaming UI still works
        if model_name in ("roberta-extractive", "scibert-extractive"):
            yield self.generate_response(
                prompt=prompt,
                model_name=model_name,
                temperature=0.0,
                context=context,
                max_tokens=max_tokens,
            )
            return

        # Groq streaming
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
