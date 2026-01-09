"""
Extractive answering models.

Supports:
- QA mode (true extractive QA): HuggingFace question-answering pipeline
- SciBERT rank mode (actual SciBERT weights, no QA head): selects best sentence from context
"""

import logging
import re
from typing import Optional, Generator, Union, List

import torch
from transformers import pipeline, AutoTokenizer, AutoModel

from src.llm.base_model import BaseLLM

logger = logging.getLogger(__name__)


def _split_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter (dependency-free)
    sents = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [s.strip() for s in sents if s and s.strip()]


class SciBERTModel(BaseLLM):
    """
    mode="qa": uses pipeline('question-answering') with a QA-finetuned checkpoint (e.g. roberta-large-squad2)
    mode="scibert_rank": uses base SciBERT weights to rank sentences from context (no QA head required)
    """

    def __init__(
        self,
        mode: str = "qa",
        model_name: str = "deepset/roberta-large-squad2",
        device: int = -1,  # keep CPU-safe default
    ):
        self.mode = mode
        self.model_name = model_name
        self.device = device

        self.qa_pipeline = None
        self._tokenizer = None
        self._encoder = None

        if self.mode == "qa":
            # QA mode (keep your current behavior, just CPU-safe)
            try:
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model=self.model_name,
                    device=self.device,
                )
                logger.info(f"Loaded QA pipeline: {self.model_name} (device={self.device})")
            except Exception as e:
                logger.error(f"Failed to load QA pipeline: {e}")
                raise

        elif self.mode == "scibert_rank":
            # SciBERT encoder mode (actual SciBERT weights)
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._encoder = AutoModel.from_pretrained(self.model_name)
                self._encoder.eval()
                logger.info(f"Loaded SciBERT encoder: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load SciBERT encoder: {e}")
                raise
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _encode(self, texts: List[str]) -> torch.Tensor:
        assert self._tokenizer is not None and self._encoder is not None

        enc = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256,
        )
        with torch.no_grad():
            out = self._encoder(**enc)
            token_emb = out.last_hidden_state  # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
            summed = torch.sum(token_emb * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            emb = summed / counts
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        if not context or not context.strip():
            text = "No context provided for extraction."
            if stream:
                def _g():
                    yield text
                return _g()
            return text

        if self.mode == "qa":
            try:
                result = self.qa_pipeline(
                    question=prompt,
                    context=context,
                    max_answer_len=int(max_tokens),
                )
                answer = result.get("answer", "") or "No answer found"
                confidence = float(result.get("score", 0.0) or 0.0)
                text = f"{answer}\n\n(Confidence: {confidence:.2%})"
            except Exception as e:
                logger.error(f"Error in extractive QA: {e}")
                text = f"Error extracting answer: {str(e)}"

        else:
            # SciBERT ranking mode: pick best matching sentence
            sentences = _split_sentences(context)
            if not sentences:
                text = "No extractable sentences found in context."
            else:
                # Bound compute
                sentences = sentences[:200]

                q_emb = self._encode([prompt])      # (1, H)
                s_emb = self._encode(sentences)     # (N, H)
                sims = (s_emb @ q_emb.T).squeeze(1) # cosine sim

                best_idx = int(torch.argmax(sims).item())
                best_sent = sentences[best_idx]
                score = float(sims[best_idx].item())
                text = f"{best_sent}\n\n(SciBERT similarity: {score:.4f})"

        if stream:
            def _g():
                yield text
            return _g()
        return text

    def get_model_info(self) -> dict:
        return {
            "name": self.model_name,
            "provider": "Hugging Face",
            "type": "extractive",
            "mode": self.mode,
            "approach": (
                "QA span extraction" if self.mode == "qa"
                else "Sentence selection via SciBERT embeddings (no QA head)"
            ),
        }
