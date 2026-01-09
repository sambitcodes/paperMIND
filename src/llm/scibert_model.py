"""
Extractive QA model used as "SciBERT mode" in the app.

Note:
- This is extractive (answers must come from context).
- Designed to work on CPU environments (Streamlit Cloud).
"""

import logging
from typing import Optional, Generator, Union

from transformers import pipeline

from src.llm.base_model import BaseLLM

logger = logging.getLogger(__name__)


class SciBERTModel(BaseLLM):
    """Extractive QA model (no hallucinations; extracts from provided context)."""

    def __init__(self, model_name: str = "deepset/roberta-large-squad2"):
        # Your previous file passed allenai/scibert_scivocab_uncased but didn't actually use it.
        # Keep behavior stable and explicit: use a QA-tuned checkpoint by default.
        self.model_name = model_name

        # Try GPU then fall back to CPU.
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model_name,
                device=0,
            )
            logger.info(f"Loaded extractive QA model on GPU: {self.model_name}")
        except Exception as e:
            logger.warning(f"GPU not available, using CPU. Error: {e}")
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model_name,
                device=-1,
            )
            logger.info(f"Loaded extractive QA model on CPU: {self.model_name}")

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        if not context or not context.strip():
            text = "No context provided for extraction."
            if stream:
                def _gen():
                    yield text
                return _gen()
            return text

        try:
            result = self.qa_pipeline(
                question=prompt,
                context=context,
                max_answer_len=int(max_tokens),
            )
            answer = result.get("answer", "") or "No answer found"
            confidence = float(result.get("score", 0.0) or 0.0)
            text = f"{answer}\n\n(Confidence: {confidence:.2%})"

            if stream:
                def _gen():
                    yield text
                return _gen()

            return text

        except Exception as e:
            logger.error(f"Error in extractive QA: {e}")
            text = f"Error extracting answer: {str(e)}"
            if stream:
                def _gen():
                    yield text
                return _gen()
            return text

    def get_model_info(self) -> dict:
        return {
            "name": self.model_name,
            "provider": "Hugging Face",
            "type": "extractive",
            "approach": "No hallucination - extracts from context only",
        }
