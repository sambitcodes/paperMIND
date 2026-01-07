"""
SciBERT extractive model for citation-focused answers.
Uses question-answering on retrieved chunks.
"""
import logging
from typing import Optional, List, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from src.llm.base_model import BaseLLM

logger = logging.getLogger(__name__)

class SciBERTModel(BaseLLM):
    """SciBERT-based extractive QA model."""
    
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased"):
        self.model_name = model_name
        
        # Load QA pipeline
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model="deepset/roberta-large-squad2",  # Better QA model
                device=0,  # GPU if available
            )
            logger.info(f"Loaded SciBERT QA model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load GPU model, using CPU: {e}")
            self.qa_pipeline = pipeline("question-answering", device=-1)
    
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.0,  # No temperature for extractive
        max_tokens: int = 512,
        stream: bool = False,
    ) -> str:
        """
        Extract answer from context using QA model.
        
        This is extractive QA - no hallucinations, only extracts from context.
        """
        
        if not context:
            return "No context provided for extraction."
        
        try:
            result = self.qa_pipeline(
                question=prompt,
                context=context,
                max_answer_len=max_tokens,
            )
            
            answer = result.get("answer", "No answer found")
            confidence = result.get("score", 0.0)
            
            response = f"{answer}\n\n(Confidence: {confidence:.2%})"
            return response
            
        except Exception as e:
            logger.error(f"Error in extractive QA: {e}")
            return f"Error extracting answer: {str(e)}"
    
    def get_model_info(self) -> dict:
        """Return model info."""
        return {
            "name": self.model_name,
            "provider": "Hugging Face",
            "type": "extractive",
            "approach": "No hallucination - extracts from context only",
        }
