import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, retriever, llm_orchestrator):
        self.retriever = retriever
        self.llm_orchestrator = llm_orchestrator

    def answer(
        self,
        question: str,
        model_name: str,
        temperature: float = 0.7,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
              "answer": str,
              "sources": List[dict],
              "retrieved_docs_count": int
            }
        """

        # Ensure retriever uses the latest top_k from UI
        if hasattr(self.retriever, "top_k"):
            self.retriever.top_k = int(top_k)
        elif hasattr(self.retriever, "topk"):
            # Backward compatibility if retriever uses topk
            self.retriever.topk = int(top_k)

        retrieval = self.retriever.retrieve(question)

        docs = getattr(retrieval, "documents", None) or []
        citations = getattr(retrieval, "citations", None) or []

        context = "\n\n".join([d for d in docs if d]).strip()

        # If nothing retrieved, still answer (LLM-only fallback) but keep sources empty
        if not context:
            answer = self.llm_orchestrator.generate_response(
                prompt=question,
                model_name=model_name,
                temperature=temperature,
                context=None,
            )
            return {
                "answer": (answer or "").strip(),
                "sources": [],
                "retrieved_docs_count": 0,
            }

        # Build sources payload (defensive: tolerate missing fields)
        sources: List[dict] = []
        for c in citations:
            sources.append(
                {
                    "document": getattr(c, "document", "Unknown"),
                    "page_number": getattr(c, "page_number", getattr(c, "pagenumber", None)),
                    "section": getattr(c, "section", None),
                    "subsection": getattr(c, "subsection", None),
                    "chunk_id": getattr(c, "chunk_id", getattr(c, "chunkid", None)),
                }
            )

        # IMPORTANT: Pass retrieved text through the `context` parameter.
        # This is required for extractive models like scibert-extractive.
        answer = self.llm_orchestrator.generate_response(
            prompt=question,
            model_name=model_name,
            temperature=temperature,
            context=context,
        )

        return {
            "answer": (answer or "").strip(),
            "sources": sources,
            "retrieved_docs_count": len(docs),
        }
