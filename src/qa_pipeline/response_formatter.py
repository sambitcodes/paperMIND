"""
Format QA responses for display.
"""
from typing import List, Generator

class ResponseFormatter:
    """Format AI responses for presentation."""
    
    @staticmethod
    def format_with_citations(
        answer: str,
        citations: List[str],
    ) -> str:
        """Format answer with citations."""
        formatted = f"{answer}\n\n"
        
        if citations:
            formatted += "**📚 Sources:**\n"
            for i, citation in enumerate(citations, 1):
                formatted += f"{i}. {citation}\n"
        else:
            formatted += "*No sources found - answer generated without document context.*"
        
        return formatted
    
    @staticmethod
    def format_streaming_chunk(chunk: str) -> str:
        """Process streaming chunk."""
        return chunk
    
    @staticmethod
    def format_error(error: str) -> str:
        """Format error message."""
        return f"❌ Error: {error}"
