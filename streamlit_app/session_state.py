"""
Streamlit session state management.
"""
import streamlit as st
from src.llm.orchestrator import LLMOrchestrator
from src.config import settings

def initialize_session_state():
    """Initialize session state."""
    
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.messages = []
        st.session_state.current_model = settings.DEFAULT_GENERATIVE_MODEL
        st.session_state.use_rag = True
        st.session_state.last_sources = []
        st.session_state.top_k = 5
        st.session_state.temperature = 0.2
        st.session_state.collection_name = None
        
        # Initialize LLM orchestrator
        st.session_state.llm_orchestrator = LLMOrchestrator()

def get_session_state():
    """Get session state."""
    return st.session_state
