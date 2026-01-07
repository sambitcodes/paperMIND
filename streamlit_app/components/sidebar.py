from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src.config import settings
from src.indexing.index_manager import IndexManager
from src.retrieval.retriever import RAGRetriever
from src.qa_pipeline.rag_pipeline import RAGPipeline
from src.llm.orchestrator import LLMOrchestrator


def render_sidebar():
    """Render sidebar controls with UX-first ordering."""

    st.markdown("### 📄 Document")

    # --------- Upload + Index (FIRST) ----------
    uploaded_file = st.file_uploader("Upload research paper (PDF)", type="pdf")

    chunk_size = st.number_input(
        "Chunk size",
        min_value=128,
        max_value=2000,
        value=int(getattr(settings, "CHUNK_SIZE", 512)),
        step=64,
        help="Number of tokens per chunk.",
    )
    chunk_overlap = st.number_input(
        "Chunk overlap",
        min_value=0,
        max_value=512,
        value=int(getattr(settings, "CHUNK_OVERLAP", 128)),
        step=16,
        help="Overlap between consecutive chunks.",
    )

    if uploaded_file:
        default_collection = uploaded_file.name.replace(".pdf", "")
        collection_name = st.text_input(
            "Collection name",
            value=st.session_state.get("collection_name", default_collection),
        )
        st.session_state.collection_name = collection_name
        st.success(f"Ready to index: {uploaded_file.name}")

        if st.button("🔍 Index document"):
            st.info("Indexing... this may take a while for long papers.")
            try:
                if "llm_orchestrator" not in st.session_state:
                    st.session_state.llm_orchestrator = LLMOrchestrator()

                with tempfile.TemporaryDirectory() as td:
                    tmp_path = Path(td) / uploaded_file.name
                    tmp_path.write_bytes(uploaded_file.getbuffer())

                    index_manager = IndexManager(
                        embedding_model=settings.EMBEDDING_MODEL,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                        persist_dir=settings.CHROMA_PERSIST_DIR,
                    )
                    index_manager.index_pdf(str(tmp_path), collection_name=collection_name)

                vector_store = index_manager.get_vector_store()
                embedder = index_manager.get_embedder()

                vector_store.create_collection(collection_name)

                retriever = RAGRetriever(
                    vector_store=vector_store,
                    embedder=embedder,
                    top_k=int(st.session_state.get("top_k", 5)),
                    similarity_threshold=settings.SIMILARITY_THRESHOLD,
                )

                st.session_state.index_manager = index_manager
                st.session_state.rag_pipeline = RAGPipeline(
                    retriever=retriever,
                    llm_orchestrator=st.session_state.llm_orchestrator,
                )

                # Light sanity check
                try:
                    count = vector_store.collection.count()
                    st.info(f"Indexed chunks in collection '{collection_name}': {count}")
                except Exception:
                    pass

                st.success(f"✓ Indexed into collection: {collection_name}")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    st.divider()

    # --------- Retrieval mode (RAG) ----------
    st.markdown("### 🔎 Retrieval")

    st.session_state.use_rag = st.toggle(
        "Use document context (RAG)",
        value=st.session_state.get("use_rag", True),
        help="When enabled, answers are grounded in the indexed paper.",
    )

    if st.session_state.use_rag:
        st.session_state.top_k = st.slider(
            "Top‑K chunks to retrieve",
            min_value=1,
            max_value=10,
            value=int(st.session_state.get("top_k", 5)),
        )

    st.divider()

    # --------- Generation settings ----------
    st.markdown("### ✨ Generation")

    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("temperature", 0.2)),
        step=0.1,
        help="Higher = more creative, lower = more deterministic.",
    )

    st.divider()

    # --------- Model selection (last) ----------
    st.markdown("### 🧠 Model")

    if "llm_orchestrator" not in st.session_state:
        st.session_state.llm_orchestrator = LLMOrchestrator()

    available_models = st.session_state.llm_orchestrator.get_available_models()
    current_model = st.session_state.get("current_model", available_models[0])

    selected = st.selectbox(
        "Choose LLM",
        available_models,
        index=available_models.index(current_model)
        if current_model in available_models
        else 0,
        key="model_select",
    )
    st.session_state.current_model = selected

    try:
        model_info = (
            st.session_state.llm_orchestrator.get_model(selected).get_model_info()
        )
        st.caption(
            f"Provider: {model_info.get('provider', 'Unknown')} · "
            f"Type: {model_info.get('type', 'Unknown')}"
        )
    except Exception:
        pass

        st.divider()

    # --------- Reset app ----------
    st.markdown("### ♻️ Session")

    if st.button("Reset app", type="secondary", help="Clear chat, state and reload app."):
        # Clear all Streamlit session state keys
        st.session_state.clear()
        # Rerun script to restore initial defaults
        st.rerun()

