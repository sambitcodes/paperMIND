"""
Chat interface component for PaperMind (defensive + streaming).

- Does NOT assume st.session_state.rag_pipeline exists.
- If RAG pipeline is missing (no PDF indexed yet), it falls back to LLM-only mode.
- Streams responses when supported (generator output).
- Uses st.session_state.get(...) to avoid KeyErrors in multipage apps.
"""
from __future__ import annotations

from typing import Generator, Iterable, Union

import streamlit as st
from src.qa_pipeline.response_formatter import ResponseFormatter
from streamlit_app.session_state import get_session_state


def _ensure_messages_initialized() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _append_message(role: str, content: str) -> None:
    _ensure_messages_initialized()
    st.session_state.messages.append({"role": role, "content": content})


def _is_stream(obj) -> bool:
    # Accept generators/iterables of strings (GroqModel.generate(stream=True) yields chunks).
    # Avoid treating plain strings as iterables.
    return obj is not None and not isinstance(obj, str) and hasattr(obj, "__iter__")


def _stream_to_markdown(stream: Iterable[str], placeholder) -> str:
    """
    Stream chunks into a single markdown element and return the final text.
    Uses st.empty() placeholder pattern so the text stays in one bubble. [web:43]
    """
    acc = ""
    for chunk in stream:
        if chunk:
            acc += str(chunk)
            placeholder.markdown(acc)
    return acc


def render_chat():
    state = get_session_state()

    # Show existing chat history
    for msg in state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Guard rails: RAG requested but not ready
    if state.use_rag:
        if "rag_pipeline" not in st.session_state or st.session_state.rag_pipeline is None:
            st.info("RAG is enabled, but no index is loaded yet. Upload + Index a PDF from the sidebar.")
        elif not getattr(state, "collection_name", None):
            st.info("RAG is enabled, but no collection is selected. Index a PDF from the sidebar.")

    prompt = st.chat_input("Ask a question about the indexed paper…")
    if not prompt:
        return

    # Add user message
    state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # ---- RAG mode ----
                if state.use_rag and ("rag_pipeline" in st.session_state) and st.session_state.rag_pipeline:
                    result = st.session_state.rag_pipeline.answer(
                        question=prompt,
                        model_name=state.current_model,
                        temperature=state.temperature,
                        top_k=state.top_k,
                    )

                    answer = result.get("answer", "").strip() or "I couldn't generate an answer."
                    sources = result.get("sources", []) or []
                    st.markdown(answer)

                    # Only warn in UI (do NOT pollute assistant message)
                    if not sources:
                        st.warning(
                            "No sources retrieved for this question. "
                            "Try lowering similarity threshold or re-indexing."
                        )

                    # Store sources for right panel
                    st.session_state.last_sources = sources

                    # Persist assistant message
                    state.messages.append({"role": "assistant", "content": answer})
                    return

                # ---- LLM-only mode ----
                answer = state.llm_orchestrator.generate_response(
                    prompt=prompt,
                    model_name=state.current_model,
                    temperature=state.temperature,
                    context=None,
                )
                answer = (answer or "").strip() or "I couldn't generate an answer."
                st.markdown(answer)

                # No sources in LLM-only mode
                st.session_state.last_sources = []
                state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                err = f"Error generating response: {e}"
                st.error(err)
                state.messages.append({"role": "assistant", "content": err})