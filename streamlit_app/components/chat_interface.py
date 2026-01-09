"""
Chat interface component for PaperMind (defensive + streaming).

- Does NOT assume st.session_state.rag_pipeline exists.
- If RAG pipeline is missing (no PDF indexed yet), it falls back to LLM-only mode.
- Streams responses when supported (generator output).
"""

from __future__ import annotations

from typing import Iterable
import re

import streamlit as st

from streamlit_app.session_state import get_session_state


def _ensure_messages_initialized() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def _normalize_latex(md: str) -> str:
    """
    Streamlit Markdown supports $...$ and $$...$$ for math. [web:388]
    Many LLMs return \\(\\) and \\[\\], so normalize those.
    """
    if not md:
        return ""

    md = md.replace(r"\(", "$").replace(r"\)", "$")
    md = md.replace(r"\[", "$$").replace(r"\]", "$$")

    # Put $$ on separate lines for better rendering stability.
    md = re.sub(r"(?<!\n)\$\$", "\n$$", md)
    md = re.sub(r"\$\$(?!\n)", "$$\n", md)
    return md


def _stream_to_markdown(stream: Iterable[str], placeholder) -> str:
    """
    Stream chunks into a single markdown element and return the final text.
    Uses st.empty() placeholder pattern. [web:43]
    """
    acc = ""
    for chunk in stream:
        if chunk:
            acc += str(chunk)
            placeholder.markdown(_normalize_latex(acc + "▌"))
    placeholder.markdown(_normalize_latex(acc))
    return acc


def _render_sources_list(sources: list[dict]) -> None:
    if not sources:
        st.write("No sources available.")
        return

    for i, s in enumerate(sources, start=1):
        st.markdown(
            _normalize_latex(
                f"**Source {i}**  \n"
                f"- Document: {s.get('document','Unknown')}  \n"
                f"- Page: {s.get('page_number','?')}  \n"
                f"- Section: {s.get('section','')}  \n"
                f"- Subsection: {s.get('subsection','')}  \n"
                f"- Chunk ID: `{s.get('chunk_id','')}`"
            )
        )
        if i != len(sources):
            st.divider()


def render_chat():
    state = get_session_state()
    _ensure_messages_initialized()

    # Render history
    for msg in st.session_state["messages"]:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")

        with st.chat_message(role):
            st.markdown(_normalize_latex(content))

            if role == "assistant" and msg.get("sources"):
                with st.expander("View sources", expanded=False):
                    _render_sources_list(msg["sources"])

    prompt = st.chat_input("Ask a question about the indexed paper…")
    if not prompt:
        return

    # User bubble
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(_normalize_latex(prompt))

    # Assistant bubble (streaming)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                message_placeholder = st.empty()

                # ---------- RAG mode ----------
                if (
                    bool(st.session_state.get("use_rag", True))
                    and st.session_state.get("rag_pipeline") is not None
                ):
                    # Use retriever for context + citations
                    rag = st.session_state["rag_pipeline"]
                    retrieval = rag.retriever.retrieve(prompt)

                    docs = retrieval.documents or []
                    citations = retrieval.citations or []

                    context = "\n\n".join([d for d in docs if d]).strip()

                    sources = []
                    for c in citations:
                        sources.append(
                            {
                                "document": c.document,
                                "page_number": c.page_number,
                                "section": c.section,
                                "subsection": c.subsection,
                                "chunk_id": c.chunk_id,
                            }
                        )

                    if context:
                        stream = state.llm_orchestrator.stream_response(
                            prompt=(
                                "Use the following context to answer the question.\n\n"
                                f"Context:\n{context}\n\n"
                                f"Question:\n{prompt}\n"
                            ),
                            model_name=state.current_model,
                            temperature=state.temperature,
                            context=None,
                        )
                        final_text = _stream_to_markdown(stream, message_placeholder)
                    else:
                        final_text = "No sources retrieved for this question."
                        message_placeholder.markdown(final_text)

                    if sources:
                        with st.expander("View sources", expanded=False):
                            _render_sources_list(sources)

                    st.session_state["messages"].append(
                        {"role": "assistant", "content": final_text, "sources": sources}
                    )
                    return

                # ---------- LLM-only mode ----------
                stream = state.llm_orchestrator.stream_response(
                    prompt=prompt,
                    model_name=state.current_model,
                    temperature=state.temperature,
                    context=None,
                )
                final_text = _stream_to_markdown(stream, message_placeholder)

                st.session_state["messages"].append(
                    {"role": "assistant", "content": final_text, "sources": []}
                )

            except Exception as e:
                err = f"Error generating response: {e}"
                st.error(err)
                st.session_state["messages"].append({"role": "assistant", "content": err, "sources": []})
