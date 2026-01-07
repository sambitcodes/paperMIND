import streamlit as st


def render_sources():
    sources = st.session_state.get("last_sources", []) or []
    if not sources:
        st.caption("No sources to display yet.")
        return

    for i, s in enumerate(sources, start=1):
        st.markdown(
            f"**Source {i}**  \n"
            f"- Document: {s.get('document','Unknown')}  \n"
            f"- Page: {s.get('page_number','?')}  \n"
            f"- Section: {s.get('section','')}  \n"
            f"- Subsection: {s.get('subsection','')}  \n"
            f"- Chunk ID: `{s.get('chunk_id','')}`"
        )
        st.divider()
