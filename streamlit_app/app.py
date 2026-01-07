"""
Single-page Streamlit application for Research Paper QA Bot.
Primary UX is in the sidebar (model select + upload + indexing + settings).
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from streamlit_app.components.css_styles import load_custom_css
from streamlit_app.session_state import initialize_session_state, get_session_state
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.chat_interface import render_chat
from streamlit_app.components.source_panel import render_sources

st.set_page_config(
    page_title="PaperMIND",
    page_icon="📚🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_custom_css()
initialize_session_state()
state = get_session_state()

def main():
    st.markdown(
    """
    <div class="main-header">
        <h1>
            📚🔍 <span style="color:#ef4444;">Paper</span><span style="color:#22c55e;">MIND</span>
        </h1>
        <p>Upload a paper, index it, and ask questions with grounded citations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        render_sidebar()

    # col1, col2 = st.columns([4, 1], gap="small")
    # with col1:
    st.markdown("### 💬 Chat")
    render_chat()

    # with col2:
    #     st.markdown("### Reset")
    #     render_sources()

if __name__ == "__main__":
    main()
