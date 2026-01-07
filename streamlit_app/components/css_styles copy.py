import streamlit as st

def load_custom_css():
    st.markdown(
        """
        <style>
        /* ---------- Base app ---------- */
        .stApp {
            background: #0e1117;
            color: #e6edf3;
        }

        /* ---------- Chat message bubbles ---------- */
        div[data-testid="stChatMessage"] {
            padding: 0.25rem 0;
        }

        /* Assistant bubble */
        div[data-testid="stChatMessage"][data-testid*="assistant"] div[data-testid="stChatMessageContent"],
        div[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] {
            background: #161b22;
            color: #e6edf3;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 12px 14px;
        }

        /* User bubble: Streamlit doesn't always expose a stable selector for role,
           so we at least make the content readable everywhere. */
        div[data-testid="stChatMessageContent"] * {
            color: #e6edf3 !important;
        }

        /* Links inside chat */
        div[data-testid="stChatMessageContent"] a {
            color: #79c0ff !important;
        }

        /* ---------- Chat input ---------- */
        div[data-testid="stChatInput"] textarea {
            background: #0b1220 !important;
            color: #e6edf3 !important;
            border: 1px solid rgba(255,255,255,0.16) !important;
        }

        /* Placeholder text */
        div[data-testid="stChatInput"] textarea::placeholder {
            color: rgba(230, 237, 243, 0.55) !important;
        }

        /* Make the send button visible */
        div[data-testid="stChatInput"] button {
            background: #238636 !important;
            color: #ffffff !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
