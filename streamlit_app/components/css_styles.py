import streamlit as st

def load_custom_css():
    st.markdown(
        """
        <style>
        /* ---------- Layout & background ---------- */
        .stApp {
            background: radial-gradient(circle at top left, #0f172a 0, #020617 55%, #0b1120 100%);
            color: #e5e7eb;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        /* Main header container */
        .main-header h1 {
            background: linear-gradient(90deg, #38bdf8, #a855f7, #f97316);
            -webkit-background-clip: text;
            color: transparent;
            font-weight: 800;
            letter-spacing: 0.03em;
        }
        .main-header p {
            color: #cbd5f5;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617, #020617 40%, #0b1120 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.35);
        }
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4 {
            color: #e5e7eb;
        }
        section[data-testid="stSidebar"] .stButton>button {
            border-radius: 999px;
            background: linear-gradient(90deg, #22c55e, #16a34a);
            color: #f9fafb;
            border: none;
            box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.25);
        }
        section[data-testid="stSidebar"] .stButton>button:hover {
            filter: brightness(1.08);
        }

        /* Inputs in sidebar */
        section[data-testid="stSidebar"] .stTextInput input,
        section[data-testid="stSidebar"] .stNumberInput input {
            background: rgba(15, 23, 42, 0.9);
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.4);
            color: #e5e7eb;
        }

        /* ---------- Chat area ---------- */
        /* Container for messages column */
        div.block-container {
            padding-top: 1.5rem;
        }

        /* Chat messages */
        div[data-testid="stChatMessage"] {
            padding: 0.25rem 0;
        }

        /* Common chat bubble styling */
        div[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] {
            border-radius: 14px;
            padding: 0.75rem 0.9rem;
            border: 1px solid rgba(148, 163, 184, 0.3);
        }

        /* User bubble: blue gradient */
        div[data-testid="stChatMessage"][data-testid*="user"] div[data-testid="stChatMessageContent"],
        div[data-testid="stChatMessage"]:has(svg[aria-label="user"]) div[data-testid="stChatMessageContent"] {
            background: linear-gradient(135deg, #1d4ed8, #22c55e);
            color: #f9fafb;
        }

        /* Assistant bubble: dark card with accent border */
        div[data-testid="stChatMessage"][data-testid*="assistant"] div[data-testid="stChatMessageContent"],
        div[data-testid="stChatMessage"]:has(svg[aria-label="assistant"]) div[data-testid="stChatMessageContent"] {
            background: radial-gradient(circle at top left, #1e293b, #020617);
            color: #e5e7eb;
            border-color: rgba(56, 189, 248, 0.55);
            box-shadow: 0 16px 40px rgba(8, 47, 73, 0.75);
        }

        /* Ensure text and code contrast inside chat */
        div[data-testid="stChatMessageContent"] * {
            color: inherit !important;
        }
        div[data-testid="stChatMessageContent"] code {
            background: rgba(15, 23, 42, 0.9);
            border-radius: 4px;
            padding: 0.1rem 0.3rem;
        }
        div[data-testid="stChatMessageContent"] a {
            color: #38bdf8 !important;
            text-decoration: none;
        }
        div[data-testid="stChatMessageContent"] a:hover {
            text-decoration: underline;
        }

        /* ---------- Chat input bar ---------- */
        div[data-testid="stChatInput"] {
            border-top: 1px solid rgba(148, 163, 184, 0.4);
            background: linear-gradient(90deg, rgba(15,23,42,0.96), rgba(15,23,42,0.98));
        }
        div[data-testid="stChatInput"] textarea {
            background: rgba(15, 23, 42, 0.95) !important;
            color: #e5e7eb !important;
            border-radius: 999px !important;
            border: 1px solid rgba(148, 163, 184, 0.5) !important;
            padding-left: 1rem !important;
        }
        div[data-testid="stChatInput"] textarea::placeholder {
            color: rgba(148, 163, 184, 0.7) !important;
        }
        div[data-testid="stChatInput"] button {
            border-radius: 999px !important;
            background: linear-gradient(135deg, #38bdf8, #a855f7) !important;
            border: none !important;
            color: #f9fafb !important;
            box-shadow: 0 12px 24px rgba(59, 130, 246, 0.4);
        }

        /* ---------- Misc cards / expanders ---------- */
        .st-expander {
            border-radius: 12px !important;
            border: 1px solid rgba(148, 163, 184, 0.4) !important;
            background: radial-gradient(circle at top left, #020617, #020617 60%, #0b1120 100%) !important;
        }
        .st-expander summary {
            color: #e5e7eb;
        }

        .stAlert {
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
