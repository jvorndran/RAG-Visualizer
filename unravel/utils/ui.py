from html import escape

import streamlit as st
import streamlit_shadcn_ui as ui


def apply_custom_styles() -> None:
    """Apply global custom styles to the Streamlit app."""
    # Reduced to minimal font imports and essential container tweaks
    st.markdown(
        """<style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Ensure streamlit-shadcn-ui components take full width */
        div[data-testid="stIFrame"] {
            width: 100% !important;
        }
        div[data-testid="stIFrame"] iframe {
            width: 100% !important;
        }

        /* Clean up main container padding */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            max-width: 1300px !important;
        }

        /* Sidebar shell */
        section[data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1.35rem;
        }

        /* Global Streamlit component polish */
        [data-testid="stVerticalBlock"] {
            gap: 0.55rem;
        }

        [data-testid="stTabs"] {
            margin-top: 0.35rem;
        }

        [data-baseweb="tab-list"] {
            gap: 0.25rem;
            border-bottom: 1px solid #dbe3ee;
        }

        [data-baseweb="tab"] {
            height: 2rem;
            padding: 0 0.1rem;
            color: #475569;
            font-size: 0.85rem;
            font-weight: 500;
        }

        [aria-selected="true"] {
            color: #020817 !important;
            font-weight: 650;
        }

        label,
        p,
        span {
            letter-spacing: 0;
        }

        [data-testid="stCaptionContainer"] {
            color: #64748b;
            line-height: 1.35;
        }

        [data-testid="stAlert"] {
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
        }

        div[data-baseweb="select"] > div,
        input {
            border-radius: 7px;
        }

        div[data-testid="stButton"] button {
            min-height: 2.35rem;
            border-radius: 7px;
            font-weight: 600;
        }

        div[data-testid="stExpander"] {
            border-color: #dbe3ee;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.58);
        }

        div[data-testid="stForm"],
        div[data-testid="stFileUploader"] section,
        div[data-testid="stCodeBlock"] {
            border-radius: 8px;
        }

        div[data-testid="stCodeBlock"] pre {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
        }

        textarea {
            border-radius: 7px !important;
        }

        hr {
            margin: 0.4rem 0;
            border-color: #e2e8f0;
        }

        .title-container {
            margin: 0.25rem 0 0.75rem;
        }

        .main-title {
            color: #020817;
            font-size: 1.45rem;
            font-weight: 750;
            letter-spacing: 0;
            line-height: 1.2;
        }

        .sidebar-app-title {
            color: #020817;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin: 0 0 0.15rem;
            text-transform: uppercase;
        }

        .sidebar-section-title {
            color: #020817;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin: 1.05rem 0 0.15rem;
            text-transform: uppercase;
        }

        .sidebar-status {
            align-items: center;
            border-radius: 999px;
            display: inline-flex;
            font-size: 0.78rem;
            font-weight: 650;
            line-height: 1;
            margin: 1rem 0 0.2rem;
            padding: 0.42rem 0.65rem;
        }

        .sidebar-status-applied {
            background: #ecfdf5;
            color: #047857;
        }

        .sidebar-status-pending {
            background: #fff7ed;
            color: #c2410c;
        }

        .page-header {
            margin: 0.75rem 0 1.35rem;
        }

        .page-eyebrow,
        .section-eyebrow {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 650;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
        }

        .page-title {
            color: #020817;
            font-size: 2rem;
            font-weight: 750;
            letter-spacing: 0;
            line-height: 1.15;
            margin: 0;
        }

        .page-caption {
            color: #64748b;
            font-size: 0.95rem;
            line-height: 1.45;
            margin-top: 0.5rem;
            max-width: 720px;
        }

        .section-heading {
            margin: 1.45rem 0 0.65rem;
        }

        .section-title {
            color: #020817;
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: 0;
            margin: 0;
        }

        .section-caption {
            color: #64748b;
            font-size: 0.88rem;
            line-height: 1.4;
            margin-top: 0.25rem;
        }

        /* Chunk highlighting specific styles (retained as it's custom HTML) */
        .chunk-highlight {
            background-color: #fee2e2;
            border-radius: 2px;
            padding: 0 2px;
        }
        
        /* Stats Bar (legacy support until fully refactored) */
        .stats-bar {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        /* Custom Primary Button Style (Black) */
        div.stButton > button[kind="primary"] {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 1px solid #000000 !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #333333 !important;
            border-color: #333333 !important;
            color: #ffffff !important;
        }
        div.stButton > button[kind="primary"]:active {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        div.stButton > button[kind="primary"]:focus {
            box-shadow: none !important;
            outline: none !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )


def render_page_header(title: str, caption: str, eyebrow: str | None = None) -> None:
    """Render a consistent page header."""
    eyebrow_html = f'<div class="page-eyebrow">{escape(eyebrow)}</div>' if eyebrow else ""
    st.markdown(
        f"""<div class="page-header">
{eyebrow_html}
<h1 class="page-title">{escape(title)}</h1>
<div class="page-caption">{escape(caption)}</div>
</div>""",
        unsafe_allow_html=True,
    )


def render_section_heading(title: str, caption: str | None = None) -> None:
    """Render a compact section heading."""
    caption_html = f'<div class="section-caption">{escape(caption)}</div>' if caption else ""
    st.markdown(
        f"""<div class="section-heading">
<div class="section-title">{escape(title)}</div>
{caption_html}
</div>""",
        unsafe_allow_html=True,
    )


def render_step_nav(active_step: str = "chunks") -> None:
    """Render the step navigation bar using shadcn tabs."""

    # Map internal IDs to Display Labels
    step_map = {
        "upload": "Upload",
        "chunks": "Text Splitting",
        "embeddings": "Vector Embedding",
        "query": "Response Generation",
        "export": "Export Code",
    }

    # Reverse map for lookup
    label_map = {v: k for k, v in step_map.items()}

    # Get current label
    default_value = step_map.get(active_step, "Text Splitting")

    # Render Tabs
    # Spacer
    selected_label = ui.tabs(
        options=list(step_map.values()),
        default_value=default_value,
        key="main_navigation_tabs",
    )

    # Handle Navigation
    selected_id = label_map.get(selected_label)

    if selected_id and selected_id != active_step:
        st.session_state.current_step = selected_id
        st.rerun(scope="fragment")
