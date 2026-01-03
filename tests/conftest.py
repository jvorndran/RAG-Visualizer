"""Shared fixtures and helpers for sidebar tests.

These fixtures provide common setup for testing Streamlit sidebar components
using streamlit.testing.v1.
"""

from unittest.mock import patch

import pytest


def element_exists(at, element_type, key):
    """Check if an element with the given key exists in the app.

    Streamlit testing API raises KeyError when element doesn't exist,
    so we use try/except to check for presence.

    Args:
        at: AppTest instance
        element_type: Element type string (e.g., "selectbox", "checkbox", "text_input")
        key: The widget key to look for

    Returns:
        True if element exists, False otherwise
    """
    try:
        element_getter = getattr(at, element_type)
        element_getter(key=key)
        return True
    except KeyError:
        return False


def get_form_submit_button(at, label="Save & Apply"):
    """Find and return the form submit button by label.

    Args:
        at: AppTest instance
        label: The button label to search for

    Returns:
        The button element if found, None otherwise
    """
    for btn in at.button:
        if btn.label == label:
            return btn
    return None


@pytest.fixture
def mock_storage_dir(tmp_path):
    """Create a temporary storage directory with test documents.

    Creates:
        - documents/document_a.pdf
        - documents/document_b.pdf
        - documents/document_c.txt
        - config/ (for save_rag_config)
    """
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir(parents=True)

    # Create test documents
    (docs_dir / "document_a.pdf").write_bytes(b"test content a")
    (docs_dir / "document_b.pdf").write_bytes(b"test content b")
    (docs_dir / "document_c.txt").write_text("test content c")

    # Create config dir (needed for save_rag_config)
    (tmp_path / "config").mkdir(parents=True)

    return tmp_path


@pytest.fixture
def sidebar_app_script():
    """Return the app script string for testing the RAG config sidebar.

    This fixture provides a minimal app script that renders the RAG config
    sidebar. Session state initialization mirrors production behavior.

    Use with AppTest.from_string(sidebar_app_script).
    """
    return '''
import streamlit as st
from rag_visualizer.ui.sidebar import render_rag_config_sidebar

# Initialize required session state (mirrors production initialization)
if "chunking_params" not in st.session_state:
    st.session_state.chunking_params = {
        "provider": "LangChain",
        "splitter": "RecursiveCharacterTextSplitter",
        "chunk_size": 500,
        "chunk_overlap": 50,
    }
if "embedding_model_name" not in st.session_state:
    st.session_state.embedding_model_name = "all-MiniLM-L6-v2"
if "parsing_params" not in st.session_state:
    st.session_state.parsing_params = {
        "pdf_parser": "pypdf",
        "output_format": "markdown",
        "normalize_whitespace": True,
        "remove_special_chars": False,
        "llamaparse_api_key": "",
        "preserve_structure": True,
        "extract_tables": True,
        "docling_enable_ocr": False,
        "docling_table_structure": True,
        "docling_threads": 4,
        "docling_filter_labels": ["PAGE_HEADER", "PAGE_FOOTER"],
        "max_characters": 40000,
    }
if "applied_parsing_params" not in st.session_state:
    st.session_state.applied_parsing_params = st.session_state.parsing_params.copy()
if "applied_chunking_params" not in st.session_state:
    st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()

with st.sidebar:
    render_rag_config_sidebar()
'''


@pytest.fixture
def patched_storage(mock_storage_dir):
    """Context manager that patches the storage directory.

    Usage:
        with patched_storage:
            at = AppTest.from_string(sidebar_app_script).run()
    """
    return patch(
        "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
    )
