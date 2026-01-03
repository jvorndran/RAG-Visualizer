"""Tests for sidebar PDF parser selection feature.

Feature 2: Changes PDF Parser (pypdf/docling/llamaparse)
- User Action: Changes PDF Parser selection
- Expected UI Change: Parser-specific options appear in Advanced section
- Session State Variables Affected: parsing_params.pdf_parser, sidebar_pdf_parser

Fixtures used from conftest.py:
- mock_storage_dir: Creates temp directory with test documents
- sidebar_app_script: App script for rendering the sidebar
- element_exists: Helper to check if element exists
"""

from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from tests.conftest import element_exists, get_form_submit_button


class TestPdfParserDropdownDisplay:
    """Tests for PDF parser dropdown display behavior."""

    def test_pdf_parser_dropdown_exists(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify PDF parser dropdown is rendered."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            pdf_parser_selectbox = at.selectbox(key="sidebar_pdf_parser")
            assert pdf_parser_selectbox is not None

    def test_pdf_parser_has_all_options(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify PDF parser dropdown has all three parser options."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            pdf_parser_selectbox = at.selectbox(key="sidebar_pdf_parser")
            options = pdf_parser_selectbox.options

            assert "pypdf" in options
            assert "docling" in options
            assert "llamaparse" in options
            assert len(options) == 3

    def test_pdf_parser_defaults_to_pypdf(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify PDF parser defaults to pypdf."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            pdf_parser_selectbox = at.selectbox(key="sidebar_pdf_parser")
            assert pdf_parser_selectbox.value == "pypdf"


class TestPdfParserSessionState:
    """Tests for PDF parser session state behavior."""

    def test_sidebar_pdf_parser_initialized(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify sidebar_pdf_parser session state is initialized."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            assert "sidebar_pdf_parser" in at.session_state
            assert at.session_state["sidebar_pdf_parser"] == "pypdf"

    def test_parsing_params_has_pdf_parser(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify parsing_params contains pdf_parser key."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            assert "parsing_params" in at.session_state
            assert "pdf_parser" in at.session_state["parsing_params"]
            assert at.session_state["parsing_params"]["pdf_parser"] == "pypdf"

    def test_preset_pdf_parser_reflected_in_dropdown(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify pre-set pdf_parser value is reflected in dropdown."""
        # Mock device detection to avoid slow CUDA/torch checks
        mock_devices = ["auto", "cpu"]
        mock_device_info = {
            "cuda_available": False,
            "mps_available": False,
            "torch_installed": False,
            "gpu_name": None,
            "cuda_version": None,
        }
        with (
            patch(
                "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
            ),
            patch(
                "rag_visualizer.utils.parsers.get_available_devices",
                return_value=mock_devices,
            ),
            patch(
                "rag_visualizer.utils.parsers.get_device_info",
                return_value=mock_device_info,
            ),
        ):
            at = AppTest.from_string(sidebar_app_script)
            parsing_params = {
                "pdf_parser": "docling",
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
            at.session_state["parsing_params"] = parsing_params
            at.session_state["applied_parsing_params"] = parsing_params.copy()
            at.run()

            pdf_parser_selectbox = at.selectbox(key="sidebar_pdf_parser")
            assert pdf_parser_selectbox.value == "docling"


class TestPdfParserUserAction:
    """Tests for user interaction with PDF parser selection."""

    def test_selectbox_accepts_docling_selection(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify selectbox accepts docling selection."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            pdf_parser_selectbox = at.selectbox(key="sidebar_pdf_parser")
            pdf_parser_selectbox.set_value("docling").run()

            pdf_parser_selectbox = at.selectbox(key="sidebar_pdf_parser")
            assert pdf_parser_selectbox.value == "docling"

    def test_selectbox_accepts_llamaparse_selection(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify selectbox accepts llamaparse selection."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            pdf_parser_selectbox = at.selectbox(key="sidebar_pdf_parser")
            pdf_parser_selectbox.set_value("llamaparse").run()

            pdf_parser_selectbox = at.selectbox(key="sidebar_pdf_parser")
            assert pdf_parser_selectbox.value == "llamaparse"


class TestDoclingParserOptions:
    """Tests for docling parser-specific options in Advanced section."""

    def test_docling_device_selector_appears(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify docling device selector appears when docling is selected."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Select docling parser
            pdf_parser_selectbox = at.selectbox(key="sidebar_pdf_parser")
            pdf_parser_selectbox.set_value("docling").run()

            # Docling device selector should exist
            docling_device = at.selectbox(key="sidebar_docling_device")
            assert docling_device is not None

    def test_docling_ocr_checkbox_appears(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify docling OCR checkbox appears when docling is selected."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            pdf_parser_selectbox = at.selectbox(key="sidebar_pdf_parser")
            pdf_parser_selectbox.set_value("docling").run()

            docling_ocr = at.checkbox(key="sidebar_docling_enable_ocr")
            assert docling_ocr is not None

    def test_docling_table_structure_checkbox_appears(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify docling table structure checkbox appears when docling is selected."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            pdf_parser_selectbox = at.selectbox(key="sidebar_pdf_parser")
            pdf_parser_selectbox.set_value("docling").run()

            docling_table = at.checkbox(key="sidebar_docling_table_structure")
            assert docling_table is not None

    def test_docling_threads_slider_appears(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify docling threads slider appears when docling is selected."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            pdf_parser_selectbox = at.selectbox(key="sidebar_pdf_parser")
            pdf_parser_selectbox.set_value("docling").run()

            docling_threads = at.slider(key="sidebar_docling_threads")
            assert docling_threads is not None

    def test_docling_extract_images_checkbox_appears(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify docling extract images checkbox appears when docling is selected."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            pdf_parser_selectbox = at.selectbox(key="sidebar_pdf_parser")
            pdf_parser_selectbox.set_value("docling").run()

            docling_images = at.checkbox(key="sidebar_docling_extract_images")
            assert docling_images is not None

    def test_docling_options_not_present_for_pypdf(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify docling-specific options are not shown when pypdf is selected."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # pypdf is the default, so docling options should not exist
            assert not element_exists(at, "selectbox", "sidebar_docling_device")


class TestLlamaparseOptions:
    """Tests for llamaparse parser-specific options in Advanced section."""

    def test_llamaparse_api_key_input_appears(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify llamaparse API key input appears when llamaparse is selected."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            pdf_parser_selectbox = at.selectbox(key="sidebar_pdf_parser")
            pdf_parser_selectbox.set_value("llamaparse").run()

            api_key_input = at.text_input(key="sidebar_llamaparse_api_key")
            assert api_key_input is not None

    def test_llamaparse_api_key_not_present_for_pypdf(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify llamaparse API key is not shown when pypdf is selected."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # pypdf is the default
            assert not element_exists(at, "text_input", "sidebar_llamaparse_api_key")

    def test_llamaparse_api_key_not_present_for_docling(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify llamaparse API key is not shown when docling is selected."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            pdf_parser_selectbox = at.selectbox(key="sidebar_pdf_parser")
            pdf_parser_selectbox.set_value("docling").run()

            assert not element_exists(at, "text_input", "sidebar_llamaparse_api_key")


class TestPypdfParserOptions:
    """Tests for pypdf parser-specific options in Advanced section."""

    def test_pypdf_preserve_structure_checkbox_appears(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify pypdf preserve structure checkbox appears when pypdf is selected."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # pypdf is the default
            preserve_structure = at.checkbox(key="sidebar_preserve_structure")
            assert preserve_structure is not None

    def test_pypdf_extract_tables_checkbox_appears(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify pypdf extract tables checkbox appears when pypdf is selected."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            extract_tables = at.checkbox(key="sidebar_extract_tables")
            assert extract_tables is not None

    def test_pypdf_options_not_present_for_docling(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify pypdf-specific options are not shown when docling is selected."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            pdf_parser_selectbox = at.selectbox(key="sidebar_pdf_parser")
            pdf_parser_selectbox.set_value("docling").run()

            assert not element_exists(at, "checkbox", "sidebar_preserve_structure")

    def test_pypdf_options_not_present_for_llamaparse(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify pypdf-specific options are not shown when llamaparse is selected."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            pdf_parser_selectbox = at.selectbox(key="sidebar_pdf_parser")
            pdf_parser_selectbox.set_value("llamaparse").run()

            assert not element_exists(at, "checkbox", "sidebar_preserve_structure")


class TestPdfParserFormSubmission:
    """Tests for PDF parser form submission updating session state.

    These tests verify the critical behavior: that selecting a parser
    and submitting the form actually updates parsing_params.pdf_parser.
    """

    def test_form_submit_updates_pdf_parser_to_docling(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify form submission updates parsing_params.pdf_parser to docling."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Initial state should be pypdf
            assert at.session_state["parsing_params"]["pdf_parser"] == "pypdf"

            # Change selectbox to docling
            at.selectbox(key="sidebar_pdf_parser").set_value("docling")

            # Submit the form
            submit_btn = get_form_submit_button(at)
            assert submit_btn is not None, "Save & Apply button not found"
            submit_btn.click().run()

            # Verify parsing_params was updated
            assert at.session_state["parsing_params"]["pdf_parser"] == "docling"

    def test_form_submit_updates_pdf_parser_to_llamaparse(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify form submission updates parsing_params.pdf_parser to llamaparse."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Change selectbox to llamaparse
            at.selectbox(key="sidebar_pdf_parser").set_value("llamaparse")

            # Submit the form
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify parsing_params was updated
            assert at.session_state["parsing_params"]["pdf_parser"] == "llamaparse"

    def test_form_submit_preserves_other_parsing_params(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify form submission preserves other parsing params when changing parser."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Get initial max_characters value
            initial_max_chars = at.session_state["parsing_params"]["max_characters"]

            # Change parser to llamaparse
            at.selectbox(key="sidebar_pdf_parser").set_value("llamaparse")
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify other params are preserved
            assert at.session_state["parsing_params"]["max_characters"] == initial_max_chars
            assert at.session_state["parsing_params"]["pdf_parser"] == "llamaparse"

    def test_form_submit_updates_applied_parsing_params(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify form submission also updates applied_parsing_params."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Change parser
            at.selectbox(key="sidebar_pdf_parser").set_value("llamaparse")
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Both should be updated
            assert at.session_state["parsing_params"]["pdf_parser"] == "llamaparse"
            assert at.session_state["applied_parsing_params"]["pdf_parser"] == "llamaparse"


class TestParserOptionStateUpdates:
    """Tests for parser-specific options updating session state on form submission."""

    def test_docling_ocr_checkbox_updates_parsing_params(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify docling OCR checkbox updates parsing_params on form submit."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Switch to docling
            at.selectbox(key="sidebar_pdf_parser").set_value("docling").run()

            # Initial OCR state should be False
            assert at.session_state["parsing_params"]["docling_enable_ocr"] is False

            # Enable OCR
            at.checkbox(key="sidebar_docling_enable_ocr").set_value(True)
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            # Verify it updated
            assert at.session_state["parsing_params"]["docling_enable_ocr"] is True

    def test_docling_table_structure_updates_parsing_params(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify docling table structure checkbox updates parsing_params."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Switch to docling
            at.selectbox(key="sidebar_pdf_parser").set_value("docling").run()

            # Initial state is True, disable it
            at.checkbox(key="sidebar_docling_table_structure").set_value(False)
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            assert at.session_state["parsing_params"]["docling_table_structure"] is False

    def test_pypdf_preserve_structure_updates_parsing_params(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify pypdf preserve structure checkbox updates parsing_params."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # pypdf is default, initial state is True
            assert at.session_state["parsing_params"]["preserve_structure"] is True

            # Disable it
            at.checkbox(key="sidebar_preserve_structure").set_value(False)
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            assert at.session_state["parsing_params"]["preserve_structure"] is False

    def test_pypdf_extract_tables_updates_parsing_params(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify pypdf extract tables checkbox updates parsing_params."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Initial state is True
            assert at.session_state["parsing_params"]["extract_tables"] is True

            # Disable it
            at.checkbox(key="sidebar_extract_tables").set_value(False)
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            assert at.session_state["parsing_params"]["extract_tables"] is False

    def test_llamaparse_api_key_updates_parsing_params(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify llamaparse API key updates parsing_params on form submit."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Switch to llamaparse
            at.selectbox(key="sidebar_pdf_parser").set_value("llamaparse").run()

            # Set API key
            at.text_input(key="sidebar_llamaparse_api_key").set_value("test-api-key-123")
            submit_btn = get_form_submit_button(at)
            submit_btn.click().run()

            assert at.session_state["parsing_params"]["llamaparse_api_key"] == "test-api-key-123"


class TestParserSwitchingBehavior:
    """Tests for parser switching state machine and conditional rendering."""

    def test_switch_pypdf_to_docling_shows_docling_options(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify switching from pypdf to docling shows docling options."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Start with pypdf - docling options should NOT exist
            assert not element_exists(at, "selectbox", "sidebar_docling_device")
            assert not element_exists(at, "checkbox", "sidebar_docling_enable_ocr")

            # Switch to docling
            at.selectbox(key="sidebar_pdf_parser").set_value("docling").run()

            # Now docling options SHOULD exist
            assert element_exists(at, "selectbox", "sidebar_docling_device")
            assert element_exists(at, "checkbox", "sidebar_docling_enable_ocr")
            assert element_exists(at, "checkbox", "sidebar_docling_table_structure")

    def test_switch_docling_to_pypdf_hides_docling_options(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify switching from docling back to pypdf hides docling options."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Switch to docling first
            at.selectbox(key="sidebar_pdf_parser").set_value("docling").run()
            assert element_exists(at, "selectbox", "sidebar_docling_device")

            # Switch back to pypdf
            at.selectbox(key="sidebar_pdf_parser").set_value("pypdf").run()

            # Docling options should be gone, pypdf options should appear
            assert not element_exists(at, "selectbox", "sidebar_docling_device")
            assert element_exists(at, "checkbox", "sidebar_preserve_structure")

    def test_switch_llamaparse_to_docling_swaps_options(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify switching from llamaparse to docling swaps parser-specific options."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Switch to llamaparse
            at.selectbox(key="sidebar_pdf_parser").set_value("llamaparse").run()
            assert element_exists(at, "text_input", "sidebar_llamaparse_api_key")
            assert not element_exists(at, "selectbox", "sidebar_docling_device")

            # Switch to docling
            at.selectbox(key="sidebar_pdf_parser").set_value("docling").run()

            # Llamaparse options gone, docling options appear
            assert not element_exists(at, "text_input", "sidebar_llamaparse_api_key")
            assert element_exists(at, "selectbox", "sidebar_docling_device")

    def test_full_parser_cycle_maintains_correct_options(
        self, mock_storage_dir, sidebar_app_script
    ):
        """Verify cycling through all parsers shows correct options at each step."""
        with patch(
            "rag_visualizer.services.storage.DEFAULT_STORAGE_DIR", mock_storage_dir
        ):
            at = AppTest.from_string(sidebar_app_script).run()

            # Step 1: pypdf (default)
            assert element_exists(at, "checkbox", "sidebar_preserve_structure")
            assert not element_exists(at, "selectbox", "sidebar_docling_device")
            assert not element_exists(at, "text_input", "sidebar_llamaparse_api_key")

            # Step 2: docling
            at.selectbox(key="sidebar_pdf_parser").set_value("docling").run()
            assert not element_exists(at, "checkbox", "sidebar_preserve_structure")
            assert element_exists(at, "selectbox", "sidebar_docling_device")
            assert not element_exists(at, "text_input", "sidebar_llamaparse_api_key")

            # Step 3: llamaparse
            at.selectbox(key="sidebar_pdf_parser").set_value("llamaparse").run()
            assert not element_exists(at, "checkbox", "sidebar_preserve_structure")
            assert not element_exists(at, "selectbox", "sidebar_docling_device")
            assert element_exists(at, "text_input", "sidebar_llamaparse_api_key")

            # Step 4: back to pypdf
            at.selectbox(key="sidebar_pdf_parser").set_value("pypdf").run()
            assert element_exists(at, "checkbox", "sidebar_preserve_structure")
            assert not element_exists(at, "selectbox", "sidebar_docling_device")
            assert not element_exists(at, "text_input", "sidebar_llamaparse_api_key")
