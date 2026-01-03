# Feature & State List: `sidebar.py`

## RAG Configuration Tab

| User Action | Expected UI Change | Session State Variables Affected |
|-------------|-------------------|----------------------------------|
| Selects a document from dropdown | Dropdown updates to show selected doc | `doc_name`, `sidebar_doc_selector` |
| Changes PDF Parser (pypdf/docling/llamaparse) | Parser-specific options appear in Advanced section | `parsing_params.pdf_parser`, `sidebar_pdf_parser` |
| Changes Output Format | Dropdown updates selection | `parsing_params.output_format`, `sidebar_output_format` |
| Toggles "Normalize Whitespace" | Checkbox state changes | `parsing_params.normalize_whitespace`, `sidebar_normalize_whitespace` |
| Toggles "Remove Special Characters" | Checkbox state changes | `parsing_params.remove_special_chars`, `sidebar_remove_special_chars` |
| Expands "Advanced Parsing Options" | Reveals parser-specific settings | (no state change) |
| Changes "Max characters to parse" | Number input updates | `parsing_params.max_characters`, `sidebar_max_characters` |
| Enters LlamaParse API Key (when llamaparse selected) | Password field updates | `parsing_params.llamaparse_api_key`, `sidebar_llamaparse_api_key` |
| Changes Docling Compute Device | Dropdown updates, device status caption changes | `parsing_params.docling_device`, `sidebar_docling_device` |
| Toggles Docling "Enable OCR" | Checkbox state changes | `parsing_params.docling_enable_ocr`, `sidebar_docling_enable_ocr` |
| Toggles Docling "Extract tables and layout" | Checkbox state changes | `parsing_params.docling_table_structure`, `sidebar_docling_table_structure` |
| Adjusts Docling worker threads slider | Slider position updates | `parsing_params.docling_threads`, `sidebar_docling_threads` |
| Selects Docling filter labels (multiselect) | Tags added/removed from multiselect | `parsing_params.docling_filter_labels`, `sidebar_docling_filter_labels` |
| Toggles Docling "Extract images from PDF" | Checkbox state changes, captioning option appears | `parsing_params.docling_extract_images`, `sidebar_docling_extract_images` |
| Toggles Docling "Caption images with LLM" | Checkbox state changes, warning may appear | `parsing_params.docling_enable_captioning`, `sidebar_docling_enable_captioning` |
| Toggles pypdf "Preserve Structure" | Checkbox state changes | `parsing_params.preserve_structure`, `sidebar_preserve_structure` |
| Toggles pypdf "Extract Tables" | Checkbox state changes | `parsing_params.extract_tables`, `sidebar_extract_tables` |
| Changes Text Splitting Library | Splitter strategy options update | `chunking_params.provider`, `sidebar_chunking_provider` |
| Changes Text Splitting Strategy | Expander info updates, dynamic params appear | `chunking_params.splitter`, `sidebar_splitter` |
| Expands splitter "About" section | Shows when/how/best-for info | (no state change) |
| Adjusts splitter parameters (chunk_size, overlap, etc.) | Number inputs/selects update | `chunking_params.<param_name>`, `sidebar_param_<param_name>` |
| Changes Embedding Model | Dropdown updates selection | `embedding_model_name`, `sidebar_embedding_model` |
| Clicks "Save & Apply" | Success toast appears, page reruns | `doc_name`, `embedding_model_name`, `parsing_params`, `chunking_params`, `applied_parsing_params`, `applied_chunking_params`; **Deletes**: `chunks`, `last_embeddings_result`, `search_results` (conditionally) |
| Clicks "Clear Session State" | Success toast appears, session cleared | **Deletes**: all keys except `session_restored`, `doc_name`, `chunking_params`, `embedding_model_name`, `llm_*`, `current_step` |

## LLM Configuration Tab

| User Action | Expected UI Change | Session State Variables Affected |
|-------------|-------------------|----------------------------------|
| Selects LLM Provider | Model options change, OpenAI-Compatible shows info card | `llm_provider`, `sidebar_provider` |
| Selects Model (standard providers) | Dropdown updates | `llm_model`, `sidebar_model_select` |
| Enters Model Name (OpenAI-Compatible) | Text input updates | `llm_model`, `sidebar_model_input` |
| Enters API Key | Password field updates (or shows env success) | `llm_api_key`, `sidebar_api_key` |
| Enters Base URL (OpenAI-Compatible) | Text input updates | `llm_base_url`, `sidebar_base_url` |
| Expands "Advanced Settings" | Reveals temperature, max tokens, system prompt | (no state change) |
| Adjusts Temperature slider | Slider position updates | `llm_temperature`, `sidebar_temperature` |
| Adjusts Max Tokens slider | Slider position updates | `llm_max_tokens`, `sidebar_max_tokens` |
| Edits System Prompt | Text area updates | `llm_system_prompt`, `sidebar_system_prompt` |
| Clicks "Save Configuration" | Config saved to storage | (persists current `llm_*` values to file) |

## Automatic State Behaviors

| Trigger | UI Change | Session State Variables Affected |
|---------|-----------|----------------------------------|
| No documents exist | Info message shown, stop form | `doc_name` set to `None` |
| `doc_name` is "Sample Text" but files exist | Auto-selects first file | `doc_name`, **Deletes**: `chunks`, `last_embeddings_result`, `search_results` |
| Current `doc_name` not in available docs | Auto-selects first available or `None` | `doc_name` |
| Invalid LLM config detected | Warning message displayed | (no state change, display only) |
| API key found in environment | Success badge shown, input hidden | `llm_api_key` cleared |

## Initial State Defaults

| Variable | Default Value |
|----------|---------------|
| `doc_name` | First available doc or `None` |
| `chunking_params` | `{provider: "LangChain", splitter: "RecursiveCharacterTextSplitter", chunk_size: 500, chunk_overlap: 50}` |
| `embedding_model_name` | `DEFAULT_MODEL` |
| `parsing_params` | `{pdf_parser: "pypdf", output_format: "markdown", normalize_whitespace: True, remove_special_chars: False, ...}` |
| `llm_provider` | `"OpenAI"` |
| `llm_model` | `""` |
| `llm_api_key` | `""` |
| `llm_base_url` | `""` |
| `llm_temperature` | `0.7` |
| `llm_max_tokens` | `1024` |
| `llm_system_prompt` | `DEFAULT_SYSTEM_PROMPT` |
