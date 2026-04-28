from typing import cast

import streamlit as st
import streamlit_shadcn_ui as ui

from unravel.services.embedders import DEFAULT_MODEL, list_available_models
from unravel.services.llm import (
    LLM_PROVIDERS,
    LLMConfig,
    get_api_key_from_env,
    get_provider_models,
    validate_config,
)
from unravel.services.storage import (
    clear_session_state,
    list_documents,
    load_llm_config,
    load_rag_config,
    save_llm_config,
    save_rag_config,
)
from unravel.ui.constants import WidgetKeys


def render_sidebar_section(title: str, caption: str | None = None) -> None:
    """Render a compact sidebar section heading."""
    st.markdown(f'<div class="sidebar-section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def render_sidebar_status(label: str, variant: str) -> None:
    """Render a lightweight sidebar status pill."""
    st.markdown(
        f'<div class="sidebar-status sidebar-status-{variant}">{label}</div>',
        unsafe_allow_html=True,
    )


def initialize_rag_sidebar_state() -> list[str]:
    """Load and initialize sidebar state for RAG controls."""
    if "rag_config_loaded" not in st.session_state:
        saved_config = load_rag_config()
        if saved_config:
            st.session_state.doc_name = saved_config.get("doc_name")
            st.session_state.embedding_model_name = saved_config.get(
                "embedding_model_name", DEFAULT_MODEL
            )
            st.session_state.chunking_params = saved_config.get("chunking_params", {})
            st.session_state.parsing_params = saved_config.get("parsing_params", {})
            st.session_state.retrieval_config = saved_config.get(
                "retrieval_config", {"strategy": "DenseRetriever", "params": {}}
            )
            st.session_state.reranking_config = saved_config.get(
                "reranking_config", {"enabled": False}
            )
        st.session_state.rag_config_loaded = True

    docs = list_documents()

    if "doc_name" not in st.session_state:
        st.session_state.doc_name = docs[0] if docs else None

    if st.session_state.doc_name == "Sample Text" and docs:
        st.session_state.doc_name = docs[0]
        for key in [
            "chunks",
            "chunks_cache_key",
            "chunk_display_cache_key",
            "chunk_display_data",
            "last_embeddings_result",
            "search_results",
        ]:
            if key in st.session_state:
                del st.session_state[key]

    if "chunking_params" not in st.session_state:
        st.session_state.chunking_params = {
            "provider": "Docling",
            "splitter": "HybridChunker",
            "max_tokens": 512,
            "chunk_overlap": 50,
            "tokenizer": "cl100k_base",
        }
    if "embedding_model_name" not in st.session_state:
        st.session_state.embedding_model_name = DEFAULT_MODEL

    if "parsing_params" not in st.session_state:
        st.session_state.parsing_params = {
            "docling_enable_ocr": False,
            "docling_table_structure": True,
            "docling_threads": 4,
            "docling_filter_labels": ["PAGE_HEADER", "PAGE_FOOTER"],
            "docling_extract_images": False,
            "docling_enable_captioning": False,
            "docling_device": "auto",
            "output_format": "markdown",
        }

    if "applied_parsing_params" not in st.session_state:
        st.session_state.applied_parsing_params = st.session_state.parsing_params.copy()
    if "applied_chunking_params" not in st.session_state:
        st.session_state.applied_chunking_params = st.session_state.chunking_params.copy()

    return docs


def render_document_selector(docs: list[str]) -> bool:
    """Render document selection and return whether RAG controls can continue."""
    render_sidebar_section("Document")
    all_docs = docs or (["Sample Text"] if st.session_state.doc_name == "Sample Text" else [])

    if st.session_state.doc_name and st.session_state.doc_name not in all_docs:
        st.session_state.doc_name = docs[0] if docs else None

    if not all_docs:
        st.info(
            "No documents available. Upload a file in the Upload step.",
            icon=":material/info:",
        )
        st.session_state.doc_name = None
        return False

    if WidgetKeys.SIDEBAR_DOC_SELECTOR not in st.session_state:
        st.session_state[WidgetKeys.SIDEBAR_DOC_SELECTOR] = st.session_state.doc_name or all_docs[0]
    elif st.session_state.get(WidgetKeys.SIDEBAR_DOC_SELECTOR) != st.session_state.doc_name:
        if st.session_state.doc_name in all_docs:
            st.session_state[WidgetKeys.SIDEBAR_DOC_SELECTOR] = st.session_state.doc_name

    st.selectbox(
        "Document",
        options=all_docs,
        key=WidgetKeys.SIDEBAR_DOC_SELECTOR,
        label_visibility="collapsed",
    )
    return True


def get_current_retrieval_config() -> dict:
    """Return a valid retrieval config from session state."""
    retrieval_config = st.session_state.get("retrieval_config")
    if isinstance(retrieval_config, dict):
        return retrieval_config
    return {"strategy": "DenseRetriever", "params": {}}


def render_retrieval_controls() -> tuple[dict, dict[str, str]]:
    """Render retrieval controls and return current config plus fusion mapping."""
    render_sidebar_section("Retrieval", "Choose how chunks are retrieved before generation.")

    current_retrieval_config = get_current_retrieval_config()
    retrieval_strategies = ["Dense", "Sparse", "Hybrid"]
    strategy_map = {
        "Dense": "DenseRetriever",
        "Sparse": "SparseRetriever",
        "Hybrid": "HybridRetriever",
    }
    reverse_strategy_map = {value: key for key, value in strategy_map.items()}
    fusion_display_map = {
        "weighted_sum": "Weighted",
        "rrf": "RRF",
    }
    reverse_fusion_map = {value: key for key, value in fusion_display_map.items()}

    current_strategy_display = reverse_strategy_map.get(
        current_retrieval_config.get("strategy", "DenseRetriever"), "Dense"
    )
    if WidgetKeys.SIDEBAR_RETRIEVAL_STRATEGY not in st.session_state:
        st.session_state[WidgetKeys.SIDEBAR_RETRIEVAL_STRATEGY] = current_strategy_display
    elif st.session_state[WidgetKeys.SIDEBAR_RETRIEVAL_STRATEGY] not in retrieval_strategies:
        st.session_state[WidgetKeys.SIDEBAR_RETRIEVAL_STRATEGY] = current_strategy_display

    retrieval_strategy = st.segmented_control(
        "Strategy",
        options=retrieval_strategies,
        key=WidgetKeys.SIDEBAR_RETRIEVAL_STRATEGY,
        help="Choose how to retrieve relevant chunks",
        label_visibility="collapsed",
    )

    if retrieval_strategy == "Hybrid":
        st.caption("Hybrid settings")
        st.slider(
            "Dense weight",
            min_value=0.0,
            max_value=1.0,
            value=current_retrieval_config.get("params", {}).get("dense_weight", 0.7),
            step=0.05,
            help="Weight for vector similarity (1-weight goes to BM25)",
            key=WidgetKeys.SIDEBAR_DENSE_WEIGHT,
        )

        current_fusion = current_retrieval_config.get("params", {}).get(
            "fusion_method", "weighted_sum"
        )
        if WidgetKeys.SIDEBAR_FUSION_METHOD not in st.session_state:
            st.session_state[WidgetKeys.SIDEBAR_FUSION_METHOD] = fusion_display_map.get(
                current_fusion, "Weighted"
            )
        elif st.session_state[WidgetKeys.SIDEBAR_FUSION_METHOD] not in [
            "Weighted",
            "RRF",
        ]:
            st.session_state[WidgetKeys.SIDEBAR_FUSION_METHOD] = fusion_display_map.get(
                current_fusion, "Weighted"
            )

        st.segmented_control(
            "Fusion method",
            options=["Weighted", "RRF"],
            key=WidgetKeys.SIDEBAR_FUSION_METHOD,
            help="RRF means Reciprocal Rank Fusion.",
        )
    elif retrieval_strategy == "Sparse":
        st.caption("Using rank-bm25 with Okapi BM25 scoring.")

    return strategy_map, reverse_fusion_map


def get_current_reranking_config() -> dict:
    """Return a valid reranking config from session state."""
    reranking_config = st.session_state.get("reranking_config")
    if isinstance(reranking_config, dict):
        return reranking_config
    return {"enabled": False}


def render_reranking_controls() -> None:
    """Render reranking controls."""
    render_sidebar_section("Reranking", "Optionally reorder retrieved chunks before answering.")
    current_reranking_config = get_current_reranking_config()

    enable_reranking = st.toggle(
        "Enable reranking",
        value=current_reranking_config.get("enabled", False),
        key=WidgetKeys.SIDEBAR_ENABLE_RERANKING,
        help="Use cross-encoder to rerank results",
    )

    if not enable_reranking:
        return

    from unravel.services.retrieval.reranking import list_available_rerankers

    all_models = list_available_rerankers()
    libraries: dict[str, list[dict]] = {}
    for model in all_models:
        libraries.setdefault(model.get("library", "Other"), []).append(model)

    library_names = list(libraries.keys())
    current_model = current_reranking_config.get("model", "ms-marco-MiniLM-L-12-v2")
    current_library = next(
        (
            library
            for library, models in libraries.items()
            if any(model["name"] == current_model for model in models)
        ),
        library_names[0],
    )

    if WidgetKeys.SIDEBAR_RERANK_LIBRARY not in st.session_state:
        st.session_state[WidgetKeys.SIDEBAR_RERANK_LIBRARY] = current_library
    elif st.session_state[WidgetKeys.SIDEBAR_RERANK_LIBRARY] not in library_names:
        st.session_state[WidgetKeys.SIDEBAR_RERANK_LIBRARY] = current_library

    selected_library = st.selectbox(
        "Model library",
        options=library_names,
        key=WidgetKeys.SIDEBAR_RERANK_LIBRARY,
    )

    available_models = [model["name"] for model in libraries[selected_library]]
    current_model_in_lib = (
        current_model if current_model in available_models else available_models[0]
    )

    st.selectbox(
        "Model",
        options=available_models,
        index=available_models.index(current_model_in_lib),
        key=WidgetKeys.SIDEBAR_RERANK_MODEL,
        help="Select reranking model",
    )

    selected_model_name = st.session_state.get(WidgetKeys.SIDEBAR_RERANK_MODEL)
    model_info = next((model for model in all_models if model["name"] == selected_model_name), None)
    if model_info:
        st.caption(model_info.get("description", ""))

    st.slider(
        "Keep top N after reranking",
        min_value=1,
        max_value=20,
        value=current_reranking_config.get("top_n", 5),
        key=WidgetKeys.SIDEBAR_RERANK_TOP_N,
    )


def render_embedding_controls() -> str:
    """Render embedding model controls and return the selected model name."""
    render_sidebar_section("Embedding model", "Select the vector model used downstream.")

    models = list_available_models()
    model_names = [model["name"] for model in models]
    display_options = []
    option_to_model = {}

    for model in models:
        display_name = model["name"].split("/")[-1]
        display_options.append(display_name)
        option_to_model[display_name] = model["name"]

    current_model_name = st.session_state.embedding_model_name
    if current_model_name not in model_names:
        current_model_name = model_names[0]

    current_selection = current_model_name.split("/")[-1]

    if WidgetKeys.SIDEBAR_EMBEDDING_MODEL not in st.session_state:
        st.session_state[WidgetKeys.SIDEBAR_EMBEDDING_MODEL] = current_selection
    elif st.session_state.get(WidgetKeys.SIDEBAR_EMBEDDING_MODEL) not in display_options:
        st.session_state[WidgetKeys.SIDEBAR_EMBEDDING_MODEL] = current_selection

    selected_option = st.selectbox(
        "Embedding model",
        options=display_options,
        key=WidgetKeys.SIDEBAR_EMBEDDING_MODEL,
        label_visibility="collapsed",
    )
    selected_model_name = option_to_model.get(selected_option, current_model_name)

    selected_model_info = next(
        (model for model in models if model["name"] == selected_model_name), None
    )
    if selected_model_info:
        provider = selected_model_info.get("library", "N/A")
        dimension = selected_model_info.get("dimension", "N/A")
        size = selected_model_info.get("size", "N/A").title()
        use_case = selected_model_info.get("use_case", "general").replace("-", " ").title()
        max_tokens = selected_model_info.get("max_seq_length", 512)
        params = selected_model_info.get("params_millions", 0)
        params_str = f"{params / 1000:.1f}B" if params >= 1000 else f"{params}M"
        st.caption(
            f"{provider} | {dimension}d | {size} | {use_case} | "
            f"{max_tokens} tokens | {params_str}"
        )

    return selected_model_name


def build_pending_rag_config(
    strategy_map: dict[str, str],
    reverse_fusion_map: dict[str, str],
) -> tuple[dict, dict]:
    """Build pending retrieval and reranking configs from widget state."""
    retrieval_strategy = st.session_state.get(
        WidgetKeys.SIDEBAR_RETRIEVAL_STRATEGY,
        "Dense",
    )
    pending_retrieval_params = {}

    if retrieval_strategy == "Hybrid":
        pending_retrieval_params = {
            "dense_weight": st.session_state.get(WidgetKeys.SIDEBAR_DENSE_WEIGHT, 0.7),
            "fusion_method": reverse_fusion_map.get(
                st.session_state.get(WidgetKeys.SIDEBAR_FUSION_METHOD, "Weighted"),
                "weighted_sum",
            ),
        }

    pending_retrieval_config = {
        "strategy": strategy_map.get(retrieval_strategy, "DenseRetriever"),
        "params": pending_retrieval_params,
    }

    enable_reranking = st.session_state.get(WidgetKeys.SIDEBAR_ENABLE_RERANKING, False)
    pending_reranking_config = {"enabled": False}

    if enable_reranking:
        pending_reranking_config = {
            "enabled": True,
            "model": st.session_state.get(
                WidgetKeys.SIDEBAR_RERANK_MODEL, "ms-marco-MiniLM-L-12-v2"
            ),
            "top_n": st.session_state.get(WidgetKeys.SIDEBAR_RERANK_TOP_N, 5),
        }

    return pending_retrieval_config, pending_reranking_config


def render_rag_actions(
    selected_model_name: str,
    pending_retrieval_config: dict,
    pending_reranking_config: dict,
) -> None:
    """Render RAG status, save, and troubleshooting actions."""
    pending_doc_name = st.session_state.get(WidgetKeys.SIDEBAR_DOC_SELECTOR)
    default_retrieval = {"strategy": "DenseRetriever", "params": {}}
    default_reranking = {"enabled": False}
    has_changes = (
        pending_doc_name != st.session_state.doc_name
        or selected_model_name != st.session_state.embedding_model_name
        or pending_retrieval_config != st.session_state.get("retrieval_config", default_retrieval)
        or pending_reranking_config != st.session_state.get("reranking_config", default_reranking)
    )

    render_sidebar_status(
        "Changes pending" if has_changes else "Configuration applied",
        "pending" if has_changes else "applied",
    )

    if st.button(
        "Save & Apply",
        type="primary",
        disabled=not has_changes,
        key=WidgetKeys.SIDEBAR_SAVE_RAG_CONFIG_BTN,
        use_container_width=True,
    ):
        doc_changed = pending_doc_name != st.session_state.doc_name
        model_changed = selected_model_name != st.session_state.embedding_model_name

        old_retrieval_config = st.session_state.get("retrieval_config", {})
        retrieval_strategy_changed = pending_retrieval_config.get(
            "strategy"
        ) != old_retrieval_config.get("strategy")

        st.session_state.doc_name = pending_doc_name
        st.session_state.embedding_model_name = selected_model_name
        st.session_state.retrieval_config = pending_retrieval_config
        st.session_state.reranking_config = pending_reranking_config

        if doc_changed:
            for key in [
                "chunks",
                "chunks_cache_key",
                "chunk_display_cache_key",
                "chunk_display_data",
                "last_embeddings_result",
                "bm25_index_data",
                "search_results",
            ]:
                if key in st.session_state:
                    del st.session_state[key]

        if model_changed:
            for key in ["last_embeddings_result", "bm25_index_data", "search_results"]:
                if key in st.session_state:
                    del st.session_state[key]

        if retrieval_strategy_changed:
            old_strategy = old_retrieval_config.get("strategy", "DenseRetriever")
            new_strategy = pending_retrieval_config.get("strategy", "DenseRetriever")

            old_uses_bm25 = old_strategy in ["SparseRetriever", "HybridRetriever"]
            new_uses_bm25 = new_strategy in ["SparseRetriever", "HybridRetriever"]

            if old_uses_bm25 != new_uses_bm25 and "bm25_index_data" in st.session_state:
                del st.session_state["bm25_index_data"]

            if "search_results" in st.session_state:
                del st.session_state["search_results"]

        current_rag_config = {
            "doc_name": st.session_state.doc_name,
            "embedding_model_name": st.session_state.embedding_model_name,
            "chunking_params": st.session_state.get("chunking_params", {}),
            "parsing_params": st.session_state.get("parsing_params", {}),
            "retrieval_config": st.session_state.get("retrieval_config", {}),
            "reranking_config": st.session_state.get("reranking_config", {}),
        }
        save_rag_config(current_rag_config)
        st.session_state["_last_saved_rag_config"] = current_rag_config.copy()

        st.success("Changes applied", icon=":material/check_circle:")
        st.rerun()

    render_sidebar_section("Troubleshooting")
    st.caption("Clear cached embeddings and session data if you encounter errors.")

    if ui.button(
        "Clear Session State", variant="outline", key=WidgetKeys.SIDEBAR_CLEAR_SESSION_BTN
    ):
        clear_session_state()

        keys_to_keep = {
            "session_restored",
            "doc_name",
            "chunking_params",
            "embedding_model_name",
            "llm_provider",
            "llm_model",
            "llm_api_key",
            "llm_base_url",
            "llm_temperature",
            "current_step",
        }
        keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_keep]
        for key in keys_to_delete:
            del st.session_state[key]

        st.success(
            "Session state cleared. Refresh the page to regenerate embeddings.",
            icon=":material/check_circle:",
        )


def render_rag_config_sidebar() -> None:
    """Render RAG configuration in the sidebar."""
    docs = initialize_rag_sidebar_state()
    if not render_document_selector(docs):
        return

    strategy_map, reverse_fusion_map = render_retrieval_controls()
    render_reranking_controls()
    selected_model_name = render_embedding_controls()
    pending_retrieval_config, pending_reranking_config = build_pending_rag_config(
        strategy_map,
        reverse_fusion_map,
    )
    render_rag_actions(
        selected_model_name,
        pending_retrieval_config,
        pending_reranking_config,
    )


def initialize_llm_sidebar_state() -> None:
    """Load and initialize sidebar state for LLM controls."""
    if "llm_config_loaded" not in st.session_state:
        saved_config = load_llm_config()
        if saved_config:
            st.session_state.llm_provider = saved_config.get("provider", "OpenAI")
            st.session_state.llm_model = saved_config.get("model", "")
            st.session_state.llm_base_url = saved_config.get("base_url", "")
            st.session_state.llm_temperature = saved_config.get("temperature", 0.7)
        st.session_state.llm_config_loaded = True

    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "OpenAI"
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = ""
    if "llm_api_key" not in st.session_state:
        st.session_state.llm_api_key = ""
    if "llm_base_url" not in st.session_state:
        st.session_state.llm_base_url = ""
    if "llm_temperature" not in st.session_state:
        st.session_state.llm_temperature = 0.7


def render_llm_sidebar() -> None:
    """Render LLM configuration in the sidebar."""
    initialize_llm_sidebar_state()

    providers = list(LLM_PROVIDERS.keys())
    current_provider = cast(str, st.session_state.llm_provider)
    if current_provider not in providers:
        current_provider = providers[0]

    if WidgetKeys.SIDEBAR_PROVIDER not in st.session_state:
        st.session_state[WidgetKeys.SIDEBAR_PROVIDER] = current_provider

    render_sidebar_section("Provider", "Choose the model endpoint used for generated answers.")
    provider = cast(
        str,
        st.selectbox(
            "Provider",
            options=providers,
            key=WidgetKeys.SIDEBAR_PROVIDER,
            label_visibility="collapsed",
        ),
    )

    previous_provider = st.session_state.get("_last_llm_provider")
    if previous_provider and previous_provider != provider:
        default_model = cast(str, LLM_PROVIDERS[provider].get("default", ""))
        st.session_state.llm_model = default_model
        st.session_state.llm_base_url = ""
        for key in (WidgetKeys.SIDEBAR_MODEL_SELECT, WidgetKeys.SIDEBAR_MODEL_INPUT):
            if key in st.session_state:
                del st.session_state[key]
        if WidgetKeys.SIDEBAR_BASE_URL in st.session_state:
            del st.session_state[WidgetKeys.SIDEBAR_BASE_URL]
    st.session_state._last_llm_provider = provider
    st.session_state.llm_provider = provider

    if provider == "OpenAI-Compatible":
        st.info(
            "**OpenAI-Compatible** allows local models such as Ollama and LM Studio.\n\n"
            "**Common Base URLs:**\n"
            "- Ollama: `http://localhost:11434/v1`\n"
            "- LM Studio: `http://localhost:1234/v1`",
            icon=":material/info:",
        )

    if provider == "OpenRouter":
        st.info(
            "**OpenRouter** provides access to many models via a single API.\n\n"
            "Enter any model identifier from [openrouter.ai/models](https://openrouter.ai/models).",
            icon=":material/info:",
        )

    if provider == "Vertex AI":
        st.info(
            "**Vertex AI** uses Google Cloud authentication.\n\n"
            "**Setup Steps:**\n"
            "1. Authenticate: `gcloud auth application-default login`\n"
            "2. Set `VERTEXAI_PROJECT=your-project-id` in `~/.unravel/.env`\n"
            "3. Optional: set `VERTEXAI_LOCATION=us-central1`",
            icon=":material/info:",
        )

    models = get_provider_models(provider)
    if provider == "OpenAI-Compatible":
        model = st.text_input(
            "Model name",
            value=st.session_state.llm_model or "llama2",
            placeholder="e.g., llama2, mistral",
            key=WidgetKeys.SIDEBAR_MODEL_INPUT,
        )
        st.session_state.llm_model = model
    elif provider == "OpenRouter":
        model = st.text_input(
            "Model name",
            value=st.session_state.llm_model or "anthropic/claude-opus-4-6",
            placeholder="e.g., anthropic/claude-opus-4-6",
            key=WidgetKeys.SIDEBAR_MODEL_INPUT,
        )
        st.session_state.llm_model = model
    else:
        default_model = cast(str, LLM_PROVIDERS[provider]["default"])
        current_model = st.session_state.llm_model or default_model
        if current_model not in models:
            current_model = default_model
            st.session_state.llm_model = default_model

        if WidgetKeys.SIDEBAR_MODEL_SELECT not in st.session_state:
            st.session_state[WidgetKeys.SIDEBAR_MODEL_SELECT] = current_model

        model = cast(
            str,
            st.selectbox(
                "Model",
                options=models,
                key=WidgetKeys.SIDEBAR_MODEL_SELECT,
                label_visibility="collapsed",
            ),
        )
        st.session_state.llm_model = model

    api_key_from_env = get_api_key_from_env(provider)
    if api_key_from_env:
        if provider == "Vertex AI":
            st.caption(f"Project ID loaded: {api_key_from_env[:20]}...")
        else:
            st.caption("API key loaded from environment.")
        api_key = api_key_from_env
    else:
        if provider == "OpenAI-Compatible":
            api_key = "not-needed"
        elif provider == "Vertex AI":
            st.caption("Set `VERTEXAI_PROJECT` in `~/.unravel/.env`.")
            st.caption("Run: `gcloud auth application-default login`.")
            api_key = ""
        else:
            st.caption(f"Set `{LLM_PROVIDERS[provider]['env_key']}` in `~/.unravel/.env`.")
            api_key = ""

    st.session_state.llm_api_key = ""

    if provider == "OpenAI-Compatible":
        default_base_url = "http://localhost:11434/v1"
        base_url = st.text_input(
            "Base URL",
            value=st.session_state.llm_base_url or default_base_url,
            placeholder=default_base_url,
            key=WidgetKeys.SIDEBAR_BASE_URL,
        )
        st.session_state.llm_base_url = base_url
    else:
        base_url = None
        st.session_state.llm_base_url = ""

    with st.expander("Advanced settings", expanded=False):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.llm_temperature,
            step=0.1,
            key=WidgetKeys.SIDEBAR_TEMPERATURE,
            help="Higher values make output more random, lower values more deterministic.",
        )
        st.session_state.llm_temperature = temperature

    if st.button(
        "Save Configuration",
        key=WidgetKeys.SIDEBAR_SAVE_CONFIG_BTN,
        use_container_width=True,
    ):
        config_data = {
            "provider": provider,
            "model": model,
            "base_url": base_url if provider == "OpenAI-Compatible" else None,
            "temperature": temperature,
        }
        save_llm_config(config_data)
        st.success("Configuration saved", icon=":material/check_circle:")

    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )
    is_valid, error_msg = validate_config(config)
    if not is_valid:
        st.warning(error_msg, icon=":material/warning:")


def render_sidebar() -> None:
    """Render the main sidebar with tabs."""
    with st.sidebar:
        st.markdown('<div class="sidebar-app-title">Configuration</div>', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["RAG Config", "LLM Config"])

        with tab1:
            render_rag_config_sidebar()

        with tab2:
            render_llm_sidebar()
