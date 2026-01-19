"""Query tester page for RAG Visualizer.

Allows users to test the full RAG pipeline with retrieval and LLM response generation.
"""

import pandas as pd
import streamlit as st
import streamlit_shadcn_ui as ui

from rag_visualizer.services.embedders import DEFAULT_MODEL, get_embedder
from rag_visualizer.services.llm import (
    DEFAULT_SYSTEM_PROMPT,
    LLMConfig,
    RAGContext,
    get_model,
)
from rag_visualizer.ui.components.chunk_viewer import (
    prepare_chunk_display_data,
    render_chunk_cards,
)
from rag_visualizer.utils.visualization import create_embedding_plot


def _get_embeddings_data() -> dict | None:
    """Retrieve embeddings data from session state."""
    if "last_embeddings_result" not in st.session_state:
        return None
    return st.session_state["last_embeddings_result"]


def _render_empty_state() -> None:
    """Render the empty state when no embeddings are available."""
    with st.container(border=True):
        st.markdown("### No embeddings found")
        st.caption(
            "Please go to the Embeddings step to generate vector representations of your document chunks first."
        )
        
        st.write("")
        if ui.button("Go to Embeddings Step", key="goto_embeddings"):
            st.session_state.current_step = "embeddings"
            st.rerun()


def _render_retrieved_chunks(
    results: list, show_scores: bool = True
) -> None:
    """Render retrieved chunks using the chunk viewer component."""
    if not results:
        return
    
    st.markdown("#### Retrieved Context")
    
    # Convert search results to chunks for the viewer
    from dataclasses import dataclass
    
    @dataclass
    class ChunkAdapter:
        """Adapter to make SearchResult compatible with chunk viewer."""
        text: str
        metadata: dict
        start_index: int = 0
        end_index: int = 0
    
    retrieved_chunks = [
        ChunkAdapter(
            text=res.text,
            metadata=res.metadata,
            start_index=i,
            end_index=i,
        )
        for i, res in enumerate(results)
    ]
    
    # Prepare display data
    retrieved_display_data = prepare_chunk_display_data(
        chunks=retrieved_chunks,
        source_text=None,
        calculate_overlap=False,
    )
    
    # Add similarity score as custom badge if requested
    custom_badges = None
    if show_scores:
        custom_badges = [
            {
                "label": "Score",
                "value": f"{res.score:.3f}",
                "color": "#d1fae5"  # Green tint for similarity
            }
            for res in results
        ]
    
    # Render using the reusable component in card mode
    render_chunk_cards(
        chunk_display_data=retrieved_display_data,
        custom_badges=custom_badges,
        show_overlap=False,
        display_mode="card",
    )


def _get_llm_config_from_sidebar() -> tuple[LLMConfig, str]:
    """Get LLM configuration from sidebar session state."""
    provider = st.session_state.get("llm_provider", "OpenAI")
    model = st.session_state.get("llm_model", "")
    api_key = st.session_state.get("llm_api_key", "")
    base_url = st.session_state.get("llm_base_url", "")
    temperature = st.session_state.get("llm_temperature", 0.7)
    max_tokens = st.session_state.get("llm_max_tokens", 1024)
    system_prompt = st.session_state.get("llm_system_prompt", DEFAULT_SYSTEM_PROMPT)
    
    # Check for env var API key
    from rag_visualizer.services.llm import get_api_key_from_env
    env_key = get_api_key_from_env(provider)
    if env_key:
        api_key = env_key
    
    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url if base_url else None,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    return config, system_prompt


def render_query_step() -> None:
    """Render the query tester step with unified RAG pipeline."""
    
    # --- Check for embeddings data ---
    embeddings_data = _get_embeddings_data()
    
    if not embeddings_data:
        _render_empty_state()
        return
    
    # Extract data from state
    vector_store = embeddings_data.get("vector_store")
    reduced_embeddings = embeddings_data.get("reduced_embeddings")
    reducer = embeddings_data.get("reducer")
    chunks = embeddings_data.get("chunks", [])
    model_name = embeddings_data.get("model", DEFAULT_MODEL)

    # Always recreate embedder on-demand (don't rely on stored object)
    embedder = get_embedder(model_name)
    
    if not vector_store or vector_store.size == 0:
        _render_empty_state()
        return
    
    # --- Initialize session state ---
    if "current_query" not in st.session_state:
        st.session_state.current_query = None
    if "current_response" not in st.session_state:
        st.session_state.current_response = None
    if "last_search_results" not in st.session_state:
        st.session_state.last_search_results = None
    
    # === Header & Metrics ===
    col_header, col_stats = st.columns([2, 1])
    with col_header:
        st.markdown("### Response Generation")
        st.caption("Retrieve context and generate answers.")
    
    with col_stats:
        # Minimal stats display
        st.markdown(
            f"""
            <div style="display: flex; gap: 1rem; justify-content: flex-end; align-items: center; height: 100%;">
                <div style="text-align: right;">
                    <span style="color: #64748b; font-size: 0.8rem;">Indexed Chunks</span><br>
                    <span style="font-weight: 600;">{vector_store.size}</span>
                </div>
                <div style="border-left: 1px solid #e2e8f0; height: 24px;"></div>
                <div style="text-align: right;">
                    <span style="color: #64748b; font-size: 0.8rem;">Model</span><br>
                    <span style="font-weight: 600;">{model_name}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")
    
    # === Query Input Section ===
    with st.container():
        # Using columns to create a "search bar + button" feel
        col_input, col_btn = st.columns([5, 1])
        
        with col_input:
            query_text = ui.input(
                placeholder="Ask a question about your documents...",
                key="query_input"
            )
            
        with col_btn:
            # Align button with input (approximate vertical alignment)
            st.markdown('<div style="margin-top: 2px;"></div>', unsafe_allow_html=True)
            # Standard streamlit button with type="primary" which we styled to be black in global css
            ask_clicked = st.button("Ask", type="primary", key="ask_button", use_container_width=True)

        # Configuration in an expander to keep UI clean
        with st.expander("Retrieval Settings", expanded=False):
            col_k, col_threshold = st.columns(2)
            with col_k:
                top_k = st.slider(
                    "Top K Results",
                    min_value=1,
                    max_value=min(20, vector_store.size),
                    value=min(5, vector_store.size),
                    key="top_k_slider"
                )
            
            with col_threshold:
                threshold = st.slider(
                    "Minimum Similarity Score",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.05,
                    key="threshold_slider"
                )

    
    # === Process Query: Retrieve + Generate ===
    if ask_clicked and query_text.strip():
        # Clear previous results
        st.session_state.current_query = query_text.strip()
        st.session_state.current_response = None
        
        # Get LLM config from sidebar
        llm_config, system_prompt = _get_llm_config_from_sidebar()
        
        # Validate LLM config
        from rag_visualizer.services.llm import validate_config
        is_valid, error_msg = validate_config(llm_config)
        
        if not is_valid:
            st.error(f"Configuration Error: {error_msg}")
            st.info("Please configure your LLM settings in the sidebar.")
            return
        
        # Step 1: Retrieve chunks
        with st.spinner("Retrieving context..."):
            # Embedder is already created above - no fallback needed
            query_embedding = embedder.embed_query(query_text.strip())
            all_results = vector_store.search(query_embedding, k=top_k)
            
            # Apply threshold filter
            search_results = [r for r in all_results if r.score >= threshold]
            
            # Store for later display
            st.session_state.last_search_results = search_results
        
        # Display retrieved chunks immediately
        if search_results:
            _render_retrieved_chunks(search_results)
            st.write("")  # Spacing
        else:
            st.warning(
                "No chunks found matching the similarity threshold. "
                "Try lowering the Minimum Score or rephrasing your query."
            )
            return
        
        # Step 2: Generate response with retrieved chunks
        st.markdown("#### Model Response")
        
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # Build RAG context
            context = RAGContext(
                query=query_text.strip(),
                chunks=[r.text for r in search_results],
                scores=[r.score for r in search_results],
            )
            
            # Get model instance
            model = get_model(
                provider=llm_config.provider,
                model=llm_config.model,
                api_key=llm_config.api_key,
                base_url=llm_config.base_url,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
            )
            
            # Stream response
            with st.spinner("Generating response..."):
                for chunk in model.stream(context, system_prompt):
                    full_response += chunk
            
            # Display Final Response in Card
            with response_placeholder.container():
                with st.container(border=True):
                    st.markdown(full_response)
            
            st.session_state.current_response = full_response
            
        except ImportError as e:
            st.error(f"Missing Dependency: {str(e)}")
            st.info("Install the required library with: `pip install rag-visualizer[llm]`")
        except Exception as e:
            st.error(f"Generation Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # === Display Previous Results ===
    elif st.session_state.current_query:
        # Show previous chunks if available
        if "last_search_results" in st.session_state:
            _render_retrieved_chunks(st.session_state.last_search_results)
            st.write("")
        
        # Show previous response if available
        if st.session_state.current_response:
            st.markdown("#### Model Response")
            with st.container(border=True):
                st.markdown(st.session_state.current_response)
    
    # === Visualization Section ===
    if st.session_state.current_query and st.session_state.last_search_results:
        st.write("")
        st.divider()
        with st.expander("Embedding Visualization", expanded=False):
            if reduced_embeddings is not None and len(reduced_embeddings) > 0:
                df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
                df["text_preview"] = [
                    c.text[:150] + "..." if len(c.text) > 150 else c.text 
                    for c in chunks
                ]
                df["chunk_index"] = range(len(chunks))
                
                # Project query to 2D if reducer available
                query_point_2d = None
                if reducer is not None and embedder:
                    try:
                        query_embedding = embedder.embed_query(st.session_state.current_query)
                        query_2d = reducer.transform(query_embedding.reshape(1, -1))
                        query_point_2d = {"x": float(query_2d[0][0]), "y": float(query_2d[0][1])}
                    except Exception:
                        pass
                
                neighbor_indices = [r.index for r in st.session_state.last_search_results]
                
                fig = create_embedding_plot(
                    df,
                    x_col="x",
                    y_col="y",
                    hover_data=["text_preview"],
                    title="",
                    query_point=query_point_2d,
                    neighbors_indices=neighbor_indices
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption(
                    "Legend: Blue dots = document chunks, Red dot = your query, "
                    "Dashed lines = retrieved chunks"
                )
            else:
                st.warning("Visualization data not available.")
