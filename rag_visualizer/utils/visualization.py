from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import umap


@st.cache_resource
def _get_umap_reducer(n_components: int = 2, random_state: int = 42) -> umap.UMAP:
    """Get or create a cached UMAP reducer."""
    return umap.UMAP(n_components=n_components, random_state=random_state)


@st.cache_data
def reduce_dimensions(
    embeddings: np.ndarray, n_components: int = 2, random_state: int = 42
) -> tuple[np.ndarray, Any]:
    """Reduce dimensionality of embeddings using UMAP.

    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        n_components: The dimension of the space to embed into
        random_state: Random state for reproducibility

    Returns:
        Tuple containing:
        - numpy array of shape (n_samples, n_components)
        - Fitted UMAP reducer object (or None if fallback used)
    """
    # Handle empty or small inputs
    if embeddings.shape[0] == 0:
        return np.zeros((0, n_components)), None

    if embeddings.shape[0] < 5:
        # Fallback for very few points where UMAP might fail or be unstable
        # Just return the first n_components columns (PCA-like) or pad with zeros
        result = np.zeros((embeddings.shape[0], n_components))
        cols = min(embeddings.shape[1], n_components)
        result[:, :cols] = embeddings[:, :cols]
        return result, None

    reducer = _get_umap_reducer(n_components, random_state)
    embedding_2d = reducer.fit_transform(embeddings)
    return embedding_2d, reducer

def create_embedding_plot(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    color_col: str | None = None,  # noqa: ARG001
    hover_data: list[str] | None = None,
    title: str = "Embedding Visualization",
    query_point: dict[str, Any] | None = None,
    neighbors_indices: list[int] | None = None
) -> go.Figure:
    """Create a 2D scatter plot of embeddings.
    
    Args:
        df: DataFrame containing the data
        x_col: Name of column for x-axis
        y_col: Name of column for y-axis
        color_col: Name of column to color by (unused, reserved for future use)
        hover_data: List of columns to show on hover
        title: Plot title
        query_point: Optional dict with 'x', 'y' coordinates for query point
        neighbors_indices: Optional list of indices in df that are neighbors
        
    Returns:
        Plotly Figure object
    """
    # Create base scatter plot for chunks
    fig = go.Figure()

    # Add chunks trace
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='markers',
            marker=dict(
                size=12,
                opacity=0.7,
                color='#7dd3fc', # Light blue/teal from screenshot
                line=dict(width=1, color='#0ea5e9')
            ),
            text=df['text_preview'] if 'text_preview' in df.columns else None,
            customdata=df[hover_data].values if hover_data else None,
            hovertemplate="<b>Chunk</b><br>%{text}<extra></extra>",
            name='Chunks'
        )
    )

    # Add query point if present
    if query_point:
        fig.add_trace(
            go.Scatter(
                x=[query_point['x']],
                y=[query_point['y']],
                mode='markers',
                marker=dict(
                    size=16,
                    opacity=1,
                    color='#f43f5e', # Pink/Red from screenshot
                    symbol='circle',
                    line=dict(width=2, color='white')
                ),
                name='Question',
                hovertemplate="<b>Question</b><br>Your query<extra></extra>"
            )
        )

        # Add connecting lines to neighbors
        if neighbors_indices:
            for idx in neighbors_indices:
                neighbor_row = df.iloc[idx]
                fig.add_trace(
                    go.Scatter(
                        x=[query_point['x'], neighbor_row[x_col]],
                        y=[query_point['y'], neighbor_row[y_col]],
                        mode='lines',
                        line=dict(
                            color='#94a3b8',
                            width=1,
                            dash='dash'
                        ),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=14)
        ),
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='white',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="sans-serif"
        )
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9', zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9', zeroline=False)
    
    return fig
