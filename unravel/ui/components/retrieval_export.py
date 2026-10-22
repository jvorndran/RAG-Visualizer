"""Download retrieval evidence without issuing another provider request."""

import streamlit as st

from unravel.services.retrieval.comparison import RetrievalSnapshot
from unravel.services.retrieval.export import serialize_retrieval_experiment
from unravel.ui.components.retrieval_comparison import BASELINE_KEY


def render_retrieval_export(current: RetrievalSnapshot) -> None:
    """Use the executed run, including an empty result and any pinned baseline."""
    if not current.query and not current.chunks:
        return
    st.download_button(
        label="Download retrieval experiment",
        data=serialize_retrieval_experiment(current, st.session_state.get(BASELINE_KEY)),
        file_name="retrieval-experiment.json",
        mime="application/json",
        key="retrieval_experiment_download",
        help="Includes full retrieved text, executed settings and any pinned baseline. "
        "Non-finite measurements are exported as null.",
    )
