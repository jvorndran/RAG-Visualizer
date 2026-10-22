from streamlit.testing.v1 import AppTest

SCRIPT = '''
from unravel.services.chunking import Chunk
from unravel.ui.components.chunk_diagnostics import render_chunk_diagnostics

chunks = [Chunk(text, i, i + len(text), {"index": i})
          for i, text in enumerate(["short", "longer chunk", "short", ""])]
render_chunk_diagnostics(chunks)
'''


def test_diagnostics_inspects_filtered_chunks_by_original_index():
    at = AppTest.from_string(SCRIPT).run(timeout=30)
    assert not at.exception
    assert [metric.value for metric in at.metric] == ["4", "5", "12", "1"]
    assert "1 empty" in at.caption[0].value

    at.selectbox(key="chunk_diagnostics_size").select("At most").run()
    at.number_input(key="chunk_diagnostics_threshold").set_value(5).run()
    assert at.selectbox(key="chunk_diagnostics_chunk").options == [
        "Chunk 1 · 5 characters", "Chunk 3 · 5 characters", "Chunk 4 · 0 characters",
    ]
    at.selectbox(key="chunk_diagnostics_chunk").select(2).run()
    assert at.code[0].value == "short"
    assert '"index":2' in at.json[0].value.replace(" ", "")

    at.selectbox(key="chunk_diagnostics_size").select("At least").run()
    at.number_input(key="chunk_diagnostics_threshold").set_value(13).run()
    assert not at.exception
    assert "No chunks match" in at.info[0].value
    assert len(at.code) == 0


def test_empty_diagnostics_and_changed_chunk_sets_are_safe():
    script = SCRIPT.replace('render_chunk_diagnostics(chunks)', '''
import streamlit as st
if st.button("Replace"):
    st.session_state["replaced"] = True
if st.session_state.get("replaced"):
    chunks = [Chunk("replacement", 0, 11, {})]
render_chunk_diagnostics(chunks)
''')
    at = AppTest.from_string(script).run(timeout=30)
    at.selectbox(key="chunk_diagnostics_chunk").select(3).run()
    at.button[0].click().run()
    assert not at.exception
    assert at.code[0].value == "replacement"

    empty = AppTest.from_string(SCRIPT.replace("render_chunk_diagnostics(chunks)",
                                           "render_chunk_diagnostics([])")).run(timeout=30)
    assert not empty.exception
    assert empty.info[0].value == "No chunks to inspect."
