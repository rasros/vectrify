"""DOT-specific prompt content; common builder behavior is covered by
tests/formats/test_code_prompts.py."""

from vectrify.formats.graphviz.prompts import build_dot_gen_prompt
from vectrify.tests.helpers import text_blocks

_IMG_URL = "data:image/png;base64,abc"


def _first_iteration_text() -> str:
    blocks = build_dot_gen_prompt(
        _IMG_URL,
        node_index=1,
        dot_prev=None,
        rasterized_dot_data_url=None,
        goal=None,
        diff_data_url=None,
    )
    return "\n".join(text_blocks(blocks))


def test_gen_prompt_system_text_mentions_digraph():
    assert "digraph" in _first_iteration_text().lower()


def test_gen_prompt_warns_about_arrow_graph_mismatch():
    assert "-> edges require digraph" in _first_iteration_text()
