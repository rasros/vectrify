"""Typst-specific prompt content; common builder behavior is covered by
tests/formats/test_code_prompts.py."""

from vectrify.formats.typst.prompts import build_typst_gen_prompt
from vectrify.tests.helpers import text_blocks

_IMG_URL = "data:image/png;base64,abc"


def _first_iteration_text() -> str:
    blocks = build_typst_gen_prompt(
        _IMG_URL,
        node_index=1,
        typst_prev=None,
        rasterized_data_url=None,
        goal=None,
        diff_data_url=None,
    )
    return "\n".join(text_blocks(blocks))


def test_gen_prompt_system_text_mentions_typst_rules():
    text = _first_iteration_text()
    assert "#set page(width: auto, height: auto, margin: 0pt)" in text
    assert "NEVER use multiple pages" in text


def test_gen_prompt_first_iteration_asks_for_code_only():
    assert "output only the typst code block" in _first_iteration_text().lower()
