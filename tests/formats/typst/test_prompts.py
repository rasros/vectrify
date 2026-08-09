"""Typst-specific prompt content; common builder behavior is covered by
tests/formats/test_code_prompts.py."""

from tests.helpers import text_blocks
from vectrify.formats.typst.prompts import build_typst_gen_prompt

_IMG_URL = "data:image/png;base64,abc"


def _first_iteration_text(canvas: tuple[int, int] = (768, 768)) -> str:
    blocks = build_typst_gen_prompt(
        _IMG_URL,
        node_index=1,
        typst_prev=None,
        rasterized_data_url=None,
        goal=None,
        canvas=canvas,
    )
    return "\n".join(text_blocks(blocks))


def test_gen_prompt_system_text_mentions_typst_rules():
    text = _first_iteration_text()
    assert "NEVER use multiple pages" in text


def test_gen_prompt_pins_the_page_to_the_canvas():
    """An auto-sized page leaves the coordinate space implicit, so the same
    #place means different positions in different candidates and crossover
    silently rescales what it grafts."""
    text = _first_iteration_text((768, 512))
    assert "#set page(width: 768pt, height: 512pt, margin: 0pt)" in text
    assert "width: auto" not in text


def test_gen_prompt_first_iteration_asks_for_code_only():
    assert "output only the typst code block" in _first_iteration_text().lower()
