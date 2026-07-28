"""Tests for the shared BaseFormatPlugin behavior, using a stub backend so no
real renderer is involved."""

import io

import pytest
from PIL import Image

from vectrify.formats.base import BaseFormatPlugin
from vectrify.tests.helpers import make_png


class _StubPlugin(BaseFormatPlugin):
    name = "stub"
    file_extension = ".stub"

    def _render_png(self, content: str) -> bytes:
        if content == "boom":
            raise RuntimeError("render failed")
        # Renders at its own natural size, like graphviz/typst do.
        return make_png("red", (40, 20))

    def _compile(self, content: str) -> None:
        if content == "bad":
            raise ValueError("syntax error")

    def extract_from_llm(self, raw: str) -> str:
        return raw.strip()


def test_rasterize_resizes_to_requested_dimensions():
    png = _StubPlugin().rasterize("x", out_w=64, out_h=48)
    assert Image.open(io.BytesIO(png)).size == (64, 48)


def test_render_png_or_none_swallows_render_errors():
    plugin = _StubPlugin()
    assert plugin.render_png_or_none("ok") is not None
    assert plugin.render_png_or_none("boom") is None


def test_validate_reports_success_and_failure():
    plugin = _StubPlugin()
    assert plugin.validate("ok") == (True, None)
    valid, err = plugin.validate("bad")
    assert valid is False
    assert err is not None
    assert "syntax error" in err


def test_apply_edit_applies_search_replace_blocks():
    result = _StubPlugin().apply_edit(
        "keep red here", "<<<SEARCH>>>\nred\n<<<REPLACE>>>\nblue\n<<<END>>>"
    )
    assert result == "keep blue here"


def test_apply_edit_falls_back_to_extraction_without_blocks():
    result = _StubPlugin().apply_edit("parent", "  a full replacement  ")
    assert result == "a full replacement"


def test_apply_search_replace_raises_when_no_block_matches():
    """Regression: str.replace is silent on a miss, so a hallucinated SEARCH
    block returned the parent unchanged and was reported as a successful edit.
    That byte-identical child then entered the pool carrying its parent's
    signature, dragging measured diversity toward zero.
    """
    from vectrify.formats.base import NoEditAppliedError, apply_search_replace

    parent = '<svg><rect fill="red"/></svg>'
    raw = (
        "<<<SEARCH>>>\n"
        '<rect fill="crimson"/>\n'
        "<<<REPLACE>>>\n"
        '<rect fill="blue"/>\n'
        "<<<END>>>"
    )
    with pytest.raises(NoEditAppliedError):
        apply_search_replace(parent, raw)


def test_apply_search_replace_returns_none_when_there_are_no_blocks():
    """Distinct from a failed match: None is the signal to fall back to parsing
    a whole file out of the response, and must keep working.
    """
    from vectrify.formats.base import apply_search_replace

    assert apply_search_replace("<svg/>", "here is some prose") is None


def test_apply_search_replace_applies_a_matching_block():
    from vectrify.formats.base import apply_search_replace

    parent = '<svg><rect fill="red"/></svg>'
    raw = (
        "<<<SEARCH>>>\n"
        '<rect fill="red"/>\n'
        "<<<REPLACE>>>\n"
        '<rect fill="blue"/>\n'
        "<<<END>>>"
    )
    assert apply_search_replace(parent, raw) == '<svg><rect fill="blue"/></svg>'


def test_apply_search_replace_allows_partial_application(caplog):
    """One block matching and another not is not a duplicate, so it is applied
    and warned about rather than rejected.
    """
    import logging

    from vectrify.formats.base import apply_search_replace

    parent = '<svg><rect fill="red"/></svg>'
    raw = (
        "<<<SEARCH>>>\n"
        '<rect fill="red"/>\n'
        "<<<REPLACE>>>\n"
        '<rect fill="blue"/>\n'
        "<<<END>>>\n"
        "<<<SEARCH>>>\n"
        '<circle r="5"/>\n'
        "<<<REPLACE>>>\n"
        '<circle r="9"/>\n'
        "<<<END>>>"
    )
    with caplog.at_level(logging.WARNING):
        out = apply_search_replace(parent, raw)
    assert out is not None
    assert 'fill="blue"' in out
    assert "1/2" in caplog.text
