"""Tests for the shared BaseFormatPlugin behavior, using a stub backend so no
real renderer is involved."""

import io

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
    assert "syntax error" in err


def test_apply_edit_applies_search_replace_blocks():
    result = _StubPlugin().apply_edit(
        "keep red here", "<<<SEARCH>>>\nred\n<<<REPLACE>>>\nblue\n<<<END>>>"
    )
    assert result == "keep blue here"


def test_apply_edit_falls_back_to_extraction_without_blocks():
    result = _StubPlugin().apply_edit("parent", "  a full replacement  ")
    assert result == "a full replacement"
