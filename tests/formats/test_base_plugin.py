"""Tests for the shared BaseFormatPlugin behavior, using a stub backend so no
real renderer is involved."""

import io

import pytest
from PIL import Image

from tests.helpers import make_png
from vectrify.formats.base import BaseFormatPlugin


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


def test_a_block_that_differs_only_in_whitespace_still_applies():
    """Measured on one run, 3 of 5 failed seed edits were blocks whose SEARCH
    text differed from the parent only in whitespace -- the edit was right and
    the transcription was not, and the whole paid call was discarded."""
    from vectrify.formats.base import apply_search_replace

    parent = '<svg>\n  <circle cx="1" cy="2" r="3" fill="#000" />\n</svg>'
    raw = (
        "<<<SEARCH>>>\n"
        '<circle cx="1"   cy="2"\n     r="3" fill="#000" />\n'
        "<<<REPLACE>>>\n"
        '<circle cx="1" cy="2" r="9" fill="#000" />\n'
        "<<<END>>>"
    )
    patched = apply_search_replace(parent, raw)
    assert patched is not None
    assert 'r="9"' in patched


def test_loose_matching_does_not_negotiate_anything_but_whitespace():
    """A block naming an element that is not there must still fail, or a paid
    call silently patches the wrong part of the drawing."""
    from vectrify.formats.base import NoEditAppliedError, apply_search_replace

    parent = '<svg>\n  <circle cx="1" cy="2" r="3" />\n</svg>'
    raw = (
        "<<<SEARCH>>>\n"
        '<circle cx="1" cy="2" r="4" />\n'
        "<<<REPLACE>>>\n"
        '<circle cx="1" cy="2" r="9" />\n'
        "<<<END>>>"
    )
    with pytest.raises(NoEditAppliedError):
        apply_search_replace(parent, raw)


def test_a_block_indented_differently_from_the_parent_applies():
    """A block copied out of the markup carries the surrounding indentation."""
    from vectrify.formats.base import apply_search_replace

    parent = '<svg>\n    <rect width="4" height="5" />\n</svg>'
    raw = (
        "<<<SEARCH>>>\n"
        '  <rect width="4" height="5" />  \n'
        "<<<REPLACE>>>\n"
        '<rect width="8" height="5" />\n'
        "<<<END>>>"
    )
    patched = apply_search_replace(parent, raw)
    assert patched is not None
    assert 'width="8"' in patched


_CIRCLE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">\n'
    '  <circle cx="1" cy="2" r="3" />\n</svg>'
)


def test_a_reply_offering_several_attempts_becomes_several_candidates():
    """An epoch's batch size and its LLM spend are the same number today: one
    call, one candidate. Most of a call is the prompt, so a second attempt in
    the same reply is nearly free."""
    from vectrify.formats.svg.plugin import SvgPlugin

    parent = _CIRCLE_SVG
    raw = (
        '<<<SEARCH>>>\n<circle cx="1" cy="2" r="3" />\n<<<REPLACE>>>\n'
        '<circle cx="1" cy="2" r="4" />\n<<<END>>>\n'
        "===ALTERNATIVE===\n"
        '<<<SEARCH>>>\n<circle cx="1" cy="2" r="3" />\n<<<REPLACE>>>\n'
        '<circle cx="1" cy="2" r="7" />\n<<<END>>>'
    )
    out = SvgPlugin().apply_edits(parent, raw)
    assert len(out) == 2
    assert 'r="4"' in out[0]
    assert 'r="7"' in out[1]


def test_one_unusable_attempt_does_not_discard_the_others():
    from vectrify.formats.svg.plugin import SvgPlugin

    parent = _CIRCLE_SVG
    raw = (
        '<<<SEARCH>>>\n<rect width="9" />\n<<<REPLACE>>>\n'
        '<rect width="8" />\n<<<END>>>\n'
        "===ALTERNATIVE===\n"
        '<<<SEARCH>>>\n<circle cx="1" cy="2" r="3" />\n<<<REPLACE>>>\n'
        '<circle cx="1" cy="2" r="5" />\n<<<END>>>'
    )
    out = SvgPlugin().apply_edits(parent, raw)
    assert len(out) == 1
    assert 'r="5"' in out[0]


def test_a_reply_where_nothing_applies_still_raises():
    """So the caller reports it and the seed retry asks for a replacement."""
    from vectrify.formats.base import NoEditAppliedError
    from vectrify.formats.svg.plugin import SvgPlugin

    parent = _CIRCLE_SVG
    raw = (
        '<<<SEARCH>>>\n<rect width="9" />\n<<<REPLACE>>>\n'
        '<rect width="8" />\n<<<END>>>\n'
        "===ALTERNATIVE===\n"
        '<<<SEARCH>>>\n<rect width="7" />\n<<<REPLACE>>>\n<rect width="6" />\n<<<END>>>'
    )
    with pytest.raises(NoEditAppliedError):
        SvgPlugin().apply_edits(parent, raw)


def test_an_ordinary_reply_is_one_candidate():
    from vectrify.formats.svg.plugin import SvgPlugin

    parent = _CIRCLE_SVG
    raw = (
        '<<<SEARCH>>>\n<circle cx="1" cy="2" r="3" />\n<<<REPLACE>>>\n'
        '<circle cx="1" cy="2" r="4" />\n<<<END>>>'
    )
    assert len(SvgPlugin().apply_edits(parent, raw)) == 1
