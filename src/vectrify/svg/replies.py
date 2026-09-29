"""Turn an LLM reply into SVG: a whole drawing, or edits to an existing one.

A reply either carries a complete ``<svg>`` fragment or search/replace blocks
against the parent, possibly several alternatives separated by a marker. Every
result is normalized into the form local search can edit.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

_SEARCH_REPLACE_RE = re.compile(
    r"<<<SEARCH>>>\n(.*?)\n<<<REPLACE>>>\n(.*?)\n<<<END>>>",
    re.DOTALL,
)


def _loose_match(haystack: str, needle: str) -> tuple[int, int] | None:
    """Where *needle* sits in *haystack*, ignoring how whitespace is written.

    Returns the half-open span, or None. Only whitespace is negotiable: every
    other character still has to match, so this cannot silently patch the wrong
    element. Leading and trailing whitespace on the needle is dropped, since a
    block copied out of the markup usually carries the surrounding indentation.
    """
    stripped = needle.strip()
    if not stripped:
        return None
    pattern = r"\s+".join(re.escape(part) for part in stripped.split())
    found = re.search(pattern, haystack)
    return found.span() if found else None


class NoEditAppliedError(ValueError):
    """Raised when diff blocks were present but none matched the parent."""


class NoUsableOutputError(ValueError):
    """Raised when a reply held neither diff blocks nor a code fragment.

    Distinguished from a malformed fragment because the two call for different
    responses and used to look identical. The extractors fall back to returning
    the whole reply when they find no fragment in it, so prose went to the
    parser and came back as "XML parse error: not well-formed (invalid token):
    line 1, column 1" -- which reads as a broken drawing when nothing was drawn
    at all. One run lost 4 of 50 calls this way while the log blamed the SVG.

    Falling back to the parent is not the remedy: that returns a byte-identical
    child, which is the waste `apply_search_replace` exists to prevent.
    """


def describe_unusable(raw: str, limit: int = 120) -> str:
    """A one-line preview of a reply, for saying what came back instead."""
    flat = " ".join(raw.split())
    if not flat:
        return "the reply was empty"
    shown = flat[:limit] + ("..." if len(flat) > limit else "")
    return f"the reply began {shown!r}"


# What separates one attempt from the next in a reply. An epoch's batch size and
# its LLM spend are the same number today: one call, one candidate. Most of a
# call is the prompt -- the image, the parent's markup, the instructions -- so a
# second attempt in the same reply is nearly free, and an epoch that opens on
# more candidates is the whole point (one measured run opened on 5, then 3, then
# 2, and the last went stale in 14 seconds).
ALTERNATIVE_MARKER = "===ALTERNATIVE==="

# Attempts taken from one reply. Each costs a rasterize and a scoring pass, so
# this is not free even though the call is already paid for; three is enough to
# widen a batch of five to a batch of fifteen.
MAX_ALTERNATIVES = 3


def split_alternatives(raw: str) -> list[str]:
    """A reply cut into the separate attempts it offers.

    Always at least one section, so a reply with no marker is what it always
    was. Empty sections are dropped: a trailing marker is not an attempt.
    """
    if ALTERNATIVE_MARKER not in raw:
        return [raw]
    parts = [part.strip() for part in raw.split(ALTERNATIVE_MARKER)]
    return [part for part in parts if part][:MAX_ALTERNATIVES] or [raw]


def apply_search_replace(parent: str, raw: str) -> str | None:
    """Apply search/replace blocks from *raw* onto *parent*.

    Returns the patched string, or ``None`` if *raw* contained no blocks at all
    -- that is the signal for callers to fall back to parsing a whole file out
    of the response. Blocks are applied in order; each replaces the first
    occurrence in the current (already-patched) text.

    Raises NoEditAppliedError if blocks were present but none of their SEARCH
    text was found. ``str.replace`` is silent in that case, so the parent used
    to come back unchanged and be reported as a successful edit: a paid LLM call
    produced a byte-identical child that still entered the pool, carrying its
    parent's signature and dragging the measured genome diversity down until it
    tripped an epoch transition.
    """
    blocks = _SEARCH_REPLACE_RE.findall(raw)
    if not blocks:
        return None

    result = parent
    applied = 0
    for search, replace in blocks:
        if search in result:
            result = result.replace(search, replace, 1)
            applied += 1
            continue
        # Exact matching failed, which is usually transcription rather than a
        # wrong edit: the model has to copy the markup back byte for byte, and
        # measured on one run 3 of 5 failed seed edits were blocks whose SEARCH
        # text differed from the parent only in whitespace. The edit itself was
        # fine. So try again treating any run of whitespace as equivalent to
        # any other, which is exactly the freedom XML already gives between
        # attributes and tags.
        span = _loose_match(result, search)
        if span is not None:
            start, end = span
            result = result[:start] + replace + result[end:]
            applied += 1
            log.debug("Applied a search/replace block on whitespace alone.")

    if applied == 0:
        raise NoEditAppliedError(
            f"none of the {len(blocks)} search/replace block(s) matched the parent"
        )
    if applied < len(blocks):
        log.warning(
            f"Applied {applied}/{len(blocks)} search/replace blocks; "
            "the rest did not match the parent."
        )
    return result


def _require_svg(fragment: str, raw: str) -> str:
    if "<svg" not in fragment.lower():
        raise NoUsableOutputError(
            f"no <svg> in the reply and no diff blocks: {describe_unusable(raw)}"
        )
    return fragment


def extract_svg(raw: str) -> str:
    """The complete drawing in *raw*, normalized."""
    from vectrify.svg.normalize import normalize_svg
    from vectrify.svg.prompts import extract_svg_fragment

    # Normalised on the way in, so local search meets one form of markup
    # rather than whichever the model reached for. Which forms it reaches
    # for is a property of the model: one model's seeds carried 147
    # elements in relative path commands, which describe an offset from
    # wherever the pen already is and so cannot be moved at all.
    return normalize_svg(_require_svg(extract_svg_fragment(raw), raw))


def apply_edit(parent: str, raw: str) -> str:
    """*parent* with the reply's search/replace blocks, or the reply's drawing."""
    from vectrify.svg.normalize import normalize_svg
    from vectrify.svg.prompts import extract_svg_fragment

    patched = apply_search_replace(parent, raw)
    if patched is None:
        patched = _require_svg(extract_svg_fragment(raw), raw)
    return normalize_svg(patched)


def apply_edits(parent: str, raw: str) -> list[str]:
    """Every attempt the reply offers, each a candidate of its own.

    A section that will not apply is dropped rather than failing the rest:
    a reply offering three attempts should not be discarded because one of
    them misquoted the markup. If none apply, the single-edit path runs
    again so the caller sees the same error it always did.
    """
    candidates: list[str] = []
    for section in split_alternatives(raw):
        try:
            candidates.append(apply_edit(parent, section))
        except Exception as exc:
            log.debug(f"Dropping one alternative: {exc}")
    if not candidates:
        return [apply_edit(parent, raw)]
    return candidates
