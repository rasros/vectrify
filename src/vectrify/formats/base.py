from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Protocol

from vectrify.image_utils import png_resize_exact

log = logging.getLogger(__name__)

_SEARCH_REPLACE_RE = re.compile(
    r"<<<SEARCH>>>\n(.*?)\n<<<REPLACE>>>\n(.*?)\n<<<END>>>",
    re.DOTALL,
)


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


class FormatPlugin(Protocol):
    name: str
    file_extension: str

    def rasterize(self, content: str, out_w: int, out_h: int) -> bytes:
        """Render content to PNG bytes at given dimensions."""
        ...

    def validate(self, content: str) -> tuple[bool, str | None]:
        """Return (is_valid, error_message_or_None)."""
        ...

    def extract_from_llm(self, raw: str) -> str:
        """Parse the LLM's raw text response to extract format content (full file)."""
        ...

    def apply_edit(self, parent: str, raw: str) -> str:
        """Apply an LLM edit response to *parent*.

        Expects search/replace diff blocks in *raw*; falls back to
        ``extract_from_llm`` if none are found.
        """
        ...

    def build_generate_prompt(
        self,
        image_data_url: str,
        node_index: int,
        content_prev: str | None,
        raster_preview_url: str | None,
        goal: str | None,
        canvas: tuple[int, int],
        source_name: str | None = None,
    ) -> list[dict]:
        """Build the LLM generation/refinement prompt as content blocks.

        *canvas* is the raster size the candidate will be rendered and scored
        at. Formats with their own coordinate space must pin it to this, so
        every candidate in the pool shares one space: the genetic operators
        graft elements between parents, and coordinates that meant different
        things in different spaces are silently misplaced by the graft.
        """
        ...

    def mutation_weights(self) -> Mapping[str, float]:
        """This backend's mutation operators and their default weights.

        The names are what a policy selects by and what results are attributed
        to, so they must be stable across a run.
        """
        ...

    def mutate(
        self,
        content: str,
        operator: str | None = None,
        targets: dict[int, float] | None = None,
    ) -> tuple[str, str]:
        """Mutate existing content. Return (new_content, origin).

        *operator* names one of ``mutation_weights``; None lets the backend
        pick for itself. *targets* weights which element to work on, by its
        position among the drawable elements.
        """
        ...

    def element_targets(self, content: str, reference_png: bytes) -> dict[int, float]:
        """How much error each element of *content* answers for.

        Backends that cannot attribute error return an empty mapping, which
        leaves mutation choosing its target uniformly.
        """
        ...

    def crossover(self, content_a: str, content_b: str) -> tuple[str, str]:
        """Crossover two contents. Return (new_content, origin)."""
        ...


class BaseFormatPlugin:
    """Shared plumbing for plugins whose renderer picks its own output size.

    Subclasses supply ``_render_png`` and ``_compile``; rasterizing, validating,
    and edit application are derived from those. ``extract_from_llm``, the
    prompt builder, and the genetic operators stay format-specific.
    """

    name: str
    file_extension: str

    def _render_png(self, content: str) -> bytes:
        """Render *content* to PNG bytes at the renderer's natural size."""
        raise NotImplementedError

    def _compile(self, content: str) -> None:
        """Raise if *content* is not syntactically valid."""
        raise NotImplementedError

    def extract_from_llm(self, raw: str) -> str:
        raise NotImplementedError

    def rasterize(self, content: str, out_w: int, out_h: int) -> bytes:
        return png_resize_exact(self._render_png(content), out_w, out_h)

    def validate(self, content: str) -> tuple[bool, str | None]:
        try:
            self._compile(content)
            return True, None
        except Exception as e:
            return False, str(e)

    def apply_edit(self, parent: str, raw: str) -> str:
        patched = apply_search_replace(parent, raw)
        return patched if patched is not None else self.extract_from_llm(raw)

    def element_targets(self, content: str, reference_png: bytes) -> dict[int, float]:
        """No attribution: mutation picks its target uniformly.

        Resolving which element owns which pixel needs a renderer that can draw
        one element at a time in a chosen colour, which these backends hand off
        to an external tool.
        """
        _ = content, reference_png
        return {}
