from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol

from vectrify.image_utils import png_resize_exact

if TYPE_CHECKING:
    import PIL.Image

_SEARCH_REPLACE_RE = re.compile(
    r"<<<SEARCH>>>\n(.*?)\n<<<REPLACE>>>\n(.*?)\n<<<END>>>",
    re.DOTALL,
)


def apply_search_replace(parent: str, raw: str) -> str | None:
    """Apply search/replace blocks from *raw* onto *parent*.

    Returns the patched string, or ``None`` if *raw* contained no blocks.
    Blocks are applied in order; each replaces the first occurrence in the
    current (already-patched) text.
    """
    blocks = _SEARCH_REPLACE_RE.findall(raw)
    if not blocks:
        return None
    result = parent
    for search, replace in blocks:
        result = result.replace(search, replace, 1)
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
        diff_data_url: str | None,
    ) -> list[dict]:
        """Build the LLM generation/refinement prompt as content blocks."""
        ...

    def mutate(self, content: str, orig_img_fast: PIL.Image.Image) -> tuple[str, str]:
        """Mutate existing content. Return (new_content, origin)."""
        ...

    def crossover(
        self,
        content_a: str,
        content_b: str,
        orig_img_fast: PIL.Image.Image,
    ) -> tuple[str, str]:
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

    def render_png_or_none(self, content: str) -> bytes | None:
        """Render without raising — the shape micro-search expects."""
        try:
            return self._render_png(content)
        except Exception:
            return None

    def validate(self, content: str) -> tuple[bool, str | None]:
        try:
            self._compile(content)
            return True, None
        except Exception as e:
            return False, str(e)

    def apply_edit(self, parent: str, raw: str) -> str:
        patched = apply_search_replace(parent, raw)
        return patched if patched is not None else self.extract_from_llm(raw)
