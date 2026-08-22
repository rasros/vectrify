from __future__ import annotations

import logging
import re
from collections.abc import Mapping

from vectrify.formats.base import BaseFormatPlugin
from vectrify.formats.mutations import operator_weights
from vectrify.formats.typst.operations import (
    MUTATIONS,
    apply_crossover,
    apply_mutation,
    canonicalize_page_setup,
    render_typst_png,
)
from vectrify.formats.typst.prompts import build_typst_gen_prompt

log = logging.getLogger(__name__)

# Built using concatenated strings to prevent Markdown parsers from
# choking on nested fences
_TYPST_FENCE = re.compile(
    "`" * 3 + r"(?:typst|typ)\s*(.*?)\s*" + "`" * 3, re.DOTALL | re.IGNORECASE
)


class TypstPlugin(BaseFormatPlugin):
    name = "typst"
    file_extension = ".typ"

    def __init__(self) -> None:
        # The worker calls build_generate_prompt before extracting the reply.
        # Retaining this run-local canvas lets extraction remove any model
        # supplied auto/multiple page settings before they enter the pool.
        self._canvas: tuple[int, int] | None = None

    def _render_png(self, content: str) -> bytes:
        return render_typst_png(content)

    def _compile(self, content: str) -> None:
        import typst

        # Compile to a throwaway PDF in memory to check syntax validity
        typst.compile(content.encode("utf-8"))

    def extract_from_llm(self, raw: str) -> str:
        m = _TYPST_FENCE.search(raw)
        content = m.group(1).strip() if m else raw.strip()
        return canonicalize_page_setup(content, self._canvas or (0, 0))

    def apply_edit(self, parent: str, raw: str) -> str:
        """Keep an LLM edit from changing the immutable page coordinate space."""
        return canonicalize_page_setup(
            super().apply_edit(parent, raw), self._canvas or (0, 0)
        )

    def build_generate_prompt(
        self,
        image_data_url: str,
        node_index: int,
        content_prev: str | None,
        raster_preview_url: str | None,
        goal: str | None,
        canvas: tuple[int, int],
        source_name: str | None = None,
        # Accepted and ignored: reporting elements that paint nothing needs a
        # renderer that can draw one element at a time, which these backends
        # hand off to an external tool. Same limit as `element_targets`.
        invisible: list[str] | None = None,  # noqa: ARG002
    ) -> list[dict]:
        self._canvas = canvas
        return build_typst_gen_prompt(
            image_data_url=image_data_url,
            node_index=node_index,
            source_name=source_name,
            typst_prev=content_prev,
            rasterized_data_url=raster_preview_url,
            goal=goal,
            canvas=canvas,
        )

    def mutation_weights(self) -> Mapping[str, float]:
        return operator_weights(MUTATIONS)

    def mutate(
        self,
        content: str,
        operator: str | None = None,
        targets: dict[int, float] | None = None,
        reference_png: bytes | None = None,
    ) -> tuple[str, str]:
        _ = targets, reference_png
        return apply_mutation(content, operator)

    def crossover(self, content_a: str, content_b: str) -> tuple[str, str]:
        return apply_crossover(content_a, content_b)
