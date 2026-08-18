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

    def _render_png(self, content: str) -> bytes:
        return render_typst_png(content)

    def _compile(self, content: str) -> None:
        import typst

        # Compile to a throwaway PDF in memory to check syntax validity
        typst.compile(content.encode("utf-8"))

    def extract_from_llm(self, raw: str) -> str:
        m = _TYPST_FENCE.search(raw)
        if m:
            return m.group(1).strip()
        return raw.strip()

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
    ) -> tuple[str, str]:
        _ = targets
        return apply_mutation(content, operator)

    def crossover(self, content_a: str, content_b: str) -> tuple[str, str]:
        return apply_crossover(content_a, content_b)
