from __future__ import annotations

import logging
import re

import PIL.Image

from vectrify.formats.base import BaseFormatPlugin
from vectrify.formats.typst.operations import (
    crossover_with_micro_search,
    mutate_with_micro_search,
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
        diff_data_url: str | None,
        canvas: tuple[int, int],
    ) -> list[dict]:
        return build_typst_gen_prompt(
            image_data_url=image_data_url,
            node_index=node_index,
            typst_prev=content_prev,
            rasterized_data_url=raster_preview_url,
            goal=goal,
            diff_data_url=diff_data_url,
            canvas=canvas,
        )

    def mutate(self, content: str, orig_img_fast: PIL.Image.Image) -> tuple[str, str]:
        return mutate_with_micro_search(
            parent_code=content, orig_img_fast=orig_img_fast, num_trials=15
        )

    def crossover(
        self, content_a: str, content_b: str, orig_img_fast: PIL.Image.Image
    ) -> tuple[str, str]:
        return crossover_with_micro_search(
            code_a=content_a,
            code_b=content_b,
            orig_img_fast=orig_img_fast,
            num_trials=15,
        )
