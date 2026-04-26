from __future__ import annotations

import io
import logging
import re

import PIL.Image

from vectrify.formats.base import apply_search_replace
from vectrify.formats.typst.operations import (
    crossover_with_micro_search,
    mutate_with_micro_search,
)
from vectrify.formats.typst.prompts import build_typst_gen_prompt

log = logging.getLogger(__name__)

# Built using concatenated strings to prevent Markdown parsers from
# choking on nested fences
_TYPST_FENCE = re.compile(
    "`" * 3 + r"(?:typst|typ)\s*(.*?)\s*" + "`" * 3, re.DOTALL | re.IGNORECASE
)


class TypstPlugin:
    name = "typst"
    file_extension = ".typ"

    def rasterize(self, content: str, out_w: int, out_h: int) -> bytes:
        import typst

        # Encode to bytes so typst-py treats it as source code, not a file path!
        png = typst.compile(content.encode("utf-8"), format="png", ppi=144)

        if isinstance(png, list):
            if not png:
                raise ValueError("Typst generated zero pages.")
            png = png[0]
        elif not isinstance(png, bytes):
            raise ValueError("Failed to rasterize Typst to PNG bytes.")

        img = PIL.Image.open(io.BytesIO(png)).convert("RGB")
        img = img.resize((out_w, out_h), PIL.Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def rasterize_fast(self, content: str, long_side: int) -> bytes | None:
        try:
            import typst

            from vectrify.image_utils import resize_long_side

            # Encode to bytes
            png = typst.compile(content.encode("utf-8"), format="png", ppi=144)

            if isinstance(png, list):
                if not png:
                    return None
                png = png[0]
            elif not isinstance(png, bytes):
                return None

            img = PIL.Image.open(io.BytesIO(png)).convert("RGB")
            img = resize_long_side(img, long_side)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return None

    def validate(self, content: str) -> tuple[bool, str | None]:
        try:
            import typst

            # Compile to dummy PDF in memory to check syntax validity
            typst.compile(content.encode("utf-8"))
            return True, None
        except Exception as e:
            return False, str(e)

    def extract_from_llm(self, raw: str) -> str:
        m = _TYPST_FENCE.search(raw)
        if m:
            return m.group(1).strip()
        return raw.strip()

    def apply_edit(self, parent: str, raw: str) -> str:
        patched = apply_search_replace(parent, raw)
        return patched if patched is not None else self.extract_from_llm(raw)

    def build_generate_prompt(
        self,
        image_data_url: str,
        node_index: int,
        content_prev: str | None,
        raster_preview_url: str | None,
        goal: str | None,
        diff_data_url: str | None,
    ) -> list[dict]:
        return build_typst_gen_prompt(
            image_data_url=image_data_url,
            node_index=node_index,
            typst_prev=content_prev,
            rasterized_data_url=raster_preview_url,
            goal=goal,
            diff_data_url=diff_data_url,
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
