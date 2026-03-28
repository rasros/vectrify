from __future__ import annotations

import io
import logging
import re

import PIL.Image

from svgizer.formats.graphviz.operations import (
    crossover_with_micro_search,
    mutate_with_micro_search,
)
from svgizer.formats.graphviz.prompts import (
    build_dot_gen_prompt,
    build_dot_summarize_prompt,
)

log = logging.getLogger(__name__)

_DOT_FENCE = re.compile(r"```dot\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
# Graph name may be unquoted (\w+), quoted ("..."), or absent.
_DOT_RAW = re.compile(
    r'(strict\s+)?(di)?graph\s+(?:\w+|"[^"]*")?\s*\{.*\}',
    re.DOTALL | re.IGNORECASE,
)


def _sanitize_dot(dot: str) -> str:
    """Fix common LLM-generated DOT mistakes before validation."""
    # Upgrade undirected `graph` to `digraph` when directed edges are present.
    if "->" in dot and not re.search(r"\bdigraph\b", dot, re.IGNORECASE):
        dot = re.sub(r"\bgraph\b", "digraph", dot, count=1, flags=re.IGNORECASE)
    return dot


class GraphvizPlugin:
    name = "graphviz"
    file_extension = ".dot"

    def rasterize(self, content: str, out_w: int, out_h: int) -> bytes:
        import graphviz

        src = graphviz.Source(content)
        png = src.pipe(format="png")
        # Resize to exact dimensions
        img = PIL.Image.open(io.BytesIO(png)).convert("RGB")
        img = img.resize((out_w, out_h), PIL.Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def rasterize_fast(self, content: str, long_side: int) -> bytes | None:
        try:
            import graphviz

            from svgizer.image_utils import resize_long_side

            src = graphviz.Source(content)
            png = src.pipe(format="png")
            img = PIL.Image.open(io.BytesIO(png)).convert("RGB")
            img = resize_long_side(img, long_side)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return None

    def validate(self, content: str) -> tuple[bool, str | None]:
        try:
            import graphviz

            graphviz.Source(content).pipe(format="svg")
            return True, None
        except Exception as e:
            return False, str(e)

    def extract_from_llm(self, raw: str) -> str:
        m = _DOT_FENCE.search(raw)
        if m:
            return _sanitize_dot(m.group(1).strip())
        m = _DOT_RAW.search(raw)
        if m:
            return _sanitize_dot(m.group(0).strip())
        return _sanitize_dot(raw.strip())

    def build_generate_prompt(
        self,
        image_data_url: str,
        node_index: int,
        content_prev: str | None,
        raster_preview_url: str | None,
        change_summary: str | None,
        diff_data_url: str | None,
    ) -> list[dict]:
        return build_dot_gen_prompt(
            image_data_url=image_data_url,
            node_index=node_index,
            dot_prev=content_prev,
            rasterized_dot_data_url=raster_preview_url,
            change_summary=change_summary,
            diff_data_url=diff_data_url,
        )

    def build_summarize_prompt(
        self,
        image_data_url: str,
        raster_preview_url: str | None,
        custom_goal: str | None,
        previous_summary: str | None,
    ) -> list[dict]:
        return build_dot_summarize_prompt(
            image_data_url=image_data_url,
            raster_preview_url=raster_preview_url,
            custom_goal=custom_goal,
            previous_summary=previous_summary,
        )

    def mutate(self, content: str, orig_img_fast: PIL.Image.Image) -> tuple[str, str]:
        return mutate_with_micro_search(
            parent_dot=content, orig_img_fast=orig_img_fast, num_trials=15
        )

    def crossover(
        self, content_a: str, content_b: str, orig_img_fast: PIL.Image.Image
    ) -> tuple[str, str]:
        return crossover_with_micro_search(
            dot_a=content_a,
            dot_b=content_b,
            orig_img_fast=orig_img_fast,
            num_trials=15,
        )
