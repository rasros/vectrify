import io

from PIL import Image

from vectrify.formats.base import apply_search_replace
from vectrify.formats.svg.operations import (
    crossover_with_micro_search,
    mutate_with_micro_search,
)
from vectrify.formats.svg.prompts import (
    build_svg_gen_prompt,
    extract_svg_fragment,
    is_valid_svg,
)
from vectrify.image_utils import rasterize_svg_to_png_bytes, resize_long_side


class SvgPlugin:
    name = "svg"
    file_extension = ".svg"

    def rasterize(self, content: str, out_w: int, out_h: int) -> bytes:
        return rasterize_svg_to_png_bytes(content, out_w=out_w, out_h=out_h)

    def rasterize_fast(self, content: str, long_side: int) -> bytes | None:
        try:
            png = rasterize_svg_to_png_bytes(content, out_w=long_side, out_h=long_side)
            img = Image.open(io.BytesIO(png)).convert("RGB")
            img = resize_long_side(img, long_side)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return None

    def validate(self, content: str) -> tuple[bool, str | None]:
        return is_valid_svg(content)

    def extract_from_llm(self, raw: str) -> str:
        return extract_svg_fragment(raw)

    def apply_edit(self, parent: str, raw: str) -> str:
        patched = apply_search_replace(parent, raw)
        return patched if patched is not None else extract_svg_fragment(raw)

    def build_generate_prompt(
        self,
        image_data_url: str,
        node_index: int,
        content_prev: str | None,
        raster_preview_url: str | None,
        goal: str | None,
        diff_data_url: str | None,
    ) -> list[dict]:
        return build_svg_gen_prompt(
            image_data_url,
            node_index,
            svg_prev=content_prev,
            rasterized_svg_data_url=raster_preview_url if content_prev else None,
            goal=goal,
            diff_data_url=diff_data_url,
        )

    def mutate(self, content: str, orig_img_fast: Image.Image) -> tuple[str, str]:
        return mutate_with_micro_search(
            parent_svg=content, orig_img_fast=orig_img_fast, num_trials=15
        )

    def crossover(
        self, content_a: str, content_b: str, orig_img_fast: Image.Image
    ) -> tuple[str, str]:
        return crossover_with_micro_search(
            svg_a=content_a,
            svg_b=content_b,
            orig_img_fast=orig_img_fast,
            num_trials=15,
        )
