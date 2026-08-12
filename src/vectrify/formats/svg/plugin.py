from vectrify.formats.base import apply_search_replace
from vectrify.formats.svg.operations import apply_crossover, apply_mutation
from vectrify.formats.svg.prompts import (
    build_svg_gen_prompt,
    extract_svg_fragment,
    is_valid_svg,
)
from vectrify.image_utils import rasterize_svg_to_png_bytes


class SvgPlugin:
    name = "svg"
    file_extension = ".svg"

    def rasterize(self, content: str, out_w: int, out_h: int) -> bytes:
        return rasterize_svg_to_png_bytes(content, out_w=out_w, out_h=out_h)

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
        canvas: tuple[int, int],
    ) -> list[dict]:
        return build_svg_gen_prompt(
            image_data_url,
            node_index,
            svg_prev=content_prev,
            rasterized_svg_data_url=raster_preview_url if content_prev else None,
            goal=goal,
            canvas=canvas,
        )

    def mutate(self, content: str) -> tuple[str, str]:
        return apply_mutation(content)

    def crossover(self, content_a: str, content_b: str) -> tuple[str, str]:
        return apply_crossover(content_a, content_b)
