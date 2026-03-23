import dataclasses
import difflib
from collections.abc import Callable

from svgizer.image_utils import make_preview_data_url, png_bytes_to_data_url
from svgizer.search import ChainState, Result, SearchNode, SearchStrategy


@dataclasses.dataclass
class SvgStatePayload:
    svg: str | None
    raster_data_url: str | None
    raster_preview_data_url: str | None
    change_summary: str | None
    invalid_msg: str | None


@dataclasses.dataclass
class SvgResultPayload:
    svg: str | None
    raster_png: bytes | None
    change_summary: str | None


def make_is_svg_stale(
    threshold: float = 0.995,
) -> Callable[[SvgStatePayload, SvgResultPayload], bool]:
    def is_svg_stale(
        prev_payload: SvgStatePayload, new_payload: SvgResultPayload
    ) -> bool:
        if prev_payload is None or prev_payload.svg is None:
            return False
        if new_payload.svg is None:
            return True
        if prev_payload.svg == new_payload.svg:
            return True

        ratio = difflib.SequenceMatcher(None, prev_payload.svg, new_payload.svg).ratio()
        return ratio >= threshold

    return is_svg_stale


class SvgStrategyAdapter:
    def __init__(
        self,
        base_strategy: SearchStrategy[SvgStatePayload],
        openai_image_long_side: int,
        write_lineage: bool,
    ):
        self.base_strategy = base_strategy
        self.openai_image_long_side = openai_image_long_side
        self.write_lineage = write_lineage

    @property
    def top_k_count(self) -> int:
        return self.base_strategy.top_k_count

    def select_parent(
        self, nodes: list[SearchNode[SvgStatePayload]], progress: float
    ) -> tuple[int, int | None]:
        return self.base_strategy.select_parent(nodes, progress)

    def create_new_state(
        self, parent_state: ChainState[SvgStatePayload], result: Result
    ) -> ChainState[SvgStatePayload]:
        new_state = self.base_strategy.create_new_state(parent_state, result)
        result_payload: SvgResultPayload = result.payload

        raster_data_url = None
        if self.write_lineage and result_payload.raster_png:
            raster_data_url = png_bytes_to_data_url(result_payload.raster_png)

        preview_data_url = None
        if result_payload.raster_png:
            preview_data_url = make_preview_data_url(
                result_payload.raster_png, self.openai_image_long_side
            )

        new_state.payload = SvgStatePayload(
            svg=result_payload.svg,
            raster_data_url=raster_data_url,
            raster_preview_data_url=preview_data_url,
            change_summary=result_payload.change_summary,
            invalid_msg=result.invalid_msg,
        )
        return new_state
