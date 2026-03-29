import dataclasses

from svgizer.formats.models import VectorResultPayload, VectorStatePayload
from svgizer.image_utils import make_preview_data_url, png_bytes_to_data_url
from svgizer.search import ChainState, Result, SearchNode, SearchStrategy


@dataclasses.dataclass
class VectorStrategyAdapter:
    base_strategy: SearchStrategy[VectorStatePayload]
    image_long_side: int
    write_lineage: bool
    save_raster: bool

    def __init__(
        self,
        base_strategy: SearchStrategy[VectorStatePayload],
        image_long_side: int,
        write_lineage: bool,
        save_raster: bool = False,
    ):
        self.base_strategy = base_strategy
        self.image_long_side = image_long_side
        self.write_lineage = write_lineage
        self.save_raster = save_raster

    @property
    def top_k_count(self) -> int:
        return self.base_strategy.top_k_count

    def select_parent(
        self, nodes: list[SearchNode[VectorStatePayload]], progress: float
    ) -> tuple[int, int | None]:
        return self.base_strategy.select_parent(nodes, progress)

    def should_diversify(
        self, pool: list[SearchNode[VectorStatePayload]]
    ) -> tuple[bool, float]:
        return self.base_strategy.should_diversify(pool)

    def epoch_seeds(
        self, pool: list[SearchNode[VectorStatePayload]], max_seeds: int
    ) -> list[SearchNode[VectorStatePayload]]:
        return self.base_strategy.epoch_seeds(pool, max_seeds)

    def create_new_state(self, result: Result) -> ChainState[VectorStatePayload]:
        new_state = self.base_strategy.create_new_state(result)
        result_payload: VectorResultPayload = result.payload

        raster_data_url = None
        if (self.write_lineage or self.save_raster) and result_payload.raster_png:
            raster_data_url = png_bytes_to_data_url(result_payload.raster_png)

        preview_data_url = result_payload.raster_preview_data_url
        if preview_data_url is None and result_payload.raster_png:
            preview_data_url = make_preview_data_url(
                result_payload.raster_png, self.image_long_side
            )

        new_state.payload = VectorStatePayload(
            content=result_payload.content,
            raster_data_url=raster_data_url,
            raster_preview_data_url=preview_data_url,
            change_summary=result_payload.change_summary,
            invalid_msg=result.invalid_msg,
        )
        return new_state
