import dataclasses

from vectrify.formats.models import VectorResultPayload, VectorStatePayload
from vectrify.image_utils import make_preview_data_url, png_bytes_to_data_url
from vectrify.search import ChainState, Result, SearchNode, SearchStrategy


@dataclasses.dataclass
class VectorStrategyAdapter:
    base_strategy: SearchStrategy[VectorStatePayload]
    resolution_llm: int
    write_lineage: bool
    save_raster: bool = False

    def select_parent(
        self, nodes: list[SearchNode[VectorStatePayload]]
    ) -> tuple[int, int | None]:
        return self.base_strategy.select_parent(nodes)

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
                result_payload.raster_png, self.resolution_llm
            )

        heatmap_data_url = None
        if result_payload.heatmap_png:
            heatmap_data_url = png_bytes_to_data_url(result_payload.heatmap_png)

        new_state.payload = VectorStatePayload(
            content=result_payload.content,
            raster_data_url=raster_data_url,
            raster_preview_data_url=preview_data_url,
            origin=result_payload.origin,
            heatmap_data_url=heatmap_data_url,
        )
        return new_state
