import dataclasses

from vectrify.formats.models import VectorResultPayload, VectorStatePayload
from vectrify.image_utils import make_preview_data_url, png_bytes_to_data_url
from vectrify.search import ChainState, Result


@dataclasses.dataclass
class VectorStateBuilder:
    """Turns a worker result into pool state, deciding which renders to keep.

    Rasters are held as data URLs on the node, so what is kept is what later
    gets written to disk and sent to the LLM; skipping them when no consumer
    is enabled is the difference between holding one preview per pool member
    and holding three.
    """

    resolution_llm: int
    write_lineage: bool
    save_raster: bool = False

    def __call__(self, result: Result) -> ChainState[VectorStatePayload]:
        payload: VectorResultPayload = result.payload

        raster_data_url = None
        if (self.write_lineage or self.save_raster) and payload.raster_png:
            raster_data_url = png_bytes_to_data_url(payload.raster_png)

        preview_data_url = payload.raster_preview_data_url
        if preview_data_url is None and payload.raster_png:
            preview_data_url = make_preview_data_url(
                payload.raster_png, self.resolution_llm
            )

        heatmap_data_url = None
        if payload.heatmap_png:
            heatmap_data_url = png_bytes_to_data_url(payload.heatmap_png)

        return ChainState(
            score=result.score,
            payload=VectorStatePayload(
                content=payload.content,
                raster_data_url=raster_data_url,
                raster_preview_data_url=preview_data_url,
                origin=payload.origin,
                heatmap_data_url=heatmap_data_url,
            ),
        )
