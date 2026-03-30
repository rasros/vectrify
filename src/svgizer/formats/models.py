import dataclasses


@dataclasses.dataclass
class VectorStatePayload:
    content: str | None
    raster_data_url: str | None
    raster_preview_data_url: str | None
    origin: str | None
    invalid_msg: str | None


@dataclasses.dataclass
class VectorResultPayload:
    content: str | None
    raster_png: bytes | None
    origin: str | None
    raster_preview_data_url: str | None = None
