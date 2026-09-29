"""The drawing a search node or worker result carries."""

import dataclasses


@dataclasses.dataclass
class VectorStatePayload:
    content: str | None
    origin: str | None


@dataclasses.dataclass
class VectorResultPayload:
    content: str | None
    # The candidate rendered at the reference's size, for the scorer thread.
    raster_png: bytes | None
    origin: str | None
