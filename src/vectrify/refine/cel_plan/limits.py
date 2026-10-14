"""Shared proposal scheduling and footprint limits, independent of execution."""

EXPANSIONS = {"fast": 4, "balanced": 8, "high": 16}
MAX_SEED_NODES = 32_000
MAX_SEED_OBJECTS = 8_192
MAX_EDIT_OBJECTS = 256

# Path nodes one surface, overlay or ink edit may touch.
DOCUMENT_NODES = 6_000
# Cumulative source-path nodes a source stroke analysis may inspect.
SOURCE_NODES = 16_384
# Path nodes and stroke width (pixels) of one filled band.
BAND_NODES = 256
BAND_WIDTH = 12
# Largest analysis or native raster a stage will scan.
RASTER_PIXELS = 1536**2
# Largest bounding box of a surface group fitted or split as one.
SURFACE_BOX_PIXELS = 262_144
# Pixels processed per chunk so a scan stays interruptible.
CHUNK_PIXELS = 65_536
