import io
import re

from PIL import Image

_VISUAL_WEIGHT = 0.7

# Regex patterns compiled once.
_RE_PATH_DATA = re.compile(r'\bd="([^"]*)"')
_RE_PATH_CMDS = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]")
_RE_PATHS = re.compile(r"<path\b")
_RE_SHAPES = re.compile(r"<(?:circle|ellipse|rect|line|polyline|polygon|text|image)\b")
_RE_COLOURS = re.compile(r'(?:fill|stroke)="(#[0-9a-fA-F]{3,8}|rgb[^"]*|[a-z]+)"')
_RE_STOPS = re.compile(r"<stop\b")
_RE_FILTERS = re.compile(r"<fe[A-Z]\w+")


def visual_complexity(png_bytes: bytes) -> float:
    """Visual complexity measured as JPEG compressed size.

    JPEG encodes spatial redundancy the same way the human visual system weighs
    detail: a flat-coloured region compresses to almost nothing; a region with
    fine detail or many colour transitions requires many more bytes.  This gives
    a perceptual complexity score that is immune to SVG structural tricks (e.g.
    a single large <rect> matching the dominant colour scores near zero even if
    the SVG file itself is small).
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return float(len(buf.getvalue()))


def content_complexity(svg_str: str) -> float:
    """Structural complexity of an SVG estimated from its source text.

    Counts path commands, element count, unique colours, and use of advanced
    features (gradients, filters) as a proxy for visual richness that is
    independent of raster rendering.
    """
    path_commands = sum(
        len(_RE_PATH_CMDS.findall(d)) for d in _RE_PATH_DATA.findall(svg_str)
    )
    paths = len(_RE_PATHS.findall(svg_str))
    shapes = len(_RE_SHAPES.findall(svg_str))
    colours = len(set(_RE_COLOURS.findall(svg_str)))
    gradient_stops = len(_RE_STOPS.findall(svg_str))
    filter_prims = len(_RE_FILTERS.findall(svg_str))

    return float(
        path_commands * 8
        + paths * 40
        + shapes * 15
        + colours * 20
        + gradient_stops * 25
        + filter_prims * 50
    )


def complexity(png_bytes: bytes, svg_str: str) -> float:
    """Blended complexity: visual (0.7) + structural content (0.3)."""
    return _VISUAL_WEIGHT * visual_complexity(png_bytes) + (
        1.0 - _VISUAL_WEIGHT
    ) * content_complexity(svg_str)
