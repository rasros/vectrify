import io
import re
import xml.etree.ElementTree as ET
import zlib

from PIL import Image

_PATH_COMMANDS = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]")


def svg_complexity(svg: str | None) -> float:
    """Complexity estimate combining three signals:

    1. zlib compressed size — Kolmogorov approximation; rewards repetitive structure.
    2. Element count — number of XML nodes.
    3. Path vertex count — total geometric commands across all path attributes.

    The weights below are chosen so that each signal contributes roughly equally
    for typical LLM-generated SVGs (~10-200 elements, ~100-2000 path commands,
    ~500-5000 compressed bytes).
    """
    if not svg:
        return 0.0

    compressed_size = len(zlib.compress(svg.encode("utf-8"), level=9))

    try:
        root = ET.fromstring(svg)
        element_count = sum(1 for _ in root.iter())
        path_vertices = sum(
            len(_PATH_COMMANDS.findall(el.get("d", "")))
            for el in root.iter()
            if el.get("d")
        )
    except ET.ParseError:
        return float(compressed_size)

    return float(compressed_size + element_count * 50 + path_vertices * 5)


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
