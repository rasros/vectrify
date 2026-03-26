"""SVG complexity scoring. Lower is simpler (better in multi-objective search)."""

import re
import xml.etree.ElementTree as ET
import zlib

# Path command letters that each represent one geometric operation.
_PATH_COMMANDS = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]")


def svg_complexity(svg: str | None) -> float:
    """Complexity estimate combining three signals:

    1. zlib compressed size — Kolmogorov approximation; rewards repetitive/simple
       structure over information-dense SVGs of the same raw byte length.
    2. Element count — number of XML nodes; penalises many-shape SVGs regardless
       of how well they compress.
    3. Path vertex count — total geometric commands across all <path d="...">
       attributes; penalises intricate curves independently of raw size.

    All three are summed. NSGA-II normalises within the population so the
    absolute scale only affects relative ordering — the weights below are chosen
    so that each signal contributes roughly equally for typical LLM-generated SVGs
    (~10-200 elements, ~100-2000 path commands, ~500-5000 compressed bytes).

    Returns 0.0 for None/empty SVGs (sentinel nodes).
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
