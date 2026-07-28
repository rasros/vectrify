"""Complexity measures used as NSGA objectives alongside visual score.

Two independent measures are reported rather than one blended number: a raster
measure of how much detail the render carries, and a source measure of how much
text it takes to say it. They are separate objectives, so nothing here has to
pick a weighting between them.
"""

import io
import re

from PIL import Image

_WHITESPACE_RE = re.compile(r"\s+")


def visual_complexity(png_bytes: bytes) -> float:
    """Visual complexity measured as JPEG compressed size.

    JPEG encodes spatial redundancy the same way the human visual system weighs
    detail: a flat-coloured region compresses to almost nothing; a region with
    fine detail or many colour transitions requires many more bytes.  This gives
    a perceptual complexity score that is immune to source-level tricks (e.g.
    a single large rectangle matching the dominant colour scores near zero even
    if the source itself is small).
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return float(len(buf.getvalue()))


def structural_complexity(source: str) -> float:
    """Source complexity as whitespace-stripped character count.

    Deliberately format-agnostic: it means the same thing for SVG, DOT and
    Typst, so no backend can be silently scored as free. Stripping whitespace
    makes it indifferent to indentation and pretty-printing, which vary with
    whatever the model happened to emit and carry no complexity information.

    A compressed size (gzip) was the obvious alternative and was rejected: it
    discounts repetitive source by ~80%, and every crossover operator injects
    elements from a *related* parent, so accumulating near-duplicate elements
    is the norm rather than the exception. That is exactly the bloat this
    objective exists to charge for, and ``visual_complexity`` already forgives
    visual repetition, so a compressible-source measure would leave redundancy
    unpenalised on both objectives at once.
    """
    return float(len(_WHITESPACE_RE.sub("", source)))
