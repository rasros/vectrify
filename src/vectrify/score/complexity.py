"""How much visual detail a render carries, as a distance from the target's.

A regulariser. The other objectives all reward agreement with the reference at
a pixel or an edge, and local search can buy that agreement by adding detail
the target does not have -- stipple a flat region and the edge map lights up in
roughly the right places. Measured on a 45-minute single-epoch run, the round
score fell 64% against a three-epoch run while the evaluator scored the two
within 0.000004 of each other: the search had been paying for detail that no
perceptual judge would credit, which is what the extra objective is here to
charge for.
"""

import io
import zlib

from PIL import Image


def detail(png_bytes: bytes) -> float:
    """Compressed size of the raw render, in bytes.

    Deflate over the raw RGB bytes rather than over the PNG, which is already
    deflated and would mostly measure the encoder's choices. A flat region
    compresses to almost nothing and fine detail does not, so this charges for
    detail the way a viewer meets it -- and reads the render, so no amount of
    rewriting the source can talk it down.
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return float(len(zlib.compress(img.tobytes(), 6)))


def detail_excess(reference_detail: float, candidate_png: bytes) -> float:
    """How much detail the candidate carries beyond the reference's, as a
    fraction of it. Zero when the candidate is no busier than the target.

    One-sided, because noise is the most incompressible thing an image can
    carry and a target is often noisy. Measured on the duck at working
    resolution: grain at sigma 2, barely visible, took the target's detail from
    7,607 to 55,664 -- 7.3x -- and sigma 12 to 12.7x. A symmetric distance
    would read a correct vector render as 0.9 wrong on this axis and leave
    adding incompressible speckle as the only way to improve it, so the
    regulariser would drive the artefacts it exists to suppress. Any generated
    or photographed input carries enough grain to trigger that.

    Charging only for excess also cannot be gamed the other way. An empty
    drawing scores 0 here, but it wins this axis alone while losing colour,
    edge and shape, and one win against three losses is dominated whatever the
    arity -- so the degenerate needs no separate guard.
    """
    if reference_detail <= 0.0:
        return 0.0
    return max(0.0, detail(candidate_png) - reference_detail) / reference_detail
