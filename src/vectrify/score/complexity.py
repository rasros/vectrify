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


def detail_distance(reference_detail: float, candidate_png: bytes) -> float:
    """How far the candidate's detail sits from the reference's, as a fraction.

    A distance rather than something to minimise, which keeps it on the same
    footing as every other objective -- 0 is perfect and lower is better -- and
    matters for more than tidiness. Minimising detail outright would make an
    empty drawing the best attainable candidate on this axis, and with four
    objectives an empty one splits 2-2 against a good candidate and so cannot be
    dominated by it. The regulariser would then protect exactly the degenerate
    it was added to prevent. Reading it as a distance charges symmetrically for
    detail the target does not have and for detail it has that is missing.
    """
    if reference_detail <= 0.0:
        return 0.0
    return abs(detail(candidate_png) - reference_detail) / reference_detail
