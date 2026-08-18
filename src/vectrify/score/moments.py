"""Shape descriptors that do not care where the shape sits.

Colour distance and edge overlap are both aggregates over a set of pixels, and
both are placement-sensitive: a feature drawn correctly in the wrong place
looks as wrong to them as a feature that has come apart in the right place. On
an emblem whose target carries one continuous gold ring, a search output that
had broken the ring into sixteen dashes scored better than every seed on both,
because its colours were closer everywhere and its dashes sat nearer the right
radius than a seed whose ring was continuous but misplaced.

Ten measures of arrangement were tried against that and every one failed the
same way, ranking the intact-but-misplaced candidate worse than the
shattered-but-placed one: counting pieces per colour band, per edge contour,
per element, per tile, gaps along the target's contours, holes, longest contour
in two variants, the geometry of the error, and a coverage-gated breakage test.
Each conflated *broken* with *moved*, and misplacement is the larger signal.

Hu's invariants do not, because they are computed about the shape's own centre
and normalised by its own size, so translation and scale fall out. What remains
describes how mass is distributed within a shape, which is what changes when a
ring becomes a row of ticks and what does not change when a ring is merely in
the wrong place.

Measured against the evaluator panel on pooled candidates, this is the only one
of the eleven that carries information the other two do not: it agrees with the
panel 52.3% of the time on pairs colour and edge already get right and 52.4% on
the pairs they get wrong. Being equally accurate either way is what independence
looks like -- every other candidate metric tracked their successes and shared
their failures. Adding it lifts agreement from 61.0% to 62.3%.
"""

import numpy as np

# Below this many pixels a mask has no shape worth describing, and the third
# order moments become numerically wild.
MIN_PIXELS = 10


def hu_moments(mask: np.ndarray) -> np.ndarray:
    """Hu's seven invariants of a binary mask, log-compressed.

    Log-compressed because the raw invariants span many orders of magnitude and
    the later ones would otherwise never influence a distance.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < MIN_PIXELS:
        return np.zeros(7)

    total = float(len(xs))
    x = xs - xs.mean()
    y = ys - ys.mean()

    def normalised(p: int, q: int) -> float:
        return float((x**p * y**q).sum()) / total ** (1.0 + (p + q) / 2.0)

    n20, n02, n11 = normalised(2, 0), normalised(0, 2), normalised(1, 1)
    n30, n03 = normalised(3, 0), normalised(0, 3)
    n21, n12 = normalised(2, 1), normalised(1, 2)

    a, b = n30 + n12, n21 + n03
    raw = np.array(
        [
            n20 + n02,
            (n20 - n02) ** 2 + 4 * n11**2,
            (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2,
            a**2 + b**2,
            (n30 - 3 * n12) * a * (a**2 - 3 * b**2)
            + (3 * n21 - n03) * b * (3 * a**2 - b**2),
            (n20 - n02) * (a**2 - b**2) + 4 * n11 * a * b,
            (3 * n21 - n03) * a * (a**2 - 3 * b**2)
            - (n30 - 3 * n12) * b * (3 * a**2 - b**2),
        ]
    )
    return np.sign(raw) * np.log1p(np.abs(raw) * 1e6)


# The log-compressed invariants differ by a few units between drawings that
# look nothing alike, where colour and edge both live in [0, 1]. Ranking would
# not care, since the objective vector is normalised by the population, but the
# measure is also written per node and read back as an absolute number, and
# an unscaled term would dominate it and mean nothing to a reader.
SHAPE_SCALE = 4.0


def shape_distance(reference: np.ndarray, candidate: np.ndarray) -> float:
    """How differently the two masks distribute their mass, ignoring position.

    Scaled into roughly [0, 1] so it sits alongside colour and edge rather than
    swamping the score they share.
    """
    raw = float(np.abs(hu_moments(candidate) - hu_moments(reference)).mean())
    return min(1.0, raw / SHAPE_SCALE)
