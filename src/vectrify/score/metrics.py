"""The names every objective and lineage column is derived from.

One place declares them, so adding a measure means adding it here rather than
editing the node model, the objective vector, lineage.csv and the analysis
scripts in turn.
"""

from collections.abc import Mapping

from vectrify.score.moments import MOMENT_WEIGHT

EDGE = "edge"
COLOUR = "colour"
SHAPE = "shape"

SCORER_METRICS: tuple[str, ...] = (EDGE, COLOUR, SHAPE)

# What selection ranks candidates by: the chromatic and structural distances,
# blended. No embedding: the round score no longer runs a model at all.
#
# It used to, weighted half and half with colour, on the strength of a
# measurement that ranked each candidate rule by how often it agreed with the
# evaluator panel. That is only as sound as the panel, and the panel then read
# whole images, which is the thing it was worst at. Against the distortion
# screen instead -- damage of a known severity in a known order, so nothing has
# to be trusted as a reference -- the ordering is different. As a share of
# level pairs ordered correctly on vector damage:
#
#     0.25 colour + 0.50 edge      96.7%    no forward pass
#     edge alone                   96.2%    no forward pass
#     colour alone                 95.8%    no forward pass
#     0.50 embedding + 0.50 colour 95.5%    one forward pass, what shipped
#     embedding at three cells     95.4%    nine forward passes
#     embedding whole              92.8%    one forward pass
#
# Adding an embedding back to the winning pair moves it by a tenth of a point
# for four to nine forward passes per candidate, which is the whole of the
# round's model cost for nothing.
#
# Edge overlap was dropped from this blend once for scoring 41.5% against the
# panel. It is the strongest single ingredient there is when measured against
# damage that is known rather than judged.
#
# Nothing counts elements or bytes. No operator adds an element, so a measure
# built on how many there are says nothing the score does not already say.
OBJECTIVE_NAMES: tuple[str, ...] = SCORER_METRICS

# Weights within the blend, from the same sweep, which searched a grid of
# 0, 0.25 and 0.5 and picked colour at 0.25 against edge at 0.50. What it chose
# is the one-to-two ratio; the pair is written normalised so the round score is
# on the scale it appears to be on, since it is recorded per node and read back
# as an absolute number. Ranking is unaffected either way -- both this and
# build_objectives are linear in the weights, so a common factor cancels.
#
# The optimum is broad: holding edge at two thirds, colour anywhere from a
# quarter to a half of it lands within a tenth of a point.
# Colour and edge keep their one-to-two ratio; the shape term takes its weight
# from what is left. See score.moments for why it earns a place and why the
# place is a small one.
COLOUR_WEIGHT = (1.0 - MOMENT_WEIGHT) / 3.0
EDGE_WEIGHT = 2.0 * (1.0 - MOMENT_WEIGHT) / 3.0
SHAPE_WEIGHT = MOMENT_WEIGHT


def round_score(colour: float, edge: float, shape: float = 0.0) -> float:
    """What a candidate is ranked by, before the population is known.

    build_objectives scales each part by its population maximum, which needs a
    population; a candidate arriving on its own needs an absolute number for
    the run's best-so-far and the lineage. Both measures already sit near
    [0, 1], so the same weights apply unscaled.
    """
    return COLOUR_WEIGHT * colour + EDGE_WEIGHT * edge + SHAPE_WEIGHT * shape


# The evaluator's verdict on a converged front member. Recorded so a run can be
# read back, and deliberately NOT an objective: it exists on a handful of nodes
# per epoch, and a metric present on only part of the population reads as 0.0
# for the rest -- the best attainable value for a minimised objective, which
# would let every unevaluated candidate dominate every evaluated one.
FRONT_SCORE = "front_score"

# Every column lineage.csv carries.
METRIC_NAMES: tuple[str, ...] = (*SCORER_METRICS, FRONT_SCORE)


def row_has_metrics(row: Mapping[str, str]) -> bool:
    """Whether a lineage.csv row actually carries metric values.

    Eviction rows are sparse: only ``id`` and ``evicted`` are set. Reading them
    as metrics would overwrite the node's real values with zeros.
    """
    return any(row.get(name) for name in METRIC_NAMES)


def read_metrics(row: Mapping[str, str]) -> dict[str, float]:
    """Pull every registered metric out of a lineage.csv row.

    Missing columns read as 0.0, so a row written before a metric existed stays
    usable.
    """
    return {name: float(row.get(name) or 0.0) for name in METRIC_NAMES}
