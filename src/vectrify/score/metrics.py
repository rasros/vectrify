"""The names every objective and lineage column is derived from.

One place declares them, so adding a measure means adding it here rather than
editing the node model, the objective vector, lineage.csv and the analysis
scripts in turn.
"""

from collections.abc import Mapping

EDGE = "edge"
COLOUR = "colour"

SCORER_METRICS: tuple[str, ...] = (EDGE, COLOUR)

# What selection ranks candidates by: the embedding distance and the chromatic
# distance, blended.
#
# This used to trade off three measures by majority vote, on the reasoning that
# three imperfect judges agree more often than the best of them alone.
# Measured against the evaluator panel on real pool populations, that is not
# what happens. As a share of candidate pairs each rule orders the way the
# panel does, where 50% is a coin:
#
#     0.5 embedding + 0.5 colour   72.6%
#     colour alone                 64.1%
#     embedding alone              60.1%
#     majority of all three        56.1%
#     edge alone                   41.5%
#
# Edge overlap is worse than chance -- it orders candidates against the panel
# more often than with it, 28% on mascot and 30% on connect-dots -- and a
# majority cannot outvote a member that is wrong more often than right, so the
# vote landed below both of its useful members. Weighting also keeps the
# magnitude of an error, which voting throws away.
#
# Nothing counts elements or bytes. No operator adds an element, so a measure
# built on how many there are says nothing the score does not already say.
OBJECTIVE_NAMES: tuple[str, ...] = SCORER_METRICS

# How the two are weighted against each other. Equal was the best of the
# blends tried and the least tuned: 0.3/0.7 scored 72.0% and 0.7/0.3 64.0%,
# so the optimum is broad and there is nothing to be gained from fitting it
# more finely to six cases.
EMBED_WEIGHT = 0.5
COLOUR_WEIGHT = 0.5

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
