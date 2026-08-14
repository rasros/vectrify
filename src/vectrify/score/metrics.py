"""The names every objective and lineage column is derived from.

One place declares them, so adding a measure means adding it here rather than
editing the node model, the objective vector, lineage.csv and the analysis
scripts in turn.
"""

from collections.abc import Mapping

EDGE = "edge"
COLOUR = "colour"

SCORER_METRICS: tuple[str, ...] = (EDGE, COLOUR)

# What build_objectives trades off, alongside score, which is the embedding
# distance. Three measures, one of each kind: semantic, structural, chromatic.
#
# They are chosen for being wrong in different places rather than for being
# individually best. Measured one mutation from a parent, each is wrong about
# 15-20% of the time on its own but any two of them are wrong together only
# 5-7% of the time, so a majority of three calls 50% of its accepted mutations
# right where the best single measure manages 35%.
#
# There is no complexity measure among them. Nothing in the operator set adds
# an element, so element count never rises and a measure built on it says
# nothing the score does not; and with no complexity objective at all, an empty
# canvas is simply far from the target on all three rather than unbeatable on a
# fourth.
OBJECTIVE_NAMES: tuple[str, ...] = SCORER_METRICS

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
