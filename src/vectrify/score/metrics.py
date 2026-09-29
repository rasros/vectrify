"""The names every objective is derived from.

One place declares them, so adding a measure means adding it here rather than
editing the node model and the objective vector in turn.
"""

EDGE = "edge"
COLOUR = "colour"
SHAPE = "shape"
DETAIL = "detail"

# Four, so the majority relation can leave a pair undecided: with three, wins
# and losses cannot split evenly and every pair is comparable, which leaves the
# top tier holding one candidate and no genuine front anywhere. A 2-2 split is
# the first time two candidates can be mutually unbeaten because they are good
# at different things.
SCORER_METRICS: tuple[str, ...] = (EDGE, COLOUR, SHAPE, DETAIL)

# No weights, and nothing to blend. Selection ranks by dominance over the
# vector of measures, which compares them component by component -- so no
# measure is privileged and a weight between them would not change a single
# verdict. The only score in the run is the evaluator's, below.
# The evaluator's verdict on a converged front member. Recorded so a run can be
# read back, and deliberately NOT an objective: it exists on a handful of nodes
# per epoch, and a metric present on only part of the population reads as 0.0
# for the rest -- the best attainable value for a minimised objective, which
# would let every unevaluated candidate dominate every evaluated one.
FRONT_SCORE = "front_score"
