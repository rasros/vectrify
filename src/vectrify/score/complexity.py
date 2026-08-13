"""Complexity measures used as NSGA objectives alongside visual score.

Independent measures are reported rather than one blended number: a raster
measure of how much detail the render carries, and a source measure of how much
text it takes to say it. They are separate objectives, so nothing here has to
pick a weighting between them.

METRICS is the single place a measure is declared. Adding one means adding an
entry here: the node model, the objective vector, lineage.csv, and the analysis
scripts all derive their columns from it.
"""

import io
import re
import zlib
from collections.abc import Callable, Mapping
from xml.etree import ElementTree as ET

from PIL import Image

_TAG_RE = re.compile(r"<[A-Za-z]")


def zip_complexity(png_bytes: bytes) -> float:
    """Visual complexity as the compressed size of the raw render.

    Deflate over the raw RGB bytes rather than over the PNG, which is already
    deflated and would mostly measure the encoder's choices. A flat region
    compresses to almost nothing; fine detail and many colour transitions do
    not, so this charges for detail the way a viewer perceives it and is immune
    to source-level tricks.
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return float(len(zlib.compress(img.tobytes(), 6)))


def node_complexity(source: str) -> float:
    """Structural complexity as element count.

    Counts drawn elements, not characters: a model that writes verbose
    attributes says the same thing at the same cost.

    Non-XML backends fall back to statement count, which has to be non-zero:
    a measure that reads as 0 for DOT or Typst would be the best attainable
    value for a minimised objective and would let those candidates dominate
    every SVG one.
    """
    try:
        root = ET.fromstring(source)
    except ET.ParseError:
        tags = len(_TAG_RE.findall(source))
        statements = len([line for line in source.splitlines() if line.strip()])
        return float(max(tags, statements))
    return float(sum(1 for _ in root.iter()))


# Metric name -> measure over (rendered PNG, source text). Every consumer derives
# its columns and objective ordering from this table, so a new metric is one
# entry rather than an edit in each of nine files.
#
# `score` is deliberately absent: it comes from the configured scorer, not from
# a complexity measure. It is an objective like any other -- dominance does not
# privilege it -- it just is not measured here.
#
# Be sparing. Dominance dilutes as objectives multiply, so past roughly four
# nearly every candidate is non-dominated and the Pareto front stops
# discriminating -- cheap to add is not the same as free to add.
METRICS: Mapping[str, Callable[[bytes, str], float]] = {
    "zip_complexity": lambda png, _source: zip_complexity(png),
    "node_complexity": lambda _png, source: node_complexity(source),
}

# Metrics that cannot live in METRICS because they are comparative: they need
# the reference image, which workers do not carry and which would mean shipping
# torch into every worker process. The scoring thread already holds the scorer
# and the prepared reference, so it fills these in after `measure_all` runs.
#
# They are still objectives like any other, so they belong in METRIC_NAMES and
# get a lineage column. Anything added here must be written for *every* scored
# node: a metric present on only part of the population reads as 0.0 for the
# rest, which is the best attainable value for a minimised objective and would
# make unmeasured candidates dominate measured ones.
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
# The complexity measures are recorded but are not objectives. Nothing in the
# operator set adds an element, so element count only ever falls and a ratio
# built on it carries no information the score does not already have; and with
# no complexity objective at all, an empty canvas is simply far from the target
# on all three measures rather than unbeatable on a fourth.
OBJECTIVE_NAMES: tuple[str, ...] = SCORER_METRICS

# The evaluator's verdict on a converged front member. Recorded so a run can be
# read back, and deliberately NOT an objective: it exists on a handful of nodes
# per epoch, and a metric present on only part of the population reads as 0.0
# for the rest -- the best attainable value for a minimised objective, which
# would let every unevaluated candidate dominate every evaluated one.
FRONT_SCORE = "front_score"

# Worker-side metrics first so the registry order (and therefore the objective
# vector and every derived column) stays stable for runs recorded before the
# scorer-side metrics existed.
# Every column lineage.csv carries: the raw measures plus the derived ones.
METRIC_NAMES: tuple[str, ...] = tuple(METRICS) + SCORER_METRICS + (FRONT_SCORE,)


def measure_all(png_bytes: bytes, source: str) -> dict[str, float]:
    """Evaluate every worker-side metric for one candidate.

    Excludes SCORER_METRICS, which need the reference image; the scoring thread
    adds those to the same dict once it has scored the candidate.
    """
    return {name: fn(png_bytes, source) for name, fn in METRICS.items()}


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
