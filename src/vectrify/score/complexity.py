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
from collections.abc import Callable, Mapping

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


# Metric name -> measure over (rendered PNG, source text). Every consumer derives
# its columns and objective ordering from this table, so a new metric is one
# entry rather than an edit in each of nine files.
#
# `score` is deliberately absent: it comes from the configured scorer, not from
# a complexity measure, and it is the constraint-gated primary objective rather
# than one of the interchangeable tiebreakers.
#
# Be sparing. Dominance dilutes as objectives multiply, so past roughly four
# nearly every candidate is non-dominated and the Pareto front stops
# discriminating -- cheap to add is not the same as free to add.
METRICS: Mapping[str, Callable[[bytes, str], float]] = {
    "visual_complexity": lambda png, _source: visual_complexity(png),
    "structural_complexity": lambda _png, source: structural_complexity(source),
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
WORST_REGION = "worst_region"

SCORER_METRICS: tuple[str, ...] = (WORST_REGION,)

# Worker-side metrics first so the registry order (and therefore the objective
# vector and every derived column) stays stable for runs recorded before the
# scorer-side metrics existed.
METRIC_NAMES: tuple[str, ...] = tuple(METRICS) + SCORER_METRICS

# Runs recorded before complexity was split into separate objectives carry a
# single blended column. It was 70% the render's JPEG size, so it is read back as
# the visual metric. Defined here so the analysis scripts do not each restate it.
LEGACY_METRIC_COLUMN = "complexity"
LEGACY_METRIC_TARGET = "visual_complexity"


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
    return any(row.get(name) for name in METRIC_NAMES) or bool(
        row.get(LEGACY_METRIC_COLUMN)
    )


def read_metrics(row: Mapping[str, str]) -> dict[str, float]:
    """Pull every registered metric out of a lineage.csv row.

    Missing columns read as 0.0, so a row written before a metric existed stays
    usable. A pre-split row is mapped through the legacy column.
    """
    metrics = {name: float(row.get(name) or 0.0) for name in METRIC_NAMES}
    if not row.get(LEGACY_METRIC_TARGET) and row.get(LEGACY_METRIC_COLUMN):
        metrics[LEGACY_METRIC_TARGET] = float(row[LEGACY_METRIC_COLUMN])
    return metrics
