"""The metric registry, and how lineage rows are read back from it."""

from vectrify.score.metrics import (
    METRIC_NAMES,
    OBJECTIVE_NAMES,
    SCORER_METRICS,
    read_metrics,
    row_has_metrics,
)


def test_registry_covers_the_declared_metrics():
    assert (*SCORER_METRICS, "front_score") == METRIC_NAMES
    assert set(METRIC_NAMES) == {"edge", "colour", "shape", "detail", "front_score"}


def test_the_evaluator_verdict_is_recorded_but_never_an_objective():
    """It exists on a handful of nodes per epoch, and a metric absent elsewhere
    reads as 0.0 -- best possible for a minimised objective -- which would let
    every unevaluated candidate dominate every evaluated one."""
    assert "front_score" in METRIC_NAMES
    assert "front_score" not in OBJECTIVE_NAMES


def test_read_metrics_defaults_missing_columns_to_zero():
    """A row written before a metric existed has to stay readable."""
    metrics = read_metrics({"edge": "0.25"})

    assert metrics["edge"] == 0.25
    assert metrics["colour"] == 0.0
    assert set(metrics) == set(METRIC_NAMES)


def test_row_has_metrics_rejects_eviction_rows():
    """Eviction rows carry only id and evicted; reading them as metrics would
    overwrite the node's real values with zeros."""
    assert row_has_metrics({"edge": "0.25"}) is True
    assert row_has_metrics({"id": "7", "evicted": "120"}) is False
    assert row_has_metrics({"edge": "", "colour": ""}) is False
