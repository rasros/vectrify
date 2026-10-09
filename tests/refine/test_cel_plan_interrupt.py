"""An interrupted export never publishes partially fitted shared boundaries."""

import time

import numpy as np
import pytest
from PIL import Image

from vectrify.refine import cel
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.export import export
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work


def test_export_checks_interruption_between_boundary_fits(monkeypatch):
    pixels = np.zeros((64, 64, 4), dtype=np.uint8)
    pixels[8:56, 8:56] = (180, 80, 50, 255)
    pixels[20:44, 20:44] = (40, 80, 180, 255)
    options = Options(palette=4, gradients=False)
    work = Work.start(10)
    evidence = collect(Image.fromarray(pixels), None, options, work)
    original = cel.curve_nodes
    calls = []

    def expire_after_one_fit(points, tolerance, **kwargs):
        calls.append(len(points))
        fitted = original(points, tolerance, **kwargs)
        work.deadline = time.monotonic() - 1
        return fitted

    monkeypatch.setattr(cel, "curve_nodes", expire_after_one_fit)
    with pytest.raises(StageInterruptedError):
        export(evidence, evidence.labels, options, work)
    assert len(calls) == 1


def test_dense_fallback_reclaims_fitting_time_only_after_valid_compaction(monkeypatch):
    from vectrify.refine.cel_plan import pipeline

    pixels = np.zeros((64, 64, 4), dtype=np.uint8)
    pixels[8:56, 8:56] = (176, 80, 48, 64)
    compact = (
        '<svg width="64" height="64"><path fill="#b05030" '
        'fill-opacity="0.250980392157" d="M8 8H56V56H8Z"/></svg>'
    )
    dense = compact.replace("M8 8H56", "M8 8L12 8L16 8L24 8H56")
    reserved = []

    def proposed(_evidence, _labels, _options, _work, **kwargs):
        return (dense if kwargs.get("conservative") else compact), {}

    def fitting(_frontier, _evidence, _options, work):
        reserved.append(work.remaining)
        return {"status": "bounded", "attempted": 0, "accepted": 0}

    monkeypatch.setattr(pipeline, "MAX_GEOMETRY_NODES", 5)
    monkeypatch.setattr(pipeline, "export", proposed)
    monkeypatch.setattr(pipeline, "refine", fitting)
    candidate = pipeline.vectorize(
        Image.fromarray(pixels), options=Options(), seconds=10
    )
    assert candidate.metrics["fitting_time_reclaimed_after_compaction"]
    assert candidate.metrics["nodes"] == 4
    assert reserved[0] > 0
    assert candidate.metrics["validation_rejections"] == []


@pytest.mark.parametrize("quality", ["balanced", "high"])
def test_high_fragmentation_defers_fitting_until_a_valid_owned_checkpoint(
    quality, monkeypatch
):
    from vectrify.refine.cel_plan import pipeline

    pixels = np.full((32, 32, 4), (176, 80, 48, 255), dtype=np.uint8)
    compact = (
        '<svg width="32" height="32"><path fill="#b05030" d="M0 0H32V32H0Z"/></svg>'
    )
    fragments = (
        '<svg width="32" height="32"><path fill="#b05030" '
        'd="M0 0H16V32H0Z"/><path fill="#b05030" '
        'd="M16 0H32V32H16Z"/></svg>'
    )
    deadlines = []

    def exported(*_args, **_kwargs):
        return fragments, {}

    def owned_checkpoint(frontier, _options, work, _operators, **kwargs):
        deadlines.append(work.deadline)
        assert frontier.add(
            compact, "Owned component", {"joint_cell_search": {"ink_fit": "source"}}
        )
        assert not kwargs["checkpoint_work"].interrupted
        return {"status": "complete", "attempted": 1, "accepted": 1, "seconds": 0}

    def fitting(_frontier, _evidence, _options, work):
        deadlines.append(work.deadline)
        return {"status": "complete", "attempted": 0, "accepted": 0}

    monkeypatch.setattr(pipeline, "MAX_EDIT_OBJECTS", 1)
    monkeypatch.setattr(pipeline, "export", exported)
    monkeypatch.setattr(pipeline, "local_search", owned_checkpoint)
    monkeypatch.setattr(pipeline, "refine", fitting)
    candidate = pipeline.vectorize(
        Image.fromarray(pixels), options=Options(quality=quality), seconds=10
    )
    assert candidate.metrics["fitting_deferred_for_fragmentation"] == (
        quality == "high"
    )
    assert candidate.metrics["fitting_time_reclaimed_after_compaction"] == (
        quality == "high"
    )
    assert len(deadlines) == 2
    assert deadlines[1] > deadlines[0]
    assert candidate.metrics["validation_rejections"] == []
    assert candidate.metrics["nodes"] == 4
