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
