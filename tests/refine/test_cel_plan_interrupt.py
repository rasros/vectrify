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
