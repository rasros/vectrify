"""Bounded region statistics agree with full-canvas sampling."""

from dataclasses import replace

import numpy as np
from PIL import Image

from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.model import Options, Work


def test_region_boxes_preserve_statistics_and_missing_label_slots():
    pixels = np.zeros((48, 64, 4), dtype=np.uint8)
    pixels[4:44, 4:60] = (130, 100, 50, 255)
    pixels[12:30, 18:45] = (180, 160, 90, 255)
    evidence = collect(
        Image.fromarray(pixels), None, Options(palette=4), Work.start(10)
    )
    labels = np.zeros_like(evidence.labels)
    labels[8:20, 12:32] = 2  # Label 1 is deliberately absent.
    graph = build(replace(evidence, labels=labels))
    assert len(graph.regions) == 3
    assert graph.regions[1].area == 0
    assert graph.regions[1].paint == (255, 255, 255)
    for region in graph.regions:
        mask = (labels == region.id) & ~evidence.empty
        paint = mask & ~evidence.line
        selected = evidence.smooth[paint if paint.any() else mask]
        expected = np.median(selected, axis=0) if len(selected) else np.full(3, 255)
        assert region.area == int(mask.sum())
        np.testing.assert_array_equal(region.paint, expected)
        assert region.texture == (
            float(evidence.texture[mask].mean()) if mask.any() else 0
        )
