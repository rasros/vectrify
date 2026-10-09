"""Evidence preserves coordinate frames, optional alpha and shared boundaries."""

import numpy as np
import pytest
from PIL import Image

from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.model import Options, Work


def reference(padding=0):
    pixels = np.zeros((80 + 2 * padding, 120 + 2 * padding, 4), dtype=np.uint8)
    pixels[padding + 10 : padding + 70, padding + 20 : padding + 100] = (
        200,
        80,
        50,
        255,
    )
    pixels[padding + 20 : padding + 50, padding + 40 : padding + 70] = (
        50,
        80,
        200,
        255,
    )
    return Image.fromarray(pixels)


def test_transparent_padding_does_not_change_analysis_or_foreground_budget():
    first = collect(reference(), None, Options(), Work.start(60))
    padded = collect(reference(50), None, Options(), Work.start(60))
    assert padded.offset == (first.offset[0] + 50, first.offset[1] + 50)
    np.testing.assert_array_equal(first.target, padded.target)
    np.testing.assert_array_equal(first.labels, padded.labels)
    assert first.foreground.sum() == padded.foreground.sum()


def test_published_evidence_is_immutable_and_reused():
    first = collect(reference(), None, Options(), Work.start(60))
    second = collect(reference(), None, Options(complexity=0), Work.start(60))
    assert first is second
    with pytest.raises(ValueError, match="read-only"):
        first.labels[0, 0] = 2


def test_unmixing_preview_white_recovers_reference_color():
    pixels = np.full((40, 40, 3), (177, 147, 137), dtype=np.uint8)
    alpha = np.full((40, 40), 0.5, dtype=np.float32)
    evidence = collect(Image.fromarray(pixels), alpha, Options(), Work.start(60))
    np.testing.assert_allclose(
        evidence.rgba[0, 0], (99 / 255, 39 / 255, 19 / 255, 0.5), atol=1e-6
    )


def test_graph_has_one_owner_per_boundary_and_protected_empty_region():
    evidence = collect(reference(), None, Options(), Work.start(60))
    graph = build(evidence)
    assert graph.hidden
    assert all(graph.labels[evidence.empty] == next(iter(graph.hidden)))
    assert len({b.id for b in graph.boundaries}) == len(graph.boundaries)
    assert all(b.left != b.right for b in graph.boundaries)
    assert all(b.left == -1 or b.left < len(graph.regions) for b in graph.boundaries)
