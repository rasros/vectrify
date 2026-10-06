"""A compact overlay spans shade junctions while retaining alpha topology."""

from dataclasses import replace

import numpy as np

from vectrify.refine.cel_plan.export import export
from vectrify.refine.cel_plan.layers import continued
from vectrify.refine.cel_plan.model import Evidence, Options, Work
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.strokes import strokes


def evidence():
    y, x = np.mgrid[:120, :120]
    labels = (x >= 60).astype(np.int32)
    labels[(x - 60) ** 2 + (y - 60) ** 2 <= 12**2] = 2
    palette = np.array(
        ((200, 150, 100), (140, 100, 65), (60, 150, 110)), dtype=np.float32
    )
    target = palette[labels]
    rgba = np.concatenate((target / 255, np.ones((120, 120, 1))), axis=-1).astype(
        np.float32
    )
    clear = np.zeros(labels.shape, dtype=bool)
    zeros = np.zeros(labels.shape, dtype=float)
    return Evidence(
        rgba,
        target,
        target,
        target,
        clear,
        ~clear,
        clear,
        clear,
        zeros,
        zeros,
        labels,
        (120, 120),
        (0, 0),
        (1, 1),
        None,
        False,
    )


def test_whole_compact_shape_can_span_multiple_base_shades():
    source = evidence()
    base, overlays = continued(source, source.labels, Options(), Work.start(10))
    assert len(overlays) == 1
    assert overlays[0].region == 2
    assert overlays[0].model.kind == "ellipse"
    assert set(np.unique(base)) == {0, 1}
    assert base[60, 56] == 0
    assert base[60, 64] == 1
    np.testing.assert_array_equal(
        source.labels[source.labels != 2], base[source.labels != 2]
    )
    assert set(np.unique(source.labels)) == {0, 1, 2}  # Immutable input graph.


def test_base_and_overlay_export_preserve_opaque_coverage():
    source = evidence()
    svg, details = export(
        source,
        source.labels,
        Options(gradients=False),
        Work.start(10),
        structure=True,
        layers=True,
    )
    actual = render(svg, source.source_size)
    assert actual[..., 3].min() >= 0.74
    assert details["overlay_models"][0]["model"] == "ellipse"
    assert Policy(source.rgba).evaluate(svg).valid
    assert np.linalg.norm(actual[60, 60, :3] * 255 - source.target[60, 60]) < 4


def test_a_shape_with_an_intentional_hole_stays_unrestricted():
    source = evidence()
    labels = source.labels.copy()
    labels[57:64, 57:64] = 0
    _, overlays = continued(source, labels, Options(), Work.start(10))
    assert not overlays


def test_silhouette_contact_cannot_become_an_unclipped_overlay():
    source = evidence()
    empty = np.zeros(source.labels.shape, dtype=bool)
    empty[48:73, 47:49] = True
    source = replace(source, empty=empty)
    _, overlays = continued(source, source.labels, Options(), Work.start(10))
    assert not overlays


def test_stopped_layer_planning_does_not_edit_a_working_state():
    source = evidence()
    work = Work.start(10)
    work.stop.set()
    base, overlays = continued(source, source.labels, Options(), work)
    np.testing.assert_array_equal(base, source.labels)
    assert not overlays


def test_adjacent_shades_can_propose_one_ink_supported_compact_surface():
    source = evidence()
    y, x = np.mgrid[:120, :120]
    own = source.labels == 2
    labels = source.labels.copy()
    labels[own & (x >= 60)] = 3
    target = source.target.copy()
    target[labels == 3] = (90, 180, 140)
    ring = own & ((x - 60) ** 2 + (y - 60) ** 2 >= 10**2)
    target[ring] = (10, 10, 10)
    rgba = source.rgba.copy()
    rgba[..., :3] = target / 255
    source = replace(source, labels=labels, target=target, smooth=target, rgba=rgba)
    base, overlays = continued(source, labels, Options(), Work.start(10))
    assert len(overlays) == 1
    assert set(overlays[0].members) == {2, 3}
    assert overlays[0].ink is not None
    assert set(np.unique(base)) == {0, 1}
    parts, details = strokes(source, Options(), overlays=overlays)
    assert parts  # Paired boundary evidence can recover a missed ink chain.
    assert any(item["operator"] == "overlay-ink" for item in details["ink_models"])


def test_shade_family_without_outline_evidence_is_not_silently_collapsed():
    source = evidence()
    _y, x = np.mgrid[:120, :120]
    labels = source.labels.copy()
    labels[(labels == 2) & (x >= 60)] = 3
    target = source.target.copy()
    target[labels == 3] = (90, 180, 140)
    source = replace(source, labels=labels, target=target, smooth=target)
    base, overlays = continued(source, labels, Options(), Work.start(10))
    assert not overlays
    np.testing.assert_array_equal(base, labels)
