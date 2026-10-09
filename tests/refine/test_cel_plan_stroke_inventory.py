"""Stroke representation is distinct from raster fidelity and fill removal."""

from dataclasses import replace
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from tests.refine.test_cel_plan_source_bands import fixture
from vectrify.document import export_svg, import_svg
from vectrify.refine.cel_plan import stroke_inventory
from vectrify.refine.cel_plan.line_fidelity import SourceLineGuard, SourceProfile
from vectrify.refine.cel_plan.model import StageInterruptedError, Work
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.stroke_inventory import StrokeInventory


def line_document(extra="", *, frame="", parent="", after=""):
    document = import_svg(
        '<svg width="96" height="96">'
        f'<g><path id="ink" {frame} d="M20.25 28.25H76.25" '
        f'fill="none" stroke="#121008" stroke-width="3"/>{after}</g></svg>'
    )
    group = document.root.children[0]
    ink = group.children[0]
    ink = replace(
        ink,
        attributes=tuple(
            (dict(ink.attributes) | ET.fromstring(f"<x {extra}/>").attrib).items()
        ),
    )
    group = replace(
        group,
        attributes=tuple(
            (dict(group.attributes) | ET.fromstring(f"<x {parent}/>").attrib).items()
        ),
        children=(ink, *group.children[1:]),
    )
    return replace(document, root=replace(document.root, children=(group,)))


@pytest.mark.parametrize("diagonal", [False, True])
def test_filled_band_cannot_pass_as_editable_ink_even_when_it_matches_pixels(diagonal):
    evidence, guard, _, _, _ = fixture(diagonal=diagonal)
    frame = 'transform="matrix(.8 .6 -.6 .8 20 -4)"' if diagonal else ""
    stroke = line_document(frame=frame)
    filled = import_svg(
        f'<svg width="96" height="96"><g><path id="ink" {frame} '
        'd="M20.25 26.75H76.25V29.75H20.25Z" fill="#121008"/></g></svg>'
    )
    if not diagonal:
        assert np.array_equal(
            render(export_svg(stroke), (96, 96)), render(export_svg(filled), (96, 96))
        )
    inventory = StrokeInventory(guard)
    actual = inventory.observe(stroke, Work.start(10))
    old = inventory.observe(filled, Work.start(10))
    assert actual["qualified_samples"] > 90
    assert actual["missing_samples"] == 0
    assert actual["stroke_objects"] == ["ink"]
    assert old["missing_samples"] == old["qualified_samples"]
    assert old["stroke_objects"] == []
    assert not actual["outline_completion_proved"]
    assert evidence.source_size == (96, 96)


@pytest.mark.parametrize(
    ("extra", "parent"),
    [
        ('stroke-opacity="0"', ""),
        ('stroke="rgba(0,0,0,0)"', ""),
        ('stroke-dasharray="2 2"', ""),
        ('clip-path="url(#clip)"', ""),
        ('stroke-width="0"', ""),
        ("", 'display="none"'),
        ("", 'visibility="hidden"'),
        ("", 'opacity="0"'),
        ("", 'filter="url(#effect)"'),
    ],
)
def test_dormant_or_unsupported_strokes_supply_no_editable_support(extra, parent):
    _, guard, _, _, _ = fixture()
    document = line_document(extra=extra, parent=parent)
    report = StrokeInventory(guard).observe(document, Work.start(10))
    assert report["missing_samples"] == report["qualified_samples"]
    assert report["excluded_stroke_objects"] == ["ink"]


def test_group_opacity_is_not_counted_once_per_overlapping_stroke():
    _, guard, _, _, _ = fixture()
    copy = (
        '<path id="copy" d="M20.25 28.25H76.25" fill="none" '
        'stroke="black" stroke-width="3"/>'
    )
    document = line_document(parent='opacity=".04"', after=copy)
    report = StrokeInventory(guard).observe(document, Work.start(10))
    assert report["missing_samples"] == report["qualified_samples"]
    assert report["stroke_objects"] == ["ink", "copy"]


def test_source_queries_are_frozen_and_literal_ports_do_not_authorize_repairs():
    source = (
        '<svg width="96" height="96"><path d="M0 0H96V96H0Z" fill="#b59977"/>'
        '<path d="M20.5 28.5H48.5H76.5" fill="none" '
        'stroke="black" stroke-width="3"/></svg>'
    )
    profiles = (
        SourceProfile.at(((20.5, 28.5), (48.5, 28.5)), 3),
        SourceProfile.at(((48.5, 28.5), (76.5, 28.5)), 3),
    )
    guard = SourceLineGuard(render(source, (96, 96)), profiles)
    document = import_svg(
        source.replace(
            'd="M20.5 28.5H48.5H76.5"', 'id="a" d="M20.5 28.5H48.5"'
        ).replace(
            "</svg>",
            '<path id="b" d="M48.5 28.5H76.5" fill="none" '
            'stroke="black" stroke-width="3"/></svg>',
        )
    )
    inventory = StrokeInventory(guard)
    before = inventory.observe(document, Work.start(10))
    assert before["qualified_samples"] > 90
    assert before["missing_samples"] == 0
    assert before["literal_shared_ports"] == [
        {
            "point": [48.5, 28.5],
            "ports": [
                {"id": "a", "contour": 0, "end": "end"},
                {"id": "b", "contour": 0, "end": "start"},
            ],
            "source_profiles": [0, 1],
        }
    ]
    profiles[0].points.flags.writeable = True
    profiles[0].points[:] += 10
    assert inventory.observe(document, Work.start(10)) == before
    assert not before["outline_completion_proved"]


def test_caps_and_fractional_native_frame_are_sampled_from_actual_export():
    _, guard, _, _, _ = fixture()
    document = line_document(
        frame='transform="translate(.375 .125)"', extra='stroke-linecap="round"'
    )
    report = StrokeInventory(guard).observe(document, Work.start(10))
    alpha = render(
        '<svg width="96" height="96"><path transform="translate(.375 .125)" '
        'd="M20.25 28.25H76.25" fill="none" stroke="white" '
        'stroke-width="3" stroke-linecap="round"/></svg>',
        (96, 96),
    )[..., 3]
    from scipy.ndimage import map_coordinates

    observed = guard.source_breaks(guard.original_profiles()[0])
    queried = map_coordinates(
        alpha,
        [observed.points[:, 1] - 0.5, observed.points[:, 0] - 0.5],
        order=1,
        mode="constant",
    )
    missing = observed.qualified & ~observed.gaps & (queried < 0.05)
    assert report["profiles"][0]["missing_indices"] == np.flatnonzero(missing).tolist()


def test_bounds_and_cancellation_cannot_publish_partial_coverage(monkeypatch):
    _, guard, document, _, _ = fixture()
    monkeypatch.setattr(stroke_inventory, "MAX_NATIVE_PIXELS", 10)
    with pytest.raises(ValueError, match="pixel bound"):
        StrokeInventory(guard).observe(document, Work.start(10))
    monkeypatch.setattr(stroke_inventory, "MAX_NATIVE_PIXELS", 96**2)
    with pytest.raises(StageInterruptedError):
        StrokeInventory(guard).observe(document, Work.start(0))
    real = stroke_inventory._native_raster
    work = Work.start(10)

    def interrupted(*args):
        result = real(*args)
        work.deadline = 0
        return result

    monkeypatch.setattr(stroke_inventory, "_native_raster", interrupted)
    with pytest.raises(StageInterruptedError):
        StrokeInventory(guard).observe(document, work)
