"""Complete curved bodies preserve source obligations beyond a chosen seed."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from vectrify.document import (
    Editor,
    Selection,
    export_svg,
    import_svg,
    load_project,
    save_project,
)
from vectrify.document.holes import reversed_subpath
from vectrify.document.join import curve_path, transformed_geometry
from vectrify.document.redraw import root_matrix
from vectrify.refine.cel_plan import span_body_fit
from vectrify.refine.cel_plan.ink_models import footprint
from vectrify.refine.cel_plan.line_fidelity import SourceLineGuard, SourceProfile
from vectrify.refine.cel_plan.local import _native_raster
from vectrify.refine.cel_plan.model import StageInterruptedError, Work
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.source_centerline import construct as centerline
from vectrify.refine.cel_plan.source_family import SourceFamily
from vectrify.refine.cel_plan.source_spans import SourceSpan
from vectrify.refine.cel_plan.span_restoration import construct as restoration


def fixture(*, opacity=1, transform="translate(0 0)", reverse=False):
    t = np.linspace(0, 1, 61)
    # A bent connected chain with an actual raw corner and a separate bar.
    left = np.column_stack((12 + 12 * t + np.sin(t * np.pi), 16 + 22 * t))
    right = np.column_stack((24 + 12 * t - np.sin(t * np.pi), 38 - 22 * t))
    guide = np.vstack((left, right[1:]))
    construction = centerline(guide, guide.copy(), Work.start(10))
    assert construction is not None
    field = footprint(construction.geometry, 6)
    head = (
        '<svg width="64" height="64"><g id="component" '
        f'opacity="{opacity}" transform="{transform}">'
    )
    base = '<path id="base" d="M4 4H60V60H4Z" fill="#b49150"/>'
    shadow = "M8 48H22V54H8Z"
    bar = (
        '<path id="bar" d="M12 16H36" fill="none" stroke="#060301" '
        'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    document = import_svg(
        head
        + base
        + f'<path id="owner" d="{field.path_data()} {shadow}" fill="#271b10"/>'
        + bar
        + "</g></svg>"
    )
    original = document.geometry_for("owner")
    retained = replace(
        original,
        subpaths=(*original.subpaths, *(reversed_subpath(s) for s in field.subpaths)),
    )
    floor = restoration(
        document,
        "owner",
        SourceSpan(field, retained, guide),
        "base",
        (64, 64),
        Work.start(10),
    )
    assert floor is not None
    raw = render(
        head
        + base
        + f'<path d="{shadow}" fill="#271b10"/>'
        + f'<path d="{construction.geometry.path_data()}" fill="none" '
        'stroke="#060301" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round"/>' + bar + "</g></svg>",
        (64, 64),
    )
    frame = root_matrix(document, "owner")
    native = transformed_geometry(construction.geometry, frame)
    # Translate the frozen observations and native seed consistently.
    translation = np.array(frame[4:])
    native_guide = guide + translation
    construction = replace(
        construction,
        geometry=native,
        points=native_guide,
        corners=tuple(tuple(np.asarray(c) + translation) for c in construction.corners),
    )
    profiles = [
        SourceProfile.at(native_guide[:61], 2),
        SourceProfile.at(native_guide[60:], 2),
        SourceProfile.at(np.linspace((12, 16), (36, 16), 20) + translation, 1.5),
    ]
    if reverse:
        profiles.reverse()
    guard = SourceLineGuard(raw, profiles)
    profile = SourceProfile.at(native_guide, 2)
    added = SourceLineGuard(raw, (profile,))
    evidence = SimpleNamespace(rgba=raw, source_size=(64, 64))
    return document, floor, construction, evidence, guard, added, profile


def fit(values, *, work=None):
    before, floor, construction, evidence, guard, added, profile = values
    return span_body_fit.SpanBodyFit(evidence, guard).fit(
        before,
        floor,
        "owner",
        construction,
        2,
        "#060301",
        added,
        profile,
        "bar",
        work or Work.start(30),
    )


@pytest.mark.parametrize(
    ("opacity", "transform", "reverse"),
    [(1, "translate(0 0)", False), (0.6, "translate(.35 .2)", True)],
)
def test_complete_curved_body_preserves_ports_corner_shadow_and_every_contract(
    opacity, transform, reverse
):
    values = fixture(opacity=opacity, transform=transform, reverse=reverse)
    before, floor, construction, evidence, guard, added, profile = values
    result = fit(values)
    assert result is not None
    candidate, proof = result
    assert proof["native_alpha_exact"]
    assert proof["outside_field_plus_actual_body_rgba_exact"]
    assert proof["native_body_absence"]
    assert not proof["accepted"]
    assert proof["parameter_count"] <= 24
    assert proof["evaluations"] <= 600
    assert proof["own_body"]["missing_samples"] == 0
    assert proof["joint_body"]["missing_samples"] == 0
    roles = [r["role"] for r in proof["affected_original_profiles"]]
    assert sorted(roles) == ["joint", "new-stroke", "new-stroke"]
    # Reordering source handles cannot supply a hard-coded joint profile.
    assert proof["affected_original_profiles"][0]["role"] == (
        "joint" if reverse else "new-stroke"
    )
    geometry = transformed_geometry(
        candidate.geometry_for("owner"), root_matrix(candidate, "owner")
    )
    np.testing.assert_allclose(
        [geometry.subpaths[0].nodes[i].endpoint for i in (0, -1)],
        proof["fixed_ports"],
        rtol=0,
        atol=1e-12,
    )
    for point in construction.corners:
        assert (
            min(
                np.linalg.norm(np.asarray(n.endpoint) - point)
                for n in geometry.subpaths[0].nodes
            )
            < 1e-12
        )
    assert candidate.element("owner").get("fill") == "none"
    assert candidate.element("owner").get("stroke-width") != "0"
    for oid in ("base", "bar"):
        assert candidate.element(oid) == before.element(oid)
        assert candidate.geometry_for(oid) == before.geometry_for(oid)
    residual = next(p.id for p in floor.parts if p.role == "residual")
    assert curve_path(candidate.geometry_for(residual)).area != 0
    family = SourceFamily.bind(
        before,
        candidate,
        "owner",
        floor.field,
        floor.parts,
        evidence.source_size,
        Work.start(10),
    )
    family.validate(candidate)
    loaded, _ = load_project(save_project(candidate))
    np.testing.assert_array_equal(
        _native_raster(ET.fromstring(export_svg(candidate)), (64, 64)).root,
        _native_raster(ET.fromstring(export_svg(loaded)), (64, 64)).root,
    )
    # Both banks retain all original observations after fitting.
    observed = added.source_breaks(profile)
    assert observed is not None
    assert observed.qualified.any()
    assert len(guard.original_profiles()) == 3


@pytest.mark.parametrize(
    "bad", ["ports", "corner", "closed", "width", "bar", "frame", "budget"]
)
def test_unsupported_or_unbounded_interpretations_cannot_return_a_drawing(
    bad, monkeypatch
):
    values: list[Any] = list(fixture())
    before, floor, construction, evidence, guard, added, profile = values
    if bad == "ports":
        nodes = list(construction.geometry.subpaths[0].nodes)
        nodes[0] = replace(nodes[0], values=(12.1, 16))
        values[2] = replace(
            construction,
            geometry=replace(
                construction.geometry,
                subpaths=(
                    replace(construction.geometry.subpaths[0], nodes=tuple(nodes)),
                ),
            ),
        )
    elif bad == "corner":
        values[2] = replace(construction, corners=((24, 30),))
    elif bad == "closed":
        values[2] = replace(
            construction,
            geometry=replace(
                construction.geometry,
                subpaths=(replace(construction.geometry.subpaths[0], closed=True),),
            ),
        )
    elif bad == "width":
        fitter = span_body_fit.SpanBodyFit(evidence, guard)
        assert (
            fitter.fit(
                before,
                floor,
                "owner",
                construction,
                np.nan,
                "#060301",
                added,
                profile,
                "bar",
                Work.start(10),
            )
            is None
        )
        return
    elif bad == "bar":
        editor = Editor(before, selection=Selection(whole_document=True))
        with editor.transaction("Unsupported bar") as tx:
            tx.set_attributes("bar", {"fill": "black"})
        values[0] = editor.snapshot.document
    elif bad == "frame":
        editor = Editor(floor.document, selection=Selection(whole_document=True))
        with editor.transaction("Nonuniform owner frame") as tx:
            tx.set_attributes("owner", {"transform": "scale(1 2)"})
        values[1] = replace(floor, document=editor.snapshot.document)
    else:
        monkeypatch.setattr(span_body_fit, "MAX_PARAMETERS", 0)
    assert fit(values) is None


def test_crop_cannot_drop_qualified_original_tail_even_when_seed_is_supported():
    values: list[Any] = list(fixture())
    raw = values[3].rgba.copy()
    # A separately qualified source tail starts on the new chain but exits its
    # body. White base paint and the well-supported fresh guide cannot pay it.
    extra = render(
        '<svg width="64" height="64"><path d="M12 16L56 42" '
        'stroke="black" stroke-width="2"/></svg>',
        (64, 64),
    )
    raw[extra[..., 3] > 0, :3] = 0
    tail = SourceProfile.at(np.linspace((12, 16), (56, 42), 60), 2)
    guard = SourceLineGuard(raw, (*values[4].original_profiles(), tail))
    values[4] = guard
    observed = guard.source_breaks(tail)
    assert observed is not None
    assert observed.qualified.sum() > 40
    assert fit(values) is None


def test_complete_native_check_rejects_unrelated_changes_and_interruption():
    values: list[Any] = list(fixture())
    floor = values[1]
    editor = Editor(floor.document, selection=Selection(whole_document=True))
    with editor.transaction("Unrelated material corruption") as tx:
        tx.set_attributes("base", {"fill": "#808080"})
    values[1] = replace(floor, document=editor.snapshot.document)
    assert fit(values) is None
    with pytest.raises(StageInterruptedError):
        fit(fixture(), work=Work.start(0))


def test_whole_guide_cannot_bridge_a_real_source_gap_or_use_a_copied_identity():
    values: list[Any] = list(fixture())
    profile = values[-1]
    raw = values[3].rgba.copy()
    raw[24:30, :, :3] = np.array((180, 145, 80)) / 255
    gap_guard = SourceLineGuard(raw, (profile,))
    observed = gap_guard.source_breaks(profile)
    assert observed is not None
    assert observed.gaps.any()
    values[5] = gap_guard
    assert fit(values) is None
    values = list(fixture())
    values[-1] = replace(values[-1])
    assert fit(values) is None


def test_raw_contrast_qualification_is_not_replaced_by_an_ink_support_ratio():
    values: list[Any] = list(fixture())
    construction, evidence, guard, profile = values[2], values[3], values[4], values[-1]
    raw = evidence.rgba.copy()
    weak = render(
        '<svg width="64" height="64"><path d="M4 4H60V60H4Z" fill="#828282"/>'
        f'<path d="{construction.geometry.path_data()}" fill="none" '
        'stroke="#787878" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round"/></svg>',
        (64, 64),
    )
    # Ten-byte raw contrast supplies matching support without qualifying the
    # stricter twelve-byte centre observation. It is not a real bright gap.
    raw[22:32] = weak[22:32]
    values[3] = SimpleNamespace(rgba=raw, source_size=(64, 64))
    values[4] = SourceLineGuard(raw, guard.original_profiles())
    added = SourceLineGuard(raw, (profile,))
    values[5] = added
    observed = added.source_breaks(profile)
    assert observed is not None
    assert 0.5 < observed.qualified.mean() < 0.9
    assert not observed.gaps.any()
    frozen = observed.qualified.copy()
    result = fit(values)
    assert result is not None
    _, proof = result
    np.testing.assert_array_equal(observed.qualified, frozen)
    assert proof["own_body"]["contracts"][-1]["qualified_samples"] == int(frozen.sum())
    assert proof["own_body"]["missing_samples"] == 0


def test_joint_body_uses_shared_group_opacity_once():
    document = import_svg(
        '<svg width="32" height="32"><g opacity=".4">'
        '<path id="a" d="M4 16H28" fill="none" stroke="black" stroke-width="4"/>'
        '<path id="b" d="M16 4V28" fill="none" stroke="black" stroke-width="4"/>'
        "</g></svg>"
    )
    root = span_body_fit._stroke_root(document, ("a", "b"), (32, 32), Work.start(10))
    actual = _native_raster(root, (32, 32)).root[..., 3]
    assert actual[16, 16] == actual[16, 8] == 102
    # Isolated bodies never acquire base opacity or double-blended ancestors.
    assert len(tuple(root.iter("g"))) == len(document.ancestry("a")) - 1


def test_body_preserves_actual_paint_alpha_caps_and_local_width():
    document = import_svg(
        '<svg width="32" height="32"><g opacity=".4" transform="scale(2)">'
        '<path id="a" d="M3 8H12" fill="none" stroke="rgba(0,0,0,.5)" '
        'stroke-opacity=".5" stroke-width="2" stroke-linecap="round"/>'
        "</g></svg>"
    )
    root = span_body_fit._stroke_root(document, ("a",), (32, 32), Work.start(10))
    actual = _native_raster(root, (32, 32)).root[..., 3]
    original = _native_raster(ET.fromstring(export_svg(document)), (32, 32)).root[
        ..., 3
    ]
    np.testing.assert_array_equal(actual, original)
    assert actual[16, 12] == 26
    assert actual[16, 5] > 0
