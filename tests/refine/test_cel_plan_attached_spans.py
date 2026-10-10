"""Fresh attached discovery cannot borrow filled ports or nearby material edges."""

from dataclasses import replace
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from vectrify.document import Editor, Selection, export_svg, import_svg
from vectrify.document.join import curve_path
from vectrify.document.redraw import root_matrix
from vectrify.refine.cel_plan import attached_spans
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.line_fidelity import SourceLineGuard, SourceProfile
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.source_family import SourceFamily
from vectrify.refine.cel_plan.span_body_fit import SpanBodyFit

OWNER = "M10 10 C25 10 40 25 50 30 H90 V70 H50 V34 C40 29 25 14 10 14 Z"
BASE = "M10 14 C25 14 40 29 50 34 H90 V80 H10 Z"
LINE = "M10 12 C25 12 40 27 50 32"


def fixture(*, opacity=1, transform="translate(0 0)", reverse=False):
    head = (
        '<svg width="160" height="160"><g id="component" '
        f'opacity="{opacity}" transform="{transform}">'
    )
    background = '<path id="bg" d="M0 0H160V160H0Z" fill="#b49150"/>'
    base = f'<path id="base" d="{BASE}" fill="#b49150"/>'
    bar = (
        '<path id="bar" d="M10 12L50 32" fill="none" stroke="#060301" '
        'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    document = import_svg(
        head
        + background
        + base
        + f'<path id="ink" d="{OWNER}" fill="#271b10"/>'
        + bar
        + "</g></svg>"
    )
    raw = render(
        head
        + background
        + base
        + '<path d="M50 34H90V70H50Z" fill="#271b10"/>'
        + f'<path d="{LINE}" fill="none" stroke="#060301" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        + bar
        + "</g></svg>",
        (160, 160),
    )
    evidence = collect(
        Image.fromarray(np.rint(raw * 255).astype(np.uint8)),
        None,
        Options(),
        Work.start(10),
    )
    t = np.linspace(0.2, 0.8, 25)[:, None]
    points = (
        (1 - t) ** 3 * [10, 12]
        + 3 * (1 - t) ** 2 * t * [25, 12]
        + 3 * (1 - t) * t**2 * [40, 27]
        + t**3 * [50, 32]
    )
    a, b, c, d, e, f = root_matrix(document, "ink")
    points = points @ np.array(((a, b), (c, d))) + (e, f)
    profiles = [
        SourceProfile.at(points, 2),
        SourceProfile.at(
            np.linspace((10, 12), (50, 32), 20) @ np.array(((a, b), (c, d))) + (e, f),
            1.5,
        ),
    ]
    if reverse:
        profiles.reverse()
    guard = SourceLineGuard(evidence.rgba, profiles)
    part = Partition(
        (Surface("bg", (1, 2), "underlay"), Surface("base", (1,)), Surface("ink", (2,)))
    )
    return document, part, evidence, guard


@pytest.mark.parametrize(
    ("opacity", "transform", "reverse"),
    [(1, "translate(0 0)", False), (0.6, "matrix(.8 .6 -.6 .8 60 20)", True)],
)
def test_generic_discovery_selects_real_ports_original_material_and_whole_curve(
    opacity, transform, reverse
):
    document, part, evidence, guard = fixture(
        opacity=opacity, transform=transform, reverse=reverse
    )
    before = guard.original_profiles()
    discovery = attached_spans.AttachedSpans(evidence, guard)
    results = discovery.discover(document, part, "ink", Work.start(30))
    assert results
    result = results[0]
    assert result.owner == "ink"
    assert result.bar == "bar"
    assert result.proof["base_material"] == "base"
    assert result.proof["shared_base_edges"] > 0
    assert result.proof["floor"]["native_alpha_exact"]
    assert result.proof["raw_support"] >= 0.9
    assert result.proof["gaps"] == 0
    assert not result.proof["positive_body_proved"]
    assert not result.proof["accepted"]
    original_curves = {
        n.values
        for s in document.geometry_for("ink").subpaths
        for n in s.nodes
        if n.command == "C"
    }
    assert {
        n.values
        for s in result.span.field.subpaths
        for n in s.nodes
        if n.command == "C"
    } == original_curves
    np.testing.assert_array_equal(
        result.guide[[0, -1]],
        np.array(
            [result.centerline.geometry.subpaths[0].nodes[i].endpoint for i in (0, -1)]
        ),
    )
    assert not result.guide.flags.writeable
    assert not result.ink.points.flags.writeable
    assert not result.ink.paint.flags.writeable
    assert guard.original_profiles() == before
    repeated = discovery.discover(document, part, "ink", Work.start(30))
    assert repeated[0].floor.parts == result.floor.parts
    assert repeated[0].span.field.path_data() == result.span.field.path_data()


def test_automatic_owner_enumeration_can_fit_and_bind_without_a_supplied_owner():
    document, part, evidence, guard = fixture()
    discovery = attached_spans.AttachedSpans(evidence, guard)
    results = list(discovery(document, part, Work.start(30)))
    own = [r for r in results if r.owner == "ink"]
    assert own
    result = own[0]
    paint = "#" + "".join(f"{int(v):02x}" for v in np.rint(result.ink.paint))
    fitted = SpanBodyFit(evidence, guard).fit(
        document,
        result.floor,
        result.owner,
        result.centerline,
        result.ink.width,
        paint,
        result.lines,
        result.profile,
        result.bar,
        Work.start(30),
    )
    assert fitted is not None
    candidate, proof = fitted
    assert proof["own_body"]["missing_samples"] == 0
    assert proof["joint_body"]["missing_samples"] == 0
    family = SourceFamily.bind(
        document,
        candidate,
        result.owner,
        result.span.field,
        result.floor.parts,
        evidence.source_size,
        Work.start(10),
        junction=result.bar,
    )
    family.validate(candidate)
    planned = part.with_family(family)
    tables = discovery._tables(candidate, planned, Work.start(10))
    assert tables is not None
    materials, ports = tables
    assert "base" in {m.id for m in materials}
    assert "bar" in {p.id for p in ports}
    assert "ink" not in {p.id for p in ports}
    assert discovery.discover(candidate, planned, "base", Work.start(10)) == ()
    assert (
        curve_path(
            candidate.geometry_for(
                next(p.id for p in family.parts if p.role == "residual")
            )
        ).area
        > 0
    )


@pytest.mark.parametrize(
    "bad",
    [
        "filled-port",
        "foreign-parent",
        "displaced-port",
        "near-edge",
        "unsupported-material",
        "gap",
    ],
)
def test_fake_ports_nearby_material_edges_and_real_gaps_do_not_authorize_a_span(bad):
    document, part, evidence, guard = fixture()
    editor = Editor(document, selection=Selection(whole_document=True))
    if bad == "foreign-parent":
        root = ET.fromstring(export_svg(document))
        component = next(e for e in root.iter() if e.get("id") == "component")
        bar = next(e for e in component if e.get("id") == "bar")
        component.remove(bar)
        ET.SubElement(component, "g", {"id": "foreign"}).append(bar)
        document = import_svg(ET.tostring(root, encoding="unicode"))
    elif bad == "gap":
        raw = evidence.rgba.copy()
        raw[19:26, 24:34] = (180 / 255, 145 / 255, 80 / 255, 1)
        guard = SourceLineGuard(raw, guard.original_profiles())
        evidence = replace(evidence, rgba=raw)
    else:
        with editor.transaction("Invalid attached source interpretation") as tx:
            if bad == "filled-port":
                tx.set_attributes("bar", {"fill": "black", "stroke": "none"})
            elif bad == "displaced-port":
                tx.set_attributes("bar", {"transform": "translate(80 0)"})
            elif bad == "near-edge":
                tx.set_attributes("base", {"transform": "translate(.1 0)"})
            else:
                tx.set_attributes("base", {"fill-opacity": ".6"})
        document = editor.snapshot.document
    discovery = attached_spans.AttachedSpans(evidence, guard)
    assert not discovery.discover(document, part, "ink", Work.start(30))


def test_shared_face_is_discovered_from_original_profiles_and_material_curves():
    head = '<svg width="96" height="96"><g id="component" opacity=".6">'
    bg = '<path id="bg" d="M0 0H96V96H0Z" fill="#b49150"/>'
    base = '<path id="base" d="M24 20V48L40 64L56 48V10H24Z" fill="#b49150"/>'
    owner = (
        '<path id="ink" d="M10 10H24V20V48L40 64L56 48V10H60V50'
        'L40 70L20 50V20H10Z" fill="#271b10"/>'
    )
    face = '<path id="face" d="M56 48L40 64L40 40L56 30Z" fill="#bd9754"/>'
    bar = (
        '<path id="bar" d="M22 20L58 10" fill="none" stroke="#060301" '
        'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    document = import_svg(head + bg + base + owner + face + bar + "</g></svg>")
    raw = render(
        head + bg + base + face + '<path d="M10 10H24V20H10Z" fill="#271b10"/>'
        '<path d="M22 20V49L40 67L58 49V10" fill="none" stroke="#060301" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        + bar
        + "</g></svg>",
        (96, 96),
    )
    evidence = collect(
        Image.fromarray(np.rint(raw * 255).astype(np.uint8)),
        None,
        Options(),
        Work.start(10),
    )
    profiles = [
        SourceProfile.at(np.linspace((54.4, 52.6), (47.2, 59.8), 12), 2),
        SourceProfile.at(
            np.array(
                ((22.0, 20.0), (22.0, 49.0), (40.0, 67.0), (58.0, 49.0), (58.0, 10.0))
            ),
            2,
        ),
    ]
    guard = SourceLineGuard(evidence.rgba, profiles)
    part = Partition(
        (
            Surface("bg", (1, 2, 3), "underlay"),
            Surface("base", (1,)),
            Surface("ink", (2,)),
            Surface("face", (3,)),
        )
    )
    results = attached_spans.AttachedSpans(evidence, guard).discover(
        document, part, "ink", Work.start(30)
    )
    assert results
    assert any(r.proof["adjoining_material"] == "face" for r in results)
    result = next(r for r in results if r.proof["adjoining_material"] == "face")
    assert result.proof["adjoining_shared_edges"] > 0
    assert result.floor.face is not None
    assert result.floor.proof["native_alpha_exact"]
    assert result.floor.proof["outside_removed_field_rgba_exact"]
    assert not result.proof["accepted"]


@pytest.mark.parametrize(
    "bound",
    [
        "MAX_SURFACES",
        "MAX_TABLE_NODES",
        "MAX_PORT_PAIRS",
        "MAX_OBSERVATIONS",
        "MAX_FIELDS",
        "MAX_CANDIDATES",
    ],
)
def test_stage_bounds_do_not_publish_partial_construction(bound, monkeypatch):
    document, part, evidence, guard = fixture()
    monkeypatch.setattr(attached_spans, bound, 0)
    discovery = attached_spans.AttachedSpans(evidence, guard)
    assert not discovery.discover(document, part, "ink", Work.start(30))


def test_native_frame_and_interruption_are_checked_before_construction():
    document, part, evidence, guard = fixture()
    discovery = attached_spans.AttachedSpans(evidence, guard)
    with pytest.raises(StageInterruptedError):
        discovery.discover(document, part, "ink", Work.start(0))
    discovery = attached_spans.AttachedSpans(
        replace(evidence, source_size=(161, 160)), guard
    )
    with pytest.raises(ValueError, match="original native source frame"):
        discovery.discover(document, part, "ink", Work.start(10))
