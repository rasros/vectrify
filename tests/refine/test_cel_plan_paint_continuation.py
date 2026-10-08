"""Paint underneath a replacement stroke without rewriting original shadows."""

from dataclasses import replace

import numpy as np
import pathops
import pytest

from vectrify.document import (
    Element,
    export_svg,
    import_svg,
    load_project,
    save_project,
)
from vectrify.document.join import curve_path, transformed_geometry
from vectrify.document.redraw import root_matrix
from vectrify.document.svg import parse_path
from vectrify.refine.cel_plan import paint_continuation
from vectrify.refine.cel_plan.model import StageInterruptedError, Work
from vectrify.refine.cel_plan.paint_continuation import PaintContinuation
from vectrify.refine.cel_plan.score import render


def fixture(*, units="userSpaceOnUse", rule="nonzero", opacity="1", effect=""):
    return import_svg(
        '<svg width="96" height="64"><defs><linearGradient id="paint" '
        f'gradientUnits="{units}" x1="0" x2="80" y1="0" y2="0">'
        '<stop offset="0" stop-color="#b88855"/>'
        '<stop offset="1" stop-color="#553322"/></linearGradient></defs>'
        '<g id="scene" opacity="0.6"><path id="bg" d="M0 0H96V64H0Z" '
        'fill="#ccddee"/><path id="ink" transform="translate(10 0)" '
        'd="M24 10V54" fill="none" stroke="#080808" stroke-width="2" '
        'stroke-linecap="butt" stroke-linejoin="round"/>'
        f'<path id="shade" transform="translate(4 0)" {effect} '
        f'fill="url(#paint)" fill-rule="{rule}" opacity="{opacity}" '
        'd="M32 10C40 12 62 10 70 10V54H32Z M46 24H55V35H46Z"/>'
        '<path id="mark" d="M75 8H85V18H75Z" fill="#554433"/></g></svg>'
    )


def propose(document, footprint=None, **kwargs):
    return PaintContinuation().extend(
        document,
        "ink",
        footprint or parse_path("M21 10H28V54H21Z"),
        "shade",
        Work.start(10),
        **kwargs,
    )


@pytest.mark.parametrize("rule", ["nonzero", "evenodd"])
def test_patch_is_exclusive_confined_and_preserves_original_curves_paint_and_frame(
    rule,
):
    document = fixture(rule=rule)
    result = propose(document)
    assert result is not None
    candidate, witness = result
    old = document.geometry_for("shade")
    actual = candidate.geometry_for("shade")
    assert actual.subpaths[: len(old.subpaths)] == old.subpaths
    assert candidate.element("shade") == document.element("shade")
    for oid in ("bg", "ink", "mark", "paint"):
        assert candidate.element(oid) == document.element(oid)
        if oid != "paint":
            assert candidate.geometry_for(oid) == document.geometry_for(oid)
    patch = replace(actual, subpaths=actual.subpaths[len(old.subpaths) :])
    shape = curve_path(transformed_geometry(patch, root_matrix(candidate, "shade")))
    domain = curve_path(parse_path("M31 10H38V54H31Z"))
    existing = curve_path(
        transformed_geometry(old, root_matrix(document, "shade")), rule
    )
    assert shape.area > 0
    assert pathops.op(shape, domain, pathops.PathOp.DIFFERENCE).area < 1e-8
    assert pathops.op(shape, existing, pathops.PathOp.INTERSECTION).area < 1e-8
    assert witness["native_bounds"] == [31, 10, 38, 54]
    assert witness["stroke_order"] == (1, 2)
    assert [c.id for c in candidate.element("scene").children] == [
        "bg",
        "shade",
        "ink",
        "mark",
    ]
    before = render(export_svg(document), (96, 64))
    after = render(export_svg(candidate), (96, 64))
    outside = np.ones((64, 96), bool)
    outside[10:54, 31:38] = False
    np.testing.assert_array_equal(before[outside], after[outside])
    # Disjoint contours in one material retain the parent's group alpha.
    np.testing.assert_array_equal(before[..., 3], after[..., 3])
    assert not np.array_equal(before[15, 31], after[15, 31])
    roundtrip, _ = load_project(save_project(candidate))
    np.testing.assert_array_equal(after, render(export_svg(roundtrip), (96, 64)))


def test_material_holes_are_preserved_and_only_footprint_part_of_hole_is_painted():
    document = fixture(rule="evenodd")
    result = propose(document, parse_path("M38 27H48V31H38Z"))
    assert result is not None
    candidate, _ = result
    old = document.geometry_for("shade")
    new = candidate.geometry_for("shade")
    assert new.subpaths[: len(old.subpaths)] == old.subpaths
    patch = replace(new, subpaths=new.subpaths[len(old.subpaths) :])
    assert curve_path(patch).area == 32


@pytest.mark.parametrize(
    ("units", "opacity", "effect"),
    [
        ("objectBoundingBox", "1", ""),
        ("userSpaceOnUse", "0.7", ""),
        ("userSpaceOnUse", "1", 'stroke-dasharray="1 2"'),
        ("userSpaceOnUse", "1", 'stroke="black"'),
    ],
)
def test_unsupported_paint_context_cannot_change_existing_material(
    units, opacity, effect
):
    document = fixture(units=units, opacity=opacity)
    if effect:
        key, value = effect.split("=", 1)
        material = document.element("shade")
        document = document.replace_element(
            replace(
                material, attributes=(*material.attributes, (key, value.strip('"')))
            )
        )
    assert propose(document) is None


def test_shared_assets_references_and_pinned_geometry_exclude_the_whole_edit():
    from vectrify.document import Editor, Selection

    document = fixture()
    editor = Editor(document, selection=Selection(whole_document=True))
    with editor.transaction("Pinned material fixture") as tx:
        g = document.geometry_for("shade")
        tx.replace_geometry(
            "shade",
            replace(
                g,
                subpaths=tuple(
                    replace(s, nodes=tuple(replace(n, pinned=True) for n in s.nodes))
                    for s in g.subpaths
                ),
            ),
        )
    assert propose(editor.snapshot.document) is None
    shared = document.replace_element(
        replace(
            document.element("mark"), geometry_id=document.element("shade").geometry_id
        )
    )
    assert propose(shared) is None
    referenced = import_svg(
        export_svg(document).replace("</svg>", '<use href="#shade" x="3"/></svg>')
    )
    assert propose(referenced) is None


def test_missing_patch_open_fill_large_scope_and_repeated_identity_are_excluded(
    monkeypatch,
):
    document = fixture()
    assert propose(document, parse_path("M30 15H50V20H30Z")) is None
    assert propose(document, parse_path("M21 10V54")) is None
    assert propose(document, parse_path("M21 10H300V54H21Z")) is None
    candidate, _ = propose(document)
    assert propose(candidate) is None
    monkeypatch.setattr(paint_continuation, "MAX_PATCH_NODES", 3)
    assert propose(document) is None


def test_stroke_already_above_material_keeps_its_order():
    from vectrify.document import Editor, Selection

    document = fixture()
    editor = Editor(document, selection=Selection(whole_document=True))
    with editor.transaction("Order fixture") as tx:
        tx.reorder_object("ink", 3)
    document = editor.snapshot.document
    candidate, witness = propose(document)
    assert witness["stroke_order"] == (3, 3)
    assert [c.id for c in candidate.element("scene").children] == [
        c.id for c in document.element("scene").children
    ]


def test_optional_core_restricts_added_material_without_extending_the_old_footprint():
    document = fixture()
    candidate, witness = propose(document, core=parse_path("M21 20H28V30H21Z"))
    old = document.geometry_for("shade")
    new = candidate.geometry_for("shade")
    patch = replace(new, subpaths=new.subpaths[len(old.subpaths) :])
    shape = curve_path(transformed_geometry(patch, root_matrix(candidate, "shade")))
    assert shape.area == 50
    assert shape.bounds == (31, 20, 36, 30)
    assert witness["opaque_core"]
    assert propose(document, core=parse_path("M1 1H2V2H1Z")) is None
    assert propose(document, core=parse_path("M21 20V30")) is None


def test_alpha_proof_rejects_unrestricted_restoration_at_a_translucent_outer_edge():
    document = import_svg(
        '<svg width="64" height="64"><g opacity="0.6">'
        '<path id="base" d="M16.5 10H50V54H16.5Z" fill="#aabbcc"/>'
        '<path id="ink" d="M16.25 10V54" fill="none" stroke="black" '
        'stroke-width="2" stroke-linecap="butt" stroke-linejoin="round"/>'
        '<path id="shade" d="M22 10H50V54H22Z" fill="#aa7744"/></g></svg>'
    )
    footprint = parse_path("M12 10H22V54H12Z")
    naive, _ = propose(document, footprint)
    safe, _ = propose(document, footprint, core=parse_path("M18 10H22V54H18Z"))
    before = render(export_svg(document), (64, 64))
    naive_pixels = render(export_svg(naive), (64, 64))
    safe_pixels = render(export_svg(safe), (64, 64))
    assert (naive_pixels[..., 3] > before[..., 3] + 1 / 255).any()
    np.testing.assert_array_equal(before[..., 3], safe_pixels[..., 3])
    assert not np.array_equal(before[20, 19], safe_pixels[20, 19])


def test_unsupported_parent_and_interruption_never_return_partial_work():
    document = fixture()
    work = Work.start(0)
    with pytest.raises(StageInterruptedError):
        PaintContinuation().extend(
            document, "ink", parse_path("M21 10H28V54H21Z"), "shade", work
        )
    scene = document.element("scene")
    nested = document.replace_element(
        replace(
            scene,
            children=tuple(
                Element("nested", "g", children=(child,))
                if child.id == "shade"
                else child
                for child in scene.children
            ),
        )
    )
    assert propose(nested) is None
    with pytest.raises(ValueError, match="supported fill rule"):
        propose(document, rule="unsupported")
