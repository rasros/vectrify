"""Selected stroke curves can simplify nearby boundaries without edge labels."""

import time
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from vectrify.document import export_svg, import_svg
from vectrify.operations.generate import Region
from vectrify.refine.crossings import bezier
from vectrify.refine.frozen import Frozen, Paths, frozen
from vectrify.refine.support import supported
from vectrify.svg_render import render_image

CONTROL = np.array([[8, 16], [16, 6], [40, 6], [48, 16]], dtype=float)
REGION = Region(0, 0, 64, 64, Image.new("RGB", (64, 64)))


def drawing(*, reverse=False, transform="translate(0 0)", opacity="1", later=True):
    t = np.linspace(0, 1, 17)
    points = bezier(CONTROL, t)[0]
    points[:, 1] += np.sin(6 * np.pi * t) * 0.25
    if reverse:
        points = points[::-1]
    data = "M" + " L".join(f"{x} {y}" for x, y in points)
    data += " L8 48 L48 48 Z" if reverse else " L48 48 L8 48 Z"
    fill = f'<path id="fill" fill="blue" d="{data}"/>'
    stroke = (
        f'<path id="ink" fill="none" stroke="black" stroke-width="4" '
        f'stroke-opacity="{opacity}" d="M8 16 C16 6 40 6 48 16"/>'
    )
    return import_svg(
        '<svg width="64" height="64">'
        f'<g transform="{transform}">{fill + stroke if later else stroke + fill}</g>'
        "</svg>"
    )


def proposal(
    document,
    *,
    fixed=None,
    tolerance=1,
    accept=lambda _: True,
    deadline=float("inf"),
    region=REGION,
):
    paths = Paths({oid: document.geometry_for(oid) for oid in ("fill", "ink")})
    return supported(
        document,
        paths,
        region,
        fixed or frozen(paths),
        tolerance,
        deadline,
        accept=accept,
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_reuses_an_open_strokes_curve_in_both_directions(reverse):
    document = drawing(reverse=reverse)
    result = proposal(document)
    before = document.geometry_for("fill").subpaths[0].nodes
    after = result.geometries["fill"].subpaths[0].nodes
    assert len(after) == 4
    assert [n.command for n in after] == ["M", "C", "L", "L"]
    control = CONTROL[::-1] if reverse else CONTROL
    assert np.allclose(after[1].values, control[1:].ravel(), atol=1e-8)
    assert after[0].id == before[0].id
    assert after[1].id == before[16].id
    assert result.geometries["ink"] == document.geometry_for("ink")
    changed = document.replace_geometry(result.geometries["fill"])
    assert np.array_equal(
        np.asarray(render_image(export_svg(document), alpha=True)),
        np.asarray(render_image(export_svg(changed), alpha=True)),
    )


@pytest.mark.parametrize("held", [0, 8, 16])
def test_a_pin_or_region_hold_prevents_replacing_its_run(held):
    document = drawing()
    geometry = document.geometry_for("fill")
    node = geometry.subpaths[0].nodes[held]
    for pin in (False, True):
        if pin:
            sub = geometry.subpaths[0]
            nodes = tuple(
                replace(n, pinned=True) if n.id == node.id else n for n in sub.nodes
            )
            current = document.replace_geometry(
                replace(geometry, subpaths=(replace(sub, nodes=nodes),))
            )
            fixed = None
        else:
            current = document
            fixed = Frozen(frozenset({node.id}))
        assert proposal(current, fixed=fixed).geometries[
            "fill"
        ] == current.geometry_for("fill")


@pytest.mark.parametrize("kwargs", [{"later": False}, {"opacity": "0.5"}])
def test_does_not_infer_support_from_a_stroke_below_or_translucent(kwargs):
    document = drawing(**kwargs)
    assert proposal(document).geometries["fill"] == document.geometry_for("fill")


def test_partial_stroke_is_copied_using_exact_subcurves():
    document = drawing()
    geometry = document.geometry_for("fill")
    sub = geometry.subpaths[0]
    points = bezier(CONTROL, np.linspace(0.2, 0.8, 17))[0]
    nodes = (
        tuple(
            replace(n, values=tuple(p))
            for n, p in zip(sub.nodes[:17], points, strict=True)
        )
        + sub.nodes[17:]
    )
    document = document.replace_geometry(
        replace(geometry, subpaths=(replace(sub, nodes=nodes),))
    )
    result = proposal(document)
    after = result.geometries["fill"].subpaths[0].nodes
    assert len(after) == 4
    control = np.vstack((after[0].endpoint, np.array(after[1].values).reshape(-1, 2)))
    assert np.allclose(
        bezier(control, np.linspace(0, 1, 25))[0],
        bezier(CONTROL, np.linspace(0.2, 0.8, 25))[0],
        atol=1e-6,
    )


def test_tolerance_is_in_reference_pixels_with_transformed_paths():
    document = drawing(transform="translate(2 1) scale(.5)")
    # The 0.25-unit noise is 0.125 reference pixels after the transform.
    assert (
        len(proposal(document, tolerance=0.15).geometries["fill"].subpaths[0].nodes)
        == 4
    )
    assert proposal(document, tolerance=0.05).geometries[
        "fill"
    ] == document.geometry_for("fill")
    shifted = replace(REGION, x=-10, y=-8, width=128, height=128)
    assert (
        len(
            proposal(document, tolerance=0.08, region=shifted)
            .geometries["fill"]
            .subpaths[0]
            .nodes
        )
        == 4
    )


def test_expired_and_rejected_proposals_leave_original_geometry():
    document = drawing()
    for kwargs in ({"deadline": time.monotonic() - 1}, {"accept": lambda _: False}):
        assert proposal(document, **kwargs).geometries["fill"] == document.geometry_for(
            "fill"
        )


def test_nonmonotone_correspondence_is_not_replaced_by_a_shortcut():
    document = drawing()
    geometry = document.geometry_for("fill")
    sub = geometry.subpaths[0]
    points = bezier(CONTROL, np.array([0, 0.2, 0.8, 0.4, 1]))[0]
    nodes = (
        tuple(
            replace(n, values=tuple(p))
            for n, p in zip(sub.nodes[:5], points, strict=True)
        )
        + sub.nodes[17:]
    )
    document = document.replace_geometry(
        replace(geometry, subpaths=(replace(sub, nodes=nodes),))
    )
    assert proposal(document).geometries["fill"] == document.geometry_for("fill")


def test_explicit_closing_alias_moves_with_the_supported_endpoint():
    document = drawing()
    geometry = document.geometry_for("fill")
    contour = geometry.subpaths[0]
    # Offset the starting endpoint, then explicitly return to it before Z.
    first = replace(contour.nodes[0], values=(8, 16.2))
    closing = replace(contour.nodes[-1], id="closing", values=first.values)
    nodes = (first, *contour.nodes[1:], closing)
    document = document.replace_geometry(
        replace(geometry, subpaths=(replace(contour, nodes=nodes),))
    )
    result = proposal(document)
    after = result.geometries["fill"].subpaths[0].nodes
    assert len(after) < len(nodes)
    assert after[0].endpoint == after[-1].endpoint
    assert after[-1].id == "closing"
    assert proposal(document, fixed=Frozen(frozenset({"closing"}))).geometries[
        "fill"
    ] == document.geometry_for("fill")


@pytest.mark.parametrize("attribute", ['opacity=".5"', 'clip-path="url(#clip)"'])
def test_a_group_context_that_changes_stroke_coverage_is_excluded(attribute):
    document = drawing()
    group = document.root.children[0]
    key, value = attribute.split("=", 1)
    document = document.replace_element(
        replace(group, attributes=(*group.attributes, (key, value.strip('"'))))
    )
    assert proposal(document).geometries["fill"] == document.geometry_for("fill")


def test_a_stroke_in_another_group_is_not_a_boundary_reference():
    document = drawing()
    group = document.root.children[0]
    stroke_group = replace(group, id="stroke_group", children=(group.children[-1],))
    fill_group = replace(group, children=(group.children[0],))
    document = replace(
        document, root=replace(document.root, children=(fill_group, stroke_group))
    )
    assert proposal(document).geometries["fill"] == document.geometry_for("fill")
