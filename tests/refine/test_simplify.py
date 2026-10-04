"""Simplify removes the points an outline does not need, within a tolerance."""

from math import cos, pi, sin, tan

import pytest
from PIL import Image

from vectrify.document import Editor, import_svg
from vectrify.operations.generate import Region
from vectrify.refine.frozen import Paths, frozen
from vectrify.refine.simplify import curved, simplify, straightened


def circle(parts: int) -> str:
    """A circle of radius 20 at (50, 50) drawn with *parts* cubics."""
    handle = 4 / 3 * tan(pi / (2 * parts)) * 20
    data = "M70 50"
    for i in range(parts):
        a, b = 2 * pi * i / parts, 2 * pi * (i + 1) / parts
        x0, y0 = 50 + 20 * cos(a), 50 + 20 * sin(a)
        x1, y1 = 50 + 20 * cos(b), 50 + 20 * sin(b)
        data += (
            f" C{x0 - handle * sin(a):.4f} {y0 + handle * cos(a):.4f}"
            f" {x1 + handle * sin(b):.4f} {y1 - handle * cos(b):.4f}"
            f" {x1:.4f} {y1:.4f}"
        )
    return data + " Z"


def document(data: str):
    return import_svg(
        '<svg width="100" height="100" viewBox="0 0 100 100">'
        f'<path id="p" fill="#000" d="{data}"/></svg>'
    )


# One reference pixel per unit.
REGION = Region(0, 0, 100, 100, Image.new("RGB", (100, 100)))


def simplified(doc, tolerance):
    start = Paths({"p": doc.geometry_for("p")})
    result = simplify(doc, start, REGION, frozen(start), tolerance)
    return result.geometries["p"].subpaths[0].nodes


def test_a_circle_drawn_with_twice_the_curves_it_needs_loses_half():
    nodes = simplified(document(circle(8)), 0.25)
    # The closing cubic ends on the first point, which cannot go.
    assert len(nodes) <= 6
    assert all(n.command in {"M", "C"} for n in nodes)


def test_the_tolerance_decides_how_much_goes():
    doc = document(circle(8))
    assert len(simplified(doc, 0.0)) == 9
    assert len(simplified(doc, 3.0)) < len(simplified(doc, 0.25))


def test_a_square_keeps_its_corners_and_loses_the_rest():
    doc = document("M10 10 L30 10 L50 10 L50 30 L50 50 L30 50 L10 50 L10 30 Z")
    corners = [n.values for n in simplified(doc, 0.5)]
    assert sorted(corners) == [(10, 10), (10, 50), (50, 10), (50, 50)]


def test_pinned_points_stay():
    doc = document("M10 10 L30 10 L50 10 L50 50 L10 50 Z")
    middle = doc.geometry_for("p").subpaths[0].nodes[1].id
    ed = Editor(doc)
    ed.pin_node("p", middle)
    ids = [n.id for n in simplified(ed.snapshot.document, 0.5)]
    assert middle in ids


def test_pinned_endpoints_stay_through_snap_and_simplify():
    from tests.document.test_topology import contour_editor
    from vectrify.refine.snap import snap

    editor = contour_editor()
    for node in editor.snapshot.document.geometry_for("fill").subpaths[0].nodes[1:3]:
        editor.pin_node("fill", node.id)
    document = editor.snapshot.document
    paths = Paths({"fill": document.geometry_for("fill")})
    fixed = frozen(paths)
    assert len(fixed.endpoints) == 2
    before = {n.id: n for s in paths.geometries["fill"].subpaths for n in s.nodes}
    x0, y0, x1, y1 = 0, 0, *document.artboard()[2:]
    region = Region(x0, y0, x1, y1, Image.new("RGB", (100, 100), "white"))
    for changed in (
        simplify(document, paths, region, fixed, 50.0),
        snap(document, paths, region, fixed, detail=True),
    ):
        after = {n.id: n for s in changed.geometries["fill"].subpaths for n in s.nodes}
        for node_id in fixed.endpoints:
            assert after[node_id].endpoint == before[node_id].endpoint


def test_a_curve_with_its_handles_on_its_line_becomes_the_line():
    doc = document("M10 10 C20 10 30 10 40 10 C40 20 60 30 40 40 Z")
    geometry = straightened(doc.geometry_for("p"), 0.5)
    commands = [n.command for n in geometry.subpaths[0].nodes]
    assert commands == ["M", "L", "C"]


def test_lines_are_curved_with_their_handles_at_the_thirds():
    doc = document("M10 10 L40 10 L40 40 Z")
    geometry = curved(doc.geometry_for("p"))
    nodes = geometry.subpaths[0].nodes
    assert [n.command for n in nodes] == ["M", "C", "C"]
    assert nodes[1].values == (20.0, 10.0, 30.0, 10.0, 40.0, 10.0)
    assert [n.id for n in nodes] == [
        n.id for n in doc.geometry_for("p").subpaths[0].nodes
    ]


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize(
    "transform", ["matrix(1 0 0 1 0 0)", "scale(0.7 1.2) rotate(12)"]
)
def test_budget_ladder_reuses_costs_without_changing_removals(pinned, transform):
    doc = import_svg(
        '<svg width="100" height="100"><g transform="' + transform + '">'
        '<path id="p" d="'
        + circle(12)
        + ' M10 10 L20 10 L30 10 L30 30 L10 30 Z"/></g></svg>'
    )
    if pinned:
        editor = Editor(doc)
        editor.pin_node("p", doc.geometry_for("p").subpaths[0].nodes[2].id)
        doc = editor.snapshot.document
    paths = Paths({"p": doc.geometry_for("p")})
    fixed = frozen(paths)
    costs = {}
    for tolerance in (1.5, 0.75, 0.375, 0, 3, 1.5):
        expected = simplify(doc, paths, REGION, fixed, tolerance)
        actual = simplify(
            doc,
            paths,
            REGION,
            fixed,
            tolerance,
            initial_costs=costs,
            cost_bound=3,
        )
        assert actual == expected
