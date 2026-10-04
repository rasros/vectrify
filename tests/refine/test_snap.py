"""Snapping moves the selected paths' points onto the reference's edges."""

import pytest
from PIL import Image, ImageDraw

from vectrify.document import DocumentError, Editor, Selection, import_svg
from vectrify.operations import Budget, Job, OperationRequest, Permissions, method
from vectrify.operations.generate import error, render_region, target_region
from vectrify.refine import snap as snap_module
from vectrify.refine.frozen import Paths, frozen
from vectrify.refine.snap import snap

# A circle of radius 20 at (46, 46) as four cubics, under transforms that
# cancel out, so mapping to pixels has to compose them. The reference circle
# is at (48, 47.5) with radius 21.5, inside the region's margin.
K = 0.5523 * 20
CIRCLE = (
    f"M46 26 C{46 + K} 26 66 {46 - K} 66 46 C66 {46 + K} {46 + K} 66 46 66 "
    f"C{46 - K} 66 26 {46 + K} 26 46 C26 {46 - K} {46 - K} 26 46 26 Z"
)
SVG = (
    '<svg width="100" height="100" viewBox="0 0 100 100">'
    '<rect id="bg" width="100" height="100" fill="#ffffff"/>'
    '<g transform="translate(-40 -40) scale(2)"><path id="p" fill="#203050" '
    f'transform="translate(20 20) scale(0.5)" d="{CIRCLE}"/></g>'
    '<path id="line" fill="none" stroke="#000" d="M5 90 L40 90"/></svg>'
)


def reference(shape="circle"):
    image = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(image)
    if shape == "circle":
        draw.ellipse((53, 52, 139, 138), fill="#203050")
    elif shape == "strand":
        # A square with a tapering strand, like a lock of hair, leaving its
        # right side and curling up.
        draw.rectangle((50, 50, 140, 140), fill="#203050")
        draw.polygon(strand(), fill="#203050")
    elif shape == "spike":
        # A square with a long thin spike out of its right side.
        draw.rectangle((50, 50, 140, 140), fill="#203050")
        draw.polygon([(140, 85), (185, 95), (140, 105)], fill="#203050")
    elif shape == "notch":
        # A square with a thin notch cut into its right side.
        draw.rectangle((50, 50, 140, 140), fill="#203050")
        draw.polygon([(141, 83), (105, 95), (141, 107)], fill="white")
    else:
        # A square with a step in one side, which one segment cannot follow.
        draw.rectangle((50, 50, 140, 140), fill="#203050")
        draw.rectangle((140, 95, 147, 140), fill="#203050")
    return image


def strand():
    """The strand's outline: a band along a curve, from 18 px wide to a point."""
    left, right = [], []
    for i in range(33):
        t = i / 32
        u = 1 - t
        # A quadratic from the side, out to the right and up.
        x = u * u * 138 + 2 * u * t * 185 + t * t * 182
        y = u * u * 108 + 2 * u * t * 108 + t * t * 48
        dx = 2 * u * (185 - 138) + 2 * t * (182 - 185)
        dy = 2 * u * (108 - 108) + 2 * t * (48 - 108)
        size = (dx * dx + dy * dy) ** 0.5
        half = 9 * u
        left.append((x - dy / size * half, y + dx / size * half))
        right.append((x + dy / size * half, y - dx / size * half))
    return left + right[::-1]


def editor(svg=SVG, *ids):
    return Editor(import_svg(svg), selection=Selection(object_ids=frozenset(ids)))


def request(ed, ref, **settings):
    return OperationRequest(
        action="improve",
        method="nodes",
        snapshot=ed.snapshot,
        editor=ed,
        permissions=Permissions(geometry=True, structure=True),
        settings={"workers": 1, **settings},
        budget=Budget(steps=50),
        reference=ref,
    )


def snapped(svg, oid, ref, *, detail=False):
    ed = editor(svg, oid)
    req = request(ed, ref)
    region = target_region(req)
    document = ed.snapshot.document
    start = Paths({oid: document.geometry_for(oid)})
    result = snap(document, start, region, frozen(start), detail=detail)
    tx = req.transaction("snap")
    tx.reshape_path(oid, result.geometries[oid])
    before = error(render_region(document, region), region.image)
    after = error(render_region(tx.preview, region), region.image)
    return start.geometries[oid], result.geometries[oid], before, after


def ids(geometry):
    return [n.id for s in geometry.subpaths for n in s.nodes]


def test_snapping_moves_a_transformed_circle_onto_the_reference():
    start, result, before, after = snapped(SVG, "p", reference())
    assert after < 0.15 * before
    assert ids(result) == ids(start)
    assert [n.command for n in result.subpaths[0].nodes] == ["M", "C", "C", "C", "C"]


def test_pinned_points_stay_put():
    ed = editor(SVG, "p")
    first = ed.snapshot.document.geometry_for("p").subpaths[0].nodes[0].id
    ed.pin_node("p", first)
    svg = ed.snapshot.document
    region = target_region(request(ed, reference()))
    start = Paths({"p": svg.geometry_for("p")})
    result = snap(svg, start, region, frozen(start))
    before, after = start.geometries["p"], result.geometries["p"]
    assert after.subpaths[0].nodes[0] == before.subpaths[0].nodes[0]
    assert after != before


def test_stroke_only_paths_are_left_alone():
    ed = editor(SVG, "line")
    document = ed.snapshot.document
    region = target_region(request(ed, reference()))
    start = Paths({"line": document.geometry_for("line")})
    assert snap(document, start, region, frozen(start)) == start


def test_detail_splits_where_one_curve_cannot_follow_the_edge():
    square = SVG.replace(CIRCLE, "M25 25 L70 25 L70 70 L25 70 Z")
    _, fixed_count, _, plain = snapped(square, "p", reference("step"))
    start, result, before, after = snapped(square, "p", reference("step"), detail=True)
    assert len(ids(fixed_count)) == len(ids(start))
    assert len(ids(start)) < len(ids(result)) <= 2 * len(ids(start))
    assert set(ids(start)) <= set(ids(result))
    assert after < plain < before


def test_detail_cuts_a_notch_without_moving_the_side_it_is_in():
    square = SVG.replace(CIRCLE, "M25 25 L70 25 L70 70 L25 70 Z")
    _, result, before, after = snapped(square, "p", reference("notch"), detail=True)
    assert after < 0.5 * before
    points = [n.values[-2:] for n in result.subpaths[0].nodes]
    # The tip reaches well into the side, and the corners stay where they were.
    for corner in [(25, 25), (70, 25), (70, 70), (25, 70)]:
        assert min(abs(x - corner[0]) + abs(y - corner[1]) for x, y in points) < 1.5
    tips = [x for x, y in points if 30 < y < 65]
    assert tips
    assert min(tips) < 58


def test_snap_alone_is_the_proposal_and_needs_a_reference():
    nodes = method("improve", "nodes")
    ed = editor(SVG, "p")
    with pytest.raises(DocumentError, match="snap to"):
        nodes.validate(request(ed, None, shape=False, snap=True))
    job = Job(nodes, request(ed, reference(), shape=False, snap=True))
    job.run()
    result = job.state()["result"]
    assert result["changed"]
    metrics = result["metrics"]
    assert set(metrics["steps"]) == {"snap"}
    assert metrics["after"]["difference"] < 0.5 * metrics["before"]["difference"]
    before = ids(ed.snapshot.document.geometry_for("p"))
    job.apply()
    assert ids(ed.snapshot.document.geometry_for("p")) == before


def test_edge_seeking_is_part_of_the_path_fit():
    ed = editor(SVG, "p")
    job = Job(method("improve", "nodes"), request(ed, reference(), snap=True))
    job.run()
    metrics = job.state()["result"]["metrics"]
    assert set(metrics["steps"]) == {"shape"}
    assert metrics["after"]["difference"] < 0.5 * metrics["before"]["difference"]


def test_detail_creeps_along_a_curling_strand():
    # A far white dot is selected too, so the reference crop takes in the
    # whole strand.
    square = SVG.replace(CIRCLE, "M25 25 L70 25 L70 70 L25 70 Z").replace(
        '<path id="line"',
        '<path id="dot" fill="#ffffff" d="M95 15 L96 15 L96 16 Z"/><path id="line"',
    )
    ed = editor(square, "p", "dot")
    region = target_region(request(ed, reference("strand")))
    document = ed.snapshot.document
    start = Paths({"p": document.geometry_for("p")})
    plain = snap(document, start, region, frozen(start))
    result = snap(document, start, region, frozen(start), detail=True)

    def difference(paths):
        tx = request(ed, reference("strand")).transaction("snap")
        tx.reshape_path("p", paths.geometries["p"])
        return error(render_region(tx.preview, region), region.image)

    assert difference(result) < 0.5 * difference(plain)
    points = [n.values[-2:] for n in result.geometries["p"].subpaths[0].nodes]
    # The outline follows the strand up past where a straight spike would end.
    assert any(x > 85 and y < 38 for x, y in points)
    for corner in [(25, 25), (70, 25), (25, 70)]:
        assert min(abs(x - corner[0]) + abs(y - corner[1]) for x, y in points) < 1.5


def test_a_wider_search_reaches_a_spike_beyond_the_default_margin():
    square = SVG.replace(CIRCLE, "M25 25 L70 25 L70 70 L25 70 Z")

    def furthest(**settings):
        ed = editor(square, "p")
        job = Job(
            method("improve", "nodes"),
            request(
                ed, reference("spike"), shape=False, snap=True, detail=True, **settings
            ),
        )
        job.run()
        job.apply()
        geometry = ed.snapshot.document.geometry_for("p")
        return max(n.values[-2] for n in geometry.subpaths[0].nodes)

    # The spike ends at x = 92; the default crop stops about 5 units past 70.
    assert furthest() < 78
    assert furthest(margin=60) > 82


def test_detail_scores_no_more_tries_than_it_may(monkeypatch):
    square = SVG.replace(CIRCLE, "M25 25 L70 25 L70 70 L25 70 Z")
    ed = editor(square, "p")
    region = target_region(request(ed, reference("step")))
    document = ed.snapshot.document
    start = Paths({"p": document.geometry_for("p")})
    fills = []
    fill = snap_module._fill

    def counted(*args):
        fills.append(args)
        return fill(*args)

    monkeypatch.setattr(snap_module, "_fill", counted)

    def points(**options):
        fills.clear()
        result = snap(document, start, region, frozen(start), detail=True, **options)
        return len(ids(result.geometries["p"]))

    assert points() > len(ids(start.geometries["p"]))
    assert len(fills) > 2
    # No tries: detail adds nothing and scores nothing.
    assert points(tries=0) == len(ids(start.geometries["p"]))
    assert not fills
    # One try: the blob's own score and that try's, at most one split.
    assert points(tries=1) <= len(ids(start.geometries["p"])) + 3
    assert len(fills) <= 2


def test_a_snap_out_of_time_leaves_the_path_as_it_is():
    square = SVG.replace(CIRCLE, "M25 25 L70 25 L70 70 L25 70 Z")
    ed = editor(square, "p")
    region = target_region(request(ed, reference("step")))
    document = ed.snapshot.document
    start = Paths({"p": document.geometry_for("p")})
    late = snap(document, start, region, frozen(start), detail=True, deadline=0.0)
    assert late.geometries["p"] == start.geometries["p"]
