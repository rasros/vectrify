"""Optimize nodes reshapes only the selected paths and proposes one ordinary edit."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from vectrify.document import DocumentError, Editor, Selection, export_svg, import_svg
from vectrify.operations import Budget, Job, OperationRequest, Permissions, method
from vectrify.operations.methods import nodes as nodes_method
from vectrify.operations.settings import read_settings
from vectrify.svg_render import render_image

# The square is drawn with more points than it needs and a little off target.
SVG = (
    '<svg width="64" height="64"><rect id="bg" width="64" height="64" '
    'fill="#ffffff"/><path id="p" fill="#000000" d="M14 14 L24 14 L34 14 '
    'L44 14 L50 14 L50 32 L50 50 L32 50 L14 50 L14 32 Z"/>'
    '<rect id="other" x="2" y="2" width="6" height="6" fill="#ff0000"/></svg>'
)


def reference():
    image = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(image).rectangle((16, 16, 47, 47), fill="black")
    return image


def editor(*ids):
    return Editor(import_svg(SVG), selection=Selection(object_ids=frozenset(ids)))


@pytest.fixture
def low_resolution(monkeypatch):
    """Fit paths at a small size, quick enough for these drawings."""
    monkeypatch.setattr(nodes_method, "RESOLUTION", 64)


# The path fit on its own unless a test says otherwise, as the tests were
# written against it.
FIT = {"shape": True, "snap": False, "detail": False, "simplify": False}


def request(ed, *, steps=4, with_reference=True, **settings):
    return OperationRequest(
        action="improve",
        method="nodes",
        snapshot=ed.snapshot,
        editor=ed,
        permissions=Permissions(geometry=True, structure=True, paint=True),
        settings={"steps": 30, **FIT, **settings},
        budget=Budget(steps=steps),
        reference=reference() if with_reference else None,
    )


def nodes_of(document):
    return [n.id for s in document.geometry_for("p").subpaths for n in s.nodes]


def test_validation_names_what_is_missing():
    nodes = method("improve", "nodes")
    with pytest.raises(DocumentError, match="Select the paths"):
        nodes.validate(request(editor()))
    with pytest.raises(DocumentError, match="Select the paths"):
        nodes.validate(request(editor("bg")))
    with pytest.raises(DocumentError, match="at least one"):
        nodes.validate(request(editor("p"), shape=False))
    with pytest.raises(DocumentError, match="fit the shape to"):
        nodes.validate(request(editor("p"), with_reference=False))
    with pytest.raises(DocumentError, match="snap to"):
        nodes.validate(
            request(editor("p"), with_reference=False, shape=False, snap=True)
        )
    ed = editor("p")
    narrow = OperationRequest(
        "improve",
        "nodes",
        ed.snapshot,
        ed,
        Permissions(geometry=True),
        settings={"shape": False, "simplify": True},
        reference=reference(),
    )
    with pytest.raises(DocumentError, match="Allow structure"):
        nodes.validate(narrow)


@pytest.mark.usefixtures("low_resolution")
def test_fitting_moves_only_the_selected_path_and_keeps_its_node_ids():
    ed = editor("p")
    original = ed.snapshot.document
    job = Job(method("improve", "nodes"), request(ed))
    job.run()
    state = job.state(preview=True)
    assert state["status"] == "ready", state
    result = state["result"]
    assert result["changed"]
    after, before = result["metrics"]["after"], result["metrics"]["before"]
    assert after["difference"] < before["difference"]
    assert set(result["previews"]) == {"reference", "before", "after"}
    job.apply()
    assert ed.undo_labels == ("Optimize nodes",)
    document = ed.snapshot.document
    assert nodes_of(document) == nodes_of(original)
    assert document.element("other") == original.element("other")
    assert document.element("bg") == original.element("bg")


@pytest.mark.usefixtures("low_resolution")
def test_tidy_fits_repeated_closing_knots_without_opening_spikes():
    # Like arena's path 425: several redundant knots after the explicit
    # closing curve. These are separate IDs, but all describe one corner.
    svg = (
        '<svg width="64" height="64"><rect width="64" height="64" fill="white"/>'
        '<path id="p" fill="#68826d" d="M20 20 L24 24 L34 20 L33 26 '
        "L42 27 L36 34 C30 33 24 33 18 35 C16 31 20 20 20 20 "
        'L20 20 L20 20 L20 20 Z"/></svg>'
    )
    document = import_svg(svg)
    ed = Editor(document, selection=Selection(object_ids=frozenset({"p"})))
    original = document.geometry_for("p").subpaths[0].nodes
    closed = [node.id for node in original if node.endpoint == original[0].endpoint]
    target = render_image(
        svg.replace('id="p"', 'id="p" transform="translate(1 -1)"'),
        (0, 0, 64, 64),
        (64, 64),
    )
    req = replace(request(ed, steps=2, snap=True), reference=target)
    job = Job(method("improve", "nodes"), req)
    job.run()
    result = job.state()["result"]
    assert result["changed"]
    assert (
        result["metrics"]["after"]["difference"]
        < (result["metrics"]["before"]["difference"])
    )
    job.apply()
    geometry = ed.snapshot.document.geometry_for("p")
    corner = geometry.node(closed[0]).endpoint
    assert all(geometry.node(node).endpoint == corner for node in closed)
    assert nodes_of(ed.snapshot.document) == nodes_of(document)
    from vectrify.refine.crossings import crossings

    assert crossings(geometry) == 0


@pytest.mark.usefixtures("low_resolution")
def test_tidy_smoothing_keeps_a_supported_unmarked_notch():
    pytest.importorskip("torch")
    source = (
        '<svg width="64" height="64"><rect width="64" height="64" fill="white"/>'
        '<path id="p" fill="black" d="M8 16 L16 16 L18 18 L20 16 L28 16 '
        'L32 24 L36 16 L42 16 L44 18 L46 16 L56 16 L56 56 L8 56 Z"/></svg>'
    )
    target = source.replace("L18 18", "L18 16").replace("L44 18", "L44 16")
    document = import_svg(source)
    ed = Editor(document, selection=Selection(object_ids=frozenset({"p"})))
    req = replace(
        request(ed, steps=2, detail=True, snap=True),
        reference=render_image(target, (0, 0, 64, 64), (64, 64)),
    )
    job = Job(method("improve", "nodes"), req)
    job.run()
    result = job.state()["result"]
    assert result["changed"]
    job.apply()
    old = document.geometry_for("p").subpaths[0].nodes
    geometry = ed.snapshot.document.geometry_for("p")
    assert {n.id for n in old} <= {n.id for s in geometry.subpaths for n in s.nodes}
    assert geometry.node(old[5].id).endpoint[1] >= 23.5
    assert geometry.node(old[2].id).endpoint[1] < 16.5
    assert geometry.node(old[8].id).endpoint[1] < 16.5


@pytest.mark.usefixtures("low_resolution")
def test_simplify_without_a_reference_removes_points_and_keeps_the_look():
    ed = editor("p")
    job = Job(
        method("improve", "nodes"),
        request(ed, with_reference=False, shape=False, simplify=True, tolerance=0.5),
    )
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    metrics = state["result"]["metrics"]
    # The square's in-between points go, and its four corners stay.
    assert metrics["after"]["nodes"] == 4
    assert metrics["after"]["difference"] < 1e-4
    job.apply()
    assert len(nodes_of(ed.snapshot.document)) == metrics["after"]["nodes"]


@pytest.mark.usefixtures("low_resolution")
def test_combined_tidy_reduces_error_without_adding_nodes():
    ed = editor("p")
    job = Job(
        method("improve", "nodes"),
        request(ed, steps=8, snap=True, simplify=True, tolerance=0.5),
    )
    job.run()
    metrics = job.state()["result"]["metrics"]
    assert {"shape", "snap"} & set(metrics["steps"])
    # The local reference budget may reject simplification of the fitted
    # curves. The default run still improves the fit without adding nodes.
    assert metrics["after"]["nodes"] <= metrics["before"]["nodes"]
    assert metrics["after"]["difference"] < 0.5 * metrics["before"]["difference"]


@pytest.mark.usefixtures("low_resolution")
def test_the_fit_gives_straight_segments_handles_where_the_reference_curves():
    ed = editor("p")
    disc = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(disc).ellipse((12, 12, 52, 52), fill="black")
    req = OperationRequest(
        action="improve",
        method="nodes",
        snapshot=ed.snapshot,
        editor=ed,
        permissions=Permissions(geometry=True, structure=True, paint=True),
        settings={
            "steps": 40,
            "movement": 6.0,
            **FIT,
        },
        budget=Budget(steps=3),
        reference=disc,
    )
    job = Job(method("improve", "nodes"), req)
    job.run()
    result = job.state()["result"]
    metrics = result["metrics"]
    assert metrics["after"]["difference"] < metrics["before"]["difference"]
    job.apply()
    commands = [
        n.command for n in ed.snapshot.document.geometry_for("p").subpaths[0].nodes
    ]
    assert "C" in commands


@pytest.mark.parametrize(
    ("shape", "snap", "simplify", "detail"),
    [
        (True, False, False, False),
        (True, True, False, True),
        (False, True, False, True),
        (False, False, True, False),
    ],
)
@pytest.mark.usefixtures("low_resolution")
def test_tidy_does_not_add_handles_to_pinned_nodes(shape, snap, simplify, detail):
    ed = editor("p")
    pins = ed.snapshot.document.geometry_for("p").subpaths[0].nodes[:2]
    for node in pins:
        ed.pin_node("p", node.id)
    job = Job(
        method("improve", "nodes"),
        request(
            ed,
            steps=2,
            shape=shape,
            snap=snap,
            simplify=simplify,
            detail=detail,
            seconds=1.0,
        ),
    )
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    if state["result"]["changed"]:
        job.apply()
    nodes = ed.snapshot.document.geometry_for("p").subpaths[0].nodes
    for original in pins:
        i = next(i for i, node in enumerate(nodes) if node.id == original.id)
        point = nodes[i]
        assert point.pinned
        assert point.endpoint == original.endpoint
        if point.command == "C":
            assert point.values[2:4] == point.endpoint
        if i + 1 < len(nodes) and nodes[i + 1].command == "C":
            assert nodes[i + 1].values[:2] == point.endpoint


def test_the_defaults_are_a_quick_tidy():
    settings = read_settings({}, nodes_method.SETTINGS, nodes_method.LABEL)
    # As the Tidy dialog has them: fit with detail, keeping every point.
    steps = ("shape", "snap", "detail", "simplify")
    assert tuple(settings[s] for s in steps) == (True, True, True, False)
    assert settings["steps"] <= 60
    assert settings["seconds"] == 10


@pytest.mark.parametrize(
    ("given", "with_reference", "expected"),
    [
        ({}, True, (True, True, True, False)),
        # Without a reference Simplify alone, as the dialog does.
        ({}, False, (False, False, False, True)),
        # Simplify alone needs no detail=False.
        (
            {"shape": False, "snap": False, "simplify": True},
            True,
            (False,) * 3 + (True,),
        ),
        ({"detail": False, "simplify": True}, True, (True, True, False, True)),
    ],
)
def test_default_steps_follow_the_request(given, with_reference, expected):
    nodes = method("improve", "nodes")
    prepared = nodes.prepare(
        replace(request(editor("p"), with_reference=with_reference), settings=given)
    )
    steps = ("shape", "snap", "detail", "simplify")
    assert tuple(prepared.settings[s] for s in steps) == expected
    nodes.validate(prepared)


@pytest.mark.usefixtures("low_resolution")
def test_detail_fit_keeps_source_points_and_adds_local_curve_controls():
    pytest.importorskip("torch")
    svg = (
        '<svg width="64" height="64"><path id="p" fill="black" '
        'd="M12 12 L52 12 L52 52 L12 52 Z"/></svg>'
    )
    original = import_svg(svg)
    ed = Editor(original, selection=Selection(object_ids=frozenset({"p"})))
    target = render_image(
        svg.replace("L52 12", "C24 16 40 16 52 12"), (0, 0, 64, 64), (64, 64)
    )
    req = replace(request(ed, steps=2, snap=True, detail=True), reference=target)
    job = Job(method("improve", "nodes"), req)
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    assert state["result"]["changed"]
    job.apply()
    after = ed.snapshot.document
    before_ids = {n.id for s in original.geometry_for("p").subpaths for n in s.nodes}
    after_ids = {n.id for s in after.geometry_for("p").subpaths for n in s.nodes}
    assert before_ids < after_ids
    assert nodes_method.crossings(after, ["p"]) == {"p": 0}
    assert (
        state["result"]["metrics"]["after"]["difference"]
        < state["result"]["metrics"]["before"]["difference"]
    )


@pytest.mark.usefixtures("low_resolution")
@pytest.mark.parametrize("amount", [-2, 2])
@pytest.mark.parametrize("in_view", [False, True])
def test_tidy_fits_white_shapes_using_reference_opacity(amount, in_view):
    svg = (
        '<svg width="64" height="64"><path id="p" fill="white" '
        'd="M14 14 L50 14 L50 50 L14 50 Z"/></svg>'
    )
    document = import_svg(svg)
    geometry = document.geometry_for("p")
    expected = document.replace_geometry(
        replace(
            geometry,
            subpaths=tuple(
                replace(
                    s,
                    nodes=tuple(
                        replace(
                            n,
                            values=tuple(
                                v + (amount if v == 14 else -amount) for v in n.values
                            ),
                        )
                        for n in s.nodes
                    ),
                )
                for s in geometry.subpaths
            ),
        )
    )
    image = render_image(export_svg(expected), alpha=True)
    ed = Editor(document, selection=Selection(object_ids=frozenset({"p"})))
    settings: dict[str, bool | int | list[int]] = {
        "shape": True,
        "snap": False,
        "simplify": False,
        "steps": 30,
    }
    if in_view:
        settings["region"] = [8, 8, 48, 48]
    job = Job(
        method("improve", "nodes"),
        OperationRequest(
            "improve",
            "nodes",
            ed.snapshot,
            ed,
            Permissions(geometry=True, structure=True),
            settings=settings,
            budget=Budget(steps=1),
            reference=image,
        ),
    )
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    metrics = state["result"]["metrics"]
    assert state["result"]["changed"]
    assert metrics["after"]["difference"] < metrics["before"]["difference"] * 0.7


def test_simplify_removes_points_only_within_the_allowance():
    # A disc drawn with 48 points, matching the reference's disc: removing
    # points within 3 px takes most of them but bends the outline.
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    points = " L".join(
        f"{32 + 20 * np.cos(a):.3f} {32 + 20 * np.sin(a):.3f}" for a in angles
    )
    disc = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(disc).ellipse((12, 12, 52, 52), fill="black")

    def tidied(**settings):
        ed = Editor(
            import_svg(
                f'<svg width="64" height="64"><path id="p" fill="#000000" '
                f'd="M{points} Z"/></svg>'
            ),
            selection=Selection(object_ids=frozenset({"p"})),
        )
        req = OperationRequest(
            action="improve",
            method="nodes",
            snapshot=ed.snapshot,
            editor=ed,
            permissions=Permissions(geometry=True, structure=True),
            settings={
                "snap": False,
                "shape": False,
                "simplify": True,
                "tolerance": 3.0,
                **settings,
            },
            budget=Budget(steps=1),
            reference=disc,
        )
        job = Job(method("improve", "nodes"), req)
        job.run()
        return job.state()["result"]["metrics"]

    loose, tight = tidied(allowance=100.0), tidied(allowance=0.5)
    assert loose["after"]["nodes"] < tight["after"]["nodes"]
    assert tight["after"]["difference"] <= tight["before"]["difference"] * 1.01
    assert loose["after"]["difference"] > tight["after"]["difference"]


@pytest.mark.usefixtures("low_resolution")
def test_a_region_tidies_only_the_points_inside_it():
    # Nothing selected: the region picks the path painting inside it, and
    # only its points left of x = 30 may move.
    ed = editor()
    original = {
        n.id: n.values
        for s in ed.snapshot.document.geometry_for("p").subpaths
        for n in s.nodes
    }
    job = Job(
        method("improve", "nodes"),
        request(
            ed,
            shape=False,
            snap=True,
            simplify=False,
            region=[0, 0, 30, 64],
        ),
    )
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    assert state["result"]["changed"]
    job.apply()
    after = {
        n.id: n.values
        for s in ed.snapshot.document.geometry_for("p").subpaths
        for n in s.nodes
    }
    assert after.keys() == original.keys()
    moved = {i for i in after if after[i] != original[i]}
    assert moved
    assert all(original[i][-2] < 30 for i in moved)


@pytest.mark.usefixtures("low_resolution")
def test_a_region_with_nothing_to_tidy_says_so():
    with pytest.raises(DocumentError, match="inside the region"):
        method("improve", "nodes").validate(
            request(editor(), shape=False, snap=True, region=[56, 56, 6, 6])
        )


# Two regions meeting along x = 32 through points at y = 24 and 40, the
# right one drawn back to its start as a cel trace is; in the reference
# they meet at x = 36 between those points.
NEIGHBOURS = (
    '<svg width="64" height="64">'
    '<rect width="64" height="64" fill="#00ff00"/>'
    '<path id="left" fill="#ff0000" '
    'd="M8 8 L32 8 L32 24 L32 40 L32 56 L8 56 Z"/>'
    '<path id="right" fill="#0000ff" '
    'd="M32 8 L56 8 L56 56 L32 56 L32 40 L32 24 L32 8 Z"/>'
    "</svg>"
)


def neighbours_reference():
    image = Image.new("RGB", (64, 64), "#00ff00")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 55, 55), fill="#0000ff")
    draw.polygon(
        [(8, 8), (32, 8), (36, 24), (36, 40), (32, 56), (8, 56)], fill="#ff0000"
    )
    return image


@pytest.mark.parametrize("selected", [("left",), ("left", "right")])
def test_tidy_keeps_straighten_handles_enabled_on_shared_points(selected):
    from vectrify.refine.simplify import curved

    document = import_svg(NEIGHBOURS)
    checked = {}
    for oid in ("left", "right"):
        geometry = curved(document.geometry_for(oid))
        point = next(
            n for s in geometry.subpaths for n in s.nodes if n.endpoint == (32, 24)
        )
        checked[oid] = point.id
        document = document.replace_geometry(
            geometry.replace_node(replace(point, handles_aligned=True))
        )
    ed = Editor(document, selection=Selection(object_ids=frozenset(selected)))
    job = Job(
        method("improve", "nodes"),
        OperationRequest(
            "improve",
            "nodes",
            ed.snapshot,
            ed,
            Permissions(geometry=True, structure=True),
            settings={"shape": False, "simplify": False},
            budget=Budget(steps=2),
            reference=neighbours_reference(),
        ),
    )
    job.run()
    assert job.state()["result"]["changed"]
    job.apply()
    for oid, node_id in checked.items():
        point = ed.snapshot.document.geometry_for(oid).node(node_id)
        assert point.endpoint != (32, 24)
        assert point.handles_aligned
    ed.undo()
    assert ed.snapshot.document == document
    ed.redo()
    assert all(
        ed.snapshot.document.geometry_for(oid).node(node_id).handles_aligned
        for oid, node_id in checked.items()
    )


def test_shared_edges_follow_when_both_paths_are_selected():
    ed = Editor(
        import_svg(NEIGHBOURS),
        selection=Selection(object_ids=frozenset({"left", "right"})),
    )
    job = Job(
        method("improve", "nodes"),
        OperationRequest(
            "improve",
            "nodes",
            ed.snapshot,
            ed,
            Permissions(geometry=True, structure=True),
            settings={"shape": False, "simplify": False},
            budget=Budget(steps=2),
            reference=neighbours_reference(),
        ),
    )
    job.run()
    result = job.state()["result"]
    assert result["changed"]
    job.apply()

    def edge(oid):
        return sorted(
            n.values[-2:]
            for s in ed.snapshot.document.geometry_for(oid).subpaths
            for n in s.nodes
            if 16 < n.values[-1] < 48
        )

    assert edge("left") == edge("right")
    assert all(x > 33 for x, _y in edge("left"))
    assert result["metrics"]["followed"] == 0


@pytest.mark.usefixtures("low_resolution")
@pytest.mark.parametrize("midpoint", [False, True])
def test_repeated_tidy_retains_an_improving_overlap_between_selected_layers(midpoint):
    pytest.importorskip("torch")
    svg = (
        '<svg width="64" height="64"><g id="pair">'
        '<path id="back" fill="white" d="M8 8 L56 8 L56 56 L32 32 L8 8 Z"/>'
        '<path id="front" fill="navy" d="M8 8 L32 32 L56 56 L8 56 L8 8 Z"/>'
        "</g></svg>"
    )
    if not midpoint:
        svg = svg.replace("L32 32 ", "")
    target = render_image(
        svg.replace(
            "L56 56 " + ("L32 32 " if midpoint else "") + "L8 8",
            "L56 56 L8 56 L8 8",
        ),
        (0, 0, 64, 64),
        (64, 64),
        alpha=True,
    )
    original = import_svg(svg)
    ed = Editor(original, selection=Selection(object_ids=frozenset({"pair"})))
    job = Job(
        method("improve", "nodes"),
        OperationRequest(
            "improve",
            "nodes",
            ed.snapshot,
            ed,
            Permissions(geometry=True, structure=True),
            settings={
                "snap": False,
                "simplify": False,
                "steps": 30,
                "seconds": 20,
            },
            budget=Budget(steps=2),
            reference=target,
        ),
    )
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    assert state["result"]["changed"]
    job.apply()
    result = ed.snapshot.document
    before = np.asarray(
        render_image(export_svg(original), (0, 0, 64, 64), (64, 64), alpha=True)
    )
    after = np.asarray(
        render_image(export_svg(result), (0, 0, 64, 64), (64, 64), alpha=True)
    )
    interior = (slice(10, 54), slice(10, 54), 3)
    assert ((255.0 - after[interior]) ** 2).sum() < (
        (255.0 - before[interior]) ** 2
    ).sum() * 0.3
    for oid in ("back", "front"):
        assert result.element(oid).attributes == original.element(oid).attributes
    assert (
        state["result"]["metrics"]["after"]["difference"]
        < state["result"]["metrics"]["before"]["difference"]
    )


@pytest.mark.usefixtures("low_resolution")
def test_region_holds_shared_junction_handles_as_well_as_endpoints():
    pytest.importorskip("torch")
    svg = (
        '<svg width="64" height="64"><g id="pair">'
        '<path id="back" fill="white" d="M8 8 L56 8 L56 56 L8 8 Z"/>'
        '<path id="front" fill="navy" d="M8 8 L56 56 L8 56 L8 8 Z"/>'
        "</g></svg>"
    )
    original = import_svg(svg)
    ed = Editor(original, selection=Selection(object_ids=frozenset({"pair"})))
    target = render_image(
        svg.replace("L56 56 L8 8", "L56 56 L8 56 L8 8"),
        (0, 0, 64, 64),
        (64, 64),
        alpha=True,
    )
    job = Job(
        method("improve", "nodes"),
        replace(request(ed, steps=2, region=[16, 16, 32, 32]), reference=target),
    )
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    # Every endpoint lies outside the user region, so none of its controls
    # may bend even though the shared edge crosses the region's interior.
    assert not state["result"]["changed"]
    assert ed.snapshot.document == original


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shape", [True, False])
@pytest.mark.parametrize("shared", [True, False])
def test_a_neighbour_s_shared_edge_moves_with_the_path(
    shared, shape, device, monkeypatch
):
    from vectrify.refine import selected

    if device == "cuda":
        problem = selected.gpu_problem()
        if problem:
            pytest.skip(problem)
    monkeypatch.setattr(selected, "fit_device", lambda: device)
    ed = Editor(
        import_svg(NEIGHBOURS), selection=Selection(object_ids=frozenset({"left"}))
    )
    req = OperationRequest(
        action="improve",
        method="nodes",
        snapshot=ed.snapshot,
        editor=ed,
        permissions=Permissions(geometry=True, structure=True),
        settings={
            "simplify": False,
            "shared": shared,
            "shape": shape,
        },
        budget=Budget(steps=2),
        reference=neighbours_reference(),
    )
    job = Job(method("improve", "nodes"), req)
    job.run()
    result = job.state()["result"]
    assert result["changed"]
    job.apply()
    document = ed.snapshot.document

    def edge(oid):
        return sorted(
            n.values[-2:]
            for s in document.geometry_for(oid).subpaths
            for n in s.nodes
            if 16 < n.values[-1] < 48
        )

    left = edge("left")
    assert left
    metrics = result["metrics"]
    assert metrics["after"]["difference"] < metrics["before"]["difference"]
    # With sharing off, the later blue path hides this boundary. The shape
    # fit can win by improving another visible edge without moving this one;
    # CPU and CUDA need not choose the same step. Snap alone still moves it.
    if shared or not shape:
        assert all(x > 33 for x, _ in left)
    if shared:
        # The two meet where they did, the corners where three meet staying.
        assert edge("right") == left
        assert result["metrics"]["followed"] == 1
    else:
        assert edge("right") == [(32.0, 24.0), (32.0, 40.0)]
        assert result["metrics"]["followed"] == 0


@pytest.mark.parametrize("paint", [True, False])
def test_fit_puts_a_stroked_line_on_its_ink_and_sets_its_width(paint):
    # The ink runs along y = 32, 4 px wide; the line is drawn 1.5 px above
    # it and half as wide.
    ed = Editor(
        import_svg(
            '<svg width="64" height="64"><path id="line" fill="none" '
            'stroke="#000000" stroke-width="2" d="M8 30.5 L24 30.5 L40 30.5 '
            'L56 30.5"/></svg>'
        ),
        selection=Selection(object_ids=frozenset({"line"})),
    )
    ink = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(ink).rectangle((8, 30, 55, 33), fill="black")
    req = OperationRequest(
        action="improve",
        method="nodes",
        snapshot=ed.snapshot,
        editor=ed,
        permissions=Permissions(geometry=True, structure=True, paint=paint),
        settings={"simplify": False},
        budget=Budget(steps=2),
        reference=ink,
    )
    job = Job(method("improve", "nodes"), req)
    job.run()
    result = job.state()["result"]
    assert result["changed"]
    assert result["metrics"]["steps"][0] == "shape"
    job.apply()
    document = ed.snapshot.document
    ys = [n.values[-1] for s in document.geometry_for("line").subpaths for n in s.nodes]
    assert all(abs(y - 32) < 0.5 for y in ys[1:-1]), ys
    width = float(document.element("line").get("stroke-width") or "nan")
    assert abs(width - 4) < 0.6 if paint else width == 2


def test_without_pytorch_tidy_runs_without_the_fit(monkeypatch):
    from vectrify.refine import selected

    monkeypatch.setattr(selected, "fit_problem", lambda: "Path fitting needs PyTorch")
    ed = editor("p")
    req = OperationRequest(
        action="improve",
        method="nodes",
        snapshot=ed.snapshot,
        editor=ed,
        permissions=Permissions(geometry=True, structure=True),
        settings={"simplify": True},
        budget=Budget(steps=2),
        reference=reference(),
    )
    nodes = method("improve", "nodes")
    nodes.validate(req)
    job = Job(nodes, req)
    job.run()
    metrics = job.state()["result"]["metrics"]
    assert metrics["steps"]
    assert "shape" not in metrics["steps"]
    with pytest.raises(DocumentError, match="PyTorch"):
        nodes.validate(request(editor("p")))


@pytest.mark.usefixtures("low_resolution")
@pytest.mark.parametrize("with_reference", [False, True])
def test_simplify_retains_removing_handles_without_removing_nodes(with_reference):
    document = import_svg(
        '<svg width="64" height="64"><path id="p" fill="black" '
        'd="M8 8 C20 8 32 8 44 8 C32 20 20 32 8 44 Z"/></svg>'
    )
    original = render_image(export_svg(document), alpha=True)
    ed = Editor(document, selection=Selection(object_ids=frozenset({"p"})))
    req = replace(
        request(ed, steps=1, shape=False, snap=False, simplify=True),
        reference=original if with_reference else None,
    )
    job = Job(method("improve", "nodes"), req)
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    assert state["result"]["changed"]
    assert state["result"]["metrics"]["steps"] == ["simplify"]
    job.apply()
    result = ed.snapshot.document
    before = document.geometry_for("p").subpaths[0].nodes
    after = result.geometry_for("p").subpaths[0].nodes
    assert [n.id for n in after] == [n.id for n in before]
    assert [n.endpoint for n in after] == [n.endpoint for n in before]
    assert [n.command for n in after] == ["M", "L", "L"]
    assert np.array_equal(
        np.asarray(render_image(export_svg(result), alpha=True)), np.asarray(original)
    )


@pytest.mark.usefixtures("low_resolution")
def test_saving_handles_still_cannot_exceed_simplifys_reference_budget():
    document = import_svg(
        '<svg width="64" height="64"><path id="p" fill="black" '
        'd="M8 8 C20 7.8 32 7.8 44 8 C32 20 20 32 8 44 Z"/></svg>'
    )
    ed = Editor(document, selection=Selection(object_ids=frozenset({"p"})))
    target = render_image(export_svg(document), alpha=True)
    job = Job(
        method("improve", "nodes"),
        replace(
            request(ed, steps=1, shape=False, snap=False, simplify=True),
            reference=target,
        ),
    )
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    # Straightening the first curve changes the reference pixels. With an
    # exact starting match its error budget is zero, so that handle must stay.
    assert state["result"]["changed"]
    job.apply()
    result = ed.snapshot.document
    assert [n.command for n in result.geometry_for("p").subpaths[0].nodes] == [
        "M",
        "C",
        "L",
    ]
    assert np.array_equal(
        np.asarray(render_image(export_svg(result), alpha=True)), np.asarray(target)
    )


def shared_clip_editor(*ids):
    # Like arena: a visible body supplies both clipping and its outline.
    document = import_svg(
        '<svg width="64" height="64"><defs><clipPath id="clip">'
        '<path id="boundary" d="M0 0H64V64H0Z"/></clipPath></defs>'
        '<path id="p" fill="#a0b384" d="M14 14L24 14L34 14L50 14'
        'L50 32L50 50L32 50L14 50L14 32Z"/>'
        '<path id="shade" fill="#658678" clip-path="url(#clip)" '
        'd="M0 32H64V64H0Z"/>'
        '<path id="outline" fill="none" stroke="#4e5f4b" '
        'stroke-width="1" d="M0 0H64V64H0Z"/></svg>'
    )
    ed = Editor(document, selection=Selection(whole_document=True))
    with ed.transaction("Share body") as tx:
        tx.share_geometry("boundary", "p")
        tx.share_geometry("outline", "p")
    ed.select(Selection(object_ids=frozenset(ids)))
    return ed


@pytest.mark.usefixtures("low_resolution")
@pytest.mark.parametrize("region", [None, [0, 0, 64, 64]])
def test_tidy_simplifies_shared_clipping_and_outline_without_detaching(region):
    ed = shared_clip_editor("p")
    original = ed.snapshot.document
    job = Job(
        method("improve", "nodes"),
        request(
            ed,
            with_reference=False,
            shape=False,
            simplify=True,
            tolerance=0.1,
            region=region,
        ),
    )
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    assert state["result"]["changed"]
    assert state["result"]["metrics"]["after"]["nodes"] == 4
    job.apply()
    after = ed.snapshot.document
    for oid in ("p", "boundary", "outline"):
        assert after.element(oid) == original.element(oid)
        assert after.geometry_for(oid).id == original.geometry_for("p").id
        assert after.geometry_for(oid) == after.geometry_for("p")
    assert after.element("shade") == original.element("shade")
    assert np.array_equal(
        np.asarray(render_image(export_svg(original), (0, 0, 64, 64), (64, 64))),
        np.asarray(render_image(export_svg(after), (0, 0, 64, 64), (64, 64))),
    )
    ed.undo()
    assert ed.snapshot.document == original


@pytest.mark.usefixtures("low_resolution")
def test_tidy_fits_a_shared_body_and_its_changing_clips_in_context():
    ed = shared_clip_editor("p")
    original = ed.snapshot.document
    geometry = original.geometry_for("p")
    moved = replace(
        geometry,
        subpaths=tuple(
            replace(
                sub,
                nodes=tuple(
                    replace(
                        node,
                        values=tuple(
                            v + (1 if i % 2 == 0 else -1)
                            for i, v in enumerate(node.values)
                        ),
                    )
                    for node in sub.nodes
                ),
            )
            for sub in geometry.subpaths
        ),
    )
    target = render_image(
        export_svg(original.replace_geometry(moved)), (0, 0, 64, 64), (64, 64)
    )
    job = Job(
        method("improve", "nodes"),
        replace(
            request(ed, steps=1, seconds=3, gain=0, snap=False),
            reference=target,
        ),
    )
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    result = state["result"]
    assert result["changed"], state
    assert not result["metrics"]["skipped"]
    assert (
        result["metrics"]["after"]["difference"]
        < result["metrics"]["before"]["difference"]
    )
    job.apply()
    after = ed.snapshot.document
    assert after.root == original.root
    assert (
        after.geometry_for("p")
        == after.geometry_for("boundary")
        == after.geometry_for("outline")
    )
    assert {n.id for s in after.geometry_for("p").subpaths for n in s.nodes} == set(
        nodes_of(original)
    )


@pytest.mark.usefixtures("low_resolution")
def test_tidy_respects_locks_on_a_shared_clipping_consumer():
    ed = shared_clip_editor("p", "shade")
    ed.set_locks("shade", frozenset({"geometry"}))
    original = ed.snapshot.document
    job = Job(
        method("improve", "nodes"),
        request(
            ed,
            with_reference=False,
            shape=False,
            simplify=True,
            tolerance=0.1,
        ),
    )
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    assert not state["result"]["changed"]
    assert "locked" in state["result"]["metrics"]["skipped"]["p"]
    assert ed.snapshot.document == original


@pytest.mark.usefixtures("low_resolution")
def test_tidy_deduplicates_selected_shared_assets():
    ed = shared_clip_editor("p", "outline")
    job = Job(
        method("improve", "nodes"),
        request(
            ed,
            with_reference=False,
            shape=False,
            simplify=True,
            tolerance=0.1,
        ),
    )
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    assert state["result"]["metrics"]["before"]["nodes"] == 9
    assert state["result"]["metrics"]["after"]["nodes"] == 4
    job.apply()


@pytest.mark.usefixtures("low_resolution")
def test_tidy_simplifies_an_instance_without_detaching():
    ed = Editor(
        import_svg(
            '<svg width="64" height="64"><defs><path id="source" '
            'transform="scale(2)" d="M0 0L4 0L8 0L8 8L0 8Z"/></defs>'
            '<use id="instance" href="#source" x="20" y="10"/></svg>'
        ),
        selection=Selection(object_ids=frozenset({"instance"})),
    )
    original = ed.snapshot.document
    job = Job(
        method("improve", "nodes"),
        request(
            ed,
            with_reference=False,
            shape=False,
            simplify=True,
            tolerance=0.1,
            region=[16, 6, 24, 24],
        ),
    )
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    assert state["result"]["changed"], state
    assert state["result"]["metrics"]["after"]["nodes"] == 4
    job.apply()
    assert ed.snapshot.document.element("instance") == original.element("instance")
    assert (
        ed.snapshot.document.geometry_for("instance").id
        == original.geometry_for("source").id
    )


@pytest.mark.usefixtures("low_resolution")
def test_tidy_region_holds_shared_nodes_visible_outside_the_region():
    ed = Editor(
        import_svg(
            '<svg width="64" height="32"><defs><path id="source" '
            'd="M2 2L6 2L10 2L10 10L2 10Z"/></defs>'
            '<use id="first" href="#source"/>'
            '<use id="second" href="#source" x="32"/></svg>'
        ),
        selection=Selection(object_ids=frozenset({"first"})),
    )
    original = ed.snapshot.document
    job = Job(
        method("improve", "nodes"),
        request(
            ed,
            with_reference=False,
            shape=False,
            simplify=True,
            tolerance=0.1,
            region=[0, 0, 20, 20],
        ),
    )
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    assert not state["result"]["changed"], state
    assert ed.snapshot.document == original
