"""Optimize nodes reshapes only the selected paths and proposes one ordinary edit."""

import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from vectrify.document import DocumentError, Editor, Selection, export_svg, import_svg
from vectrify.image_utils import on_white
from vectrify.operations import Budget, Job, OperationRequest, Permissions, method
from vectrify.operations.generate import Region
from vectrify.operations.methods import nodes as nodes_method
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


# The path fit on its own unless a test says otherwise, as the tests were
# written against it.
FIT = {"shape": True, "snap": False, "simplify": False}


def request(ed, *, steps=4, with_reference=True, **settings):
    return OperationRequest(
        action="improve",
        method="nodes",
        snapshot=ed.snapshot,
        editor=ed,
        permissions=Permissions(geometry=True, structure=True, paint=True),
        settings={"workers": 1, "resolution": 64, "steps": 30, **FIT, **settings},
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


def test_tidy_cleans_unsupported_dents_even_when_gradient_fitting_stalls(monkeypatch):
    pytest.importorskip("torch")
    source = (
        '<svg width="64" height="64"><rect width="64" height="64" fill="white"/>'
        '<path id="p" fill="black" d="M8 16 L16 16 L18 18 L20 16 L28 16 '
        'L32 20 L36 16 L42 16 L44 18 L46 16 L56 16 L56 56 L8 56 Z"/></svg>'
    )
    target = source.replace("L18 18", "L18 16").replace("L44 18", "L44 16")
    document = import_svg(source)
    ed = Editor(document, selection=Selection(object_ids=frozenset({"p"})))
    monkeypatch.setattr(
        "vectrify.refine.paths.fit_filled_svg", lambda svg, *_args, **_kwargs: svg
    )
    req = replace(
        request(ed, steps=1),
        reference=render_image(target, (0, 0, 64, 64), (64, 64)),
    )
    job = Job(method("improve", "nodes"), req)
    job.run()
    result = job.state()["result"]
    assert result["changed"]
    job.apply()
    old = document.geometry_for("p").subpaths[0].nodes
    new = ed.snapshot.document.geometry_for("p").subpaths[0].nodes
    assert [n.id for n in new] == [n.id for n in old]
    assert new[2].endpoint == pytest.approx((18, 16), abs=1e-9)
    assert new[8].endpoint == pytest.approx((44, 16), abs=1e-9)
    assert new[5].endpoint == (32, 20)


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


def test_fitting_and_simplification_can_both_be_retained(monkeypatch):
    from vectrify.refine import selected
    from vectrify.refine.simplify import curved

    ed = editor("p")
    # Use a known bounded fit to check how Tidy combines the two steps.
    # The real optimizer is exercised by the fitting tests below.
    geometry = curved(ed.snapshot.document.geometry_for("p"))
    targets = {
        node.id: 16 + (np.asarray(node.values).reshape(-1, 2) - 14) * 32 / 36
        for subpath in geometry.subpaths
        for node in subpath.nodes
    }

    def fitting(document, _selection, _target, options, **_kwargs):
        values = {}
        for subpath in document.geometry_for("p").subpaths:
            for node in subpath.nodes:
                previous = np.asarray(node.values).reshape(-1, 2)
                delta = targets[node.id] - previous
                length = np.maximum(np.linalg.norm(delta, axis=1), 1e-12)
                scale = np.minimum(1, options.displacement / length)[:, None]
                values[node.id] = tuple((previous + delta * scale).ravel())
        return SimpleNamespace(values=values)

    monkeypatch.setattr(selected, "fit_selected_path", fitting)
    job = Job(
        method("improve", "nodes"),
        request(ed, steps=8, snap=True, simplify=True, tolerance=0.5),
    )
    job.run()
    result = job.state()["result"]
    metrics = result["metrics"]
    assert "simplify" in metrics["steps"], metrics
    assert {"shape", "snap"} & set(metrics["steps"])
    assert metrics["after"]["nodes"] == 4
    assert metrics["after"]["difference"] < 0.5 * metrics["before"]["difference"]


def test_workers_run_the_steps_side_by_side_to_the_same_result():
    alone, together = (
        Job(
            method("improve", "nodes"),
            request(editor("p"), steps=3, snap=True, simplify=True, workers=n),
        )
        for n in (1, 3)
    )
    alone.run()
    together.run()
    first, second = alone.state()["result"], together.state()["result"]
    assert first["metrics"]["steps"] == second["metrics"]["steps"]
    assert first["metrics"]["after"] == second["metrics"]["after"]


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
            "workers": 1,
            "resolution": 64,
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


def bow_tie(document):
    """*document* with the square turned into a bow-tie."""
    twisted = import_svg(
        '<svg width="64" height="64"><path id="p" fill="#000000" '
        'd="M16 16 L48 16 L16 48 L48 48 Z"/></svg>'
    ).geometry_for("p")
    return document.replace_geometry(replace(twisted, id=document.geometry_for("p").id))


@pytest.fixture
def snap_twists(monkeypatch):
    """Snap turns the square into a bow-tie and claims a perfect match."""
    run_step = nodes_method._run_step

    def twisting(step, task, stop=None, progress=None):
        if step == "snap":
            return bow_tie(task.document), np.asarray(task.region.image), {}
        return run_step(step, task, stop, progress)

    monkeypatch.setattr(nodes_method, "_run_step", twisting)


def test_a_step_that_makes_a_path_cross_itself_is_not_kept(monkeypatch):
    from vectrify.refine import selected

    calls = []

    def twisting(document, *_args, **_kwargs):
        # Preserve IDs and commands, but make the fitter return a bow-tie.
        # Repeated knots fill out the existing topology without affecting the
        # crossing. Exercise the actual fit acceptance guard, independent of
        # where snapping happens or how many improving rounds a CPU finishes.
        nodes = document.geometry_for("p").subpaths[0].nodes
        corners = np.array([(16, 16), (48, 48), (16, 48), (48, 16), (16, 16)])
        previous = corners[0]
        values = {}
        for i, node in enumerate(nodes):
            end = corners[4 * i // (len(nodes) - 1)]
            if node.command == "C":
                step = end - previous
                value = np.array([previous + step / 3, previous + 2 * step / 3, end])
            else:
                value = end
            values[node.id] = tuple(float(v) for v in value.ravel())
            previous = end
        calls.append(values)
        return SimpleNamespace(values=values)

    monkeypatch.setattr(selected, "fit_selected_path", twisting)
    ed = editor("p")
    job = Job(method("improve", "nodes"), request(ed, steps=2))
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    assert calls
    assert not state["result"]["changed"]
    metrics = state["result"]["metrics"]
    assert metrics["steps"] == []
    assert nodes_method._crossings(ed.snapshot.document, ["p"]) == {"p": 0}
    assert metrics["after"]["difference"] == metrics["before"]["difference"]


@pytest.mark.usefixtures("snap_twists")
def test_only_crossing_results_leave_the_paths_as_they_were():
    job = Job(method("improve", "nodes"), request(editor("p"), shape=False, snap=True))
    job.run()
    state = job.state()
    assert not state["result"]["changed"]
    assert state["result"]["metrics"]["folded"] == {"snap": 1}
    assert "cross itself" in state["message"]


def test_the_defaults_are_a_quick_tidy():
    from vectrify.operations.settings import read_settings

    settings = read_settings({}, nodes_method.SETTINGS, nodes_method.LABEL)
    assert (settings["snap"], settings["simplify"], settings["shape"]) == (
        True,
        True,
        True,
    )
    # The fit is cheap enough to be on: a reduced resolution, few steps, and
    # it stops once it stalls.
    assert settings["resolution"] <= 384
    assert settings["steps"] <= 60
    assert settings["stall"] > 0
    assert not settings["detail"]
    assert settings["seconds"] == 10
    assert nodes_method.DEFAULT_ROUNDS <= 4


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
    assert nodes_method._crossings(after, ["p"]) == {"p": 0}
    assert (
        state["result"]["metrics"]["after"]["difference"]
        < state["result"]["metrics"]["before"]["difference"]
    )


def test_detail_fit_holds_new_knots_where_the_curve_leaves_the_view(monkeypatch):
    from vectrify.refine import selected

    document = import_svg(
        '<svg width="64" height="64"><path id="p" '
        'd="M10 10 C10 60 50 60 50 10 L50 5 L10 5 Z"/></svg>'
    )
    seen = []

    def fitting(current, selection, *_args, **_kwargs):
        nodes = current.geometry_for("p").subpaths[0].nodes
        outside = {n.id for n in nodes if n.endpoint[1] > 20}
        assert outside
        assert not outside & selection.node_ids
        assert nodes[0].id in selection.node_ids
        seen.append(outside)
        return SimpleNamespace(values={})

    monkeypatch.setattr(selected, "fit_selected_path", fitting)
    target = Image.new("RGB", (64, 64), "white")
    task = nodes_method._Task(
        document,
        Region(0, 0, 64, 64, target),
        nodes_method.read_settings(
            {"detail": True, "region": [0, 0, 64, 20]}, nodes_method.SETTINGS, "Tidy"
        ),
        ("p",),
        target,
    )
    result, skipped = nodes_method._fit(task, None, None)
    assert seen
    assert not skipped
    assert result == document


def moved(document, where):
    """*document* with path p's points moved: *where* maps each point's
    values to new ones."""
    geometry = document.geometry_for("p")
    subpaths = tuple(
        replace(s, nodes=tuple(replace(n, values=where(n.values)) for n in s.nodes))
        for s in geometry.subpaths
    )
    return document.replace_geometry(replace(geometry, subpaths=subpaths))


def test_a_small_local_fix_on_a_large_selection_is_kept(monkeypatch):
    # A large square with one point pushed 2 px in from its edge, and a disc
    # in the reference the path cannot match: next to the difference the
    # disc leaves, putting the point back is a sliver.
    def fixing(_step, task, _stop=None, _progress=None):
        document = moved(
            task.document, lambda v: (136.0, 16.0) if v == (136.0, 18.0) else v
        )
        return document, nodes_method._pixels(document, task.region), {}

    monkeypatch.setattr(nodes_method, "_run_step", fixing)
    ed = Editor(
        import_svg(
            '<svg width="256" height="256"><path id="p" fill="#000000" '
            'd="M16 16 L128 16 L136 18 L144 16 L240 16 L240 240 L16 240 Z"/></svg>'
        ),
        selection=Selection(object_ids=frozenset({"p"})),
    )
    target = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(target).rectangle((16, 16, 239, 239), fill="black")
    ImageDraw.Draw(target).ellipse((48, 48, 208, 208), fill="white")
    req = OperationRequest(
        action="improve",
        method="nodes",
        snapshot=ed.snapshot,
        editor=ed,
        permissions=Permissions(geometry=True, structure=True),
        settings={"workers": 1, "simplify": False, "shape": False},
        budget=Budget(steps=1),
        reference=target,
    )
    job = Job(method("improve", "nodes"), req)
    job.run()
    metrics = job.state()["result"]["metrics"]
    before, after = metrics["before"]["difference"], metrics["after"]["difference"]
    # Of the whole region the fix is under the old bar of 0.1%, yet it is
    # kept: where it acted it fixed all there was.
    assert 0 < (before - after) / before < 0.001
    assert metrics["steps"] == ["snap"]


def test_a_step_is_judged_by_the_pixels_it_changed():
    region = Region(0, 0, 40, 40, Image.new("RGB", (40, 40), "black"))
    start = np.zeros((40, 40, 3), dtype=np.uint8)
    start[:, 20:] = 255
    fixed, worse = start.copy(), start.copy()
    fixed[10:14, 20:24] = 0
    worse[10:14, 10:14] = 255
    now = nodes_method._Scored.of(start, region)
    assert nodes_method._Scored.of(fixed, region).fixed(now) > 0.3
    assert nodes_method._Scored.of(worse, region).fixed(now) < 0
    assert nodes_method._Scored.of(start, region).fixed(now) == 0


def test_white_paint_and_a_transparent_gap_have_different_tidy_scores():
    rgb = Image.new("RGB", (40, 40), "white")
    alpha = np.zeros((40, 40), np.float32)
    alpha[10:30, 10:30] = 1
    region = Region(0, 0, 40, 40, rgb, alpha)
    correct = np.full((40, 40, 4), 255, np.uint8)
    correct[:, :, 3] = (alpha * 255).astype(np.uint8)
    spill = correct.copy()
    spill[0:10, 10:30, 3] = 255
    gap = correct.copy()
    gap[15:25, 15:25, 3] = 0
    best = nodes_method._Scored.of(correct, region)
    assert best.difference == 0
    assert nodes_method._Scored.of(spill, region).difference > 0
    assert nodes_method._Scored.of(gap, region).difference > 0
    wrong_colour = correct.copy()
    wrong_colour[15:25, 15:25, :3] = 0
    # A transparent hole in opaque white is as wrong as black paint there.
    assert nodes_method._Scored.of(gap, region).difference == pytest.approx(
        nodes_method._Scored.of(wrong_colour, region).difference
    )


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
        "resolution": 64,
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


def test_the_time_limit_ends_the_run_and_keeps_the_best_so_far(monkeypatch):
    # Each snap takes a while and brings the square 1 px nearer: only the
    # time limit ends the run.
    def nearer(_step, task, _stop=None, _progress=None):
        time.sleep(0.2)
        document = moved(task.document, lambda v: tuple(x + 1 for x in v))
        return document, nodes_method._pixels(document, task.region), {}

    monkeypatch.setattr(nodes_method, "_run_step", nearer)
    ed = Editor(
        import_svg(
            '<svg width="64" height="64"><path id="p" fill="#000000" '
            'd="M8 8 L40 8 L40 40 L8 40 Z"/></svg>'
        ),
        selection=Selection(object_ids=frozenset({"p"})),
    )
    started = time.monotonic()
    job = Job(
        method("improve", "nodes"),
        request(ed, steps=50, shape=False, snap=True, seconds=0.5),
    )
    job.run()
    assert time.monotonic() - started < 3
    metrics = job.state()["result"]["metrics"]
    assert metrics["out_of_time"]
    assert 1 <= len(metrics["steps"]) < 8
    assert metrics["after"]["difference"] < metrics["before"]["difference"]
    job.apply()
    corner = ed.snapshot.document.geometry_for("p").subpaths[0].nodes[0].values
    assert corner == (8 + len(metrics["steps"]),) * 2


def cutting(_step, task, _stop=None, _progress=None):
    """Every step cuts the square's corner at (50, 50) off: a point fewer,
    and a worse match."""
    geometry = task.document.geometry_for("p")
    subpaths = tuple(
        replace(s, nodes=tuple(n for n in s.nodes if n.values != (50.0, 50.0)))
        for s in geometry.subpaths
    )
    document = task.document.replace_geometry(replace(geometry, subpaths=subpaths))
    return document, nodes_method._pixels(document, task.region), {}


def test_a_step_that_makes_the_match_worse_is_not_kept(monkeypatch):
    monkeypatch.setattr(nodes_method, "_run_step", cutting)
    job = Job(
        method("improve", "nodes"),
        request(editor("p"), shape=False, snap=False, simplify=True),
    )
    job.run()
    state = job.state()
    assert not state["result"]["changed"]
    assert state["result"]["metrics"]["after"]["nodes"] == 10
    # Allowed to cost that much, the point goes.
    job = Job(
        method("improve", "nodes"),
        request(editor("p"), shape=False, snap=False, simplify=True, allowance=100.0),
    )
    job.run()
    metrics = job.state()["result"]["metrics"]
    assert metrics["steps"] == ["simplify"]
    assert metrics["after"]["nodes"] == 9


def test_simplify_removes_points_only_within_the_error_budget():
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
                "workers": 1,
                "snap": False,
                "shape": False,
                "tolerance": 3.0,
                "allowance": 100.0,
                **settings,
            },
            budget=Budget(steps=1),
            reference=disc,
        )
        job = Job(method("improve", "nodes"), req)
        job.run()
        return job.state()["result"]["metrics"]

    loose, tight = tidied(budget=100.0), tidied(budget=0.5)
    assert loose["after"]["nodes"] < tight["after"]["nodes"]
    assert tight["after"]["difference"] <= tight["before"]["difference"] * 1.01
    assert loose["after"]["difference"] > tight["after"]["difference"]


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
                "resolution": 64,
                "steps": 30,
                "seconds": 20,
                "workers": 1,
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


def test_individual_and_group_selection_use_the_same_drawing_order():
    # IDs deliberately sort in the opposite order to the drawing.
    document = import_svg(
        NEIGHBOURS.replace('<path id="left"', '<g id="pair"><path id="z"')
        .replace('id="right"', 'id="a"')
        .replace("</svg>", "</g></svg>")
    )
    for ids in ({"pair"}, {"z", "a"}):
        ed = Editor(document, selection=Selection(object_ids=frozenset(ids)))
        assert nodes_method.selected_paths(request(ed)) == ["z", "a"]


@pytest.mark.parametrize("first_spends", [1, 5])
def test_a_slow_fill_cannot_spend_the_later_fills_time(first_spends, monkeypatch):
    from vectrify.refine import selected

    now = [0.0]
    calls = []

    def slow(_document, selection, _reference, _options, *, stop, progress, corners):
        assert progress is None
        oid = next(iter(selection.object_ids))
        assert corners <= {
            n.id for s in _document.geometry_for(oid).subpaths for n in s.nodes
        }
        calls.append((oid, stop.deadline))
        now[0] += first_spends if oid == "left" else stop.deadline - now[0]
        return SimpleNamespace(values={})

    monkeypatch.setattr(nodes_method.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(selected, "fit_selected_path", slow)
    document = import_svg(NEIGHBOURS)
    target = neighbours_reference()
    task = nodes_method._Task(
        document,
        Region(0, 0, 64, 64, target),
        nodes_method.read_settings({}, nodes_method.SETTINGS, "Tidy"),
        ("left", "right"),
        target,
    )
    after, skipped = nodes_method._fit(task, nodes_method._Until(10), None)
    assert calls == [("left", 5), ("right", 10)]
    assert not skipped
    assert after == document


def test_fitting_a_later_path_sees_the_already_followed_shared_edge(monkeypatch):
    from vectrify.refine import selected
    from vectrify.refine.shared import frozen_points

    def fit(document, selection, *_args, **_kwargs):
        oid = next(iter(selection.object_ids))
        geometry = document.geometry_for(oid)
        if oid == "right":
            assert any(
                n.values[-2:] == (36.0, 24.0)
                for subpath in geometry.subpaths
                for n in subpath.nodes
            )
            return SimpleNamespace(values={})
        values = {}
        for subpath in geometry.subpaths:
            for node in subpath.nodes:
                if node.values[-2:] in {(32.0, 24.0), (32.0, 40.0)}:
                    values[node.id] = (*node.values[:-2], 36.0, node.values[-1])
        return SimpleNamespace(values=values)

    monkeypatch.setattr(selected, "fit_selected_path", fit)
    document = import_svg(NEIGHBOURS)
    target = neighbours_reference()
    shared = nodes_method._shared_edges(document, ("left", "right"))
    task = nodes_method._Task(
        document,
        Region(0, 0, 64, 64, target),
        nodes_method.read_settings({}, nodes_method.SETTINGS, "Tidy"),
        ("left", "right"),
        target,
        held=frozen_points(document, shared),
        shared=tuple(shared),
    )
    after, skipped = nodes_method._fit(task, None, None)
    assert not skipped
    assert after != document
    assert (
        nodes_method._pixels(after, task.region).tolist()
        != nodes_method._pixels(document, task.region).tolist()
    )


def test_a_bad_later_fit_does_not_discard_an_earlier_improvement(monkeypatch):
    from vectrify.refine import selected

    document = import_svg(
        '<svg width="64" height="64">'
        '<path id="first" fill="black" d="M8 8 L24 8 L24 24 L8 24 Z"/>'
        '<path id="last" fill="black" d="M40 40 L56 40 L56 56 L40 56 Z"/>'
        "</svg>"
    )
    first = document.geometry_for("first")
    target_document = document.replace_geometry(
        replace(
            first,
            subpaths=tuple(
                replace(
                    s,
                    nodes=tuple(
                        replace(n, values=tuple(v + 2 for v in n.values))
                        for n in s.nodes
                    ),
                )
                for s in first.subpaths
            ),
        )
    )
    region = Region(0, 0, 64, 64, Image.new("RGB", (64, 64)))
    target = Image.fromarray(nodes_method._pixels(target_document, region))

    def misplaced(current, selection, *_args, **_kwargs):
        oid = next(iter(selection.object_ids))
        shift = 2 if oid == "first" else 8
        return SimpleNamespace(
            values={
                n.id: tuple(v + shift for v in n.values)
                for s in current.geometry_for(oid).subpaths
                for n in s.nodes
            }
        )

    monkeypatch.setattr(selected, "fit_selected_path", misplaced)
    task = nodes_method._Task(
        document,
        replace(region, image=target),
        nodes_method.read_settings({}, nodes_method.SETTINGS, "Tidy"),
        ("first", "last"),
        target,
    )
    after, skipped = nodes_method._fit(task, None, None)
    assert not skipped
    assert after.geometry_for("first") == target_document.geometry_for("first")
    assert after.geometry_for("last") == document.geometry_for("last")


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
            "workers": 1,
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
        settings={"workers": 1, "simplify": False},
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


def test_a_path_the_fit_cannot_take_is_skipped_not_fatal(monkeypatch):
    from vectrify.refine import selected
    from vectrify.refine.paths import UnsupportedPathError

    def refusing(*_args, **_kwargs):
        raise UnsupportedPathError("no opaque filled cubic paths to optimise")

    monkeypatch.setattr(selected, "fit_selected_path", refusing)
    job = Job(method("improve", "nodes"), request(editor("p"), snap=True))
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    assert "opaque" in state["result"]["metrics"]["skipped"]["p"]


def test_without_pytorch_the_default_tidy_runs_without_the_fit(monkeypatch):
    from vectrify.refine import selected

    monkeypatch.setattr(selected, "fit_problem", lambda: "Path fitting needs PyTorch")
    ed = editor("p")
    req = OperationRequest(
        action="improve",
        method="nodes",
        snapshot=ed.snapshot,
        editor=ed,
        permissions=Permissions(geometry=True, structure=True),
        settings={"workers": 1},
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


def test_fitting_reclaims_time_unused_by_simplify(monkeypatch):
    document = editor("p").snapshot.document
    region = Region(0, 0, 64, 64, reference())
    task = nodes_method._Task(
        document, region, {}, ("p",), reference(), deadline=10, share=3
    )
    now = [0.0]
    calls = []
    monkeypatch.setattr(nodes_method.time, "monotonic", lambda: now[0])

    def step(name, given, _stop, _report):
        assert given.document == document
        calls.append((name, given.share))
        now[0] += 0.2
        return document, np.asarray(region.image), {}

    monkeypatch.setattr(nodes_method, "_run_step", step)
    result = nodes_method._round(["shape", "simplify"], task, None, None, None)
    assert calls == [("simplify", 3), ("shape", 9.55)]
    # Candidate priority stays identical to the parallel execution order.
    assert list(result) == ["shape", "simplify"]


def test_straightening_cannot_discard_a_better_fitted_curve(monkeypatch):
    from vectrify.refine import selected, simplify

    document = editor("p").snapshot.document
    original = document.geometry_for("p")
    expected = moved(document, lambda v: tuple(x + 2 for x in v))
    region = Region(0, 0, 64, 64, reference())
    target = Image.fromarray(nodes_method._pixels(expected, region))

    def fitting(current, _selection, *_args, **_kwargs):
        return SimpleNamespace(
            values={
                n.id: tuple(v + 2 for v in n.values)
                for s in current.geometry_for("p").subpaths
                for n in s.nodes
            }
        )

    monkeypatch.setattr(selected, "fit_selected_path", fitting)
    monkeypatch.setattr(simplify, "straightened", lambda *_args: original)
    task = nodes_method._Task(
        document,
        replace(region, image=target),
        nodes_method.read_settings({}, nodes_method.SETTINGS, "Tidy"),
        ("p",),
        target,
    )
    after, skipped = nodes_method._fit(task, None, None)
    assert not skipped
    assert np.array_equal(nodes_method._pixels(after, region), np.asarray(target))
    assert after.geometry_for("p") != original


def test_tidy_fits_adjacent_fills_jointly_when_individual_fits_cannot_improve(
    monkeypatch,
):
    pytest.importorskip("torch")
    svg = (
        '<svg width="64" height="64"><g id="g">'
        '<path id="left" fill="#a23" d="M10 10 L30 10 L30 50 L10 50 Z"/>'
        '<path id="right" fill="#38b" d="M34 10 L54 10 L54 50 L34 50 Z"/>'
        "</g></svg>"
    )
    target_svg = (
        svg.replace("L30 10 L30 50", "L32 10 L32 50")
        .replace("M34 10", "M32 10")
        .replace("L34 50", "L32 50")
    )
    image = render_image(target_svg, (0, 0, 64, 64), (64, 64), alpha=True)
    document = import_svg(svg)
    ed = Editor(document, selection=Selection(object_ids=frozenset({"g"})))
    monkeypatch.setattr(
        "vectrify.refine.selected.fit_selected_path",
        lambda *_args, **_kwargs: SimpleNamespace(values={}),
    )
    req = OperationRequest(
        "improve",
        "nodes",
        ed.snapshot,
        ed,
        Permissions(geometry=True, structure=True, paint=True),
        settings={
            **FIT,
            "workers": 1,
            "resolution": 64,
            "steps": 20,
            "movement": 2,
            "seconds": 10,
        },
        budget=Budget(steps=1),
        reference=image,
    )
    job = Job(method("improve", "nodes"), req)
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    assert state["result"]["changed"]
    metrics = state["result"]["metrics"]
    assert metrics["after"]["difference"] < metrics["before"]["difference"] * 0.7
    job.apply()
    assert ed.snapshot.document.root == document.root


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


@pytest.mark.parametrize("stroke_width", [0.2, 4])
def test_stroke_supported_simplification_checks_the_actual_reference(
    stroke_width, monkeypatch
):
    from vectrify.refine import simplify

    document = import_svg(
        '<svg width="64" height="64"><g id="pair">'
        '<path id="fill" fill="blue" d="M8 16 L16 16.25 L24 15.75 '
        'L32 16.25 L40 15.75 L48 16 L48 48 L8 48 Z"/>'
        f'<path id="ink" fill="none" stroke="black" stroke-width="{stroke_width}" '
        'd="M8 16 L48 16"/></g></svg>'
    )
    target = render_image(export_svg(document), alpha=True)
    ed = Editor(document, selection=Selection(object_ids=frozenset({"pair"})))
    # Isolate the additional model: the ordinary joining model offers no change.
    monkeypatch.setattr(
        simplify, "simplify", lambda _d, paths, *_args, **_kwargs: paths
    )
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
    assert state["result"]["changed"] == (stroke_width == 4)
    if state["result"]["changed"]:
        job.apply()
        assert len(ed.snapshot.document.geometry_for("fill").subpaths[0].nodes) == 4
        assert ed.snapshot.document.geometry_for("ink") == document.geometry_for("ink")
        assert np.array_equal(
            np.asarray(render_image(export_svg(ed.snapshot.document), alpha=True)),
            np.asarray(target),
        )


def test_simplifys_budget_includes_an_unselected_shared_neighbour():
    from vectrify.refine.frozen import Frozen, Paths
    from vectrify.refine.shared import frozen_points

    document = import_svg(
        '<svg width="64" height="64">'
        '<path id="ghost" fill="blue" fill-opacity="0" '
        'd="M8 8 L32 8 L32.5 24 L31.5 40 L32 56 L8 56 Z"/>'
        '<path id="visible" fill="navy" '
        'd="M32 8 L56 8 L56 56 L32 56 L31.5 40 L32.5 24 Z"/></svg>'
    )
    target = render_image(export_svg(document), alpha=True)
    region = Region(0, 0, 64, 64, on_white(target), np.asarray(target)[:, :, 3] / 255)
    shared = nodes_method._shared_edges(document, ("ghost",))
    assert shared
    assert shared[0].neighbour == "visible"
    task = nodes_method._Task(
        document,
        region,
        nodes_method.read_settings({"tolerance": 3}, nodes_method.SETTINGS, "Tidy"),
        ("ghost",),
        target,
        shared=tuple(shared),
    )
    original = Paths({"ghost": document.geometry_for("ghost")})
    result = nodes_method._simplified(
        task, original, Frozen(frozen_points(document, shared)), float("inf")
    )
    # The selected transparent path's pixels cannot expose the budget violation;
    # its opaque neighbour following the simpler edge does.
    assert result == original
