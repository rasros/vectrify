"""Optimize nodes reshapes only the selected paths and proposes one ordinary edit."""

import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from vectrify.document import DocumentError, Editor, Selection, import_svg
from vectrify.operations import Budget, Job, OperationRequest, Permissions, method
from vectrify.operations.methods import nodes as nodes_method

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


def test_the_steps_mix_to_fit_a_rough_shape_with_fewer_points():
    ed = editor("p")
    job = Job(
        method("improve", "nodes"),
        request(ed, steps=8, snap=True, simplify=True, tolerance=0.5),
    )
    job.run()
    result = job.state()["result"]
    metrics = result["metrics"]
    assert "simplify" in metrics["steps"]
    assert {"shape", "snap"} & set(metrics["steps"])
    assert metrics["after"]["nodes"] < metrics["before"]["nodes"]
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


@pytest.mark.usefixtures("snap_twists")
def test_a_step_that_makes_a_path_cross_itself_is_not_kept():
    job = Job(method("improve", "nodes"), request(editor("p"), steps=2, snap=True))
    job.run()
    metrics = job.state()["result"]["metrics"]
    assert metrics["steps"] == ["shape", "shape"]
    assert metrics["folded"] == {"snap": 2}
    assert metrics["after"]["difference"] < metrics["before"]["difference"]


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
    assert (settings["snap"], settings["simplify"]) == (True, True)
    assert not settings["shape"]
    assert not settings["detail"]
    assert settings["seconds"] == 10
    assert nodes_method.DEFAULT_ROUNDS <= 4


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
        settings={"workers": 1, "simplify": False},
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
    region = SimpleNamespace(image=Image.new("RGB", (40, 40), "black"))
    start = np.zeros((40, 40, 3), dtype=np.uint8)
    start[:, 20:] = 255
    fixed, worse = start.copy(), start.copy()
    fixed[10:14, 20:24] = 0
    worse[10:14, 10:14] = 255
    now = nodes_method._Scored.of(start, region)
    assert nodes_method._Scored.of(fixed, region).fixed(now) > 0.3
    assert nodes_method._Scored.of(worse, region).fixed(now) < 0
    assert nodes_method._Scored.of(start, region).fixed(now) == 0


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
