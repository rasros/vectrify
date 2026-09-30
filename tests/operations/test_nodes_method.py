"""Optimize nodes reshapes only the selected paths and proposes one ordinary edit."""

import pytest
from PIL import Image, ImageDraw

from vectrify.document import DocumentError, Editor, Selection, import_svg
from vectrify.operations import Budget, Job, OperationRequest, Permissions, method

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


def request(ed, *, steps=4, with_reference=True, **settings):
    return OperationRequest(
        action="improve",
        method="nodes",
        snapshot=ed.snapshot,
        editor=ed,
        permissions=Permissions(geometry=True, structure=True, paint=True),
        settings={"workers": 1, "resolution": 64, "steps": 30, **settings},
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
        settings={"workers": 1, "resolution": 64, "steps": 40, "movement": 6.0},
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
