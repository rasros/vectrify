"""Search improvements climbs over the selection and proposes ordinary edits."""

import threading

import pytest
from PIL import Image

from vectrify.document import DocumentError, Editor, Selection, import_svg
from vectrify.operations import Budget, Job, OperationRequest, Permissions, method

SVG = (
    '<svg width="48" height="48"><rect id="bg" width="48" height="48" '
    'fill="#ffffff"/><rect id="a" x="8" y="8" width="16" height="16" '
    'fill="#660000"/><rect id="b" x="28" y="28" width="12" height="12" '
    'fill="#000066"/></svg>'
)


def reference():
    image = Image.new("RGB", (48, 48), "white")
    image.paste((200, 30, 30), (10, 10, 26, 26))
    image.paste((0, 0, 102), (28, 28, 40, 40))
    return image


def request(editor, *, steps=120, **permissions):
    return OperationRequest(
        action="improve",
        method="search",
        snapshot=editor.snapshot,
        editor=editor,
        permissions=Permissions(**permissions),
        settings={"workers": 1, "resolution": 64},
        budget=Budget(steps=steps),
        reference=reference(),
    )


def editor(*ids):
    return Editor(import_svg(SVG), selection=Selection(object_ids=frozenset(ids)))


def test_validation_needs_a_reference_scope_and_permission():
    search = method("improve", "search")
    with pytest.raises(DocumentError, match="Select objects"):
        search.validate(request(editor(), paint=True))
    with pytest.raises(DocumentError, match="at least one"):
        search.validate(request(editor("a")))
    with pytest.raises(DocumentError, match="Allow geometry"):
        search.validate(request(editor("a"), transform=True))
    ed = editor("a")
    bad = OperationRequest(
        "improve",
        "search",
        ed.snapshot,
        ed,
        Permissions(paint=True),
        settings={"llm": True},
        reference=reference(),
    )
    with pytest.raises(DocumentError, match="Unknown search setting"):
        search.validate(bad)


def test_search_improves_only_the_selected_object():
    ed = editor("a")
    original = ed.snapshot.document
    job = Job(method("improve", "search"), request(ed, geometry=True, paint=True))
    job.run()
    state = job.state(preview=True)
    assert state["status"] == "ready", state
    result = state["result"]
    assert result["changed"]
    after, before = result["metrics"]["after"], result["metrics"]["before"]
    assert after["difference"] < before["difference"]
    assert set(result["previews"]) == {"reference", "before", "after"}
    job.apply()
    assert ed.undo_labels == ("Search improvements",)
    document = ed.snapshot.document
    assert document.element("a") != original.element("a")
    assert document.element("b") == original.element("b")
    assert document.element("bg") == original.element("bg")


def test_stop_keeps_the_best_result_so_far():
    ed = editor("a")
    job = Job(method("improve", "search"), request(ed, steps=100_000, paint=True))
    timer = threading.Timer(3, job.stop.set)
    timer.start()
    try:
        job.run()
    finally:
        timer.cancel()
    state = job.state()
    assert state["status"] == "ready", state
    assert state["result"]["metrics"]["tasks"] < 100_000
