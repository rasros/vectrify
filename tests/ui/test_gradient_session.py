"""Fit gradient through the editor session, and back to a flat fill."""

from tests.operations.test_gradient_fit import DOC, ramp_reference
from tests.ui.test_improve import reference, wait
from vectrify.document import Selection, import_svg
from vectrify.ui.session import Session


def gradients(session):
    return [
        e.id
        for e in session.editor.snapshot.document.elements()
        if e.tag == "linearGradient"
    ]


def test_fit_gradient_then_a_flat_colour_from_the_paint_panel():
    session = Session(
        import_svg(DOC.format(stroke="")), reference=reference(ramp_reference())
    )
    session.editor.select(Selection(object_ids=frozenset({"a"})))
    job = session.operation(
        {
            "command": "start",
            "action": "improve",
            "method": "colours",
            "epoch": session.epoch,
            "revision": session.editor.snapshot.revision,
            "permissions": {"paint": True},
            "settings": {"passes": 1, "resolution": 100, "fill": "linear"},
        }
    )
    state = wait(session, job)
    assert state["status"] == "ready", state
    assert state["result"]["metrics"]["gradients"] == 1
    session.operation({"command": "apply", "job": job["id"]})
    [gradient] = gradients(session)
    shown = session.state()
    assert f'fill="url(#{gradient})"' in shown["svg"]
    rows = {o["id"]: o for o in shown["objects"]}
    assert rows[gradient]["label"] == "Linear gradient 1"
    assert rows[gradient]["resource"]
    assert session.editor.undo_labels == ("Fit gradients",)

    session.action(
        {
            "command": "paint",
            "changes": {"fill": "#336699"},
            "epoch": session.epoch,
            "revision": session.editor.snapshot.revision,
        }
    )
    assert session.editor.snapshot.document.element("a").get("fill") == "#336699"
    assert gradients(session) == []
    session.editor.undo()
    assert gradients(session) == [gradient]
