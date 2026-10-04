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
    assert gradient not in rows
    assert not any(o["tag"] in {"defs", "stop"} for o in rows.values())
    assert rows["a"]["fill_gradient"]["private"]
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


def test_private_fill_properties_edit_stops_endpoints_and_undo():
    from tests.document.test_gradients import private_editor

    session = Session(private_editor().snapshot.document)
    session.editor.select(Selection(object_ids=frozenset({"p"})))
    before = session.editor.snapshot.document
    gradient = gradients(session)[0]
    session.action(
        {
            "command": "paint",
            "epoch": session.epoch,
            "revision": session.editor.snapshot.revision,
            "changes": {
                "fill": {
                    "start": [1, 2],
                    "end": [5, 6],
                    "stops": [
                        {"offset": 0, "colour": "#112233"},
                        {"offset": 0.5, "colour": "#445566", "opacity": 0.5},
                        {"offset": 1, "colour": "#778899"},
                    ],
                }
            },
        }
    )
    after = session.editor.snapshot.document
    assert gradients(session) == [gradient]
    assert after.element(gradient).get("x1") == "1.0"
    shown = next(o for o in session.state()["objects"] if o["id"] == "p")
    assert shown["fill_gradient"]["stops"][1]["stop-opacity"] == "0.5"
    assert len(shown["fill_gradient"]["stops"]) == 3
    session.editor.undo()
    assert session.editor.snapshot.document == before
    session.editor.redo()
    assert session.editor.snapshot.document == after


def test_imported_shared_gradients_remain_in_the_tree():
    from tests.document.test_gradients import SVG

    session = Session(import_svg(SVG))
    rows = {o["id"]: o for o in session.state()["objects"]}
    assert rows["g"]["resource"]
    assert rows["p"]["fill_gradient"]["private"] is False
