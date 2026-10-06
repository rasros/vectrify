"""Fit gradient through the editor session, and back to a flat fill."""

import pytest

from tests.operations.test_gradient_fit import DOC, ramp_reference
from tests.ui.test_improve import reference, wait
from tests.ui.test_session import send
from vectrify.document import DocumentError, EditRejectedError, Selection, import_svg
from vectrify.ui.session import Session


def gradients(session):
    return [
        e.id
        for e in session.editor.snapshot.document.elements()
        if e.tag == "linearGradient"
    ]


@pytest.mark.parametrize("paint", ["", 'fill="none" stroke="none"'])
def test_create_gradient_without_a_reference_keeps_no_stroke_and_undo(paint):
    session = Session(
        import_svg(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            f'<rect id="shape" width="100" height="100" {paint}/></svg>'
        )
    )
    session.editor.select(Selection(object_ids=frozenset({"shape"})))
    before = session.editor.snapshot.document
    session.action(
        {
            "command": "paint",
            "epoch": session.epoch,
            "revision": session.editor.snapshot.revision,
            "changes": {
                "fill": {
                    "start": [0, 50],
                    "end": [100, 50],
                    "stops": [
                        {"offset": 0, "colour": "#336699"},
                        {"offset": 1, "colour": "#ffffff"},
                    ],
                }
            },
        }
    )
    after = session.editor.snapshot.document
    [gradient] = gradients(session)
    assert after.element(gradient).paint_owner == "shape"
    assert after.element("shape").get("stroke") == before.element("shape").get("stroke")
    assert next(o for o in session.state()["objects"] if o["id"] == "shape")[
        "fill_gradient"
    ]["private"]
    session.editor.undo()
    assert session.editor.snapshot.document == before
    session.editor.redo()
    assert session.editor.snapshot.document == after


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


SHARED = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">
<defs><linearGradient id="ramp" x2="100%" gradientTransform="rotate(15)"
spreadMethod="reflect"><stop id="first" offset="0" stop-color="#ff0000"/>
<stop id="last" offset="100%" stop-color="#0000ff"/></linearGradient></defs>
<g fill="url(#ramp)"><rect id="a" width="80" height="80"/></g>
<rect id="b" x="100" width="80" height="80" fill="url(#ramp)"/>
<rect id="other" x="190" width="5" height="5" fill="#000000"/></svg>"""


def shared_edit():
    return {
        "id": "ramp",
        "attributes": {"x1": "10%", "y1": "0%", "x2": "90%", "y2": "100%"},
        "stops": [
            {"offset": 0, "colour": "#112233", "opacity": 0},
            {"offset": 0.5, "colour": "#445566", "opacity": 0.5},
            {"offset": 1, "colour": "#778899"},
        ],
    }


@pytest.mark.parametrize("selected", ["a", "b", "ramp"])
def test_shared_gradient_edit_keeps_users_settings_identities_and_undo(selected):
    session = Session(import_svg(SHARED))
    send(session, "select", objects=[selected])
    before = session.editor.snapshot.document
    shown = {o["id"]: o for o in session.state()["objects"]}
    for oid in ("a", "b", "ramp"):
        assert shown[oid]["fill_gradient"]["id"] == "ramp"
        assert shown[oid]["fill_gradient"]["users"] == 2
        assert not shown[oid]["fill_gradient"]["private"]
    send(session, "paint", gradient=shared_edit())
    after = session.editor.snapshot.document
    gradient = after.element("ramp")
    assert gradient.paint_owner is None
    assert gradients(session) == ["ramp"]
    assert gradient.get("x1") == "10%"
    assert gradient.get("gradientUnits") is None
    assert gradient.get("gradientTransform") == "rotate(15)"
    assert gradient.get("spreadMethod") == "reflect"
    assert [stop.id for stop in gradient.children[:2]] == ["first", "last"]
    assert gradient.children[0].get("stop-opacity") == "0.0"
    assert gradient.children[1].get("stop-opacity") == "0.5"
    for oid in ("a", "b", "other"):
        assert after.element(oid) == before.element(oid)
    assert session.editor.snapshot.selection.object_ids == frozenset({selected})
    session.editor.undo()
    assert session.editor.snapshot.document == before
    session.editor.redo()
    assert session.editor.snapshot.document == after
    edit = shared_edit()
    edit["stops"] = edit["stops"][:2]
    send(session, "paint", gradient=edit)
    assert len(session.editor.snapshot.document.element("ramp").children) == 2


@pytest.mark.parametrize(
    ("locked", "lock"),
    [("a", "paint"), ("b", "paint"), ("ramp", "paint"), ("last", "stop-color")],
)
def test_shared_gradient_edits_respect_locks_on_all_users_and_stops(locked, lock):
    session = Session(import_svg(SHARED))
    session.editor.set_locks(locked, frozenset({lock}))
    send(session, "select", objects=["a"])
    before = session.editor.snapshot.document
    with pytest.raises(EditRejectedError, match="locked"):
        send(session, "paint", gradient=shared_edit())
    assert session.editor.snapshot.document == before


def test_shared_gradient_edit_rejects_an_unrelated_selection_and_invalid_settings():
    session = Session(import_svg(SHARED))
    before = session.editor.snapshot.document
    send(session, "select", objects=["other"])
    with pytest.raises(DocumentError, match="using its fill"):
        send(session, "paint", gradient=shared_edit())
    send(session, "select", objects=["ramp"])
    edit = shared_edit()
    edit["attributes"]["x2"] = "NaN"
    with pytest.raises(DocumentError, match="finite"):
        send(session, "paint", gradient=edit)
    assert session.editor.snapshot.document == before


def test_private_gradient_uses_the_same_resource_edit_and_keeps_ownership():
    from tests.document.test_gradients import private_editor

    session = Session(private_editor().snapshot.document)
    send(session, "select", objects=["p"])
    [gradient] = gradients(session)
    edit = shared_edit()
    edit["id"] = gradient
    send(session, "paint", gradient=edit)
    assert gradients(session) == [gradient]
    assert session.editor.snapshot.document.element(gradient).paint_owner == "p"
    assert next(o for o in session.state()["objects"] if o["id"] == "p")[
        "fill_gradient"
    ]["private"]
