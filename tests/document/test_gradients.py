"""Linear gradient fills: import, export, references and owned definitions."""

import json
from dataclasses import replace
from xml.etree import ElementTree as ET

import pytest

from vectrify.document import (
    DocumentError,
    EditKind,
    Editor,
    EditRejectedError,
    Selection,
    UnsupportedSvgError,
    export_svg,
    import_svg,
    load_project,
    save_project,
)
from vectrify.document.join import average_paint
from vectrify.document.paint import (
    GradientStop,
    LinearGradient,
    mean_colour,
    solid_paint,
)
from vectrify.operations.generate import fresh_ids

SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" '
    'xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 10 10">'
    '<defs id="defs"><linearGradient id="base">'
    '<stop id="s0" offset="0" stop-color="#ff0000"/>'
    '<stop id="s1" offset="100%" stop-color="#0000ff" stop-opacity="0.5"/>'
    "</linearGradient>"
    '<linearGradient id="g" xlink:href="#base" gradientUnits="userSpaceOnUse" '
    'x1="0" y1="0" x2="10" y2="0" spreadMethod="reflect"/></defs>'
    '<path id="p" d="M0 0 H10 V10 H0 Z" fill="url(#g)"/>'
    '<rect id="r" width="2" height="2" fill="#00ff00"/></svg>'
)
RAMP = LinearGradient(
    (0, 0), (2, 1), (GradientStop(0, "#000000"), GradientStop(1, "#ffffff"))
)


def gradients(document):
    return [e for e in document.elements() if e.tag == "linearGradient"]


def test_import_keeps_linear_gradients_and_resolves_their_links():
    document = import_svg(SVG)
    linked = document.element("g")
    assert linked.get("href") is None
    assert linked.get("spreadMethod") == "reflect"
    assert [c.get("stop-color") for c in linked.children] == ["#ff0000", "#0000ff"]
    # The copied stops get identities of their own.
    assert {c.id for c in linked.children}.isdisjoint({"s0", "s1"})
    again = import_svg(export_svg(document))
    assert export_svg(again) == export_svg(document)
    loaded, _ = load_project(save_project(document))
    assert loaded == document


def test_gradients_in_use_cannot_be_deleted_and_ids_are_renamed_together():
    document = import_svg(SVG)
    editor = Editor(document, selection=Selection.all())
    with (
        pytest.raises(EditRejectedError, match="references"),
        editor.transaction("Delete") as tx,
    ):
        tx.delete_objects(frozenset({"g"}))
    renamed = import_svg(fresh_ids(SVG))
    path = next(e for e in renamed.elements() if e.tag == "path")
    target = (path.get("fill") or "")[5:-1]
    assert renamed.element(target).tag == "linearGradient"
    assert target != "g"


@pytest.mark.parametrize(
    "body",
    [
        # Radial gradients stay unsupported, reported as before.
        '<defs><radialGradient id="q"><stop offset="0"/></radialGradient></defs>'
        '<path d="M0 0 H1 V1 Z" fill="url(#q)"/>',
        '<linearGradient id="q"><stop offset="0"/></linearGradient>',
        '<defs><linearGradient id="q"><rect width="1" height="1"/>'
        "</linearGradient></defs>",
        '<defs><stop offset="0"/></defs>',
        '<defs><linearGradient id="q" x1="a"/></defs>',
        '<defs><linearGradient id="q"><stop stop-color="none"/></linearGradient>'
        "</defs>",
        '<path d="M0 0 H1 V1 Z" fill="url(#q) red"/>',
        '<defs><clipPath id="q"><rect width="1" height="1"/></clipPath></defs>'
        '<path d="M0 0 H1 V1 Z" fill="url(#q)"/>',
    ],
)
def test_unsupported_or_misplaced_gradients_are_reported(body):
    with pytest.raises(DocumentError):
        import_svg('<svg xmlns="http://www.w3.org/2000/svg">' + body + "</svg>")


def test_radial_gradients_are_listed_as_import_issues():
    with pytest.raises(UnsupportedSvgError, match="radialGradient"):
        import_svg(
            '<svg xmlns="http://www.w3.org/2000/svg"><defs><radialGradient id="q">'
            '<stop offset="0"/></radialGradient></defs></svg>'
        )


def test_set_fill_creates_updates_and_removes_a_private_gradient():
    original = import_svg(
        '<svg xmlns="http://www.w3.org/2000/svg"><rect id="r" width="2" '
        'height="2" fill="#00ff00"/><rect id="other" width="1" height="1"/></svg>'
    )
    editor = Editor(original, selection=Selection(object_ids=frozenset({"r"})))
    with editor.transaction("Gradient", allowed=frozenset({EditKind.PAINT})) as tx:
        tx.set_fill("r", RAMP)
    document = editor.snapshot.document
    [gradient] = gradients(document)
    assert gradient.paint_owner == "r"
    assert document.root.children[0].tag == "defs"
    assert document.element("r").get("fill") == f"url(#{gradient.id})"
    assert gradient.get("gradientUnits") == "userSpaceOnUse"
    stops = [s.id for s in gradient.children]

    changed = LinearGradient(
        (1, 1), (3, 3), (GradientStop(0, "#112233"), GradientStop(1, "#445566"))
    )
    with editor.transaction("Again") as tx:
        tx.set_fill("r", changed)
    [updated] = gradients(editor.snapshot.document)
    assert updated.id == gradient.id
    assert [s.id for s in updated.children] == stops
    assert updated.get("x2") == "3.0"

    with editor.transaction("Flat") as tx:
        tx.set_fill("r", "#123456")
    assert editor.snapshot.document.element("r").get("fill") == "#123456"
    assert not gradients(editor.snapshot.document)
    editor.undo()
    assert gradients(editor.snapshot.document) == [updated]
    editor.undo()
    editor.undo()
    assert editor.snapshot.document == original
    editor.redo()
    assert gradients(editor.snapshot.document)[0].id == gradient.id


def test_set_fill_leaves_shared_gradients_and_needs_paint_permission():
    original = import_svg(SVG.replace('fill="#00ff00"', 'fill="url(#g)"'))
    editor = Editor(original, selection=Selection(object_ids=frozenset({"r"})))
    with editor.transaction("Gradient") as tx:
        tx.set_fill("r", RAMP)
    document = editor.snapshot.document
    assert document.element("p").get("fill") == "url(#g)"
    assert document.element("g") == original.element("g")
    own = (document.element("r").get("fill") or "")[5:-1]
    assert own != "g"
    with (
        pytest.raises(EditRejectedError, match="not permitted"),
        editor.transaction("No paint", allowed=frozenset({EditKind.GEOMETRY})) as tx,
    ):
        tx.set_fill("r", "#000000")
    with (
        pytest.raises(EditRejectedError, match="unselected"),
        editor.transaction("Unselected") as tx,
    ):
        tx.set_fill("p", RAMP)
    editor.set_locks("r", frozenset({"fill"}))
    with (
        pytest.raises(EditRejectedError, match="locked"),
        editor.transaction("Locked") as tx,
    ):
        tx.set_fill("r", "#000000")


def test_deleting_the_only_user_keeps_an_explicit_shared_gradient():
    editor = Editor(import_svg(SVG), selection=Selection(object_ids=frozenset({"p"})))
    with editor.transaction("Delete") as tx:
        tx.delete_objects(frozenset({"p"}))
    ids = {e.id for e in editor.snapshot.document.elements()}
    assert "g" in ids
    # A gradient no deleted object used stays.
    assert "base" in ids


def test_a_gradient_counts_as_its_mean_colour_where_one_colour_is_needed():
    document = import_svg(SVG)
    r, g, b, a = mean_colour(document.element("g"))
    assert (round(r, 3), round(g, 3), round(b, 3), round(a, 3)) == (
        0.5,
        0.0,
        0.5,
        0.75,
    )
    assert solid_paint(document, "url(#g)") == "#800080"
    assert solid_paint(document, "#abcdef") == "#abcdef"
    styles = [
        {
            "fill": "url(#g)",
            "fill-opacity": "1",
            "stroke": "none",
            "stroke-opacity": "1",
            "opacity": "1",
            "stroke-width": "1",
            "stroke-miterlimit": "4",
        },
        {
            "fill": "#800080",
            "fill-opacity": "0.5",
            "stroke": "none",
            "stroke-opacity": "1",
            "opacity": "1",
            "stroke-width": "1",
            "stroke-miterlimit": "4",
        },
    ]
    assert average_paint(styles, [1.0, 1.0], document)["fill"] == "#800080"


def private_editor():
    editor = Editor(
        import_svg(
            '<svg width="10" height="10"><path id="p" d="M0 0 H10 V10 H0 Z"/>'
            '<rect id="r" width="1" height="1"/></svg>'
        ),
        selection=Selection.all(),
    )
    with editor.transaction("Private fill") as tx:
        tx.set_fill("p", RAMP)
    return editor


def test_private_gradients_round_trip_in_projects_svg_and_fresh_ids():
    document = private_editor().snapshot.document
    [gradient] = gradients(document)
    assert load_project(save_project(document))[0] == document
    again = import_svg(export_svg(document))
    assert again.element(gradient.id).paint_owner == "p"
    assert export_svg(again) == export_svg(document)
    root = ET.fromstring(export_svg(document))
    exported_gradient = root.find("{*}defs/{*}linearGradient")
    assert exported_gradient is not None
    assert exported_gradient.get("id") == gradient.id
    renamed = import_svg(fresh_ids(export_svg(document)))
    [copied] = gradients(renamed)
    assert copied.id != gradient.id
    assert copied.paint_owner != "p"
    assert renamed.element(copied.paint_owner).get("fill") == f"url(#{copied.id})"


def test_old_projects_keep_shared_gradient_identity_and_appearance():
    original = import_svg(SVG)
    data = json.loads(save_project(original))
    data["version"] = 3

    def strip(element):
        element.pop("paint_owner")
        for child in element["children"]:
            strip(child)

    strip(data["root"])
    loaded, _ = load_project(json.dumps(data))
    assert loaded == original
    assert all(g.paint_owner is None for g in gradients(loaded))
    assert export_svg(loaded) == export_svg(original)


def test_replacing_a_single_user_shared_gradient_preserves_the_shared_asset():
    original = import_svg(SVG)
    editor = Editor(original, selection=Selection.all())
    with editor.transaction("Private fill") as tx:
        tx.set_fill("p", RAMP)
    after = editor.snapshot.document
    assert after.element("g") == original.element("g")
    assert after.element("p").get("fill") != "url(#g)"
    editor.undo()
    assert editor.snapshot.document == original
    editor.redo()
    assert editor.snapshot.document == after


def test_copying_a_shape_copies_its_private_fill_and_stroke():
    editor = private_editor()
    with editor.transaction("Outline") as tx:
        tx.set_attributes("p", {"stroke": tx.preview.element("p").get("fill")})
    original = editor.snapshot.document
    source = original.element("p")
    with editor.transaction("Copy") as tx:
        tx.insert_object(original.root.id, replace(source, id="copy"))
    copied = editor.snapshot.document
    assert copied.element("copy").get("fill") != source.get("fill")
    assert copied.element("copy").get("stroke") == copied.element("copy").get("fill")
    [copy_gradient] = [g for g in gradients(copied) if g.paint_owner == "copy"]
    editor.select(Selection(object_ids=frozenset({"copy"})))
    with editor.transaction("Recolour copy") as tx:
        tx.set_fill("copy", LinearGradient((2, 2), (3, 3), RAMP.stops))
    assert editor.snapshot.document.element(copy_gradient.id).get("x1") == "2.0"
    assert editor.snapshot.document.element("p") == source
    with editor.transaction("Delete copy") as tx:
        tx.delete_objects(frozenset({"copy"}))
    assert gradients(editor.snapshot.document) == gradients(original)
    editor.undo()
    assert editor.snapshot.document.element(copy_gradient.id).paint_owner == "copy"
    editor.redo()
    assert gradients(editor.snapshot.document) == gradients(original)


def test_cutting_private_paint_copies_it_and_joining_collects_old_assets():
    editor = private_editor()
    editor.select(Selection(object_ids=frozenset({"p"})))
    before = editor.snapshot.document
    with editor.transaction("Knife") as tx:
        pieces = tx.cut_paths((-1, 5), (11, 5))
    cut = editor.snapshot.document
    assert len(pieces) == 2
    assert {g.paint_owner for g in gradients(cut)} == set(pieces)
    assert len({cut.element(oid).get("fill") for oid in pieces}) == 2
    editor.select(Selection(object_ids=frozenset(pieces)))
    with editor.transaction("Join") as tx:
        joined = tx.join_paths(frozenset(pieces), color_source="p")
    after = editor.snapshot.document
    [gradient] = gradients(after)
    assert gradient.paint_owner == joined
    editor.undo()
    assert editor.snapshot.document == cut
    editor.undo()
    assert editor.snapshot.document == before


def test_removing_private_fill_keeps_it_while_the_owners_stroke_uses_it():
    editor = private_editor()
    fill = editor.snapshot.document.element("p").get("fill")
    with editor.transaction("Outline") as tx:
        tx.set_attributes("p", {"stroke": fill})
        tx.set_fill("p", "none")
    assert len(gradients(editor.snapshot.document)) == 1
    with editor.transaction("Flat outline") as tx:
        tx.set_attributes("p", {"stroke": "black"})
    assert not gradients(editor.snapshot.document)
    editor.undo()
    assert len(gradients(editor.snapshot.document)) == 1


def test_invalid_private_ownership_is_rejected_on_load():
    document = private_editor().snapshot.document
    [gradient] = gradients(document)
    for invalid in ("missing", "r", document.root.id):
        broken = document.replace_element(replace(gradient, paint_owner=invalid))
        with pytest.raises(DocumentError, match="Private gradients"):
            export_svg(broken)
    broken = document.replace_element(
        replace(document.element("r"), attributes=(("fill", f"url(#{gradient.id})"),))
    )
    with pytest.raises(DocumentError, match="cannot be shared"):
        export_svg(broken)


def test_split_parts_gives_each_shape_its_own_gradient_under_the_wrapper():
    editor = Editor(
        import_svg(
            '<svg width="10" height="10"><path id="p" transform="scale(2)" '
            'd="M0 0 H2 V2 H0 Z M3 0 H5 V2 H3 Z"/></svg>'
        ),
        selection=Selection.all(),
    )
    with editor.transaction("Private paint") as tx:
        tx.set_fill("p", RAMP)
        tx.set_attributes("p", {"stroke": tx.preview.element("p").get("fill")})
    before = editor.snapshot.document
    with editor.transaction("Split parts") as tx:
        pieces = tx.split_disconnected("p")
    after = editor.snapshot.document
    assert len(pieces) == 2
    assert {g.paint_owner for g in gradients(after)} == set(pieces)
    assert after.ancestry(pieces[0])[-2].get("fill") is None
    assert after.ancestry(pieces[0])[-2].get("transform") == "scale(2)"
    for oid in pieces:
        assert after.element(oid).get("fill") == after.element(oid).get("stroke")
    editor.undo()
    assert editor.snapshot.document == before
    editor.redo()
    assert editor.snapshot.document == after
