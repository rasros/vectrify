"""Hidden objects: display="none" saved with the drawing, never hit."""

import io
from dataclasses import replace

import pytest
from PIL import Image
from shapely.geometry import box

from tests.document.test_document import select
from vectrify.document import (
    DocumentError,
    Editor,
    HitIndex,
    export_svg,
    import_svg,
)
from vectrify.document.project import load_project, save_project
from vectrify.svg_render import render_png

SVG = """<svg width="100" height="100"><g id="layer">
<path id="a" fill="red" d="M0 0 L20 0 L20 20 Z"/>
<rect id="b" x="30" width="20" height="20" fill="blue" opacity="0.5"/>
</g><rect id="c" x="60" width="20" height="20" fill="green"/>
<defs><clipPath id="clip"><rect id="edge" width="10" height="10"/></clipPath></defs>
</svg>"""


def pixel(svg: str, x: int, y: int) -> tuple[int, ...]:
    image = Image.open(io.BytesIO(render_png(svg, (0, 0, 100, 100), (100, 100))))
    colour = image.convert("RGB").getpixel((x, y))
    assert isinstance(colour, tuple)
    return colour


def test_display_none_round_trips_through_svg_and_project():
    source = SVG.replace('id="b"', 'id="b" display="none"')
    document = import_svg(source)
    assert document.element("b").get("display") == "none"
    assert document.hidden("b")
    assert not document.hidden("a")
    again = import_svg(export_svg(document))
    assert again.element("b").get("display") == "none"
    assert load_project(save_project(document))[0] == document


def test_only_display_none_and_inline_on_drawing_objects_are_supported():
    import_svg(SVG.replace('id="b"', 'id="b" display="inline"'))
    with pytest.raises(DocumentError, match="display"):
        import_svg(SVG.replace('id="b"', 'id="b" display="block"'))
    with pytest.raises(DocumentError, match="display"):
        import_svg(SVG.replace('id="clip"', 'id="clip" display="none"'))


def test_a_hidden_group_hides_and_unhits_everything_it_holds():
    document = import_svg(SVG.replace('id="layer"', 'id="layer" display="none"'))
    assert document.hidden("a")
    assert document.hidden("b")
    assert not document.hidden("c")
    hits = HitIndex(document)
    assert hits.area("a") is None
    assert hits.area("layer") is None
    assert not hits.wholly_inside("a", box(-1, -1, 101, 101))
    assert hits.area("c") is not None
    svg = export_svg(document)
    assert pixel(svg, 15, 2) == (255, 255, 255)
    assert pixel(svg, 70, 10) == (0, 128, 0)


def test_hiding_is_one_undoable_edit_that_keeps_paint_and_opacity():
    editor = Editor(import_svg(SVG))
    before = editor.snapshot.document
    with editor.transaction("Hide objects", selection=select("b", "a")) as tx:
        tx.set_hidden("b", True)
        tx.set_hidden("a", True)
    document = editor.snapshot.document
    assert document.element("b").get("display") == "none"
    assert document.element("b").get("opacity") == "0.5"
    assert document.element("b").get("fill") == "blue"
    assert editor.undo_labels == ("Hide objects",)
    with editor.transaction("Show objects", selection=select("b")) as tx:
        tx.set_hidden("b", False)
    assert editor.snapshot.document.element("b").get("display") is None
    editor.undo()
    editor.undo()
    assert editor.snapshot.document == before


def test_hiding_respects_paint_locks_and_refuses_definitions():
    document = import_svg(SVG)
    layer = document.element("layer")
    document = document.replace_element(replace(layer, locks=frozenset({"paint"})))
    editor = Editor(document)
    for target, selected, refusal in (
        ("a", "a", "paint is locked"),
        ("edge", "edge", "definitions"),
        ("c", "b", "Select"),
    ):
        tx = editor.transaction("Hide objects", selection=select(selected))
        with pytest.raises(DocumentError, match=refusal), tx:
            tx.set_hidden(target, True)
    assert all(e.get("display") is None for e in editor.snapshot.document.elements())
