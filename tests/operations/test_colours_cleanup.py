"""Closed-form colour fitting and selection-limited geometry cleanup."""

import pytest
from PIL import Image

from vectrify.document import DocumentError, Editor, Selection, import_svg
from vectrify.operations import Job, OperationRequest, Permissions, method

DOC = (
    '<svg width="100" height="100"><rect id="bg" width="100" height="100" '
    'fill="#ffffff"/><g opacity="0.5"><rect id="a" x="10" y="10" width="40" '
    'height="40" fill="#000000"/></g><rect id="front" x="30" y="30" width="40" '
    'height="40" fill="#0000ff"/></svg>'
)


def render(svg, size=100):
    import io

    import cairosvg

    png = cairosvg.svg2png(
        bytestring=svg.encode(), output_width=size, output_height=size
    )
    return Image.open(io.BytesIO(png)).convert("RGB")


def request(editor, name, selection, permissions, reference=None, **settings):
    snapshot = editor.snapshot
    return OperationRequest(
        action="improve" if name == "colours" else "simplify",
        method=name,
        snapshot=type(snapshot)(snapshot.revision, snapshot.document, selection),
        editor=editor,
        permissions=permissions,
        settings=settings,
        reference=reference,
        bounds=(0, 0, 100, 100),
    )


def test_colour_fit_recovers_the_fill_through_opacity_and_occlusion():
    target = render(DOC.replace('fill="#000000"', 'fill="#c83214"'))
    ed = Editor(import_svg(DOC))
    job = Job(
        method("improve", "colours"),
        request(
            ed,
            "colours",
            Selection(object_ids=frozenset({"a"})),
            Permissions(paint=True),
            target,
            resolution=100,
        ),
    )
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    assert state["result"]["metrics"]["after"]["error"] < 1e-4
    job.apply()
    fill = ed.snapshot.document.element("a").get("fill")
    r, g, b = (int(fill[i : i + 2], 16) for i in (1, 3, 5))
    assert abs(r - 0xC8) <= 2
    assert abs(g - 0x32) <= 2
    assert abs(b - 0x14) <= 2
    assert ed.snapshot.document.element("front").get("fill") == "#0000ff"
    assert ed.undo_labels == ("Fit colours",)


def test_colour_fit_needs_paint_permission_and_fillable_objects():
    ed = Editor(import_svg(DOC))
    fit = method("improve", "colours")
    selection = Selection(object_ids=frozenset({"a"}))
    with pytest.raises(DocumentError, match="paint"):
        fit.validate(request(ed, "colours", selection, Permissions(), render(DOC)))
    ed = Editor(import_svg(DOC.replace('fill="#000000"', 'fill="none"')))
    with pytest.raises(DocumentError, match="solid fill"):
        fit.validate(
            request(ed, "colours", selection, Permissions(paint=True), render(DOC))
        )


CLEAN = (
    '<svg width="100" height="100"><g id="g">'
    '<path id="p" d="M10 10 L20 10 L30 10 L30 20 L10 20 Z" fill="#ff0000"/>'
    '<path id="q" d="M50 10 L60 10 L60 20 L50 20 Z" fill="#ff0000"/>'
    '<path id="fixed" d="M70 10 L75 10 L80 10 L80 20 L70 20 Z" fill="#ff0000"/>'
    '<path id="r" d="M85 10 L90 10 L90 20 L85 20 Z" fill="#ff0000"/>'
    "</g></svg>"
)


def test_cleanup_merges_and_simplifies_only_the_selection():
    ed = Editor(import_svg(CLEAN))
    fixed = ed.snapshot.document.geometry_for("fixed")
    job = Job(
        method("simplify", "cleanup"),
        request(
            ed,
            "cleanup",
            Selection(object_ids=frozenset({"p", "q", "r"})),
            Permissions(geometry=True, structure=True),
        ),
    )
    job.start()
    state = job.state(preview=True)
    assert state["status"] == "ready", state
    assert state["result"]["changed"]
    job.apply()
    document = ed.snapshot.document
    ids = [c.id for c in document.element("g").children]
    assert "fixed" in ids
    assert "r" in ids
    assert len(ids) == 3
    assert document.geometry_for("fixed") == fixed
    merged = document.geometry_for(ids[0])
    assert len(merged.subpaths) == 2
    assert len(merged.subpaths[0].nodes) == 4


def test_cleanup_needs_geometry_and_structure():
    ed = Editor(import_svg(CLEAN))
    with pytest.raises(DocumentError, match="geometry and structure"):
        method("simplify", "cleanup").validate(
            request(ed, "cleanup", Selection.all(), Permissions(geometry=True))
        )
