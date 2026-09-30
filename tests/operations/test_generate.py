"""Generate methods insert reference-space SVG into the document as one edit."""

import time

import pytest
from PIL import Image

from vectrify.document import (
    DocumentError,
    Editor,
    Selection,
    import_svg,
)
from vectrify.operations import Job, OperationRequest, Permissions, method
from vectrify.operations.generate import (
    generated_result,
    render_region,
    target_region,
)

DOC = (
    '<svg width="200" height="100" viewBox="10 20 200 100">'
    '<rect id="bg" x="10" y="20" width="200" height="100" fill="#ffffff"/>'
    '<g id="layer"/></svg>'
)


def reference():
    # 400x200 px over a 200x100 artboard: two pixels per document unit.
    image = Image.new("RGB", (400, 200), "white")
    image.paste((200, 0, 0), (100, 50, 300, 150))
    return image


def request(editor, selection, **extra):
    snapshot = editor.snapshot
    return OperationRequest(
        action="generate",
        method=extra.pop("method", "samvg"),
        snapshot=type(snapshot)(snapshot.revision, snapshot.document, selection),
        editor=editor,
        permissions=Permissions(structure=True),
        reference=reference(),
        **extra,
    )


SQUARE = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200">'
    '<path d="M100 50H300V150H100Z" fill="#c80000"/></svg>'
)


def test_generated_svg_maps_reference_pixels_onto_the_artboard():
    original = import_svg(DOC)
    editor = Editor(original)
    req = request(editor, Selection.all())
    region = target_region(req)
    assert (region.x, region.y, region.width, region.height) == (10, 20, 200, 100)
    result = generated_result(req, SQUARE, region, label="Generate", name="Trace")
    proposal = result.recommended
    assert proposal.changed
    assert proposal.metrics["after"]["error"] < proposal.metrics["before"]["error"]
    assert proposal.metrics["after"]["error"] < 1e-3
    assert proposal.metrics["shapes"] == 1
    assert set(proposal.previews) == {"reference", "before", "after"}
    proposal.transaction.commit()
    assert editor.undo_labels == ("Generate",)
    group = editor.snapshot.document.root.children[-1]
    assert (group.tag, group.name) == ("g", "Trace")
    assert group.get("transform") == "matrix(0.5 0 0 0.5 10.0 20.0)"
    editor.undo()
    assert editor.snapshot.document == original


def test_the_region_is_the_selection_with_a_margin_or_else_the_artboard():
    doc = DOC.replace(
        '<g id="layer"/>',
        '<rect id="spot" x="60" y="45" width="100" height="50"/><g id="layer"/>',
    )
    editor = Editor(import_svg(doc))
    req = request(editor, Selection(object_ids=frozenset({"spot"})))
    region = target_region(req)
    # 10% of the longer side on every edge.
    assert (region.x, region.y, region.width, region.height) == (50, 35, 120, 70)
    assert region.image.size == (240, 140)
    assert region.image.getpixel((0, 0)) == (255, 255, 255)
    assert render_region(req.snapshot.document, region).size == region.image.size
    for selection in (
        Selection(whole_document=True),
        Selection(object_ids=frozenset({"layer"})),
    ):
        region = target_region(request(editor, selection))
        assert (region.x, region.y, region.width, region.height) == (10, 20, 200, 100)


def test_generation_goes_into_a_selected_group_or_needs_the_whole_drawing():
    editor = Editor(import_svg(DOC))
    req = request(editor, Selection(object_ids=frozenset({"layer"})))
    result = generated_result(
        req, SQUARE, target_region(req), label="Generate", name="Trace"
    )
    result.recommended.transaction.commit()
    layer = editor.snapshot.document.element("layer")
    assert [c.tag for c in layer.children] == ["g"]
    with pytest.raises(DocumentError, match="whole drawing or one group"):
        method("generate", "samvg").validate(
            request(editor, Selection(object_ids=frozenset({"bg"})))
        )


def test_samvg_job_inserts_the_trace(monkeypatch):
    seen = {}

    def fake(image, **kwargs):
        seen.update(kwargs, size=image.size)
        return SQUARE

    monkeypatch.setattr("vectrify.refine.samvg.generate_svg", fake)
    editor = Editor(import_svg(DOC))
    job = Job(
        method("generate", "samvg"),
        request(editor, Selection.all(), settings={"max_layers": 8, "max_side": 400}),
    )
    job.run()
    state = job.state(preview=True)
    assert state["status"] == "ready", state
    assert seen["max_layers"] == 8
    assert seen["size"] == (400, 200)
    job.apply()
    assert editor.undo_labels == ("Generate with SAMVG",)


def test_samvg_traces_a_small_reference_enlarged_and_places_it_the_same(
    monkeypatch,
):
    seen = {}

    def fake(image, **kwargs):
        seen.update(kwargs, size=image.size)
        # The red square, in the enlarged image's pixels.
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="400">'
            '<path d="M200 100H600V300H200Z" fill="#c80000"/></svg>'
        )

    monkeypatch.setattr("vectrify.refine.samvg.generate_svg", fake)
    editor = Editor(import_svg(DOC))
    job = Job(
        method("generate", "samvg"),
        request(
            editor,
            Selection.all(),
            settings={"max_side": 800},
        ),
    )
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    # Twice the size, so pixel settings double and areas quadruple.
    assert seen["size"] == (800, 400)
    assert seen["min_width"] == 6
    assert seen["min_pixels"] == 128
    assert seen["tolerance"] == 1.0
    # Placed exactly over the reference's square: no error left.
    assert state["result"]["metrics"]["after"]["error"] < 1e-3


def test_samvg_segments_at_the_reference_size_by_default(monkeypatch):
    seen = {}

    def fake(image, **kwargs):
        seen.update(kwargs, size=image.size)
        return SQUARE

    monkeypatch.setattr("vectrify.refine.samvg.generate_svg", fake)
    job = Job(
        method("generate", "samvg"),
        request(Editor(import_svg(DOC)), Selection.all(), settings={}),
    )
    job.run()
    assert job.state()["status"] == "ready"
    # The 400x200 reference is neither shrunk nor enlarged.
    assert seen["size"] == (400, 200)
    assert seen["max_side"] == 400


def test_samvg_rejects_bad_settings_and_missing_permission():
    editor = Editor(import_svg(DOC))
    samvg = method("generate", "samvg")
    with pytest.raises(DocumentError, match="Unknown SAMVG setting"):
        samvg.validate(request(editor, Selection.all(), settings={"ocr": True}))
    with pytest.raises(DocumentError, match="whole number"):
        samvg.validate(request(editor, Selection.all(), settings={"max_layers": 2.5}))
    with pytest.raises(DocumentError, match="structure"):
        samvg.validate(
            OperationRequest(
                action="generate",
                method="samvg",
                snapshot=editor.snapshot,
                editor=editor,
                reference=reference(),
            )
        )


def test_artboard_defaults_without_a_viewbox():
    assert import_svg('<svg width="30" height="40"/>').artboard() == (0, 0, 30, 40)
    assert import_svg('<svg viewBox="1,2 3 4"/>').artboard() == (1, 2, 3, 4)


def test_session_scope_drawing_generates_without_selecting_everything(monkeypatch):
    import base64
    import io

    from vectrify.ui.session import Session

    monkeypatch.setattr("vectrify.refine.samvg.generate_svg", lambda *_a, **_k: SQUARE)
    stream = io.BytesIO()
    reference().save(stream, format="PNG")
    session = Session(
        import_svg(DOC),
        reference={
            "name": "ref.png",
            "opacity": 0.5,
            "data_url": "data:image/png;base64,"
            + base64.b64encode(stream.getvalue()).decode(),
        },
    )
    payload = {
        "command": "start",
        "action": "generate",
        "method": "samvg",
        "epoch": session.epoch,
        "revision": 0,
        "permissions": {"structure": True},
    }
    with pytest.raises(DocumentError, match="whole drawing or one group"):
        session.operation(payload)
    job = session.operation(dict(payload, scope="drawing"))
    for _ in range(500):
        state = session.operation({"command": "status", "job": job["id"]})
        if state["status"] != "running":
            break
        time.sleep(0.01)
    assert state["status"] == "ready", state
    session.operation({"command": "apply", "job": job["id"]})
    assert session.editor.snapshot.selection == Selection()
    assert session.editor.undo_labels == ("Generate with SAMVG",)


def test_fresh_ids_rename_definitions_and_every_reference():
    from vectrify.operations.generate import fresh_ids

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink">'
        '<defs><path id="r" d="M0 0H4V4Z"/><clipPath id="c"><use href="#r"/>'
        '</clipPath></defs><use xlink:href="#r" clip-path="url(#c)" fill="red"/>'
        "</svg>"
    )
    renamed = fresh_ids(svg)
    assert 'id="r"' not in renamed
    assert "#c)" not in renamed
    document = import_svg(renamed)
    ids = {e.id for e in document.elements()}
    uses = [e for e in document.elements() if e.tag == "use"]
    assert all((u.get("href") or "")[1:] in ids for u in uses)
    clipped = next(u for u in uses if u.get("clip-path"))
    assert (clipped.get("clip-path") or "")[5:-1] in ids


def test_colour_regions_job_inserts_generated_regions(monkeypatch):
    torch = pytest.importorskip("torch")
    seen = {}

    def fake(image, **kwargs):
        seen.update(kwargs, size=image.size)
        return SQUARE, {"seconds": 0.1, "palette_colours": 2}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("vectrify.refine.colour_regions.vectorize", fake)
    editor = Editor(import_svg(DOC))
    job = Job(
        method("generate", "colour-regions"),
        request(
            editor,
            Selection.all(),
            method="colour-regions",
            settings={"colours": 8, "outline_style": "clean"},
        ),
    )
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    assert state["result"]["metrics"]["colours"] == 2
    assert (seen["colours"], seen["outline_style"], seen["size"]) == (
        8,
        "clean",
        (400, 200),
    )
    job.apply()
    assert editor.undo_labels == ("Generate colour regions",)
    with pytest.raises(DocumentError, match="Choose a supported outline style"):
        method("generate", "colour-regions").validate(
            request(editor, Selection.all(), settings={"outline_style": "wavy"})
        )
