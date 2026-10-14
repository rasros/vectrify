"""Set clip and Remove clip: the person names the clipped objects and the boundary."""

import io

import numpy as np
import pytest
from PIL import Image

from vectrify.document import DocumentError, export_svg, import_svg
from vectrify.svg_render import render_png
from vectrify.ui.agent import Agent
from vectrify.ui.session import Session

# A body with a hole, inside a moved group; shading inside a rotated and a
# scaled group, covering the whole drawing; a red rectangle on top; and a
# rectangle and an open line, neither of which can be a boundary.
SVG = """<svg width="100" height="100">
<g id="region" transform="translate(10 5)"><path id="body" fill="green"
 fill-rule="evenodd" d="M10 15 L60 15 L60 65 L10 65 Z M25 30 L45 30 L45 50 L25 50 Z"/>
</g>
<g id="outer" transform="translate(100 0) rotate(90)"><g id="paint"
 transform="scale(2)"><path id="shade" fill="blue" d="M0 0 L50 0 L50 50 L0 50 Z"/>
</g></g>
<rect id="spot" x="0" y="0" width="30" height="30" fill="red"/>
<rect id="box" x="70" y="70" width="20" height="20" fill="none"/>
<path id="stroke" fill="none" stroke="black" d="M0 90 L40 90"/>
</svg>"""

BLUE, GREEN, RED, WHITE = (0, 0, 255), (0, 128, 0), (255, 0, 0), (255, 255, 255)


def send(session, command, **data):
    return session.action(
        {
            "command": command,
            "epoch": session.epoch,
            "revision": session.editor.snapshot.revision,
            **data,
        }
    )


def pixels(document_or_svg):
    svg = (
        document_or_svg
        if isinstance(document_or_svg, str)
        else export_svg(document_or_svg)
    )
    png = render_png(svg, (0, 0, 100, 100), (100, 100))
    return np.array(Image.open(io.BytesIO(png)).convert("RGB"))


def at(image, x, y):
    return tuple(int(v) for v in image[y, x])


def clip_of(document, oid):
    return document.element(document.element(oid).get("clip-path")[5:-1])


def set_clip(session, targets, boundaries, linked=True):
    send(session, "select", objects=[*targets, *boundaries])
    return send(
        session, "set_clip", targets=targets, boundaries=boundaries, linked=linked
    )


def move_body_corner(session, x, y):
    """Drag the body's first corner in Nodes, as the person would."""
    node = session.editor.snapshot.document.geometry_for("body").subpaths[0].nodes[0]
    send(session, "select", objects=["body"], nodes=[node.id])
    send(session, "move_nodes", changes={"body": {node.id: [x, y]}})


def test_set_clip_clips_through_nested_transforms_keeping_holes_and_one_undo():
    session = Session(import_svg(SVG))
    before = session.editor.snapshot.document
    result = set_clip(session, ["paint"], ["body"])
    document = session.editor.snapshot.document
    assert result["undo"] == ["Set clip"]
    assert result["selection"]["objects"] == ["paint"]
    # The boundary is read, never moved or repainted; the shading keeps its
    # paths, its place in the stack and its groups' transforms.
    for oid in ("body", "shade", "region", "spot"):
        assert document.element(oid) == before.element(oid)
    assert document.element("outer").children[0].id == "paint"
    # The body sits at 20..70 on the page with its hole at 35..55; the
    # shading is clipped to it there although its groups rotate and scale.
    image = pixels(document)
    assert at(image, 30, 25) == BLUE
    assert at(image, 65, 65) == BLUE
    assert at(image, 45, 45) == WHITE  # the hole
    assert at(image, 75, 50) == WHITE  # outside the body
    assert at(image, 10, 10) == RED  # the rectangle on top is not clipped
    send(session, "undo")
    assert session.editor.snapshot.document == before
    send(session, "redo")
    assert np.array_equal(pixels(session.editor.snapshot.document), image)


def test_linked_boundary_reshapes_every_clip_it_bounds_and_copied_keeps_its_outline():
    session = Session(import_svg(SVG))
    set_clip(session, ["paint", "spot"], ["body"])
    document = session.editor.snapshot.document
    # One boundary, shared: each target has a clip of its own, in its own
    # coordinates, and both share the body's points.
    paint, spot = clip_of(document, "paint"), clip_of(document, "spot")
    assert paint.id != spot.id
    body = document.geometry_for("body")
    for clip in (paint, spot):
        assert document.geometry_for(clip.children[0].id) is body
    image = pixels(document)
    assert at(image, 25, 25) == RED
    assert at(image, 10, 10) == WHITE
    # Pulling the body's corner out to the page's corner reshapes both clips.
    move_body_corner(session, -10, -5)
    image = pixels(session.editor.snapshot.document)
    assert at(image, 10, 10) == RED
    assert at(image, 15, 8) == RED

    copied = Session(import_svg(SVG))
    set_clip(copied, ["paint", "spot"], ["body"], linked=False)
    document = copied.editor.snapshot.document
    assert document.geometry_for(clip_of(document, "spot").children[0].id) != body
    move_body_corner(copied, -10, -5)
    image = pixels(copied.editor.snapshot.document)
    assert at(image, 10, 10) == GREEN  # the body, with nothing clipped there
    assert at(image, 25, 25) == RED


def test_clip_survives_project_save_and_svg_export_and_stays_linked():
    session = Session(import_svg(SVG))
    set_clip(session, ["paint"], ["body"])
    document = session.editor.snapshot.document
    image = pixels(document)

    reopened = Session(import_svg("<svg/>"))
    reopened.open(session.project(), "clipped.vectrify")
    assert export_svg(reopened.editor.snapshot.document) == export_svg(document)

    svg = export_svg(document)
    assert 'clip-path="url(#' in svg
    imported = Session(import_svg(svg))
    loaded = imported.editor.snapshot.document
    assert np.array_equal(pixels(loaded), image)
    boundary = clip_of(loaded, "paint").children[0].id
    assert loaded.geometry_for(boundary) is loaded.geometry_for("body")
    # Still linked after reopening: reshaping the body reshapes the clip.
    move_body_corner(imported, -10, -5)
    assert at(pixels(imported.editor.snapshot.document), 40, 15) == BLUE


def test_a_boundary_can_be_inside_its_target():
    session = Session(import_svg(SVG))
    set_clip(session, ["region", "outer"], ["body"])
    document = session.editor.snapshot.document
    document.validate()
    assert document.element("region").get("clip-path")
    image = pixels(document)
    assert at(image, 30, 25) == BLUE
    assert at(image, 75, 50) == WHITE


def test_locks_on_the_boundary_do_not_block_it_being_read():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["body"])
    send(session, "locks", object="body", locks=["geometry", "paint", "structure"])
    set_clip(session, ["outer"], ["body"])
    document = session.editor.snapshot.document
    assert document.element("outer").get("clip-path")
    assert at(pixels(document), 45, 45) == WHITE


@pytest.mark.parametrize(
    ("selected", "data", "message"),
    [
        (["paint", "box"], {"targets": ["paint"], "boundaries": ["box"]}, "Convert"),
        (
            ["paint", "stroke"],
            {"targets": ["paint"], "boundaries": ["stroke"]},
            "closed",
        ),
        (
            ["paint", "body", "spot"],
            {"targets": ["paint"], "boundaries": ["body"]},
            "one role",
        ),
        (
            ["paint", "body"],
            {"targets": ["paint", "body"], "boundaries": ["body"]},
            "not both",
        ),
        (["paint", "body"], {"targets": ["paint"], "boundaries": []}, "boundary"),
        (["paint", "body"], {"boundaries": ["body"]}, "objects to clip"),
        (
            ["paint", "body"],
            {"targets": ["paint"], "boundaries": ["body"], "linked": "yes"},
            "linked or copied",
        ),
        (
            ["outer", "paint", "body"],
            {"targets": ["outer", "paint"], "boundaries": ["body"]},
            "parent and child",
        ),
    ],
)
def test_set_clip_refusals_change_nothing(selected, data, message):
    session = Session(import_svg(SVG))
    send(session, "select", objects=selected)
    before = session.editor.snapshot
    with pytest.raises(DocumentError, match=message):
        send(session, "set_clip", **data)
    assert session.editor.snapshot == before


def test_a_locked_target_refuses_set_and_remove():
    session = Session(import_svg(SVG))
    set_clip(session, ["paint"], ["body"])
    send(session, "select", objects=["paint"])
    send(session, "locks", object="paint", locks=["structure"])
    before = session.editor.snapshot
    with pytest.raises(DocumentError, match="locked"):
        send(session, "remove_clip")
    with pytest.raises(DocumentError, match="locked"):
        set_clip(session, ["paint"], ["body"])
    assert session.editor.snapshot.document == before.document


def test_remove_clip_unclips_only_the_clipped_and_keeps_the_boundary():
    session = Session(import_svg(SVG))
    set_clip(session, ["paint"], ["body"])
    clipped = session.editor.snapshot.document
    clip = clip_of(clipped, "paint").id
    send(session, "select", objects=["spot"])
    send(session, "locks", object="spot", locks=["structure"])
    send(session, "select", objects=["paint", "spot"])
    before = session.editor.snapshot.document
    result = send(session, "remove_clip")
    document = session.editor.snapshot.document
    assert result["undo"][-1] == "Remove clip"
    assert result["selection"]["objects"] == ["paint", "spot"]
    assert document.element("paint").get("clip-path") is None
    assert clip not in {e.id for e in document.elements()}
    assert document.element("body") == clipped.element("body")
    assert at(pixels(document), 75, 50) == BLUE
    send(session, "undo")
    assert session.editor.snapshot.document == before
    send(session, "select", objects=["body", "spot"])
    with pytest.raises(DocumentError, match="None of the selected objects"):
        send(session, "remove_clip")


def test_editor_and_mcp_clip_alike():
    session = Session(import_svg(SVG))
    set_clip(session, ["paint", "spot"], ["body"])
    agent = Agent(Session(import_svg(SVG)))
    agent.call(
        "clip",
        {
            "seen": [agent.session.epoch, 0],
            "ids": ["paint", "spot"],
            "boundary": ["body"],
        },
    )
    ours, theirs = (s.editor.snapshot.document for s in (session, agent.session))
    assert np.array_equal(pixels(ours), pixels(theirs))
    for oid in ("paint", "spot"):
        mine, other = clip_of(ours, oid), clip_of(theirs, oid)
        assert [c.attributes for c in mine.children] == [
            c.attributes for c in other.children
        ]
