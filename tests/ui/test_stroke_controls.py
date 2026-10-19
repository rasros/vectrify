"""Stroke controls use paint transactions and preserve editable endpoints."""

import xml.etree.ElementTree as ET

import pytest
from shapely.geometry import Point

from tests.ui.helpers import send
from vectrify.document import DocumentError, HitIndex, import_svg
from vectrify.document.arrows import END, OWNER, START, triangles
from vectrify.document.project import load_project, save_project
from vectrify.document.svg import export_svg
from vectrify.ui.agent import Agent
from vectrify.ui.session import Session

SVG = """<svg width="100" height="100"><g stroke="red" stroke-width="2">
<path id="p" fill="none" d="M20 50 C20 50 60 50 80 50"/>
<path id="closed" d="M0 0 L10 0 L10 10 Z"/></g></svg>"""


def test_caps_joins_arrows_ui_undo_and_project_roundtrip():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["p"])
    before = session.editor.snapshot.document
    nodes = before.geometry_for("p")
    send(
        session,
        "paint",
        changes={
            "stroke-linecap": "round",
            "stroke-linejoin": "bevel",
            START: "arrow",
            END: "arrow",
        },
    )
    doc = session.editor.snapshot.document
    assert doc.geometry_for("p") == nodes
    assert session.state()["objects"][1]["open_path"]
    assert not next(e for e in session.state()["objects"] if e["id"] == "closed")[
        "open_path"
    ]
    assert len(triangles(doc, doc.element("p"), 2)) == 2
    reopened, _ = load_project(save_project(doc))
    assert reopened == doc
    svg = export_svg(doc)
    decorations = [e for e in ET.fromstring(svg).iter() if e.get(OWNER)]
    assert len(decorations) == 2
    assert {e.get("fill") for e in decorations} == {"red"}
    reimported = import_svg(svg)
    assert len(reimported.elements()) == len(doc.elements())
    assert reimported.element("p").get(START) == "arrow"
    send(session, "undo")
    assert session.editor.snapshot.document == before
    send(session, "redo")
    assert session.editor.snapshot.document == doc


def test_arrow_tangents_update_with_nodes_and_painted_hits_include_heads():
    session = Session(import_svg(SVG))
    agent = Agent(session)
    agent.call(
        "properties",
        {
            "seen": [session.epoch, 0],
            "ids": ["p"],
            "start_arrow": "arrow",
            "end_arrow": "arrow",
            "stroke_linecap": "square",
            "stroke_linejoin": "round",
            "stroke_miterlimit": 5,
        },
    )
    doc = session.editor.snapshot.document
    hit = HitIndex(doc)
    area = hit.area("p")
    assert area is not None
    assert area.covers(Point(26, 52))
    assert hit.contours_in("p", Point(26, 52).buffer(0.1))
    g = doc.geometry_for("p")
    end = g.subpaths[0].nodes[-1]
    agent.call(
        "set_points",
        {
            "seen": [session.epoch, 1],
            "changes": {"p": {end.id: [20, 50, 80, 80, 80, 80]}},
        },
    )
    doc = session.editor.snapshot.document
    _, arrow = triangles(doc, doc.element("p"), 2)[-1]
    assert arrow[0] == (80, 80)
    assert arrow[1][1] < 80
    send(session, "select", objects=["p"])
    send(
        session,
        "paint",
        changes={"stroke": "blue", "stroke-width": "3", "stroke-opacity": "0.5"},
    )
    svg = ET.fromstring(session.state()["svg"])
    arrows = [e for e in svg.iter() if e.get(OWNER)]
    assert {e.get("fill") for e in arrows} == {"blue"}
    assert {e.get("fill-opacity") for e in arrows} == {"0.5"}


@pytest.mark.parametrize(
    "changes",
    [
        {"stroke-linecap": "arrow"},
        {"stroke-linejoin": "square"},
        {"stroke-miterlimit": "0.5"},
        {"stroke-miterlimit": "bad"},
        {START: "bad"},
    ],
)
def test_invalid_settings_are_atomic(changes):
    session = Session(import_svg(SVG))
    send(session, "select", objects=["p"])
    before = session.editor.snapshot
    with pytest.raises(DocumentError):
        send(session, "paint", changes=changes)
    assert session.editor.snapshot == before


def test_locked_paint_refuses_endpoint_edits():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["p"])
    send(session, "locks", object="p", locks=["paint"])
    with pytest.raises(DocumentError, match="locked"):
        send(session, "paint", changes={END: "arrow"})


def test_zero_length_and_closed_contours_have_no_arrow_geometry():
    doc = import_svg(
        '<svg width="10" height="10"><path id="p" d="M0 0 L0 0 M1 1 L5 1 L5 5 Z"'
        ' data-vectrify-arrow-end="arrow" stroke="black"/></svg>'
    )
    assert triangles(doc, doc.element("p"), 1) == []


def test_transformed_clipped_arrows_select_the_original_path():
    doc = import_svg(
        '<svg width="100" height="100"><defs><clipPath id="clip">'
        '<rect width="15" height="15"/></clipPath></defs>'
        '<g transform="translate(20 20)"><line id="p" x1="0" y1="10"'
        ' x2="20" y2="10" stroke="red" stroke-width="2" clip-path="url(#clip)"'
        ' data-vectrify-arrow-start="arrow"/></g></svg>'
    )
    hit = HitIndex(doc)
    area = hit.area("p")
    assert area is not None
    assert area.covers(Point(26, 32))
    assert not area.covers(Point(39, 30))
    reopened = import_svg(export_svg(doc))
    assert reopened.element("p").tag == "line"
    assert reopened.element("p").get(START) == "arrow"


def test_converting_arrowed_stroke_to_fill_preserves_arrow_footprint():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["p"])
    send(session, "paint", changes={START: "arrow", END: "arrow"})
    before = HitIndex(session.editor.snapshot.document).area("p")
    send(session, "line_to_fill")
    after = HitIndex(session.editor.snapshot.document).area("p")
    assert before is not None
    assert after is not None
    assert before.symmetric_difference(after).area < 0.1
