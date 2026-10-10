from __future__ import annotations

import io

import numpy as np
from PIL import Image

from vectrify.document import export_svg, import_svg
from vectrify.svg_render import render_png
from vectrify.ui.agent import Agent
from vectrify.ui.session import Session

SVG = """<svg width="100" height="100">
<path id="sand" fill="tan" d="M0 65 L100 65 L100 100 L0 100 Z"/>
<g transform="translate(10 5)">
<path id="tuft" fill="green" d="M10 65 L15 10 L25 65 L35 5 L40 65 Z"/>
<path id="light" fill="lime" stroke="yellow" d="M10 65 L15 10 L25 65 Z"/></g></svg>"""


def test_outline_shares_geometry_renders_above_highlight_and_roundtrips():
    agent = Agent(Session(import_svg(SVG)))
    seen = [agent.session.epoch, 0]
    result = agent.call("linked_outline", {"seen": seen, "id": "tuft", "width": 3}).data
    doc = agent.session.editor.snapshot.document
    outline = result["relationship"]["outline"]
    assert doc.geometry_for(outline) is doc.geometry_for("tuft")
    assert doc.element(outline).get("fill") == "none"
    assert doc.element("light").get("stroke") == "yellow"
    assert doc.root.children[-1].name == "Outlines"
    png = render_png(export_svg(doc), (0, 0, 100, 100), (100, 100))
    pixels = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    assert pixels[15, 25].max() < 50
    node = doc.geometry_for("tuft").subpaths[0].nodes[1]
    agent.call(
        "set_points",
        {
            "seen": [seen[0], 1],
            "coords": "local",
            "changes": {"tuft": {node.id: [15, 8]}},
        },
    )
    doc = agent.session.editor.snapshot.document
    assert doc.geometry_for(outline).node(node.id).endpoint == (15, 8)
    assert doc.geometry_for(outline) is doc.geometry_for("tuft")
    loaded = Session(import_svg(SVG))
    loaded.open(agent.session.project(), "tuft.vectrify")
    assert (
        loaded.editor.snapshot.document.geometry_for(outline).id
        == doc.geometry_for("tuft").id
    )
    updated = agent.call(
        "linked_outline",
        {
            "seen": [seen[0], 2],
            "id": "tuft",
            "colour": "navy",
            "width": 2,
            "layer": "Ink",
        },
    ).data
    assert updated["relationship"]["outline"] == outline
    assert (
        agent.session.editor.snapshot.document.element(outline).get("stroke") == "navy"
    )


def test_outline_in_transformed_existing_layer_stays_on_source_geometry():
    from vectrify.document.model import Element
    from vectrify.document.regions import object_matrix

    document = import_svg(SVG)
    layer = Element(
        "outlines", "g", attributes=(("transform", "translate(20 0)"),), name="Outlines"
    )
    from dataclasses import replace

    document = document.replace_element(
        replace(document.root, children=(*document.root.children, layer))
    )
    agent = Agent(Session(document))
    reply = agent.call(
        "linked_outline", {"seen": [agent.session.epoch, 0], "id": "tuft"}
    ).data
    after = agent.session.editor.snapshot.document
    outline = reply["relationship"]["outline"]
    assert object_matrix(after, outline) == object_matrix(after, "tuft")
