from __future__ import annotations

import pytest

from vectrify.document import DocumentError, export_svg, import_svg
from vectrify.ui.agent import Agent
from vectrify.ui.session import Session

SVG = """<svg width="100" height="100">
<g fill="green" transform="translate(2 1)">
<path id="fills" d="M8 69 L13 9 L23 69 Z M68 69 L73 9 L83 69 Z"/>
</g>
<path id="sand" fill="tan" d="M0 72 L100 72 L100 100 L0 100 Z"/>
<path id="shading" fill="darkgreen" d="M10 70 L15 10 L18 70 Z M70 70 L75 10 L78 70 Z"/>
<path id="ink" fill="none" stroke="black"
d="M10 70 L15 10 L25 70 Z M70 70 L75 10 L85 70 Z"/>
</svg>"""


def fresh():
    agent = Agent(Session(import_svg(SVG)))
    return agent, [agent.session.epoch, 0]


def membership(agent):
    doc = agent.session.editor.snapshot.document
    return [
        {
            "object": oid,
            "contours": [doc.geometry_for(oid).subpaths[0].id],
            "role": role,
            "include": True,
        }
        for oid, role in [("fills", "fill"), ("shading", "shading"), ("ink", "outline")]
    ]


def test_isolate_complete_tuft_across_large_paths_without_changing_neighbor():
    from vectrify.ui.agent import render_document

    agent, seen = fresh()
    before = agent.session.editor.snapshot.document
    preview = agent.call("isolate_feature", {"seed": [17, 40], "radius": 34}).data
    assert {"fills", "shading", "ink"} <= {c["object"] for c in preview["candidates"]}
    assert all(c["role"] is None for c in preview["candidates"])
    batch = agent.call(
        "isolate_feature",
        {
            "seen": seen,
            "action": "stage",
            "region": [5, 5, 25, 66],
            "members": membership(agent),
            "name": "Left tuft",
        },
    ).data
    assert agent.session.editor.snapshot.document == before
    assert len(batch["feature"]["source_mapping"]) == 3
    assert not batch["feature"]["boundary_cuts"]
    agent.call("edit_batch", {"seen": seen, "action": "apply", "id": batch["id"]})
    after = agent.session.editor.snapshot.document
    group = after.element(batch["feature"]["group"])
    assert group.name == "Left tuft"
    assert after.element("sand") == before.element("sand")
    for oid in ("fills", "shading", "ink"):
        assert after.geometry_for(oid).subpaths == before.geometry_for(oid).subpaths[1:]
    for side in (100, 300):
        a = render_document(export_svg(before), before.artboard(), (side, side))
        b = render_document(export_svg(after), after.artboard(), (side, side))
        assert a.tobytes() == b.tobytes()
    loaded = Session(import_svg("<svg/>"))
    loaded.open(agent.session.project(), "feature.vectrify")
    assert loaded.editor.snapshot.document.element(group.id).get(
        "data-vectrify-feature-sources"
    )


def test_membership_is_explicit_and_excluded_outline_stays_in_place():
    agent, seen = fresh()
    before = agent.session.editor.snapshot.document
    chosen = membership(agent)
    chosen[2]["include"] = False
    reply = agent.call(
        "isolate_feature",
        {"seen": seen, "action": "stage", "region": [5, 5, 25, 66], "members": chosen},
    ).data
    assert len(reply["feature"]["source_mapping"]) == 2
    assert agent._batches[reply["id"]].after.geometry_for("ink") == before.geometry_for(
        "ink"
    )
    chosen[0].pop("include")
    with pytest.raises(DocumentError, match="explicit role"):
        agent.call(
            "isolate_feature",
            {
                "seen": seen,
                "action": "stage",
                "region": [5, 5, 25, 66],
                "members": chosen,
            },
        )


def test_grouping_refuses_to_cross_overlapping_unrelated_geometry():
    agent, seen = fresh()
    agent.call(
        "add_path",
        {"seen": seen, "d": "M12 20 L30 20 L30 40 L12 40 Z", "fill": "red", "index": 2},
    )
    before = agent.session.editor.snapshot.document
    with pytest.raises(DocumentError, match="stacking"):
        agent.call(
            "isolate_feature",
            {
                "seen": [seen[0], 1],
                "action": "stage",
                "region": [5, 5, 25, 66],
                "members": membership(agent),
            },
        )
    assert agent.session.editor.snapshot.document == before


def test_boundary_fragments_report_cuts_and_preserve_outside_geometry():
    agent, seen = fresh()
    chosen = membership(agent)[:1]
    with pytest.raises(DocumentError, match="not contained"):
        agent.call(
            "isolate_feature",
            {
                "seen": seen,
                "action": "stage",
                "region": [5, 5, 25, 40],
                "members": chosen,
            },
        )
    batch = agent.call(
        "isolate_feature",
        {
            "seen": seen,
            "action": "stage",
            "region": [5, 5, 25, 40],
            "members": chosen,
            "cut": True,
        },
    ).data
    assert batch["feature"]["boundary_cuts"][0]["object"] == "fills"
    before = agent.session.editor.snapshot.document
    after = agent._batches[batch["id"]].after
    assert (
        after.geometry_for("fills").subpaths[0]
        == before.geometry_for("fills").subpaths[1]
    )


def test_isolation_transfers_protected_tip_and_retains_single_undo():
    agent, seen = fresh()
    node = (
        agent.session.editor.snapshot.document.geometry_for("fills")
        .subpaths[0]
        .nodes[1]
    )
    agent.call(
        "protect_features",
        {"seen": seen, "points": [["fills", node.id]], "kind": "tip"},
    )
    before = agent.session.editor.snapshot.document
    batch = agent.call(
        "isolate_feature",
        {
            "seen": [seen[0], 1],
            "action": "stage",
            "region": [5, 5, 25, 66],
            "members": membership(agent),
        },
    ).data
    applied = agent.call(
        "edit_batch", {"seen": [seen[0], 1], "action": "apply", "id": batch["id"]}
    ).data
    after = agent.session.editor.snapshot.document
    piece = next(
        m["object"]
        for m in batch["feature"]["source_mapping"]
        if m["source"] == "fills"
    )
    assert after.geometry_for(piece).node(node.id).feature == "tip"
    agent.call("undo", {"seen": [seen[0], 2], "ids": [applied["edit_id"]]})
    assert agent.session.editor.snapshot.document == before
