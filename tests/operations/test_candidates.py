"""Mutated SVG is accepted only as ordinary, scope-checked transaction commands."""

import pytest
from PIL import Image

from vectrify.document import (
    DocumentError,
    Editor,
    Selection,
    export_svg,
    import_svg,
)
from vectrify.operations.candidates import CandidateRejectedError, replay

SVG = (
    '<svg width="40" height="40"><g id="g">'
    '<path id="a" d="M0 0L10 0L10 10Z" fill="#ff0000"/>'
    '<rect id="b" x="1" y="1" width="3" height="3" fill="#0000ff"/></g>'
    '<circle id="c" cx="30" cy="30" r="4" fill="#00ff00"/></svg>'
)


def editor(*ids, **permissions):
    ed = Editor(import_svg(SVG), selection=Selection(object_ids=frozenset(ids)))
    allowed = frozenset(k for k, v in permissions.items() if v)
    return ed, ed.transaction("Improve", allowed=allowed)


def exported(ed):
    return export_svg(ed.snapshot.document)


def test_paint_and_node_changes_replay_in_place_keeping_node_ids():
    ed, tx = editor("a", geometry=True, paint=True)
    before = ed.snapshot.document.geometry_for("a")
    svg = export_svg(ed.snapshot.document)
    svg = svg.replace('fill="#ff0000"', 'fill="#ee1100"').replace(
        "L10.0 0.0", "L11.5 0.5"
    )
    assert replay(tx, svg).edits == 2
    tx.commit()
    after = ed.snapshot.document.geometry_for("a")
    assert [n.id for s in after.subpaths for n in s.nodes] == [
        n.id for s in before.subpaths for n in s.nodes
    ]
    assert after.subpaths[0].nodes[1].values == (11.5, 0.5)
    assert ed.snapshot.document.element("a").get("fill") == "#ee1100"
    assert ed.undo_labels == ("Improve",)


def test_changes_outside_the_selection_are_rejected():
    ed, tx = editor("a", geometry=True, paint=True)
    svg = exported(ed).replace('fill="#00ff00"', 'fill="#00ee00"')
    with pytest.raises(DocumentError, match="unselected"):
        replay(tx, svg)


def test_changes_outside_the_permissions_are_rejected():
    ed, tx = editor("a", geometry=True)
    svg = exported(ed).replace('fill="#ff0000"', 'fill="#ee1100"')
    with pytest.raises(DocumentError, match="not permitted"):
        replay(tx, svg)


def test_pinned_endpoints_cannot_move():
    ed, _ = editor("a", geometry=True)
    node = ed.snapshot.document.geometry_for("a").subpaths[0].nodes[1]
    ed.pin_node("a", node.id)
    tx = ed.transaction("Improve", allowed=frozenset({"geometry"}))
    svg = exported(ed).replace("L10.0 0.0", "L11.0 0.0")
    with pytest.raises(DocumentError, match="pinned"):
        replay(tx, svg)


def test_sibling_reorder_replays_as_a_structure_edit():
    ed, tx = editor("g", structure=True)
    svg = exported(ed)
    a = svg[svg.index("<path") : svg.index("/>", svg.index("<path")) + 2]
    b = svg[svg.index("<rect") : svg.index("/>", svg.index("<rect")) + 2]
    svg = svg.replace(a, "\0").replace(b, a).replace("\0", b)
    replay(tx, svg)
    tx.commit()
    assert [c.id for c in ed.snapshot.document.element("g").children] == ["b", "a"]


def test_structural_changes_are_rejected():
    ed, tx = editor("a", geometry=True, structure=True)
    svg = exported(ed).replace("L10.0 10.0 Z", "L10.0 10.0 L5.0 12.0 Z")
    with pytest.raises(CandidateRejectedError, match="structure"):
        replay(tx, svg)
    ed, tx = editor("g", geometry=True)
    removed = exported(ed)
    start = removed.index("<rect")
    removed = removed[:start] + removed[removed.index("/>", start) + 2 :]
    with pytest.raises(DocumentError, match="not permitted"):
        replay(tx, removed)
    ed, tx = editor("g", structure=True)
    replay(tx, removed)
    tx.commit()
    assert [c.id for c in ed.snapshot.document.element("g").children] == ["a"]


def test_unchanged_candidate_makes_no_edits():
    ed, tx = editor("a", geometry=True, paint=True)
    assert replay(tx, exported(ed)).edits == 0
    assert tx.preview == ed.snapshot.document


def test_scoped_search_pool_replays_through_the_transaction():
    from vectrify.image_utils import rasterize_svg as rasterize
    from vectrify.operations import OperationRequest, Permissions
    from vectrify.operations.candidates import mutation_scope
    from vectrify.vector.reference import Reference
    from vectrify.vector.search import SearchSettings, run_search, seed_node
    from vectrify.vector.worker import WorkerContext

    ed = Editor(import_svg(SVG), selection=Selection(object_ids=frozenset({"a"})))
    request = OperationRequest(
        action="improve",
        method="nsga",
        snapshot=ed.snapshot,
        editor=ed,
        permissions=Permissions(paint=True),
    )
    source = export_svg(ed.snapshot.document)
    png = rasterize(source, 40, 40)
    reference = Reference.build(
        Image.new("RGB", (40, 40), "white"),
        score_resolution=40,
        segment_count=1,
    )
    context = WorkerContext(
        scope=mutation_scope(request),
        original_png_bytes=reference.png,
        original_w=40,
        original_h=40,
        log_level="ERROR",
        random_seed=3,
    )
    seed = seed_node(reference, source, png, node_id=1, origin="drawing")
    outcome = run_search(
        reference,
        [seed],
        context,
        SearchSettings(pool_size=4, max_total_tasks=40, epochs=1),
    )
    assert outcome.pool
    changed = 0
    for node in outcome.pool:
        tx = request.transaction("Improve")
        changed += replay(tx, node.state.payload.content).edits > 0
        for other in ("b", "c"):
            assert tx.preview.element(other) == ed.snapshot.document.element(other)
    assert changed
