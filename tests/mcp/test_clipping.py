from __future__ import annotations

import io

import anyio
import numpy as np
import pytest
from mcp import Client
from PIL import Image

from vectrify.document import DocumentError, Selection, export_svg, import_svg
from vectrify.document.model import Element
from vectrify.mcp.server import build_server
from vectrify.svg_render import cached_rendering, render_image, render_png
from vectrify.ui.agent import Agent
from vectrify.ui.session import Session

SVG = """<svg width="100" height="100">
<g id="region" transform="translate(10 5)">
<path id="body" fill="green" fill-rule="evenodd"
 d="M10 10 L60 10 L60 60 L10 60 Z M25 25 L45 25 L45 45 L25 45 Z"/>
<g id="paint"><path id="shade" fill="blue"
 d="M0 0 L90 0 L90 90 L0 90 Z"/></g>
<path id="outline" fill="none" stroke="black" stroke-width="2"
 d="M10 10 L60 10 L60 60 L10 60 Z"/></g>
<path id="other" fill="red" d="M80 80 L95 80 L95 95 L80 95 Z"/>
</svg>"""


def make(svg=SVG):
    return Agent(Session(import_svg(svg)))


def call(agent, tool, **args):
    return agent.call(
        tool,
        {"seen": [agent.session.epoch, agent.session.editor.snapshot.revision], **args},
    ).data


def pixels(document):
    png = render_png(export_svg(document), (0, 0, 100, 100), (100, 100))
    return np.array(Image.open(io.BytesIO(png)).convert("RGBA"))


def test_existing_body_clips_oversized_and_new_shadows_with_holes_and_outline_last():
    agent = make()
    before = agent.session.editor.snapshot.document
    reply = call(agent, "clip", ids=["paint"], boundary=["body"])
    after = agent.session.editor.snapshot.document
    assert after.element("region").children[-1].id == "outline"
    assert after.element("paint").children == before.element("paint").children
    for oid in ("body", "shade", "outline", "other"):
        assert before.element(oid) == after.element(oid)
    clip = after.element(reply["clipping"][0]["clip"])
    assert after.geometry_for(clip.children[0].id) is after.geometry_for("body")
    p = pixels(after)
    assert tuple(p[20, 30, :3]) == (0, 0, 255)
    assert tuple(p[40, 45, :3]) == (255, 255, 255)  # hole
    assert tuple(p[10, 10, :3]) == (255, 255, 255)  # outside body
    assert tuple(p[15, 40, :3]) == (0, 0, 0)  # outline remains above clipping
    call(agent, "add_path", parent="paint", d="M0 0 L100 0 L100 100 Z", fill="yellow")
    assert tuple(pixels(agent.session.editor.snapshot.document)[10, 10, :3]) == (
        255,
        255,
        255,
    )
    call(agent, "undo", ids=[reply["edit_id"]])  # independent later insertion remains
    assert (
        agent.session.editor.snapshot.document.element("paint").get("clip-path") is None
    )


def test_boundary_can_be_inside_target_without_cyclic_references():
    agent = make()
    call(agent, "clip", ids=["region"], boundary=["body"])
    doc = agent.session.editor.snapshot.document
    doc.validate()
    assert tuple(pixels(doc)[10, 10, :3]) == (255, 255, 255)
    loaded = Session(import_svg(SVG))
    loaded.open(agent.session.project(), "clipped.vectrify")
    assert export_svg(loaded.editor.snapshot.document) == export_svg(doc)


def test_inset_matches_beveled_stroke_at_acute_inward_corner():
    agent = make("""<svg width="100" height="100">
    <path id="body" fill="green" stroke="black" stroke-width="2"
      d="M10 10 L45 10 L50 70 L55 10 L90 10 L90 90 L10 90 Z"/>
    <path id="shade" fill="blue" d="M0 0 L100 0 L100 100 L0 100 Z"/>
    </svg>""")
    before = agent.session.editor.snapshot.document
    call(agent, "clip", ids=["shade"], boundary=["body"], inset=1)
    after = agent.session.editor.snapshot.document
    assert after.element("body") == before.element("body")
    p = pixels(after)
    assert tuple(p[68, 50, :3]) == (0, 0, 0)  # outline stays visible
    # SVG bevels a miter exceeding its limit; the clip must stop there too,
    # rather than extending a rectangular unshaded tab below the notch.
    assert tuple(p[72, 50, :3]) == (0, 0, 255)


@pytest.mark.parametrize("linked", [True, False])
def test_boundary_node_updates_follow_only_linked_geometry(linked):
    agent = make()
    clipped = call(agent, "clip", ids=["paint"], boundary=["body"], linked=linked)
    doc = agent.session.editor.snapshot.document
    clip = doc.element(clipped["clipping"][0]["clip"])
    shape = clip.children[0].id
    node = doc.geometry_for("body").subpaths[0].nodes[1]
    call(agent, "set_points", coords="local", changes={"body": {node.id: [70, 10]}})
    after = agent.session.editor.snapshot.document
    expected = 70 if linked else 60
    assert after.geometry_for(shape).subpaths[0].nodes[1].endpoint == (expected, 10)
    assert (after.geometry_for(shape).id == after.geometry_for("body").id) == linked


def test_union_of_boundaries_and_differently_transformed_targets():
    agent = make("""<svg width="100" height="100">
    <g transform="translate(10 20) scale(2)"><path id="a"
      d="M0 0 L10 0 L10 10 L0 10 Z"/></g>
    <path id="b" d="M50 20 L70 20 L70 40 L50 40 Z"/>
    <g id="target" transform="translate(5 0)"><path id="paint"
      fill="blue" d="M0 0 L100 0 L100 100 L0 100 Z"/></g>
    <path id="second" transform="scale(2)" fill="red"
      d="M0 0 L50 0 L50 50 L0 50 Z"/></svg>""")
    call(agent, "clip", ids=["target", "second"], boundary=["a", "b"])
    doc = agent.session.editor.snapshot.document
    p = pixels(doc)
    assert tuple(p[25, 15, :3]) == (255, 0, 0)
    assert tuple(p[25, 55, :3]) == (255, 0, 0)
    assert tuple(p[25, 40, :3]) == (255, 255, 255)
    assert doc.element("target").get("clip-path") != doc.element("second").get(
        "clip-path"
    )


def test_replacing_and_removing_only_collects_owned_unreferenced_clips():
    agent = make()
    first = call(agent, "clip", ids=["paint"], boundary=["body"])["clipping"][0]["clip"]
    second = call(agent, "clip", ids=["paint"], boundary=["body"], linked=False)[
        "clipping"
    ][0]["clip"]
    doc = agent.session.editor.snapshot.document
    assert first not in {e.id for e in doc.elements()}
    removed = call(agent, "clip", ids=["paint"], action="remove")
    doc = agent.session.editor.snapshot.document
    assert second not in {e.id for e in doc.elements()}
    assert doc.element("paint").get("clip-path") is None
    assert not call(agent, "clip", ids=["paint"], action="remove")["changed"]
    call(agent, "undo", ids=[removed["edit_id"]])
    assert agent.session.editor.snapshot.document.element("paint").get("clip-path") == (
        f"url(#{second})"
    )


def test_remove_preserves_imported_shared_boundary():
    agent = make("""<svg width="100" height="100"><defs>
    <clipPath id="old"><path d="M0 0 L50 0 L50 50 Z"/></clipPath></defs>
    <g id="one" clip-path="url(#old)"/><g id="two" clip-path="url(#old)"/></svg>""")
    old = agent.session.editor.snapshot.document.element("old")
    call(agent, "clip", ids=["one"], action="remove")
    doc = agent.session.editor.snapshot.document
    assert doc.element("old") == old
    assert doc.element("two").get("clip-path") == "url(#old)"


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ({"ids": ["paint"]}, "boundary"),
        ({"ids": ["paint"], "boundary": ["shade"], "action": "remove"}, "without"),
        ({"ids": ["paint"], "boundary": ["outline"], "action": "bad"}, "action"),
        ({"ids": ["region", "paint"], "boundary": ["body"]}, "parent and child"),
        ({"ids": ["paint"], "boundary": ["paint"]}, "must be paths"),
        ({"ids": ["paint"], "boundary": ["body"], "linked": "yes"}, "boolean|true"),
    ],
)
def test_invalid_requests_do_not_change_drawing(args, match):
    agent = make()
    before = agent.session.editor.snapshot.document
    with pytest.raises(DocumentError, match=match):
        call(agent, "clip", **args)
    assert agent.session.editor.snapshot.document == before


def test_open_boundary_and_singular_transforms_refuse_without_partial_edit():
    agent = make("""<svg width="100" height="100">
    <path id="open" d="M0 0 L40 0 L40 40"/>
    <path id="closed" d="M0 0 L40 0 L40 40 Z"/>
    <g id="good"/><g id="singular" transform="scale(0)"/></svg>""")
    before = agent.session.editor.snapshot.document
    with pytest.raises(DocumentError, match="closed"):
        call(agent, "clip", ids=["good"], boundary=["open"])
    with pytest.raises(DocumentError):
        call(agent, "clip", ids=["good", "singular"], boundary=["closed"])
    assert agent.session.editor.snapshot.document == before


def test_structure_locks_refuse_and_unrelated_locks_do_not_block():
    agent = make()
    call(agent, "properties", ids=["other"], locks=["structure"])
    call(agent, "clip", ids=["paint"], boundary=["body"])
    call(agent, "properties", ids=["paint"], locks=["structure"])
    before = agent.session.editor.snapshot.document
    with pytest.raises(DocumentError, match="locked"):
        call(agent, "clip", ids=["paint"], action="remove")
    assert agent.session.editor.snapshot.document == before


def test_document_command_respects_explicit_scope_and_preserves_imported_resources():
    agent = make()
    editor = agent.session.editor
    with (
        pytest.raises(DocumentError, match="unselected"),
        editor.transaction("Clipping", selection=Selection(frozenset({"other"}))) as tx,
    ):
        tx.set_clip("paint", Element("clip", "clipPath"))
    assert editor.snapshot.revision == 0


def test_batch_preview_is_private_apply_and_undo_are_single_steps():
    agent = make()
    before = agent.session.editor.snapshot.document
    staged = call(
        agent,
        "edit_batch",
        edits=[{"tool": "clip", "args": {"ids": ["paint"], "boundary": ["body"]}}],
    )
    assert agent.session.editor.snapshot.document == before
    applied = call(agent, "edit_batch", action="apply", id=staged["id"])
    assert agent.session.editor.snapshot.revision == 1
    assert agent.session.editor.snapshot.document.element("paint").get("clip-path")
    call(agent, "undo", ids=[applied["edit_id"]])
    assert agent.session.editor.snapshot.document == before


def test_mcp_tool_schema_and_file_target(tmp_path):
    path = tmp_path / "test.svg"
    path.write_text(SVG)

    async def run():
        async with Client(build_server()) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            assert "clip" in tools
            assert "linked" in tools["clip"].input_schema["properties"]
            assert "inset" in tools["clip"].input_schema["properties"]
            await client.call_tool("open", {"path": str(path)})
            clipped = await client.call_tool(
                "clip", {"ids": ["paint"], "boundary": ["body"], "inset": 2}
            )
            assert not clipped.is_error
            await client.call_tool("save", {})
            assert 'clip-path="url(#' in path.read_text()
            assert 'data-vectrify-clip-inset="2.0"' in path.read_text()

    anyio.run(run)


def test_reflected_overlapping_boundaries_union_and_cairo_previews_agree():
    agent = make("""<svg width="100" height="100">
    <path id="a" d="M10 10 L60 10 L60 60 L10 60 Z"/>
    <path id="b" transform="translate(80 0) scale(-1 1)"
      d="M10 10 L60 10 L60 60 L10 60 Z"/>
    <g id="paint"><path fill="blue" d="M0 0 L100 0 L100 100 L0 100 Z"/></g>
    </svg>""")
    call(agent, "clip", ids=["paint"], boundary=["a", "b"])
    doc = agent.session.editor.snapshot.document
    p = pixels(doc)
    assert tuple(p[30, 30, :3]) == (0, 0, 255)  # overlap of opposite windings
    assert tuple(p[30, 65, :3]) == (0, 0, 255)
    with cached_rendering():
        for box in (None, (0, 0, 100, 100)):
            image = render_image(export_svg(doc), box, (100, 100))
            assert np.array_equal(np.array(image), p[:, :, :3])


def test_locked_definitions_refuse_without_changing_targets():
    agent = make(
        '<svg width="100" height="100"><defs id="defs"/>'
        '<path id="body" d="M0 0 L50 0 L50 50 Z"/><g id="paint"/></svg>'
    )
    call(agent, "properties", ids=["defs"], locks=["structure"])
    before = agent.session.editor.snapshot.document
    with pytest.raises(DocumentError, match="locked"):
        call(agent, "clip", ids=["paint"], boundary=["body"])
    assert agent.session.editor.snapshot.document == before


def test_new_path_in_transformed_clipped_group_keeps_root_coordinates_and_index():
    agent = make()
    call(agent, "clip", ids=["paint"], boundary=["body"])
    added = call(
        agent,
        "add_path",
        parent="paint",
        index=0,
        name="New shadow",
        d="M24 18 L34 18 L34 28 L24 28 Z",
        fill="yellow",
    )
    doc = agent.session.editor.snapshot.document
    created = added["result"]["objects"][0]
    assert doc.element("paint").children[0].id == created
    assert doc.element(created).name == "New shadow"
    # First draw is covered by the original blue shadow. Move it forward
    # within the same clipped parent, then verify no transformed displacement.
    call(agent, "arrange", ids=[created], to="front")
    p = pixels(agent.session.editor.snapshot.document)
    assert tuple(p[22, 28, :3]) == (255, 255, 0)
    assert tuple(p[22, 40, :3]) == (0, 0, 255)


@pytest.mark.parametrize("inset", [0, 2])
def test_persons_node_drag_updates_linked_boundary_and_keeps_selection(inset):
    agent = make()
    call(agent, "clip", ids=["paint"], boundary=["body"], inset=inset)
    doc = agent.session.editor.snapshot.document
    node = doc.geometry_for("body").subpaths[0].nodes[1]
    selection = Selection(frozenset({"body"}), frozenset({node.id}))
    agent.session.editor.select(selection)
    agent.session.action(
        {
            "command": "node",
            "object": "body",
            "node": node.id,
            "values": [75, 10],
            "epoch": agent.session.epoch,
            "revision": agent.session.editor.snapshot.revision,
        }
    )
    doc = agent.session.editor.snapshot.document
    clip = doc.element(doc.element("paint").get("clip-path")[5:-1])
    assert doc.geometry_for(clip.children[0].id).node(node.id).endpoint == (75, 10)
    assert agent.session.editor.snapshot.selection == selection


def test_inset_keeps_body_outline_visible_below_shadows_and_expands_holes():
    from shapely.geometry import Point

    from vectrify.document.hit_test import HitIndex
    from vectrify.document.join import clip_outline

    agent = make()
    call(agent, "properties", ids=["body"], stroke="black", stroke_width=4)
    # The fixture's separate outline is unnecessary; keep only the body's stroke.
    call(agent, "delete", ids=["outline"])
    before = agent.session.editor.snapshot.document
    reply = call(agent, "clip", ids=["paint"], boundary=["body"], inset=2)
    doc = agent.session.editor.snapshot.document
    assert doc.element("body") == before.element("body")
    assert doc.element("region").children[-1].id == "paint"
    p = pixels(doc)
    assert tuple(p[16, 40, :3]) == (0, 0, 0)  # inner half of the body's top stroke
    assert tuple(p[20, 40, :3]) == (0, 0, 255)
    assert tuple(p[29, 40, :3]) == (0, 0, 0)  # inner-hole outline
    index = HitIndex(doc, tolerance=0.01)
    area = index.area("shade")
    assert area is not None
    assert not area.covers(Point(40, 16))
    assert not area.covers(Point(40, 29))
    assert area.covers(Point(40, 20))
    clip = doc.element(reply["clipping"][0]["clip"])
    assert clip_outline(doc, clip).area == pytest.approx(46 * 46 - 24 * 24)


CORNER_SVG = """<svg width="100" height="100">
<g id="style" stroke-linejoin="round" stroke-miterlimit="4">
<path id="body" fill="green" stroke="black" stroke-width="8"
 d="M10 10 H90 V90 H50 V50 H10 Z"/></g>
<path id="shade" fill="blue" d="M0 0 H100 V100 H0 Z"/>
</svg>"""


@pytest.mark.parametrize("join", ["miter", "round", "bevel"])
@pytest.mark.parametrize("limit", [1, 4])
def test_inset_matches_inherited_corner_join_and_miter_limit(join, limit):
    agent = make(CORNER_SVG)
    call(
        agent,
        "properties",
        ids=["style"],
        stroke_linejoin=join,
        stroke_miterlimit=limit,
    )
    call(agent, "clip", ids=["shade"], boundary=["body"], inset=4)
    doc = agent.session.editor.snapshot.document
    # At a reflex corner the inset must leave exactly the body's stroke
    # exposed: a hardcoded miter leaves green tabs beside rounded outlines.
    expected = (0, 0, 0) if join == "miter" and limit == 4 else (0, 0, 255)
    assert tuple(pixels(doc)[46, 53, :3]) == expected
    svg = export_svg(doc)
    loaded = import_svg(svg)
    assert export_svg(loaded) == svg
    assert np.array_equal(pixels(doc), pixels(loaded))
    reopened = Session(import_svg(CORNER_SVG))
    reopened.open(agent.session.project(), "corners.vectrify")
    assert export_svg(reopened.editor.snapshot.document) == svg


@pytest.mark.parametrize("linked", [True, False])
def test_inset_corner_settings_follow_linked_source_changes_and_undo(linked):
    from vectrify.document.clipping import clip_geometry

    agent = make(CORNER_SVG)
    result = call(
        agent, "clip", ids=["shade"], boundary=["body"], inset=4, linked=linked
    )
    doc = agent.session.editor.snapshot.document
    boundary = doc.element(result["clipping"][0]["clip"]).children[0]
    rounded = clip_geometry(doc, boundary).path_data()
    changed = call(agent, "properties", ids=["style"], stroke_linejoin="miter")
    doc = agent.session.editor.snapshot.document
    mitered = clip_geometry(doc, doc.element(boundary.id)).path_data()
    assert (mitered != rounded) == linked
    call(agent, "properties", ids=["style"], stroke_miterlimit=1)
    doc = agent.session.editor.snapshot.document
    assert (
        clip_geometry(doc, doc.element(boundary.id)).path_data() != mitered
    ) == linked
    call(agent, "undo", ids=[changed["edit_id"]])
    doc = agent.session.editor.snapshot.document
    assert clip_geometry(doc, doc.element(boundary.id)).path_data() == rounded


@pytest.mark.parametrize("linked", [True, False])
def test_inset_follows_source_edits_only_when_linked_and_survives_round_trips(linked):
    from vectrify.document.clipping import clip_geometry

    agent = make()
    reply = call(
        agent, "clip", ids=["paint"], boundary=["body"], inset=2, linked=linked
    )
    doc = agent.session.editor.snapshot.document
    boundary_id = doc.element(reply["clipping"][0]["clip"]).children[0].id
    prior = clip_geometry(doc, doc.element(boundary_id)).path_data()
    node = doc.geometry_for("body").subpaths[0].nodes[1]
    call(agent, "set_points", coords="local", changes={"body": {node.id: [75, 10]}})
    doc = agent.session.editor.snapshot.document
    assert (clip_geometry(doc, doc.element(boundary_id)).path_data() != prior) == linked
    reopened = Session(import_svg(SVG))
    reopened.open(agent.session.project(), "inset.vectrify")
    assert export_svg(reopened.editor.snapshot.document) == export_svg(doc)
    svg = export_svg(doc)
    loaded = import_svg(svg)
    assert export_svg(loaded) == svg  # no double inset
    assert np.array_equal(pixels(doc), pixels(loaded))
    if linked:
        assert loaded.geometry_for(boundary_id) is loaded.geometry_for("body")
        again = Agent(Session(loaded))
        node = loaded.geometry_for("body").subpaths[0].nodes[2]
        call(again, "set_points", coords="local", changes={"body": {node.id: [80, 60]}})
        changed = again.session.editor.snapshot.document
        assert clip_geometry(changed, changed.element(boundary_id)).path_data() != (
            clip_geometry(loaded, loaded.element(boundary_id)).path_data()
        )


@pytest.mark.parametrize("large", [20, 30, 100, 1e300])
def test_large_inset_collapses_curved_shapes_without_inverting(large):
    agent = make("""<svg width="100" height="100"><path id="body"
    d="M10 30 C10 5 60 5 60 30 C60 55 10 55 10 30 Z"/>
    <path id="shade" fill="blue" d="M0 0 L100 0 L100 100 L0 100 Z"/></svg>""")
    reply = call(agent, "clip", ids=["shade"], boundary=["body"], inset=2)
    doc = agent.session.editor.snapshot.document
    boundary = doc.element(reply["clipping"][0]["clip"]).children[0]
    assert doc.geometry_for(boundary.id) is doc.geometry_for("body")
    assert any(
        n.command == "C" for s in doc.geometry_for("body").subpaths for n in s.nodes
    )
    p = pixels(doc)
    assert tuple(p[14, 35, :3]) == (0, 0, 255)
    assert tuple(p[12, 35, :3]) == (0, 0, 0)
    call(agent, "clip", ids=["shade"], boundary=["body"], inset=large)
    assert not np.any(
        np.all(
            pixels(agent.session.editor.snapshot.document)[:, :, :3] == [0, 0, 255],
            axis=2,
        )
    )


@pytest.mark.parametrize("inset", [-1, float("nan"), float("inf")])
def test_invalid_insets_refuse_atomically(inset):
    agent = make()
    before = agent.session.editor.snapshot.document
    with pytest.raises(DocumentError):
        call(agent, "clip", ids=["paint"], boundary=["body"], inset=inset)
    assert agent.session.editor.snapshot.document == before


def test_inset_transforms_with_source_and_batch_undo_restores_original():
    agent = make("""<svg width="100" height="100">
    <g transform="translate(10 10) scale(2 3)"><path id="body"
      fill="green" stroke="black" stroke-width="2"
      d="M0 0 L20 0 L20 20 L0 20 Z"/></g>
    <g id="paint" transform="translate(5 0)"><path id="shade" fill="blue"
      d="M0 0 L100 0 L100 100 L0 100 Z"/></g></svg>""")
    before = agent.session.editor.snapshot.document
    stage = call(
        agent,
        "edit_batch",
        edits=[
            {
                "tool": "clip",
                "args": {"ids": ["paint"], "boundary": ["body"], "inset": 1},
            }
        ],
    )
    assert agent.session.editor.snapshot.document == before
    reply = call(agent, "edit_batch", action="apply", id=stage["id"])
    p = pixels(agent.session.editor.snapshot.document)
    assert tuple(p[11, 30, :3]) == (0, 0, 0)
    assert tuple(p[15, 30, :3]) == (0, 0, 255)
    call(agent, "undo", ids=[reply["edit_id"]])
    assert agent.session.editor.snapshot.document == before
