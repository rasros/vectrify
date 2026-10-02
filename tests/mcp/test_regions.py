"""Finding what is under a spot, reading and taking the contours in a region,
and the looks that map images onto the drawing, through the MCP client."""

from __future__ import annotations

import io
import math
import re

import anyio
import numpy as np
from mcp import Client
from PIL import Image

from tests.mcp.helpers import data, error, images
from vectrify.mcp.server import Vectrify, build_server
from vectrify.mcp.target import FileTarget

# A cel trace's layout: a face fill and one figure-wide ink path of many
# contours, in a group scaled by a half and moved; a square on top elsewhere.
# The ink's left eye (contour 1) is the box 70-90 x 85-105 in the document,
# and contour 4 is a stroke across it at y 95 from x 50 to 120.
SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" \
viewBox="0 0 200 200">
<rect id="bg" width="200" height="200" fill="#eeeeee"/>
<g id="cel" transform="matrix(0.5 0 0 0.5 20 10)">
<path id="face" d="M40 40 L360 40 L360 360 L40 360 Z" fill="#f0c090"/>
<path id="ink" d="M60 60 L340 60 M100 150 L140 150 L140 190 L100 190 Z \
M260 150 L300 150 L300 190 L260 190 Z M100 300 L300 300 M60 170 L200 170" \
fill="none" stroke="#111111" stroke-width="6"/>
</g>
<path id="top" d="M150 0 L200 0 L200 50 L150 50 Z" fill="#ff0000"/>
<path id="blob" d="M120 120 L180 120 L180 180 L120 180 Z" fill="#0000ff"/>
<path id="ring" d="M0 150 L40 150 L40 190 L0 190 Z M10 160 L10 180 L30 180 L30 160 Z" \
fill="#00aa00"/>
</svg>
"""
EYE = [60, 80, 40, 30]


def dark_disc_png() -> bytes:
    """White, with a black disc of radius 30 at (100, 100)."""
    ys, xs = np.mgrid[0:200, 0:200]
    pixels = np.full((200, 200, 3), 255, dtype=np.uint8)
    pixels[(xs + 0.5 - 100) ** 2 + (ys + 0.5 - 100) ** 2 <= 30**2] = 10
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, "PNG")
    return buffer.getvalue()


def run(tmp_path, body, state: Vectrify | None = None) -> None:
    drawing = tmp_path / "cel.svg"
    drawing.write_text(SVG)

    async def session():
        async with Client(build_server(state)) as client:

            async def call(tool_name: str, /, **args):
                return await client.call_tool(tool_name, args)

            data(await call("open", path=str(drawing)))
            await body(call)

    anyio.run(session)


def test_pick_finds_painted_objects_and_contours_front_to_back(tmp_path):
    async def body(call):
        # On the eye's left stroke: the ink's contour 1, over the face.
        on_stroke = data(await call("pick", x=70, y=88))
        assert [o["id"] for o in on_stroke["objects"]] == ["ink", "face", "bg"]
        ink = on_stroke["objects"][0]
        assert ink["groups"] == ["cel"]
        assert [c["index"] for c in ink["contours"]] == [1]
        eye = ink["contours"][0]
        assert eye["paints"] == "stroke"
        assert eye["closed"]
        assert eye["count"] == 4
        # The eye's bounds in the document, not the ink's own coordinates.
        assert eye["bounds"] == [70, 85, 20, 20]
        # Inside the eye the ink paints nothing: its bounds are not a hit.
        inside = data(await call("pick", x=80, y=100))
        assert [o["id"] for o in inside["objects"]] == ["face", "bg"]
        # The stroke's width counts: 1.4 units off the line is still on it.
        assert data(await call("pick", x=68.6, y=100))["objects"][0]["id"] == "ink"
        assert data(await call("pick", x=67, y=100))["objects"][0]["id"] == "face"
        # A radius reaches it; overlapping objects come front to back.
        assert data(await call("pick", x=67, y=100, radius=2))["objects"][0]["id"] == (
            "ink"
        )
        top = data(await call("pick", x=175, y=25))
        assert [o["id"] for o in top["objects"]] == ["top", "bg"]
        # describe(region) lists what paints there, with the contours.
        described = data(await call("describe", region=EYE))
        assert [o["id"] for o in described["objects"]] == ["ink", "face", "bg"]
        assert {c["index"] for c in described["objects"][0]["contours"]} == {1, 4}

    run(tmp_path, body)


def test_points_filters_to_a_region_in_document_coordinates(tmp_path):
    async def body(call):
        listed = data(await call("points", id="ink", region=EYE))
        assert listed["contours_total"] == 5
        assert [c["index"] for c in listed["contours"]] == [1, 4]
        assert listed["transform"] == "matrix(0.5 0 0 0.5 20 10)"
        eye, across = listed["contours"]
        assert [n["values"] for n in eye["nodes"]] == [
            [70, 85],
            [90, 85],
            [90, 105],
            [70, 105],
        ]
        # Only the nodes inside the region: the stroke across has none there.
        assert "nodes" not in across
        assert across["bounds"] == [50, 95, 70, 0]
        local = data(await call("points", id="ink", contours=[1], coords="both"))
        first = local["contours"][0]["nodes"][0]
        assert first["values"] == [70, 85]
        assert first["local"] == [100, 150]
        # set_points takes document coordinates too.
        data(await call("set_points", changes={"ink": {first["id"]: [72, 86]}}))
        moved = data(await call("points", id="ink", contours=[1], coords="local"))
        assert moved["contours"][0]["nodes"][0]["values"] == [104, 152]
        # Pages, with a marker saying there is more.
        paged = data(await call("points", id="ink", page_size=5))
        assert paged["nodes_total"] == 14
        assert paged["pages"] == 3
        assert "page=1" in paged["more"]
        assert sum(len(c["nodes"]) for c in paged["contours"]) == 5
        summary = data(await call("points", id="ink", nodes=False))
        assert len(summary["contours"]) == 5
        assert "nodes" not in summary["contours"][0]

    run(tmp_path, body)


def test_extract_takes_a_region_out_cutting_strokes_across_it(tmp_path):
    async def body(call):
        before = data(await call("history"))["undo_total"]
        taken = data(await call("extract", region=EYE))
        assert taken["step"] == "Agent: Extract region"
        assert data(await call("history"))["undo_total"] == before + 1
        (piece,) = taken["extracted"]
        assert piece["from"] == "ink"
        assert piece["contours"] == 2
        new = piece["path"]
        # The face is filled around the region, not cut by it.
        assert "face" not in taken.get("removed", [])
        moved = data(await call("points", id=new))
        bounds = sorted(c["bounds"] for c in moved["contours"])
        # The eye whole, and the stroke across cut at the region's edges.
        assert bounds == [[60, 95, 40, 0], [70, 85, 20, 20]]
        rest = data(await call("points", id="ink"))
        assert sorted(c["bounds"] for c in rest["contours"])[:2] == [
            [50, 40, 140, 0],
            [50, 95, 10, 0],
        ]
        assert rest["contours_total"] == 5
        described = data(await call("describe", within="cel"))
        ids = [o["id"] for o in described["objects"]]
        # Same paint and group, just above the ink.
        assert ids == ["face", "ink", new]
        assert described["objects"][2]["paint"]["stroke"] == "#111111"
        # Without cutting only whole contours go.
        data(await call("undo"))
        whole = data(await call("extract", region=EYE, cut=False))
        assert whole["extracted"][0]["contours"] == 1
        data(await call("undo"))
        # A region may be a polygon.
        triangle = [[40, 70], [140, 70], [90, 140]]
        inside = data(await call("extract", region=triangle, ids=["ink"], cut=False))
        assert inside["extracted"][0]["contours"] == 1
        data(await call("undo"))
        # A fill crossing the edge is split along it when named.
        split = data(await call("extract", region=[100, 100, 50, 100], ids=["blob"]))
        (blob,) = split["extracted"]
        left = data(await call("points", id=blob["path"]))
        assert left["contours"][0]["bounds"] == [120, 120, 30, 60]
        right = data(await call("points", id="blob"))
        assert right["contours"][0]["bounds"] == [150, 120, 30, 60]
        refused = await call("extract", region=[0, 150, 10, 10], ids=["blob"])
        assert "No contour" in error(refused)
        # A shape around the region is not carved by it, nor its hole taken.
        around = await call("extract", region=[5, 155, 30, 30], ids=["ring"])
        assert "No contour" in error(around)
        # Deleting in a region: the eye goes, the stroke across stays whole.
        deleted = data(await call("delete", region=EYE))
        assert deleted["step"] == "Agent: Delete in region"
        after = data(await call("points", id="ink"))
        assert after["contours_total"] == 4

    run(tmp_path, body)


def test_renders_say_how_pixels_map_and_can_carry_a_grid(tmp_path):
    async def body(call):
        rendered = await call("render", region=[0, 0, 100, 50], max_side=200)
        mapping = data(rendered)["mapping"]
        assert mapping["units_per_pixel"] == [0.5, 0.5]
        assert mapping["pixels"] == [200, 100]
        assert "document (0 + px * 0.5, 0 + py * 0.5)" in mapping["text"]
        gridded = await call("render", region=[0, 0, 100, 50], max_side=200, grid=True)
        assert images(gridded)[0] != images(rendered)[0]

    reference = tmp_path / "disc.png"
    reference.write_bytes(dark_disc_png())

    async def with_reference(call):
        await body(call)
        data(await call("load_reference", path=str(reference)))
        side = data(await call("render", overlay="side", max_side=400, grid=True))
        assert "The reference starts 204 pixels in" in side["mapping"]["text"]
        compared = data(await call("compare", region=[0, 0, 100, 100], max_side=64))
        assert compared["mapping"]["units_per_pixel"] == [1.5625, 1.5625]

    run(tmp_path, with_reference)


def test_trace_reference_outlines_a_dark_blob_and_pick_reads_colours(tmp_path):
    reference = tmp_path / "disc.png"
    reference.write_bytes(dark_disc_png())

    async def body(call):
        data(await call("load_reference", path=str(reference)))
        traced = data(await call("trace_reference", region=[50, 50, 100, 100]))
        (shape,) = traced["shapes"]
        assert abs(shape["area"] - math.pi * 900) < 0.03 * math.pi * 900
        assert all(
            abs(a - b) < 1.5
            for a, b in zip(shape["bounds"], [70, 70, 60, 60], strict=True)
        )
        numbers = [float(v) for v in re.findall(r"-?[\d.]+", shape["d"])]
        assert shape["d"].startswith("M")
        assert shape["d"].endswith("Z")
        assert min(numbers) > 68
        assert max(numbers) < 132
        assert "No reference" not in traced.get("note", "")
        nothing = data(
            await call("trace_reference", region=[0, 0, 40, 40], colour="#00ff00")
        )
        assert nothing["shapes"] == []
        assert "note" in nothing

        # pick gives the colours at a spot along with what paints there.
        dark = data(await call("pick", x=100, y=100, radius=3))
        assert dark["colour"]["reference"] == "#0a0a0a"
        assert dark["colour"]["drawing"] == "#f0c090"
        assert dark["colour"]["difference"] > 0.5
        assert dark["objects"][0]["id"] == "face"
        red = data(await call("pick", x=175, y=25))
        assert red["colour"]["drawing"] == "#ff0000"
        assert [o["id"] for o in red["objects"]] == ["top", "bg"]
        stroke = data(await call("pick", x=70, y=88))
        ink = stroke["objects"][0]
        assert ink["id"] == "ink"
        assert [(c["index"], c["paints"]) for c in ink["contours"]] == [(1, "stroke")]
        assert stroke["colour"]["drawing"] == "#111111"

    run(tmp_path, body)


def test_each_call_is_one_revision_and_new_paths_go_where_they_are_drawn(tmp_path):
    async def body(call):
        start = data(await call("history"))["revision"]
        added = data(
            await call(
                "add_path",
                d="M75 98 L85 98 L85 104 Z",
                fill="#00ff00",
                stroke="#000000",
                name="Pupil",
            )
        )
        assert added["revision"] == start + 1
        (pupil,) = added["created"]
        # Under it is the face, in the cel group: it goes just above that.
        assert added["placed"]["parent"] == "cel"
        assert added["placed"]["above"] == "face"
        described = data(await call("describe", within="cel"))
        assert [o["id"] for o in described["objects"]] == ["face", pupil, "ink"]
        # Where it was drawn, in the document.
        drawn = data(await call("points", id=pupil))["contours"][0]
        assert drawn["bounds"] == [75, 98, 10, 6]
        painted = data(await call("properties", ids=[pupil], fill="#123456"))
        assert painted["revision"] == start + 2
        undone = data(await call("undo", steps=2))
        assert undone["revision"] == start + 3
        # Nothing under it: the top level, in front.
        alone = data(await call("add_path", d="M0 0 L1 0 L1 1 Z"))
        assert alone["revision"] == start + 4
        assert alone["placed"]["parent"] == data(await call("describe"))["root"]

    run(tmp_path, body)


def test_view_reads_what_the_window_shows_and_render_draws_it(tmp_path):
    state = Vectrify()

    async def body(call):
        headless = data(await call("view"))
        assert headless["window"] is False
        assert headless["region"] == [0, 0, 200, 200]
        assert "view" in error(await call("render", region="view"))
        target = state.target
        assert isinstance(target, FileTarget)
        target.session.set_view(
            {
                "region": [50, 60, 40, 30],
                "zoom": 8,
                "pixels": [320, 240],
                "tool": "nodes",
                "entered": "cel",
                "reference_view": "overlay",
            }
        )
        seen = data(await call("view"))
        assert seen["window"] is True
        assert seen["region"] == [50, 60, 40, 30]
        assert seen["zoom"] == 8
        assert seen["tool"] == "nodes"
        assert seen["entered_group"] == "cel"
        rendered = await call("render", region="view")
        shown = data(rendered)
        assert shown["pixels"] == [320, 240]
        assert shown["mapping"]["units_per_pixel"] == [0.125, 0.125]
        (png,) = images(rendered)
        with Image.open(io.BytesIO(png)) as image:
            # The eye's left stroke at x 70 is 160 pixels in.
            grey = np.asarray(image.convert("L"))
        assert grey[220, 160] < 60

    run(tmp_path, body, state)
