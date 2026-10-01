"""The cel tracer: line-bounded regions, centrelines and line-aware merging."""

import io
import re
from dataclasses import replace

import cairosvg
import numpy as np
import pytest
from PIL import Image

from vectrify.document.model import PathNode, Subpath
from vectrify.refine import cel


def cel_image() -> Image.Image:
    """Two flat regions, red and blue, each outlined in black."""
    pixels = np.full((80, 120, 3), 255, dtype=np.uint8)
    pixels[10:70, 10:60] = (220, 60, 50)
    pixels[10:70, 60:110] = (60, 90, 210)
    pixels[9:12, 9:111] = pixels[68:71, 9:111] = 20
    pixels[9:71, 9:12] = pixels[9:71, 108:111] = pixels[9:71, 59:62] = 20
    return Image.fromarray(pixels)


def test_a_small_gap_in_a_line_keeps_the_regions_either_side_apart():
    free = np.ones((40, 60), dtype=bool)
    free[:, 29:31] = False
    # A gap three pixels high, narrower than the largest ball.
    free[18:21, 29:31] = True
    labels = cel.trapped_ball_fill(free)
    left, right = labels[20, 10], labels[20, 50]
    assert left
    assert right
    assert left != right
    assert (labels[:, :29] == left).mean() > 0.9
    assert (labels[:, 31:] == right).mean() > 0.9


def test_thinning_leaves_a_one_pixel_centreline():
    mask = np.zeros((20, 50), dtype=bool)
    mask[8:13, 5:45] = True
    skeleton = cel.thin(mask)
    middle = skeleton[:, 10:40]
    assert (middle.sum(0) == 1).all()
    assert (np.argmax(middle, axis=0) == 10).all()
    assert skeleton[mask].sum() == skeleton.sum()


def test_merging_spares_the_boundary_a_line_runs_along():
    # Three columns of one colour: a line between the first two, none
    # between the last two.
    labels = np.repeat(np.repeat(np.arange(3), 10)[None], 10, axis=0)
    line = np.zeros(labels.shape, dtype=bool)
    line[:, 9:11] = True
    target = np.full((*labels.shape, 3), 120, dtype=np.float32)
    merged = cel.merge_regions(labels, target, line, 2)
    assert merged[0, 0] != merged[0, 15]
    assert merged[0, 15] == merged[0, 25]
    # The line is crossed only when nothing else is left to merge.
    merged = cel.merge_regions(labels, target, line, 1)
    assert len(np.unique(merged)) == 1


def test_shared_edges_meet_exactly():
    labels = np.zeros((30, 40), dtype=int)
    labels[5:25, 8:30] = 1
    labels[12:18, 14:22] = 2
    outlines = cel.region_outlines(labels, 0.75)
    assert set(outlines) == {0, 1, 2}

    def numbers(data):
        return {
            tuple(map(float, p)) for p in re.findall(r"(-?\d+\.\d+) (-?\d+\.\d+)", data)
        }

    # The inner region's outline is the middle one's hole, point for point.
    assert numbers(outlines[2]) <= numbers(outlines[1])


def test_a_cel_image_traces_as_filled_regions_and_stroked_lines():
    image = cel_image()
    svg, details = cel.vectorize(image, regions=3)
    fills = re.findall(r'<path d="[^"]+" fill="(#[0-9a-f]{6})"', svg)
    strokes = re.findall(r'<path d="[^"]+" fill="none" stroke="(#[0-9a-f]{6})"', svg)
    assert details["regions"] == 3
    assert {"#dc3c32", "#3c5ad2", "#ffffff"} <= set(fills)
    assert strokes
    assert all(int(s[1:3], 16) < 80 for s in strokes)
    assert details["line_style"] == "strokes"
    png = cairosvg.svg2png(bytestring=svg.encode(), background_color="white")
    assert png is not None
    rendered = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=float)
    reference = np.asarray(image, dtype=float)
    assert np.mean((rendered - reference) ** 2) < 400


def test_filled_lines_replace_strokes_when_asked():
    svg, details = cel.vectorize(cel_image(), regions=3, strokes=False)
    assert details["line_style"] == "filled"
    assert 'fill="none"' not in svg


@pytest.mark.parametrize("regions", [0, -1])
def test_rejects_a_region_count_below_one(regions):
    with pytest.raises(ValueError, match="region count"):
        cel.vectorize(cel_image(), regions=regions)


def run(*points) -> Subpath:
    nodes = [PathNode(f"n{i}", "L", p) for i, p in enumerate(points)]
    return Subpath("s", (replace(nodes[0], command="M"), *nodes[1:]))


def test_runs_meeting_at_a_junction_carry_on_the_straightest_way():
    runs = [run((0, 0), (10, 0)), run((10, 0), (20, 0)), run((10, 0), (10, 10))]
    joined = cel._joined_runs(runs, 0)
    assert sorted([n.endpoint for n in c.nodes] for c in joined) == [
        [(0, 0), (10, 0), (20, 0)],
        [(10, 0), (10, 10)],
    ]


def test_a_gap_in_a_line_is_bridged_only_within_reach():
    runs = [run((0, 0), (10, 0)), run((12, 0), (20, 0))]
    assert len(cel._joined_runs(runs, 3)) == 1
    assert len(cel._joined_runs(runs, 1)) == 2


def covered(skeleton, runs) -> bool:
    """Whether every pixel of *skeleton* but its junctions lies on a run."""
    on = np.zeros_like(skeleton)
    for r in runs:
        xs, ys = np.floor(r).astype(int).T
        on[ys, xs] = True
    missed = skeleton & ~on
    return int(missed.sum()) <= 2


def test_a_line_leaving_a_junction_along_a_staircase_is_followed_on():
    # A ring standing on a line: the line leaves the junction where they
    # meet by a step beside it, and is followed on, not back into it.
    y, x = np.mgrid[:40, :60]
    mask = np.zeros((40, 60), dtype=bool)
    mask[20:23, 2:58] = True
    distance = np.hypot(x - 30, y - 16)
    mask |= (distance <= 5) & (distance > 3)
    skeleton = cel.thin(mask)
    runs = cel.line_runs(skeleton, 0.1)
    assert all(not np.array_equal(r[0], r[-1]) for r in runs)
    assert covered(skeleton, runs)
    lengths = sorted(len(r) for r in runs)
    # The ring, and the line either side of it: no short pieces.
    assert len(runs) == 4
    assert lengths[0] > 5


def test_the_outlines_runs_join_through_their_junctions():
    _, details = cel.vectorize(cel_image(), regions=3)
    assert details["line_pieces"] < details["line_runs"]


def test_a_square_region_keeps_its_corners_with_few_points():
    labels = np.zeros((40, 40), dtype=int)
    labels[10:30, 10:30] = 1
    data = cel.region_outlines(labels, 0.75)[1]
    points = re.findall(r"(-?\d+\.\d+) (-?\d+\.\d+)", data)
    ends = {(float(x), float(y)) for x, y in points}
    assert {(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)} <= ends
    assert len(re.findall(r"[MLC]", data)) <= 5


def test_a_round_region_comes_out_smooth_with_few_points():
    y, x = np.mgrid[:80, :80]
    labels = (np.hypot(x - 39.5, y - 39.5) <= 25).astype(int)
    data = cel.region_outlines(labels, 0.75)[1]
    assert len(re.findall(r"[MLC]", data)) <= 10
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80">'
        f'<path d="{data}" fill="#000"/></svg>'
    )
    png = cairosvg.svg2png(bytestring=svg.encode(), background_color="white")
    assert png is not None
    drawn = np.asarray(Image.open(io.BytesIO(png)).convert("L")) < 128
    assert (drawn != labels.astype(bool)).sum() < 0.03 * labels.sum()


def test_a_line_is_cut_where_its_width_steps():
    widths = np.concatenate((np.full(30, 2.0), np.full(30, 6.0)))
    pieces = cel.width_pieces(widths, 8)
    assert pieces == [(0, 30), (30, 59)]
    assert cel.width_pieces(np.full(60, 3.0) + np.sin(np.arange(60)), 8) == [(0, 59)]
    # A step too short to stand on its own is no cut.
    widths = np.concatenate((np.full(40, 2.0), np.full(5, 6.0), np.full(15, 2.0)))
    assert cel.width_pieces(widths, 8) == [(0, 59)]


def test_a_tapering_line_stays_a_stroke():
    pixels = np.full((60, 200, 3), 255, dtype=np.uint8)
    # Two pixels wide, then five.
    pixels[29:31, 10:100] = 20
    pixels[28:33, 100:190] = 20
    svg, details = cel.vectorize(Image.fromarray(pixels), regions=1)
    assert details["line_style"] == "strokes"
    widths = sorted(float(w) for w in re.findall(r'stroke-width="([\d.]+)"', svg))
    assert len(widths) == 2
    assert widths[1] > 2 * widths[0]
