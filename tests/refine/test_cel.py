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


def test_merging_keeps_a_shadow_past_the_region_count():
    # A lit surface in two shades too close to tell, and a shadow on it.
    labels = np.repeat(np.repeat(np.arange(3), 20)[None], 20, axis=0)
    target = np.full((*labels.shape, 3), 150, dtype=np.float32)
    target[:, 20:40] = 146
    target[:, 40:] = 120
    line = np.zeros(labels.shape, dtype=bool)
    merged = cel.merge_regions(labels, target, line, 1)
    assert merged[0, 0] == merged[0, 25]
    assert merged[0, 0] != merged[0, 50]
    # Too small to be a shadow, it merges like any region.
    merged = cel.merge_regions(labels, target, line, 1, shadow_least=500)
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


def test_a_neutral_line_on_a_navy_fill_of_its_own_luminance_is_found():
    # Navy and the line have about the same luminance; only the brightest
    # channel tells the line is darker.
    pixels = np.full((40, 60, 3), (25, 30, 55), dtype=np.float32)
    pixels[:, 40:] = 235
    pixels[:, 19:21] = (34, 31, 32)
    pixels[:, 39:41] = (34, 31, 32)
    line, _ = cel.detect_lines(pixels, 3)
    assert line[5:35, 19:21].all()
    assert line[5:35, 39:41].all()
    assert not line[5:35, 25:35].any()


def test_a_line_only_a_little_darker_than_a_dark_side_is_found():
    pixels = np.full((40, 60, 3), (35, 45, 75), dtype=np.float32)
    pixels[:, 31:] = (25, 30, 55)
    pixels[:, 30] = (30, 28, 30)
    line, _ = cel.detect_lines(pixels, 3)
    assert line[5:35, 30].all()


def test_shading_against_a_line_does_not_take_the_line_with_it():
    # A narrow grey shadow touching a black line, on white.
    pixels = np.full((40, 60, 3), 235, dtype=np.float32)
    pixels[:, 20:24] = 200
    pixels[:, 24:26] = 20
    line, _ = cel.detect_lines(pixels, 3)
    assert line[5:35, 24:26].all()
    assert not line[5:35, 20:22].any()


def test_a_line_runs_on_where_it_meets_a_dark_fill():
    # A thin grey line on white running into a navy fill: the fill must not
    # take the line as a notch of itself.
    pixels = np.full((40, 60, 3), 235, dtype=np.float32)
    pixels[:, 40:] = (25, 30, 55)
    pixels[20, :40] = 120
    line, _ = cel.detect_lines(pixels, 3)
    assert line[20, 5:38].all()


def test_grain_is_measured():
    rng = np.random.default_rng(0)
    pixels = np.asarray(cel_image(), dtype=np.float32)
    assert cel.noise_level(pixels) < cel.NOISE
    noisy = np.clip(pixels + rng.normal(0, 8, pixels.shape), 0, 255)
    assert cel.noise_level(noisy) > cel.NOISE


def test_lines_are_found_through_grain():
    rng = np.random.default_rng(1)
    pixels = np.asarray(cel_image(), dtype=np.float32)
    noisy = np.clip(pixels + rng.normal(0, 10, pixels.shape), 0, 255)
    svg, details = cel.vectorize(Image.fromarray(noisy.astype(np.uint8)), regions=3)
    assert details["line_style"] == "strokes"
    stroked = re.sub(r'<rect[^>]*>|<path d="[^"]+" fill="#[^>]*>', "", svg)
    png = cairosvg.svg2png(bytestring=stroked.encode(), background_color="white")
    assert png is not None
    drawn = np.asarray(Image.open(io.BytesIO(png)).convert("L")) < 128
    truth = np.asarray(cel_image().convert("L")) < 60
    # The strokes lie on the lines, and few lie anywhere else.
    assert drawn[truth].mean() > 0.6
    assert (drawn & ~truth).sum() < 0.3 * drawn.sum()


def test_a_thin_antialiased_line_is_drawn_in_its_ink_and_thin():
    # One bold black line, and a thin one antialiased to grey.
    pixels = np.full((60, 200, 3), 255, dtype=np.uint8)
    pixels[10:16, 10:190] = 0
    pixels[40, 10:190] = 90
    pixels[41, 10:190] = 200
    svg, _ = cel.vectorize(Image.fromarray(pixels), regions=1)
    strokes = re.findall(r'stroke="#([0-9a-f]{6})" stroke-width="([\d.]+)"', svg)
    assert {code for code, _ in strokes} == {"000000"}
    widths = sorted(float(w) for _, w in strokes)
    assert widths[0] < 1
    assert widths[-1] > 4


def test_strokes_are_grouped_by_width():
    widths = np.array([1.0, 1.1, 1.2, 2.0, 2.2, 4.0, 4.3])
    groups = cel.width_groups(widths, np.ones(7))
    assert [sorted(widths[g].tolist()) for g in groups] == [
        [1.0, 1.1, 1.2],
        [2.0, 2.2],
        [4.0, 4.3],
    ]
    # A seldom-used width joins the group nearest it.
    lengths = np.array([100.0, 100, 100, 1, 100, 100, 100])
    widths = np.array([1.0, 1.1, 1.2, 2.0, 4.0, 4.1, 4.2])
    assert len(cel.width_groups(widths, lengths)) == 2


def test_a_dark_shape_among_lines_is_filled_not_stroked():
    line = np.zeros((60, 80), dtype=bool)
    line[30, 2:78] = True
    y, x = np.mgrid[:60, :80]
    line |= np.hypot(x - 40, y - 30) <= 12
    kept = cel.without_shapes(line)
    assert not kept[30, 35:45].any()
    assert kept[30, 2:20].all()


def test_regions_of_one_colour_merge_across_a_line_before_different_ones():
    labels = np.repeat(np.repeat(np.arange(3), 10)[None], 10, axis=0)
    line = np.zeros(labels.shape, dtype=bool)
    line[:, 9:11] = True
    target = np.full((*labels.shape, 3), 120, dtype=np.float32)
    target[:, 20:] = 200
    merged = cel.merge_regions(labels, target, line, 2)
    assert merged[0, 0] == merged[0, 15]
    assert merged[0, 15] != merged[0, 25]


def broken_disc() -> Image.Image:
    """A red disc on white, its black outline 4 px wide broken in two
    places, with a line across its middle."""
    y, x = np.mgrid[:160, :200]
    r = np.hypot(x - 100, y - 80)
    pixels = np.full((160, 200, 3), 255, dtype=np.uint8)
    pixels[r <= 60] = (220, 60, 50)
    ring = (r > 56) & (r <= 60)
    ring &= ~((x > 150) & (abs(y - 80) < 5)) & ~((y < 30) & (abs(x - 100) < 4))
    pixels[ring] = 20
    pixels[78:81, 42:158] = 20
    return Image.fromarray(pixels)


def stroke_paths(svg: str) -> list[tuple[str, str, float]]:
    return [
        (d, paint, float(width))
        for d, paint, width in re.findall(
            r'<path d="([^"]+)" fill="none" stroke="(#[0-9a-f]{6})" '
            r'stroke-width="([\d.]+)"',
            svg,
        )
    ]


def test_the_outer_outline_is_one_closed_stroke_through_the_gaps():
    svg, details = cel.vectorize(broken_disc(), regions=3, outline=True)
    assert details["outline"] == 1
    assert details["outline_inked"]
    d, paint, width = stroke_paths(svg)[0]
    assert d.count("M") == 1
    assert d.endswith("Z")
    assert int(paint[1:3], 16) < 80
    assert 3 <= width <= 5.5
    png = cairosvg.svg2png(bytestring=svg.encode(), background_color="white")
    assert png is not None
    drawn = np.asarray(Image.open(io.BytesIO(png)).convert("L"))
    # Dark where the outline was broken, and on the drawn ring's middle.
    assert drawn[80, 158] < 100
    assert drawn[22, 100] < 100
    assert drawn[80, 42] < 100
    # Not drawn twice: the only other line is the one across, and it runs
    # on to the outline at both ends.
    others = " ".join(d for d, _, _ in stroke_paths(svg)[1:])
    assert others.count("M") == 1
    points = np.array(re.findall(r"(-?\d+\.\d+) (-?\d+\.\d+)", others), dtype=float)
    radii = np.hypot(points[:, 0] - 100, points[:, 1] - 80)
    assert abs(radii[0] - 58) < 2.5
    assert abs(radii[-1] - 58) < 2.5


def test_the_outer_outline_is_off_by_default_and_needs_a_background():
    _, details = cel.vectorize(broken_disc(), regions=3)
    assert "outline" not in details
    full = np.full((60, 80, 3), (220, 60, 50), dtype=np.uint8)
    full[:, 38:42] = 20
    assert cel.silhouette(full.astype(np.float32)) is None
    _, details = cel.vectorize(Image.fromarray(full), regions=2, outline=True)
    assert details.get("outline", 0) == 0


def test_the_outer_outline_is_a_stroke_with_filled_lines_too():
    svg, details = cel.vectorize(broken_disc(), regions=3, outline=True, strokes=False)
    assert details["line_style"] == "filled"
    strokes = stroke_paths(svg)
    assert len(strokes) == 1
    assert strokes[0][0].endswith("Z")
