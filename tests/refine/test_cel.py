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


def test_merging_keeps_a_small_mark_of_a_clearly_different_colour():
    # Skin in two close shades and, in it, a small blue iris: too small to be
    # a shadow, but far from the skin's colour.
    labels = np.zeros((40, 40), dtype=np.int64)
    labels[:, 20:] = 1
    labels[10:16, 5:11] = 2
    target = np.full((40, 40, 3), (240, 200, 170), dtype=np.float32)
    target[:, 20:] = (236, 196, 166)
    target[10:16, 5:11] = (60, 110, 200)
    line = np.zeros(labels.shape, dtype=bool)
    merged = cel.merge_regions(labels, target, line, 1)
    assert merged[0, 0] == merged[0, 30]
    assert merged[12, 8] != merged[0, 0]
    # A speck that small of a colour near the skin's merges.
    target[10:16, 5:11] = (220, 180, 150)
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


def test_filled_lines_keep_a_grainy_outline_whole_in_its_ink():
    # A grey panel outlined two pixels wide in black, antialiased, with a
    # grey line on it, and grain: the outline's pixels fall nearest
    # different inks, yet it is filled whole in black, and the grey line
    # stays grey.
    y, x = np.mgrid[:120, :160].astype(float)
    distance = np.maximum(abs(x - 80) - 50, abs(y - 60) - 40)
    pixels = np.full((120, 160, 3), 245.0)
    pixels[distance <= 0] = (170, 175, 185)
    ink = np.clip(1.5 - abs(distance + 1), 0, 1)[..., None]
    pixels = pixels * (1 - ink) + 15 * ink
    pixels[50:56, 40:120] = 95
    rng = np.random.default_rng(3)
    noisy = np.clip(pixels + rng.normal(0, 8, pixels.shape), 0, 255)
    image = Image.fromarray(noisy.astype(np.uint8))
    svg, details = cel.vectorize(image, regions=3, strokes=False)
    assert details["line_style"] == "filled"
    drawn = rendered(svg)
    ring = abs(distance + 1) <= 1
    assert drawn[ring].mean() < 70
    assert 70 < np.median(drawn[52:54, 50:110]) < 120


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


def test_a_short_mark_off_a_line_stays_but_a_whisker_goes():
    # A thick line with a bump, which thins to a whisker, and a thin line
    # with a short mark leaving it, as a mouth leaves a jaw's outline.
    mask = np.zeros((60, 80), dtype=bool)
    mask[10:18, 5:75] = True
    mask[18:21, 36:44] = True
    mask[40:43, 5:75] = True
    mask[34:40, 40:42] = True
    skeleton = cel.thin(mask)
    depth = cel.distance_transform_edt(mask)

    def free_ends(runs):
        return {float(p[1]) for r in runs for p in (r[0], r[-1])}

    # Both are shorter than the spur: without the line's depth, both go.
    runs = cel.line_runs(skeleton, 10)
    assert len(runs) == 4
    assert not any(14 < y < 41 for y in free_ends(runs))
    # The mark's free end is beyond its line's ink and stays; the whisker's
    # is within the thick line's and goes.
    runs = cel.line_runs(skeleton, 10, depth)
    assert len(runs) == 5
    assert 35.5 in free_ends(runs)
    assert not any(14 < y < 20 for y in free_ends(runs))


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


def test_a_line_an_even_number_of_pixels_wide_is_stroked_down_its_middle():
    # Thinning leaves a line four pixels wide on one of its two middle rows,
    # half a pixel off; the stroke runs down the ink's middle.
    pixels = np.full((60, 200, 3), 255, dtype=np.uint8)
    pixels[28:32, 10:190] = 20
    svg, _ = cel.vectorize(Image.fromarray(pixels), regions=1)
    (d,) = re.findall(r'<path d="([^"]+)" fill="none"', svg)
    ys = np.array([float(v) for v in re.findall(r"-?\d+\.\d+ (-?\d+\.\d+)", d)])
    assert abs(np.median(ys) - 30) < 0.15


def test_a_line_tapering_to_a_point_steps_down_in_width_as_strokes():
    # Calligraphic ink, eight pixels wide at one end and a point at the other.
    y, x = np.mgrid[:60, :260].astype(float)
    half = np.clip((240 - x) / 230, 0, 1) * 4
    pixels = np.full((60, 260, 3), 255.0)
    ink = np.clip(half + 0.5 - abs(y - 30), 0, 1) * (x >= 10)
    pixels = pixels * (1 - ink[..., None]) + 20 * ink[..., None]
    svg, details = cel.vectorize(Image.fromarray(pixels.astype(np.uint8)), regions=1)
    assert details["line_style"] == "strokes"
    assert details.get("ink_fills", 0) == 0
    widths = sorted(float(w) for w in re.findall(r'stroke-width="([\d.]+)"', svg))
    assert len(widths) >= 5
    assert widths[-1] > 3 * widths[0]
    drawn = rendered(svg).astype(float)
    assert (
        np.abs(
            drawn - np.asarray(Image.fromarray(pixels.astype(np.uint8)).convert("L"))
        ).mean()
        < 3
    )


def test_the_thin_end_of_a_tapering_line_is_drawn_solid_not_faint():
    # Three pixels wide, then tapering to a point over forty.
    y, x = np.mgrid[:60, :260].astype(float)
    half = np.where(x < 180, 1.5, np.clip((220 - x) / 40, 0, 1) * 1.5)
    ink = np.clip(half + 0.5 - abs(y - 30), 0, 1) * (x >= 10)
    pixels = np.repeat(255 - 235 * ink[..., None], 3, -1).astype(np.uint8)
    svg, _ = cel.vectorize(Image.fromarray(pixels), regions=1)
    assert "stroke-opacity" not in svg
    widths = [float(w) for w in re.findall(r'stroke-width="([\d.]+)"', svg)]
    assert min(widths) >= cel.TIP_LEAST
    drawn = rendered(svg).astype(float)
    truth = np.asarray(Image.fromarray(pixels).convert("L")).astype(float)
    # The tip stays as dark as ink along its length, and no heavier overall.
    assert (drawn[:, 180:220].min(0) < 128).mean() > 0.9
    assert np.abs(drawn - truth).mean() < 1


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


def test_a_fold_wedge_is_shading_and_a_wide_black_line_is_not():
    # A brown robe with a black line five pixels wide and, beside it, a
    # tapering dark-brown fold wedge six pixels at its widest.
    pixels = np.full((60, 80, 3), (125, 100, 87), dtype=np.float32)
    pixels[:, 10:15] = (20, 16, 14)
    y, x = np.mgrid[:60, :80]
    wedge = (y >= 5) & (x >= 40) & (x < 40 + 6 * (1 - (y - 5) / 55))
    pixels[wedge] = (65, 52, 48)
    line, _ = cel.detect_lines(pixels, 3)
    shaded, _ = cel.detect_lines(pixels, 3, shading=False)
    assert line[5:55, 10:15].all()
    # Without the shading test the wedge reads as a line; with it, its wide
    # part and much of its tip go to the fills.
    assert shaded[8:35, 41:44].mean() > 0.5
    assert not line[8:22, 40:47].any()
    assert line[:, 35:].sum() < 0.4 * shaded[:, 35:].sum()


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


def test_a_one_pixel_line_is_found_through_grain():
    # A 3 x 3 median gives a one-pixel line to the surface; the line is
    # found again with it kept, along a row and a diagonal alike.
    rng = np.random.default_rng(2)
    pixels = np.asarray(cel_image(), dtype=np.float32).copy()
    pixels[25, 15:55] = 30
    for i in range(30):
        pixels[30 + i, 65 + i] = 30
    noisy = np.clip(pixels + rng.normal(0, 10, pixels.shape), 0, 255)
    assert cel.noise_level(noisy) > cel.NOISE
    image = noisy.astype(np.float32)
    lost, _ = cel.detect_lines(cel.median_filter(image, size=(3, 3, 1)), 3)
    kept, _ = cel.detect_lines(cel.denoised(image), 3)
    row = np.zeros(lost.shape, dtype=bool)
    row[25, 18:52] = True
    diagonal = np.zeros(lost.shape, dtype=bool)
    for i in range(3, 27):
        diagonal[30 + i, 65 + i] = True
    assert kept[row].mean() > 0.9
    assert kept[diagonal].mean() > 0.9
    assert lost[row].mean() < 0.5
    svg, _ = cel.vectorize(Image.fromarray(noisy.astype(np.uint8)), regions=3)
    stroked = re.sub(r'<rect[^>]*>|<path d="[^"]+" fill="#[^>]*>', "", svg)
    drawn = rendered(stroked) < 200
    assert drawn[24:27, 18:52].any(0).mean() > 0.8


def test_hatching_on_a_dark_fill_is_found_through_grain():
    # A navy panel hatched with one-pixel black lines, blurred, with grain:
    # each hatching pixel is hardly darker than the grain around it, but
    # the lines stand out of it along their length.
    from scipy.ndimage import gaussian_filter

    pixels = np.full((120, 160, 3), 245.0)
    pixels[10:110, 10:150] = (40, 50, 80)
    y, x = np.mgrid[:120, :160]
    hatch = ((x + y) % 9 == 0) & (y > 20) & (y < 100) & (x > 20) & (x < 140)
    pixels[hatch] = 35
    pixels = gaussian_filter(pixels, (1, 1, 0))
    rng = np.random.default_rng(4)
    noisy = np.clip(pixels + rng.normal(0, 8, pixels.shape), 0, 255)
    assert cel.noise_level(noisy.astype(np.float32)) > cel.NOISE
    svg, _ = cel.vectorize(Image.fromarray(noisy.astype(np.uint8)), regions=3)
    stroked = re.sub(r'<rect[^>]*>|<path d="[^"]+" fill="#[^>]*>', "", svg)
    drawn = cel.binary_dilation(rendered(stroked) < 200, np.ones((3, 3)))
    assert drawn[hatch].mean() > 0.7
    # Little is drawn off the hatching.
    off = ~cel.binary_dilation(hatch, np.ones((5, 5)))
    assert (drawn & off)[20:100, 20:140].mean() < 0.1


def test_a_thin_antialiased_line_is_drawn_in_its_ink_and_thin():
    # One bold black line, and a thin one antialiased to grey.
    pixels = np.full((60, 200, 3), 255, dtype=np.uint8)
    pixels[10:16, 10:190] = 0
    pixels[40, 10:190] = 90
    pixels[41, 10:190] = 200
    svg, _ = cel.vectorize(Image.fromarray(pixels), regions=1)
    strokes = re.findall(
        r'stroke="#([0-9a-f]{6})" stroke-width="([\d.]+)"'
        r'(?: stroke-opacity="([\d.]+)")?',
        svg,
    )
    assert {code for code, _, _ in strokes} == {"000000"}
    # The thin one holds under a pixel of ink across, drawn solid, not as a
    # sliver.
    widths = sorted(float(w) for _, w, _ in strokes)
    assert widths[0] < 1
    assert widths[-1] > 4
    assert min(widths) >= cel.TIP_LEAST
    assert not any(o for _, _, o in strokes)


def test_a_black_line_on_navy_takes_the_black_ink_not_a_lighter_grey():
    # A blurred black line on a navy fill is only a little darker than the
    # fill; a grey ink lighter than the fill explains none of that darkening,
    # however close its colour, and must not win as the ink covering least.
    palette = np.array([[26.0, 30.0, 46.0], [111.0, 120.0, 119.0]])
    surface = np.array([24.0, 30.0, 54.0])
    middle = np.array([19.0, 26.0, 48.0])
    assert cel._ink_of(palette, middle, surface) == 0
    # A thin black line antialiased to grey on white is still black.
    white = np.array([250.0, 250.0, 250.0])
    assert cel._ink_of(palette, np.array([150.0, 152.0, 155.0]), white) == 0


def test_a_hairline_is_drawn_solid_with_about_its_ink():
    # A line holding two thirds of a pixel of black ink, antialiased to grey.
    pixels = np.full((60, 200, 3), 255, dtype=np.uint8)
    pixels[10:16, 10:190] = 0
    pixels[40, 10:190] = 90
    svg, _ = cel.vectorize(Image.fromarray(pixels), regions=1)
    assert "stroke-opacity" not in svg
    drawn = rendered(svg).astype(float)
    # About as much ink across as the line had, as dark as ink at its middle.
    ink = (255 - drawn[37:44, 50:150]).sum(0).mean()
    assert abs(ink - (255 - 90)) < 40
    assert drawn[37:44, 50:150].min(0).mean() < 90


def test_a_hairline_in_grain_is_drawn_a_pixel_wide_and_faint():
    rng = np.random.default_rng(3)
    pixels = np.full((60, 200, 3), 225.0)
    pixels[10:16, 10:190] = 0
    pixels[40, 10:190] = 80
    noisy = np.clip(pixels + rng.normal(0, 8, pixels.shape), 0, 255)
    assert cel.noise_level(noisy) > cel.NOISE
    svg, _ = cel.vectorize(Image.fromarray(noisy.astype(np.uint8)), regions=1)
    faint = re.findall(r'stroke-width="([\d.]+)" stroke-opacity="([\d.]+)"', svg)
    assert faint
    width, opacity = (float(v) for v in faint[0])
    assert width == cel.STROKE_LEAST
    assert opacity < 1


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


def test_bold_lettering_keeps_its_own_width_among_thin_lines():
    # Many thin lines of a few widths, and a little bold lettering: too
    # little to hold a twentieth of the length, but far wider than the rest.
    widths = np.array([1.0, 1.2, 1.4, 1.7, 2.0, 3.6, 3.8, 4.0])
    lengths = np.array([400.0, 400, 400, 400, 400, 20, 20, 20])
    groups = cel.width_groups(widths, lengths)
    assert sorted(widths[groups[-1]].tolist()) == [3.6, 3.8, 4.0]
    # Once it is shorter still, it is a stray and joins its neighbour.
    lengths[5:] = 5
    groups = cel.width_groups(widths, lengths)
    assert 2.0 in widths[groups[-1]]


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


def hemmed_shape() -> Image.Image:
    """A green panel on white outlined in black on three sides, its bottom
    edge a cut hem with no ink."""
    pixels = np.full((160, 200, 3), 255, dtype=np.uint8)
    pixels[20:140, 30:170] = (80, 140, 120)
    pixels[20:24, 30:170] = 20
    pixels[20:140, 30:34] = 20
    pixels[20:140, 166:170] = 20
    return Image.fromarray(pixels)


def rendered(svg: str) -> np.ndarray:
    png = cairosvg.svg2png(bytestring=svg.encode(), background_color="white")
    assert png is not None
    return np.asarray(Image.open(io.BytesIO(png)).convert("L"))


def test_the_outer_outline_leaves_a_cut_edge_with_no_ink_open():
    svg, details = cel.vectorize(hemmed_shape(), regions=3, outline=True)
    assert details["outline_inked"]
    d, _, _ = stroke_paths(svg)[0]
    assert not d.endswith("Z")
    drawn = rendered(svg)
    # The inked sides are drawn, the hem is not.
    assert drawn[80, 31] < 100
    assert drawn[80, 168] < 100
    assert drawn[21, 100] < 100
    assert drawn[137:141, 50:150].min() > 100


def test_the_outer_outline_follows_the_width_of_its_ink():
    y, x = np.mgrid[:160, :200]
    r = np.hypot(x - 100, y - 80)
    pixels = np.full((160, 200, 3), 255, dtype=np.uint8)
    pixels[r <= 60] = (220, 60, 50)
    # Thin on the left half, bold on the right.
    pixels[(r > 57) & (r <= 60) & (x < 100)] = 20
    pixels[(r > 52) & (r <= 60) & (x >= 100)] = 20
    svg, _ = cel.vectorize(Image.fromarray(pixels), regions=3, outline=True)
    widths = sorted(width for _, paint, width in stroke_paths(svg) if paint < "#3")
    assert widths[0] <= 4
    assert widths[-1] >= 6.5
    drawn = rendered(svg)
    # The bold side's inner edge, which a stroke of one width would miss.
    assert drawn[80, 100 + 53] < 100
    assert drawn[80, 100 - 55] > 100


def small_face() -> Image.Image:
    """A skin-coloured face outlined thinly in black, with an eyelid line
    and below it a dark eye too wide to be a line and too small to be
    filled, as in a character's face, and a thin mouth."""
    y, x = np.mgrid[:120, :140]
    pixels = np.full((120, 140, 3), 255, dtype=np.uint8)
    r = np.hypot(x - 70, y - 60)
    pixels[r <= 50] = (245, 200, 165)
    pixels[(r > 48) & (r <= 50)] = 25
    pixels[45:47, 45:68] = 25
    pixels[np.hypot((x - 55) / 4.5, (y - 50) / 3.5) <= 1] = (20, 18, 15)
    pixels[80:82, 60:80] = 25
    return Image.fromarray(pixels)


def test_a_small_dark_mark_wider_than_a_line_stays_dark():
    svg, _ = cel.vectorize(small_face(), regions=4)
    png = cairosvg.svg2png(bytestring=svg.encode(), background_color="white")
    assert png is not None
    drawn = np.asarray(Image.open(io.BytesIO(png)).convert("L"), dtype=float)
    y, x = np.mgrid[:120, :140]
    eye = np.hypot((x - 55) / 3.5, (y - 51) / 2.5) <= 1
    assert drawn[eye].mean() < 80
    assert drawn[80:82, 70].min() < 120


def test_a_fill_is_fitted_under_the_lines_not_from_its_median():
    # Region 0's paint is mostly 100 with a fifth at 200: the fitted fill is
    # the least-squares one, nearer their mean, where a median keeps 100.
    # Part of a column is under a line of cover 0.5 in black, and region 1
    # keeps its fallback colour, the lines hiding it whole.
    labels = np.zeros((10, 20), dtype=np.int64)
    labels[:, 10:] = 1
    target = np.full((10, 20, 3), 100.0, dtype=np.float32)
    target[1:9, 1:3] = 200
    cover = np.zeros((10, 20, 1), dtype=np.float32)
    cover[1:9, 5] = 0.5
    target[1:9, 5] = 50
    cover[:, 10:] = 1
    painted = np.zeros((10, 20, 3), dtype=np.float32)
    fallback = np.array([[100.0] * 3, [7.0] * 3])
    fitted = cel.fitted_fills(target, labels, cover, painted, fallback)
    # Region 0 less its edge column beside region 1; the half-covered pixels
    # show 50 = 0.5 * 100, so they agree with the rest.
    seen = 1 - cover[:, :9, 0]
    expected = (seen * target[:, :9, 0]).sum() / (seen * seen).sum()
    assert fitted[0, 0] == pytest.approx(expected)
    assert expected > 115
    assert fitted[1].tolist() == [7.0] * 3


def test_a_region_of_two_shades_is_split_into_them():
    # One region holding a dark patch in a light one, with some grain: its
    # fitted colour would be a grey between them.
    rng = np.random.default_rng(3)
    labels = np.zeros((60, 80), dtype=np.int64)
    target = np.full((60, 80, 3), (200, 170, 150), dtype=np.float32)
    target[15:45, 20:50] = (70, 55, 50)
    target += rng.normal(0, 3, target.shape).astype(np.float32)
    line = np.zeros(labels.shape, dtype=bool)
    split = cel.split_mixed(labels, target, line)
    assert len(np.unique(split)) == 2
    assert (split[16:44, 21:49] == split[30, 35]).all()
    assert (split[:10] == split[0, 0]).all()
    assert split[0, 0] != split[30, 35]


def test_a_ramp_or_one_shade_is_not_split():
    labels = np.zeros((60, 80), dtype=np.int64)
    line = np.zeros(labels.shape, dtype=bool)
    ramp = np.linspace(60, 220, 80, dtype=np.float32)[None, :, None]
    target = np.broadcast_to(ramp, (60, 80, 3)).copy()
    assert len(np.unique(cel.split_mixed(labels, target, line))) == 1
    flat = np.full((60, 80, 3), 150, dtype=np.float32)
    flat += np.random.default_rng(4).normal(0, 6, flat.shape).astype(np.float32)
    assert len(np.unique(cel.split_mixed(labels, flat, line))) == 1


def ramped_image() -> Image.Image:
    """A square outlined in black, its paint ramping left to right from dark
    to light blue."""
    pixels = np.full((90, 120, 3), 255, dtype=np.uint8)
    ramp = np.linspace(0, 1, 100)[None, :, None]
    dark, light = np.array([40, 60, 140]), np.array([170, 200, 250])
    pixels[10:80, 10:110] = (dark + ramp * (light - dark)).astype(np.uint8)
    pixels[8:11, 8:112] = pixels[79:82, 8:112] = 20
    pixels[8:82, 8:11] = pixels[8:82, 109:112] = 20
    return Image.fromarray(pixels)


def test_a_region_whose_colour_ramps_takes_a_gradient():
    image = ramped_image()
    flat, _ = cel.vectorize(image, regions=2, gradients=False)
    ramped, details = cel.vectorize(image, regions=2)
    assert details["gradients"] >= 1
    assert "<linearGradient" in ramped
    assert "url(#ramp" in ramped
    reference = np.asarray(image, dtype=float)

    def error(svg):
        return float(((rendered_rgb(svg) - reference) ** 2).mean())

    assert error(ramped) < 0.7 * error(flat)


def test_flat_regions_stay_flat():
    svg, details = cel.vectorize(cel_image(), regions=3)
    assert details["gradients"] == 0
    assert "linearGradient" not in svg


def rendered_rgb(svg: str) -> np.ndarray:
    png = cairosvg.svg2png(bytestring=svg.encode(), background_color="white")
    assert png is not None
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=float)


def test_a_stroke_of_closed_lines_only_has_no_ends_to_join():
    from vectrify.document.lines import end_pairs

    assert end_pairs([], 3.0) == []
