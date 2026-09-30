"""The cel tracer: line-bounded regions, centrelines and line-aware merging."""

import io
import re

import cairosvg
import numpy as np
import pytest
from PIL import Image

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
