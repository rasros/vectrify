"""Representation checks for the experimental CUDA colour-region benchmark."""

import importlib.util
import io
import itertools
from pathlib import Path
from xml.etree import ElementTree as ET

import cairosvg
import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")
spec = importlib.util.spec_from_file_location(
    "bench_colour_regions",
    Path(__file__).resolve().parents[2] / "scripts" / "bench_colour_regions.py",
)
assert spec is not None
assert spec.loader is not None
regions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regions)


def render_png(svg: str) -> bytes:
    png = cairosvg.svg2png(bytestring=svg.encode())
    assert png is not None
    return png


@pytest.mark.parametrize("clean", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_final_geometry_cleanup_runs_after_generation(monkeypatch, clean, enabled):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test device")
    monkeypatch.setattr(
        regions, "fit_palette", lambda pixels, *_: np.zeros(pixels.shape[:2], dtype=int)
    )
    monkeypatch.setattr(
        regions,
        "clean_region_svg",
        lambda *_a, **_kw: (
            '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0 L1 0 L2 0"/></svg>',
            {},
        ),
    )
    original = regions.cleanup_svg_geometry
    calls = []

    def cleanup(svg):
        calls.append(svg)
        return original(svg)

    monkeypatch.setattr(regions, "cleanup_svg_geometry", cleanup)
    options = {} if enabled else {"geometry_cleanup": False}
    svg, metrics = regions.vectorize(
        Image.new("RGB", (8, 8), "red"),
        preserve_outlines=clean,
        outline_style="clean" if clean else "preserve",
        min_pixels=1,
        **options,
    )
    assert len(calls) == int(enabled)
    assert ("geometry_cleanup" in metrics) == enabled
    assert metrics["bytes"] == len(svg.encode())


def test_fragment_cleanup_retains_regions_and_fills_removed_pixels():
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[:, 6:] = 1
    labels[1, 1] = 2
    result = regions.remove_fragments(labels, 4)
    assert np.array_equal(result[:, :6], np.zeros((12, 6)))
    assert np.array_equal(result[:, 6:], np.ones((12, 6)))
    assert labels[1, 1] == 2  # Input is not mutated.


def test_outline_detection_keeps_closed_loops_and_connected_faint_segments():
    pixels = np.full((32, 32, 3), 200, dtype=np.float32)
    ink = np.zeros((32, 32), dtype=bool)
    ink[8:24, 8] = ink[8:24, 23] = True
    ink[8, 8:24] = ink[23, 8:24] = True
    pixels[ink] = 40
    pixels[8, 12:18] = 196  # Weak edge connects to stronger linework.
    pixels[2, 2:6] = 196  # Isolated weak texture should not become an outline.

    actual = regions.dark_outline_mask(pixels, radius=2, contrast=6)

    assert np.array_equal(actual, ink)


def test_outline_overlay_leaves_a_flat_image_unchanged():
    source = '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"/>'
    actual, metadata = regions.append_outlines(
        source, np.full((8, 8, 3), 200, dtype=np.float32)
    )
    assert actual == source
    assert metadata["outline_pixels"] == 0


def test_clean_boundaries_share_junctions_and_draw_each_edge_once():
    labels = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 2, 2]])
    traces = regions.boundary_chains(labels)
    edges = [
        tuple(sorted((tuple(a), tuple(b))))
        for trace in traces
        for a, b in itertools.pairwise(trace)
    ]
    assert len(traces) == 3
    assert len(edges) == len(set(edges)) == 6
    assert all(
        np.array_equal(trace[0], [2, 2]) or np.array_equal(trace[-1], [2, 2])
        for trace in traces
    )


def test_clean_stroke_preserves_closed_loops():
    labels = np.zeros((12, 12), dtype=int)
    labels[3:9, 3:9] = 1
    traces = regions.boundary_chains(labels)
    assert len(traces) == 1
    assert np.array_equal(traces[0][0], traces[0][-1])
    path = regions.polygon_path(traces[0])
    assert path.endswith("Z")
    assert "L" in path


def test_clean_outlines_do_not_invent_borders_for_unsupported_shading(monkeypatch):
    pixels = np.full((64, 96, 3), 200, dtype=np.float32)
    pixels[12:44, 8:36] = 30
    pixels[14:42, 10:34] = 200
    labels = np.zeros((64, 96), dtype=int)
    labels[13:43, 9:35] = 1
    labels[12:44, 58:86] = 2  # Palette shading region without source linework.
    monkeypatch.setattr(regions, "fit_palette", lambda *_args: labels)

    _actual, metadata = regions.clean_region_svg(
        pixels,
        colours=3,
        steps=4,
        minimum=1,
        radius=3,
        contrast=6,
        width=1.5,
        regions=3,
    )

    assert metadata["outline_paths"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA benchmark")
def test_clean_outline_is_a_single_stroke_not_two_borders_of_the_ink():
    pixels = np.full((64, 64, 3), 240, dtype=np.float32)
    pixels[14:50, 14:50] = 30
    pixels[16:48, 16:48] = (220, 70, 60)

    actual, metadata = regions.clean_region_svg(
        pixels, colours=2, steps=4, minimum=1, radius=3, contrast=6, width=2, regions=2
    )

    group = ET.fromstring(actual).find(
        "{http://www.w3.org/2000/svg}g[@id='clean-outlines']"
    )
    assert metadata["outline_paths"] == 1
    assert group is not None
    assert group.get("fill") == "none"
    assert group.get("stroke-width") == "2"
    assert group[0].attrib["d"].endswith("Z")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA benchmark")
def test_outline_overlay_restores_a_thin_ring_without_filling_its_hole():
    pixels = np.full((32, 32, 3), 200, dtype=np.float32)
    pixels[8:24, 8:24] = 30
    pixels[10:22, 10:22] = 200
    source = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32">'
        '<rect width="32" height="32" fill="#c8c8c8"/></svg>'
    )
    actual, _ = regions.append_outlines(source, pixels, radius=3, colours=2)
    png = render_png(actual)
    rendered = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))

    assert np.array_equal(rendered, pixels)
    group = ET.fromstring(actual).find("{http://www.w3.org/2000/svg}g")
    assert group is not None
    assert group.get("id") == "preserved-outlines"
    assert all(path.get("stroke") is None for path in group)


def test_trace_retains_a_small_hole_when_simplification_would_collapse_it():
    mask = np.ones((12, 12), dtype=bool)
    mask[5:7, 5:7] = False
    path = regions.trace(mask, 4)
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12">'
        f'<path d="{path}" fill="red" fill-rule="evenodd"/></svg>'
    )
    png = render_png(svg)
    alpha = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))[:, :, 3]
    assert np.array_equal(alpha > 0, mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA benchmark")
def test_gpu_colour_regions_preserve_two_colour_geometry_and_full_coverage():
    pixels = np.zeros((24, 24, 3), dtype=np.uint8)
    pixels[:] = (200, 40, 20)
    pixels[4:20, 4:20] = (20, 100, 200)
    pixels[9:15, 9:15] = (200, 40, 20)
    svg, _ = regions.vectorize(
        Image.fromarray(pixels),
        colours=2,
        steps=4,
        min_pixels=1,
        tolerance=0,
        smooth_sigma=0,
    )
    root = ET.fromstring(svg)
    assert all(el.tag.split("}")[-1] in {"svg", "rect", "path"} for el in root.iter())
    png = render_png(svg)
    rgba = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
    assert np.array_equal(rgba[:, :, :3], pixels)
    assert np.all(rgba[:, :, 3] == 255)


@pytest.mark.parametrize("denoise", [False, True])
def test_shared_mesh_reuses_every_stroke_edge_in_both_fill_contours(denoise):
    labels = np.zeros((40, 60), dtype=int)
    labels[8:32, 8:40] = 1
    labels[20:, 40:] = 2
    # An acute triangle tip must survive the simplification unchanged.
    for x in range(40, 54):
        half = max(0, (53 - x) // 3)
        labels[15 - half : 16 + half, x] = 1
    contours, interfaces = regions.shared_region_geometry(labels, 1.5, denoise=denoise)
    directed = [
        (tuple(a), tuple(b))
        for loops in contours.values()
        for loop in loops
        for a, b in itertools.pairwise(loop)
    ]
    for _, points in interfaces:
        for a, b in itertools.pairwise(points):
            assert (tuple(a), tuple(b)) in directed
            assert (tuple(b), tuple(a)) in directed
    vertices = np.concatenate([p for _, p in interfaces])
    assert np.any(
        (vertices[:, 0] == 54) & (vertices[:, 1] >= 15) & (vertices[:, 1] <= 16)
    )  # No rounding or retreat of the acute tip.
    assert all(
        np.array_equal(p[0], p[-1]) for loops in contours.values() for p in loops
    )


def test_shared_mesh_preserves_holes_and_canvas_edges():
    labels = np.zeros((20, 20), dtype=int)
    labels[4:16, 4:16] = 1
    labels[8:12, 8:12] = 0
    contours, _ = regions.shared_region_geometry(labels, 0)
    paths = "".join(
        f'<path d="{" ".join(regions.polygon_path(p) for p in loops)}" '
        f'fill="{("red", "blue")[i]}" fill-rule="evenodd"/>'
        for i, loops in contours.items()
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">{paths}</svg>'
    )
    rendered = np.asarray(
        Image.open(io.BytesIO(render_png(svg))).convert("RGBA")
    )
    assert np.all(rendered[:, :, 3] == 255)
    assert np.array_equal(rendered[:, :, 2] == 255, labels == 1)


def test_contour_denoising_removes_wobble_without_moving_an_acute_tip():
    # A long clean feature and one-pixel edge chatter coexist in one contour.
    top = np.column_stack((np.arange(41), (np.arange(41) % 2) * 0.8))
    legs = [top]
    corners = [(40, 0), (40, 10), (80, 20), (40, 30), (40, 40), (0, 40), (0, 0)]
    for first, last in itertools.pairwise(corners):
        length = int(np.linalg.norm(np.subtract(last, first)))
        legs.append(np.linspace(first, last, length + 1)[1:])
    noisy = np.concatenate(legs)

    clean = regions.simplify_detail_chain(noisy, 0.85)

    assert np.any(np.all(clean == [80, 20], axis=1))
    assert np.array_equal(clean[0], clean[-1])
    assert np.linalg.norm(np.diff(clean, axis=0), axis=1).sum() < (
        np.linalg.norm(np.diff(noisy, axis=0), axis=1).sum() - 5
    )


def test_ink_assignment_retains_a_thin_island_without_eroded_interior():
    labels = np.zeros((32, 48), dtype=int)
    labels[15:17, 14:34] = 1
    pixels = np.full((32, 48, 3), 240, dtype=np.float32)
    pixels[labels == 1] = (180, 50, 40)

    actual = regions.follow_ink_boundaries(pixels, labels, 1)

    assert np.any(actual[15:17, 14:34] == 1)
    assert np.all(actual[:8] == 0)
    assert np.all(actual[-8:] == 0)
    assert set(np.unique(actual)) == {0, 1}


def test_stroke_support_rejects_a_parallel_edge_beside_real_ink():
    luminance = np.full((40, 80), 200, dtype=np.float32)
    luminance[19:21, 4:76] = 40
    real = np.array([[8, 20], [72, 20]], dtype=float)
    shading_edge = real + np.array([0, 3])

    assert len(regions.supported_stroke_runs(real, luminance, 6)) == 1
    assert regions.supported_stroke_runs(shading_edge, luminance, 6) == []


def test_stroke_support_preserves_two_genuinely_inked_nearby_edges():
    luminance = np.full((40, 80), 200, dtype=np.float32)
    luminance[19:21, 4:76] = 40
    luminance[24:26, 4:76] = 40
    for y in (20, 25):
        points = np.array([[8, y], [72, y]], dtype=float)
        assert len(regions.supported_stroke_runs(points, luminance, 6)) == 1


def test_stroke_support_bridges_short_weak_gaps_but_removes_long_false_branches():
    luminance = np.full((40, 80), 200, dtype=np.float32)
    luminance[19:21, 4:76] = 40
    luminance[:, 24:26] = 200  # Two-pixel break should stay connected.
    points = np.column_stack((np.arange(8, 73, 2), np.full(33, 20)))
    runs = regions.supported_stroke_runs(points, luminance, 6)
    assert len(runs) == 1
    luminance[:, 24:40] = 200
    runs = regions.supported_stroke_runs(points, luminance, 6)
    assert len(runs) == 2
    assert runs[0][-1, 0] <= 26
    assert runs[1][0, 0] >= 38


def test_outline_branch_deduplication_removes_only_the_uninked_narrow_detour():
    luminance = np.full((48, 80), 200, dtype=np.float32)
    luminance[19:21, 4:76] = 40
    real = np.array([[8, 20], [72, 20]], dtype=float)
    detour = np.array([[8, 20], [24, 16], [56, 16], [72, 20]], dtype=float)

    actual = regions.deduplicate_outline_branches([real, detour[::-1]], luminance, 6)

    assert len(actual) == 1
    assert np.array_equal(actual[0], real)
    # A faint unpaired contour should not be fragmented by this cleanup.
    assert np.array_equal(
        regions.deduplicate_outline_branches([detour], luminance, 6)[0], detour
    )


def test_outline_branch_deduplication_preserves_both_inked_sides_of_a_thin_shape():
    from PIL import ImageDraw

    image = Image.new("L", (80, 48), 200)
    draw = ImageDraw.Draw(image)
    real = np.array([[8, 20], [72, 20]], dtype=float)
    other = np.array([[8, 20], [24, 12], [56, 12], [72, 20]], dtype=float)
    for points in (real, other):
        draw.line([(float(x), float(y) - 0.5) for x, y in points], fill=40, width=3)

    actual = regions.deduplicate_outline_branches(
        [real, other], np.asarray(image, dtype=np.float32), 6
    )

    assert len(actual) == 2


def test_outline_branch_deduplication_preserves_a_large_region_with_faint_ink():
    luminance = np.full((140, 80), 200, dtype=np.float32)
    luminance[19:21, 4:76] = 40
    real = np.array([[8, 20], [72, 20]], dtype=float)
    opposite = np.array([[8, 20], [8, 120], [72, 120], [72, 20]], dtype=float)

    actual = regions.deduplicate_outline_branches([real, opposite], luminance, 6)

    assert len(actual) == 2


def test_outline_branch_deduplication_follows_a_contour_across_junctions():
    luminance = np.full((48, 80), 200, dtype=np.float32)
    luminance[19:21, 4:76] = 40
    first = np.array([[8, 20], [40, 20]], dtype=float)
    second = np.array([[40, 20], [72, 20]], dtype=float)
    detour = np.array([[8, 20], [24, 16], [56, 16], [72, 20]], dtype=float)

    actual = regions.deduplicate_outline_branches([first, second, detour], luminance, 6)

    assert len(actual) == 2
    assert np.array_equal(actual[0], first)
    assert np.array_equal(actual[1], second)


def test_clean_method_simplifies_texture_without_changing_contours(monkeypatch):
    coarse = np.zeros((64, 96), dtype=int)
    coarse[8:56, 8:88] = 1
    texture = coarse.copy()
    for y in range(16, 48):
        texture[y, 24 + (y % 4) * 2 : 72] = 2
    pixels = np.full((64, 96, 3), 230, dtype=np.float32)
    pixels[coarse == 1] = (180, 130, 80)
    pixels[texture == 2] = (182, 132, 82)
    monkeypatch.setattr(
        regions,
        "fit_palette",
        lambda _pixels, count, _steps: coarse if count == 2 else texture,
    )
    results = []
    for tolerance in (1.25, 5):
        svg, metadata = regions.clean_region_svg(
            pixels,
            colours=3,
            steps=1,
            minimum=1,
            radius=3,
            contrast=6,
            width=1.2,
            regions=2,
            texture_tolerance=tolerance,
        )
        root = ET.fromstring(svg)
        results.append((root, metadata))
    assert results[1][1]["texture_vertices"] < results[0][1]["texture_vertices"]
    # Everything except the texture geometry is byte-for-byte identical.
    assert [ET.tostring(el) for el in results[0][0] if not el.get("clip-path")] == [
        ET.tostring(el) for el in results[1][0] if not el.get("clip-path")
    ]


def test_clean_method_reuses_region_geometry_and_preserves_holes(monkeypatch):
    labels = np.zeros((64, 64), dtype=int)
    labels[8:56, 8:56] = 1
    labels[24:40, 24:40] = 0
    pixels = np.full((64, 64, 3), 230, dtype=np.float32)
    pixels[labels == 1] = (190, 80, 60)
    monkeypatch.setattr(regions, "fit_palette", lambda *_args: labels)
    svg, metadata = regions.clean_region_svg(
        pixels,
        colours=2,
        steps=1,
        minimum=1,
        radius=3,
        contrast=6,
        width=1.2,
        regions=2,
    )
    root = ET.fromstring(svg)
    namespace = "{http://www.w3.org/2000/svg}"
    definitions = root.findall(f"{namespace}defs/{namespace}path")
    assert len(definitions) == metadata["shared_region_definitions"] == 2
    for definition in definitions:
        reference = "#" + definition.attrib["id"]
        uses = [
            el
            for el in root.iter(namespace + "use")
            if el.get("{http://www.w3.org/1999/xlink}href") == reference
        ]
        assert len(uses) == 2  # One clip and one painted fill share the path.
    rendered = np.asarray(
        Image.open(io.BytesIO(render_png(svg))).convert("RGB")
    )
    assert np.array_equal(rendered[32, 32], [230, 230, 230])
    assert np.array_equal(rendered[16, 16], [190, 80, 60])


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_clean_method_rejects_invalid_texture_tolerance(tolerance):
    with pytest.raises(ValueError, match="texture tolerance"):
        regions.clean_region_svg(
            np.zeros((8, 8, 3), dtype=np.float32),
            colours=2,
            steps=1,
            minimum=1,
            radius=3,
            contrast=6,
            width=1,
            texture_tolerance=tolerance,
        )
