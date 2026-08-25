import io
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from vectrify.image_utils import rasterize_svg_to_png_bytes
from vectrify.refine.paths import (
    _composite_opaque_fills,
    _fill_batched_windings,
    _fill_coverage,
    _fill_coverages,
    _fill_path_coverage,
    _large_path_tile_boundary_candidates,
    _large_path_tile_candidates,
    _pad_fused_cubics,
    _tiled_large_path_coverage,
    _torch_compile_enabled,
    _xing_loss,
    fit_filled_svg,
    fit_opaque_fills_locally,
    parse_filled_cubics,
)


def _sixteen_cubic_circle(torch):
    angle = torch.arange(17, device="cuda", dtype=torch.float32) * (2 * torch.pi / 16)
    points = torch.stack((12 + 7 * torch.cos(angle), 13 + 6 * torch.sin(angle)), -1)
    return torch.stack(
        (
            points[:-1],
            points[:-1] * (2 / 3) + points[1:] * (1 / 3),
            points[:-1] * (1 / 3) + points[1:] * (2 / 3),
            points[1:],
        ),
        1,
    )[None]


def test_torch_compile_can_be_explicitly_disabled(monkeypatch):
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
    assert not _torch_compile_enabled()


@pytest.mark.parametrize("samples", [8, 16, 32])
def test_native_winding_matches_torch_forward_and_gradient(samples, monkeypatch):
    """Release-wheel CUDA path agrees with the portable sampled renderer."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from vectrify.refine import cuda_renderer

    if not cuda_renderer.available():
        pytest.skip("optional SAMVG CUDA extension is not installed")
    controls = _sixteen_cubic_circle(torch).requires_grad_()
    native = _fill_batched_windings(
        controls, (0, 0, 24, 24), samples=samples, x_offset=0.25, y_offset=0.25
    )
    extension = cuda_renderer._extension
    monkeypatch.setattr(cuda_renderer, "_extension", lambda: None)
    portable = _fill_batched_windings(
        controls, (0, 0, 24, 24), samples=samples, x_offset=0.25, y_offset=0.25
    )
    upstream = torch.randn_like(native)
    (native * upstream).sum().backward()
    native_gradient = controls.grad.detach().clone()
    controls.grad = None
    (portable * upstream).sum().backward()

    assert torch.allclose(native, portable, atol=1e-5, rtol=1e-5)
    assert torch.allclose(native_gradient, controls.grad, atol=1e-4, rtol=1e-4)
    monkeypatch.setattr(cuda_renderer, "_extension", extension)


def test_native_winding_falls_back_without_the_optional_extension(monkeypatch):
    torch = pytest.importorskip("torch")
    from vectrify.refine import cuda_renderer

    monkeypatch.setattr(cuda_renderer, "_extension", lambda: None)
    controls = torch.zeros((1, 16, 4, 2))
    assert (
        cuda_renderer.winding(
            controls, (0, 0, 8, 8), samples=16, x_offset=0.5, y_offset=0.5
        )
        is None
    )


def test_native_subpixel_windings_share_the_separate_winding_result():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from vectrify.refine.cuda_renderer import available, winding, windings

    if not available():
        pytest.skip("optional SAMVG CUDA extension is not installed")
    controls = _sixteen_cubic_circle(torch).requires_grad_()
    fused = windings(controls, (0, 0, 24, 24), samples=16, subpixels=2)
    separate = torch.stack(
        [
            winding(
                controls,
                (0, 0, 24, 24),
                samples=16,
                x_offset=(x + 0.5) / 2,
                y_offset=(y + 0.5) / 2,
            )
            for y in range(2)
            for x in range(2)
        ],
        dim=1,
    )
    upstream = torch.randn_like(fused)
    (fused * upstream).sum().backward()
    fused_gradient = controls.grad.detach().clone()
    controls.grad = None
    (separate * upstream).sum().backward()

    assert torch.allclose(fused, separate, atol=1e-5, rtol=1e-5)
    assert torch.allclose(fused_gradient, controls.grad, atol=1e-4, rtol=1e-4)


def test_native_even_odd_coverage_stays_cairo_validated():
    """The native winding path preserves SVG hole coverage, not just tensors."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from vectrify.refine.cuda_renderer import available

    if not available():
        pytest.skip("optional SAMVG CUDA extension is not installed")
    size = 96
    contours = [
        torch.tensor(contour, dtype=torch.float32, device="cuda")
        for contour in parse_filled_cubics(DONUT_PATH)
    ]
    native = (
        _fill_path_coverage(contours, (0, 0, size, size), fill_rule="evenodd")
        .cpu()
        .numpy()
    )
    head = f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">'
    blank = f"{head}</svg>"
    drawn = f'{head}<path d="{DONUT_PATH}" fill="#000" fill-rule="evenodd" /></svg>'
    real = (
        np.asarray(
            Image.open(
                io.BytesIO(rasterize_svg_to_png_bytes(blank, out_w=size, out_h=size))
            ).convert("L"),
            dtype=np.float32,
        )
        - np.asarray(
            Image.open(
                io.BytesIO(rasterize_svg_to_png_bytes(drawn, out_w=size, out_h=size))
            ).convert("L"),
            dtype=np.float32,
        )
    ) / 255.0

    assert np.abs(native - real).mean() < 0.002
    assert native[48, 48] < 0.01


def test_native_analytic_cubic_coverage_stays_cairo_validated():
    """The production single-contour path uses cubic intersections, not samples."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from vectrify.refine.cuda_renderer import available

    if not available():
        pytest.skip("optional SAMVG CUDA extension is not installed")
    size = 96
    contour = torch.tensor(
        parse_filled_cubics("M 12 48 C 12 5 84 5 84 48 C 84 91 12 91 12 48 Z")[0],
        dtype=torch.float32,
        device="cuda",
    )
    contour = torch.cat((contour, contour[:1].expand(16 - len(contour), -1, -1)))[None]
    native = _fill_coverages(contour, (0, 0, size, size), subpixels=4).cpu().numpy()
    head = f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">'
    blank = f"{head}</svg>"
    path = "M 12 48 C 12 5 84 5 84 48 C 84 91 12 91 12 48 Z"
    drawn = f'{head}<path d="{path}" fill="#000" /></svg>'
    real = (
        np.asarray(
            Image.open(
                io.BytesIO(rasterize_svg_to_png_bytes(blank, out_w=size, out_h=size))
            ).convert("L"),
            dtype=np.float32,
        )
        - np.asarray(
            Image.open(
                io.BytesIO(rasterize_svg_to_png_bytes(drawn, out_w=size, out_h=size))
            ).convert("L"),
            dtype=np.float32,
        )
    ) / 255.0

    assert np.abs(native - real).mean() < 0.002


def test_native_analytic_multi_contour_coverage_preserves_a_hole():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from vectrify.refine.cuda_renderer import available, multi_coverage

    if not available():
        pytest.skip("optional SAMVG CUDA extension is not installed")
    contours = [
        torch.tensor(contour, dtype=torch.float32, device="cuda")
        for contour in parse_filled_cubics(DONUT_PATH)
    ]
    controls = torch.cat(
        [
            torch.cat((contour, contour[:1].expand(16 - len(contour), -1, -1)))[None]
            for contour in contours
        ]
    ).requires_grad_()
    coverage = multi_coverage(
        controls, [0, 2], (0, 0, 96, 96), subpixels=4, fill_rule="evenodd"
    )
    assert coverage is not None
    assert coverage[0, 48, 48] < 0.01
    coverage.sum().backward()
    assert controls.grad is not None
    assert controls.grad.abs().sum() > 0


SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24">'
    '<path d="M 4 4 C 8 4 12 4 16 4 C 16 8 16 12 16 16 '
    'C 12 16 8 16 4 16 C 4 12 4 8 4 4 Z" fill="#0000ff" />'
    "</svg>"
)


def _mse(svg: str, target: Image.Image) -> float:
    rendered = Image.open(
        io.BytesIO(rasterize_svg_to_png_bytes(svg, out_w=24, out_h=24))
    ).convert("RGB")
    return float(
        (
            (
                np.asarray(rendered, dtype=np.float32)
                - np.asarray(target, dtype=np.float32)
            )
            ** 2
        ).mean()
    )


def test_filled_path_fit_moves_fill_colour_toward_target():
    target = Image.new("RGB", (24, 24), "black")
    target.paste("red", (4, 4, 16, 16))

    fitted = fit_filled_svg(
        SVG,
        target,
        steps=4,
        point_learning_rate=0.0,
        color_learning_rate=0.5,
    )

    assert _mse(fitted, target) < _mse(SVG, target)


def test_filled_path_fit_can_learn_fill_opacity():
    source = SVG.replace('#0000ff"', '#ff0000" fill-opacity="1"')
    target = Image.new("RGB", (24, 24), "black")
    target.paste("#400000", (4, 4, 16, 16))

    fitted = fit_filled_svg(
        source,
        target,
        steps=12,
        point_learning_rate=0.0,
        color_learning_rate=0.1,
        learn_alpha=True,
    )

    root = ET.fromstring(fitted)
    path = next(element for element in root.iter() if element.get("d"))
    assert 0 < float(path.get("fill-opacity", "0")) < 1


def test_sparse_fill_replay_matches_dense_replay_update():
    source = SVG.replace(
        "</svg>",
        '<path d="M 8 8 L 20 8 L 20 20 L 8 20 Z" fill="#00ff00" /></svg>',
    )
    target = Image.new("RGB", (24, 24), "black")
    target.paste("red", (4, 4, 16, 16))

    dense = fit_filled_svg(
        source,
        target,
        steps=1,
        point_learning_rate=0.0,
        color_learning_rate=0.1,
    )
    sparse = fit_filled_svg(
        source,
        target,
        steps=1,
        point_learning_rate=0.0,
        color_learning_rate=0.1,
        sparse_replay=True,
    )

    assert abs(_mse(dense, target) - _mse(sparse, target)) < 2


def test_filled_path_fit_preserves_a_closed_contours_segment_count():
    target = Image.new("RGB", (24, 24), "black")
    target.paste("red", (4, 4, 16, 16))

    fitted = fit_filled_svg(
        SVG,
        target,
        steps=1,
        point_learning_rate=0.5,
        color_learning_rate=0.0,
    )

    root = ET.fromstring(fitted)
    path = next(element for element in root.iter() if element.get("d"))
    assert [len(contour) for contour in parse_filled_cubics(path.get("d", ""))] == [4]


def test_filled_path_fit_preserves_each_compound_contours_closure():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24">'
        '<path d="M 2 2 L 22 2 L 22 22 L 2 22 Z '
        'M 8 8 L 16 8 L 16 16 L 8 16 Z" fill="#ff0000" fill-rule="evenodd" />'
        "</svg>"
    )

    fitted = fit_filled_svg(
        svg,
        Image.new("RGB", (24, 24), "black"),
        steps=1,
        point_learning_rate=0.5,
        color_learning_rate=0.0,
    )

    root = ET.fromstring(fitted)
    path = next(element for element in root.iter() if element.get("d"))
    assert [len(contour) for contour in parse_filled_cubics(path.get("d", ""))] == [
        4,
        4,
    ]


def test_local_fill_fit_changes_only_one_bounded_group():
    paths = []
    for index in range(20):
        x = (index % 5) * 20 + 2
        y = (index // 5) * 20 + 2
        paths.append(
            f'<path d="M {x} {y} L {x + 12} {y} L {x + 12} {y + 12} '
            f'L {x} {y + 12} Z" fill="#ff0000" />'
        )
    source = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="80">'
        + "".join(paths)
        + "</svg>"
    )
    target = Image.new("RGB", (100, 80), "black")
    reference = io.BytesIO()
    target.save(reference, format="PNG")

    fitted = fit_opaque_fills_locally(
        source,
        reference.getvalue(),
        steps=1,
        maximum_paths=4,
        rasterize=lambda markup, width, height: rasterize_svg_to_png_bytes(
            markup, out_w=width, out_h=height
        ),
    )

    before = [
        element.get("fill")
        for element in ET.fromstring(source).iter()
        if element.get("d")
    ]
    after = [
        element.get("fill")
        for element in ET.fromstring(fitted).iter()
        if element.get("d")
    ]
    assert len(before) == len(after) == 20
    assert sum(left != right for left, right in zip(before, after, strict=True)) <= 4


DONUT_PATH = (
    "M 12 48 C 12 5 84 5 84 48 C 84 91 12 91 12 48 Z "
    "M 34 48 C 34 30 62 30 62 48 C 62 66 34 66 34 48 Z"
)


def test_even_odd_filled_renderer_matches_exported_svg_with_a_hole():
    torch = pytest.importorskip("torch")
    size = 96
    contours = [
        torch.tensor(contour, dtype=torch.float32)
        for contour in parse_filled_cubics(DONUT_PATH)
    ]
    soft = _fill_path_coverage(
        contours, (0, 0, size, size), fill_rule="evenodd"
    ).numpy()
    head = f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">'
    blank = f"{head}</svg>"
    drawn = f'{head}<path d="{DONUT_PATH}" fill="#000" fill-rule="evenodd" /></svg>'

    def luminance(svg: str) -> np.ndarray:
        png = rasterize_svg_to_png_bytes(svg, out_w=size, out_h=size)
        return np.asarray(Image.open(io.BytesIO(png)).convert("L"), dtype=np.float32)

    real = (luminance(blank) - luminance(drawn)) / 255.0

    agreement = np.minimum(soft, real).sum() / np.maximum(soft, real).sum()
    # This is a curved multi-contour SVG rendered through CairoSVG, rather
    # than a renderer self-comparison.  Keep the differentiable rasterizer's
    # antialias coverage within one percent of the exported geometry.
    assert agreement > 0.99
    assert np.abs(soft - real).mean() < 0.002
    assert soft[48, 48] < 0.01


def test_filled_fit_preserves_even_odd_contours_at_zero_steps():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96">'
        f'<path d="{DONUT_PATH}" fill="#0000ff" fill-rule="evenodd" />'
        "</svg>"
    )
    target = Image.new("RGB", (96, 96), "white")

    fitted = fit_filled_svg(svg, target, steps=0)

    assert fitted.count("M ") == 2
    assert 'fill-rule="evenodd"' in fitted


def test_filled_fit_can_optimise_an_even_odd_path_without_losing_its_hole():
    source = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96">'
        f'<path d="{DONUT_PATH}" fill="#0000ff" fill-rule="evenodd" />'
        "</svg>"
    )
    target_svg = source.replace("#0000ff", "#ff0000")
    target = Image.open(
        io.BytesIO(rasterize_svg_to_png_bytes(target_svg, out_w=96, out_h=96))
    ).convert("RGB")

    fitted = fit_filled_svg(
        source,
        target,
        steps=4,
        point_learning_rate=0.0,
        color_learning_rate=0.5,
        optimisation_long_side=96,
    )
    rendered = Image.open(
        io.BytesIO(rasterize_svg_to_png_bytes(fitted, out_w=96, out_h=96))
    ).convert("RGB")

    def error(candidate: Image.Image) -> float:
        difference = np.asarray(candidate, dtype=np.float32) - np.asarray(
            target, dtype=np.float32
        )
        return float((difference**2).mean())

    assert error(rendered) < error(
        Image.open(
            io.BytesIO(rasterize_svg_to_png_bytes(source, out_w=96, out_h=96))
        ).convert("RGB")
    )
    assert fitted.count("M ") == 2


def test_batched_fill_coverage_matches_each_individual_contour():
    torch = pytest.importorskip("torch")
    contour = torch.tensor(parse_filled_cubics(SVG.split('d="')[1].split('"')[0])[0])
    controls = torch.stack((contour, contour + torch.tensor([2.0, 1.0])))
    box = (0, 0, 32, 32)

    batched = _fill_coverages(controls, box)
    separate = torch.stack([_fill_coverage(control, box) for control in controls])

    assert torch.allclose(batched, separate, atol=1e-6)


def test_large_path_tile_candidates_keep_every_possible_ray_crossing():
    torch = pytest.importorskip("torch")
    # The second contour is horizontally left of the right tile and cannot
    # cross rays from it; the tall first contour must remain in both tiles.
    contours = [
        torch.tensor([[2.0, 2.0], [2.0, 30.0], [30.0, 2.0], [30.0, 30.0]])
        .expand(16, -1, -1)
        .clone(),
        torch.tensor([[1.0, 2.0], [1.0, 30.0], [7.0, 2.0], [7.0, 30.0]])
        .expand(16, -1, -1)
        .clone(),
    ]

    tiles = _large_path_tile_candidates(contours, 32, 32, tile_size=16, margin=0)
    by_origin = {(left, top): candidates for left, top, _w, _h, candidates in tiles}

    assert 0 in by_origin[(0, 0)]
    assert 0 in by_origin[(16, 0)]
    assert 1 not in by_origin[(16, 0)]


def test_large_path_boundary_candidates_are_a_local_subset_of_ray_candidates():
    torch = pytest.importorskip("torch")
    contours = [
        torch.tensor([[2.0, 2.0], [2.0, 14.0], [30.0, 2.0], [30.0, 14.0]])
        .expand(16, -1, -1)
        .clone(),
        torch.tensor([[20.0, 2.0], [20.0, 14.0], [30.0, 2.0], [30.0, 14.0]])
        .expand(16, -1, -1)
        .clone(),
    ]
    tiles = _large_path_tile_candidates(contours, 32, 16, tile_size=16, margin=0)
    boundary = _large_path_tile_boundary_candidates(contours, tiles, margin=0)
    by_origin = {
        (left, top): (candidates, nearby)
        for (left, top, _w, _h, candidates), nearby in zip(tiles, boundary, strict=True)
    }
    ray, nearby = by_origin[(0, 0)]
    assert ray == (0, 1)
    assert nearby == (0,)


def test_tiled_analytic_large_path_matches_untiled_coverage_and_gradients():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from vectrify.refine.cuda_renderer import available, multi_coverage

    if not available():
        pytest.skip("optional SAMVG CUDA extension is not installed")
    contours = []
    for row in range(4):
        for column in range(4):
            x, y = 2 + column * 14, 2 + row * 14
            contours.append(
                torch.tensor(
                    [[x, y], [x + 4, y], [x + 4, y + 4], [x, y + 4]],
                    device="cuda",
                    dtype=torch.float32,
                )
                .expand(16, -1, -1)
                .clone()
            )
    controls = torch.stack(contours).requires_grad_()
    full = multi_coverage(
        controls, [0, len(controls)], (0, 0, 64, 64), subpixels=2, fill_rule="evenodd"
    )
    assert full is not None
    tiles = _large_path_tile_candidates(list(controls), 64, 64)
    boundary_global = _large_path_tile_boundary_candidates(list(controls), tiles)
    tiled = _tiled_large_path_coverage(
        list(controls),
        (0, 0, 64, 64),
        tiles,
        fill_rule="evenodd",
        subpixels=2,
        packed_contours=controls,
        candidate_indices=[
            torch.tensor(candidates, dtype=torch.long, device="cuda")
            for _left, _top, _width, _height, candidates in tiles
        ],
        boundary_candidate_indices=[
            torch.tensor(
                [candidates.index(candidate) for candidate in nearby],
                dtype=torch.long,
                device="cuda",
            )
            for (_left, _top, _width, _height, candidates), nearby in zip(
                tiles, boundary_global, strict=True
            )
        ],
    )
    assert tiled is not None
    assert torch.allclose(tiled, full[0], atol=1e-6)
    tiled.sum().backward()
    assert controls.grad is not None
    assert controls.grad.abs().sum() > 0


def test_analytic_multi_coverage_reuses_topology_workspace_between_steps():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from vectrify.refine.cuda_renderer import available, multi_coverage

    if not available():
        pytest.skip("optional SAMVG CUDA extension is not installed")
    controls = torch.tensor(
        [[[[0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]]] * 16],
        device="cuda",
        requires_grad=True,
    )
    workspace = torch.empty((1, 4, 4), dtype=torch.uint16, device="cuda")
    pointer = workspace.data_ptr()
    for _ in range(2):
        controls.grad = None
        coverage = multi_coverage(
            controls,
            [0, 1],
            (0, 0, 4, 4),
            subpixels=2,
            fill_rule="nonzero",
            topology_workspace=workspace,
        )
        assert coverage is not None
        coverage.sum().backward()
        assert torch.isfinite(controls.grad).all()
        assert workspace.data_ptr() == pointer


@pytest.mark.parametrize("fill_rule", ["evenodd", "nonzero"])
def test_tiled_analytic_large_compound_path_matches_cairo(fill_rule):
    """The 16-contour production tile path preserves holes in Cairo terms."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from vectrify.refine.cuda_renderer import available

    if not available():
        pytest.skip("optional SAMVG CUDA extension is not installed")
    size = 64
    pieces = []
    for row in range(2):
        for column in range(4):
            x, y = 2 + column * 16, 6 + row * 28
            # Reverse the inner contour, so it is a hole for both SVG rules.
            pieces.extend(
                (
                    f"M {x} {y} L {x + 12} {y} L {x + 12} {y + 12} L {x} {y + 12} Z",
                    (
                        f"M {x + 3} {y + 3} L {x + 3} {y + 9} "
                        f"L {x + 9} {y + 9} L {x + 9} {y + 3} Z"
                    ),
                )
            )
    path = " ".join(pieces)
    contours = [
        torch.tensor(contour, dtype=torch.float32, device="cuda")
        for contour in parse_filled_cubics(path)
    ]
    assert len(contours) == 16
    padded = [_pad_fused_cubics(contour[None])[0] for contour in contours]
    native = _tiled_large_path_coverage(
        padded,
        (0, 0, size, size),
        _large_path_tile_candidates(padded, size, size),
        fill_rule=fill_rule,
        subpixels=4,
    )
    assert native is not None
    head = f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">'
    blank = f"{head}</svg>"
    drawn = f'{head}<path d="{path}" fill="#000" fill-rule="{fill_rule}" /></svg>'
    cairo = (
        np.asarray(
            Image.open(
                io.BytesIO(rasterize_svg_to_png_bytes(blank, out_w=size, out_h=size))
            ).convert("L"),
            dtype=np.float32,
        )
        - np.asarray(
            Image.open(
                io.BytesIO(rasterize_svg_to_png_bytes(drawn, out_w=size, out_h=size))
            ).convert("L"),
            dtype=np.float32,
        )
    ) / 255.0
    rendered = native.detach().cpu().numpy()
    assert np.abs(rendered - cairo).mean() < 0.002
    assert rendered[12, 8] < 0.01


def test_filled_fit_uses_analytic_tiles_for_large_cuda_paths(monkeypatch):
    """A supported large path must not silently enter sampled winding."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from vectrify.refine import cuda_renderer
    from vectrify.refine import paths as filled_paths

    if not cuda_renderer.available():
        pytest.skip("optional SAMVG CUDA extension is not installed")
    pieces = []
    for row in range(2):
        for column in range(8):
            x, y = 2 + column * 7, 8 + row * 24
            pieces.append(f"M {x} {y} L {x + 5} {y} L {x + 5} {y + 5} L {x} {y + 5} Z")
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">'
        f'<path d="{" ".join(pieces)}" fill="#4080c0" fill-rule="nonzero" />'
        "</svg>"
    )

    def unexpected_sampled_fallback(*_args, **_kwargs):
        raise AssertionError("supported large CUDA path entered sampled winding")

    monkeypatch.setattr(
        filled_paths, "_fill_path_coverage", unexpected_sampled_fallback
    )
    fitted = fit_filled_svg(
        svg,
        Image.new("RGB", (64, 64), "white"),
        steps=1,
        point_learning_rate=0.0,
        color_learning_rate=0.0,
        optimisation_long_side=64,
    )
    assert "path" in fitted


def test_bounded_compositing_gradient_matches_monolithic_render():
    """The memory-bounded fit pass must retain the full painter's-order MSE gradient."""
    torch = pytest.importorskip("torch")
    contour = torch.tensor(
        parse_filled_cubics(SVG.split('d="')[1].split('"')[0])[0],
        dtype=torch.float32,
    )
    controls = [
        contour.clone().requires_grad_(),
        (contour + torch.tensor([3.0, 2.0])).requires_grad_(),
    ]
    colours = [
        torch.tensor([0.1, 0.4, 0.8], requires_grad=True),
        torch.tensor([0.7, 0.2, 0.3], requires_grad=True),
    ]
    target = torch.linspace(0, 1, 16 * 16 * 3).reshape(16, 16, 3)
    box = (0, 0, 16, 16)

    alphas = [_fill_coverage(control, box) for control in controls]
    rendered = torch.zeros_like(target)
    for alpha, colour in zip(alphas, colours, strict=True):
        rendered = rendered * (1 - alpha[..., None]) + colour * alpha[..., None]
    direct = torch.autograd.grad(
        ((rendered - target) ** 2).mean(), [*controls, *colours], retain_graph=True
    )

    with torch.no_grad():
        frozen_alphas = [alpha.detach() for alpha in alphas]
        canvases = []
        frozen_rendered = torch.zeros_like(target)
        for alpha, colour in zip(frozen_alphas, colours, strict=True):
            canvases.append(frozen_rendered)
            frozen_rendered = (
                frozen_rendered * (1 - alpha[..., None])
                + colour.detach() * alpha[..., None]
            )
        suffixes = []
        transparency = torch.ones_like(frozen_alphas[0])
        for alpha in reversed(frozen_alphas):
            suffixes.append(transparency)
            transparency = transparency * (1 - alpha)
        suffixes.reverse()
        image_gradient = 2 * (frozen_rendered - target) / frozen_rendered.numel()

    surrogate = torch.zeros(())
    for index, (alpha, colour) in enumerate(zip(alphas, colours, strict=True)):
        alpha_gradient = (
            image_gradient
            * suffixes[index][..., None]
            * (colour.detach() - canvases[index])
        ).sum(dim=-1)
        colour_gradient = (
            image_gradient
            * suffixes[index][..., None]
            * frozen_alphas[index][..., None]
        ).sum(dim=(0, 1))
        surrogate = surrogate + (alpha * alpha_gradient).sum()
        surrogate = surrogate + (colour * colour_gradient).sum()
    bounded = torch.autograd.grad(surrogate, [*controls, *colours])

    for actual, expected in zip(bounded, direct, strict=True):
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_tensorised_opaque_compositing_matches_layer_loop():
    torch = pytest.importorskip("torch")
    alphas = torch.tensor(
        [
            [[0.2, 0.7], [0.3, 0.5]],
            [[0.6, 0.1], [0.8, 0.4]],
            [[0.9, 0.2], [0.4, 0.3]],
        ]
    )
    colours = torch.tensor([[0.1, 0.3, 0.9], [0.9, 0.2, 0.4], [0.2, 0.8, 0.5]])
    expected = torch.zeros(2, 2, 3)
    for alpha, colour in zip(alphas, colours, strict=True):
        expected = expected * (1 - alpha[..., None]) + colour * alpha[..., None]

    actual = _composite_opaque_fills(alphas, colours)

    assert torch.allclose(actual, expected)


def test_xing_loss_is_normalized_and_penalizes_either_turn_direction():
    torch = pytest.importorskip("torch")
    controls = torch.tensor(
        [
            [[0, 0], [10, 0], [0, 0], [0, 5]],
            [[0, 0], [2, 0], [0, 0], [0, -7]],
            [[0, 0], [3, 0], [0, 0], [4, 0]],
        ],
        dtype=torch.float32,
    )

    loss = _xing_loss(controls)

    # The two perpendicular curves each contribute one despite different
    # handle lengths; the collinear curve contributes zero.
    assert torch.isclose(loss, torch.tensor(2 / 3))


def test_even_odd_uses_winding_parity_for_a_double_wound_contour():
    torch = pytest.importorskip("torch")
    double_loop = (
        "M 12 48 C 12 5 84 5 84 48 C 84 91 12 91 12 48 "
        "C 12 5 84 5 84 48 C 84 91 12 91 12 48 Z"
    )
    contours = [
        torch.tensor(contour, dtype=torch.float32)
        for contour in parse_filled_cubics(double_loop)
    ]

    even_odd = _fill_path_coverage(contours, (0, 0, 96, 96), fill_rule="evenodd")
    nonzero = _fill_path_coverage(contours, (0, 0, 96, 96), fill_rule="nonzero")
    batched_even_odd = _fill_coverages(
        torch.stack(contours), (0, 0, 96, 96), fill_rule="evenodd"
    )
    head = '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96">'
    blank = head + "</svg>"
    drawn = f'{head}<path d="{double_loop}" fill="#000" fill-rule="evenodd" /></svg>'
    real = (
        np.asarray(
            Image.open(
                io.BytesIO(rasterize_svg_to_png_bytes(blank, out_w=96, out_h=96))
            ).convert("L"),
            dtype=np.float32,
        )
        - np.asarray(
            Image.open(
                io.BytesIO(rasterize_svg_to_png_bytes(drawn, out_w=96, out_h=96))
            ).convert("L"),
            dtype=np.float32,
        )
    ) / 255.0

    assert even_odd[48, 48] < 0.01
    assert nonzero[48, 48] > 0.99
    assert torch.allclose(batched_even_odd[0], even_odd, atol=1e-6)
    assert real[48, 48] < 0.01
