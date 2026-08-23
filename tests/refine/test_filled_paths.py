import io

import numpy as np
import pytest
from PIL import Image

from vectrify.image_utils import rasterize_svg_to_png_bytes
from vectrify.refine.paths import (
    _fill_coverage,
    _fill_coverages,
    _fill_path_coverage,
    _xing_loss,
    fit_filled_svg,
    parse_filled_cubics,
)

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
        return np.asarray(
            Image.open(io.BytesIO(png)).convert("L"), dtype=np.float32
        )

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
    assert "fill-rule=\"evenodd\"" in fitted


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
        difference = (
            np.asarray(candidate, dtype=np.float32)
            - np.asarray(target, dtype=np.float32)
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

    even_odd = _fill_path_coverage(
        contours, (0, 0, 96, 96), fill_rule="evenodd"
    )
    nonzero = _fill_path_coverage(
        contours, (0, 0, 96, 96), fill_rule="nonzero"
    )
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
