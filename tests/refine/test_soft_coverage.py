"""Portable coverage must resolve thin fills and strokes at any orientation."""

import io

import cairosvg
import numpy as np
import pytest
from PIL import Image

from vectrify.refine.paths import parse_filled_cubics
from vectrify.refine.soft_coverage import soft_coverage


@pytest.mark.parametrize("width", [0.3, 0.7, 1.4])
def test_subpixel_width_fills_match_cairo_and_keep_their_area(width):
    torch = pytest.importorskip("torch")
    left, right = 4.5 - width / 2, 4.5 + width / 2
    path = f"M{left} 1 L{right} 1 L{right} 7 L{left} 7 Z"
    controls = [torch.tensor(c, dtype=torch.float32) for c in parse_filled_cubics(path)]
    actual = soft_coverage(controls, (0, 0, 9, 9)).detach().numpy()
    png = cairosvg.svg2png(
        bytestring=(
            f'<svg width="9" height="9"><path d="{path}" fill="black"/></svg>'
        ).encode()
    )
    assert png is not None
    expected = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))[:, :, 3] / 255
    assert np.max(abs(actual - expected)) < 0.025
    assert actual.sum() == pytest.approx(6 * width, abs=0.025)


def test_thin_fill_area_has_a_gradient_for_both_boundaries():
    torch = pytest.importorskip("torch")
    width = torch.tensor(0.3, requires_grad=True)
    y0, y1 = width.new_tensor(1), width.new_tensor(7)
    left, right = 4.5 - width / 2, 4.5 + width / 2
    points = [
        torch.stack((x, y))
        for x, y in [(left, y0), (right, y0), (right, y1), (left, y1)]
    ]
    controls = torch.stack(
        [
            torch.stack((a, (2 * a + b) / 3, (a + 2 * b) / 3, b))
            for a, b in zip(points, points[1:] + points[:1], strict=True)
        ]
    )
    area = soft_coverage([controls], (0, 0, 9, 9)).sum()
    area.backward()
    assert width.grad is not None
    assert float(width.grad) == pytest.approx(6, abs=0.05)


@pytest.mark.parametrize("width", [0.3, 0.7, 1.4])
@pytest.mark.parametrize("end", [(5.5, 27.5), (27.5, 27.5), (27.5, 14.5)])
def test_subpixel_strokes_match_cairo_across_orientations(width, end):
    torch = pytest.importorskip("torch")
    from vectrify.refine.soft_coverage import soft_stroke_coverage

    start = (5.5, 5.5)
    a, b = (torch.tensor(p) for p in (start, end))
    control = torch.stack((a, (2 * a + b) / 3, (a + 2 * b) / 3, b))[None]
    path = f"M{start[0]} {start[1]} L{end[0]} {end[1]}"
    svg = (
        f'<svg width="32" height="32"><path d="{path}" fill="none" '
        f'stroke="black" stroke-width="{width}" stroke-linecap="round"/></svg>'
    )
    png = cairosvg.svg2png(bytestring=svg.encode())
    assert png is not None
    expected = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))[:, :, 3] / 255
    actual = soft_stroke_coverage([control], (0, 0, 32, 32), width).detach().numpy()
    single = (
        soft_stroke_coverage([control], (0, 0, 32, 32), width, subpixels=1)
        .detach()
        .numpy()
    )
    assert np.max(abs(actual - expected)) < 0.065
    assert np.mean((actual - expected) ** 2) < np.mean((single - expected) ** 2) * 0.9
    # Thin diagonal lines used to lose about 30% of their actual ink area.
    assert actual.sum() == pytest.approx(expected.sum(), rel=0.025, abs=0.2)


def test_thin_diagonal_stroke_gradient_moves_toward_the_reference():
    torch = pytest.importorskip("torch")
    from vectrify.refine.soft_coverage import soft_stroke_coverage

    svg = (
        '<svg width="32" height="32"><path d="M5.5 5.5 L27.5 27.5" '
        'fill="none" stroke="black" stroke-width="0.3" '
        'stroke-linecap="round"/></svg>'
    )
    png = cairosvg.svg2png(bytestring=svg.encode())
    assert png is not None
    reference = torch.tensor(
        np.asarray(Image.open(io.BytesIO(png)))[:, :, 3] / 255, dtype=torch.float32
    )
    shift = torch.tensor(0.35, requires_grad=True)
    a, b = torch.tensor((5.5, 5.5)), torch.tensor((27.5, 27.5))
    control = torch.stack((a, (2 * a + b) / 3, (a + 2 * b) / 3, b))[None]
    moved = control + shift * torch.tensor((1, -1)) / 2**0.5
    loss = (
        (soft_stroke_coverage([moved], (0, 0, 32, 32), 0.3) - reference).square().mean()
    )
    loss.backward()
    assert shift.grad is not None
    assert torch.isfinite(shift.grad)
    assert shift.grad > 0
