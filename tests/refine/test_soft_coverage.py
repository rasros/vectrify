"""Portable coverage must resolve thin fills instead of widening them."""

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
