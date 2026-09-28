"""GPU miter silhouettes, bevel fallback and gradients against SVG rendering."""

import io

import cairosvg
import numpy as np
import pytest
from PIL import Image

from vectrify.refine.miter import incoming_directions, miter_stroke_coverage


def gpu():
    torch = pytest.importorskip("torch")
    from vectrify.refine.cuda_renderer import available

    if not torch.cuda.is_available() or not available():
        pytest.skip("CUDA extension required")
    return torch


def controls(points):
    torch = gpu()
    p = torch.tensor(points, dtype=torch.float32, device="cuda")
    q = p.roll(-1, 0)
    return torch.stack((p, (2 * p + q) / 3, (p + 2 * q) / 3, q), dim=1)


@pytest.mark.parametrize(
    ("points", "limit"),
    [
        ([(12, 12), (48, 12), (48, 48), (12, 48)], 4),
        ([(32, 20), (40, 52), (24, 52)], 8),
        ([(32, 20), (40, 52), (24, 52)], 2),
        ([(24, 52), (40, 52), (32, 20)], 2),
    ],
)
def test_miter_matches_svg_and_has_geometry_and_width_gradients(points, limit):
    torch = gpu()
    c = controls(points).requires_grad_()
    widths = torch.tensor([8.0], device="cuda", requires_grad=True)
    result = miter_stroke_coverage(
        c[None],
        incoming_directions(c)[None],
        widths,
        (0, 0, 64, 64),
        miter_limit=limit,
        subpixels=4,
    )
    d = "M" + " L".join(f"{x} {y}" for x, y in points) + "Z"
    svg = (
        f'<svg width="64" height="64"><path d="{d}" fill="none" '
        f'stroke="black" stroke-width="8" stroke-miterlimit="{limit}"/></svg>'
    )
    png = cairosvg.svg2png(bytestring=svg.encode())
    assert png is not None
    assert result is not None
    expected = np.asarray(Image.open(io.BytesIO(png)))[:, :, 3] / 255
    actual = result[0].detach().cpu().numpy()
    assert abs(actual - expected).mean() < 0.008
    assert np.count_nonzero((actual > 0.5) != (expected > 0.5)) < 12
    weights = torch.linspace(0.1, 1, 4096, device="cuda").reshape(64, 64)
    (result[0] * weights).sum().backward()
    assert torch.isfinite(c.grad).all()
    assert c.grad.abs().sum() > 0
    assert torch.isfinite(widths.grad).all()
    assert widths.grad.abs().sum() > 0


def test_miter_chunks_keep_seam_and_skip_zero_length_closure():
    torch = gpu()
    c = controls([(12, 12), (48, 12), (48, 48), (12, 48)])
    c = torch.cat((c, c[:1, :1].expand(1, 4, 2)))
    incoming = incoming_directions(c)
    width = c.new_tensor([8.0])
    whole = miter_stroke_coverage(
        c[None], incoming[None], width, (0, 0, 64, 64), subpixels=4
    )
    chunks = [
        miter_stroke_coverage(
            c[a:b][None], incoming[a:b][None], width, (0, 0, 64, 64), subpixels=4
        )
        for a, b in [(0, 2), (2, 5)]
    ]
    assert torch.equal(whole, torch.stack(chunks).amax(0))


def test_curved_miter_outline_matches_svg_with_degenerate_handles():
    torch = gpu()
    c = torch.tensor(
        [
            [[12, 40], [12, 40], [20, 4], [32, 16]],
            [[32, 16], [46, 18], [58, 46], [48, 48]],
            [[48, 48], [36, 60], [12, 40], [12, 40]],
        ],
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    result = miter_stroke_coverage(
        c[None],
        incoming_directions(c)[None],
        c.new_tensor([6.0]),
        (0, 0, 64, 64),
        subpixels=4,
    )
    svg = (
        '<svg width="64" height="64"><path '
        'd="M12 40 C12 40 20 4 32 16 C46 18 58 46 48 48 C36 60 12 40 12 40 Z" '
        'fill="none" stroke="black" stroke-width="6"/></svg>'
    )
    png = cairosvg.svg2png(bytestring=svg.encode())
    assert png is not None
    assert result is not None
    expected = np.asarray(Image.open(io.BytesIO(png)))[:, :, 3] / 255
    actual = result[0].detach().cpu().numpy()
    assert abs(actual - expected).mean() < 0.01
    result.sum().backward()
    assert torch.isfinite(c.grad).all()
