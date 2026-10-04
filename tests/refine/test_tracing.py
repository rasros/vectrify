import io
import re

import cairosvg
import numpy as np
from PIL import Image

import vectrify.refine.tracing as tracing
from vectrify.refine.tracing import _fit_cubic, _simplified_data, mask_path


def test_mask_path_keeps_a_hole_as_a_second_even_odd_subpath():
    mask = np.ones((8, 8), dtype=bool)
    mask[2:6, 2:6] = False

    path = mask_path(mask)

    assert path is not None
    assert path.count("M ") == 2
    assert path.count(" Z") == 2


def test_smoothing_takes_the_steps_out_of_an_enlarged_raster_edge():
    # A diagonal edge made at a third of the size and enlarged: steps three
    # pixels wide.
    small = np.tril(np.ones((40, 40), dtype=bool), -1)
    small[:, 30:] = False
    mask = (
        np.asarray(
            Image.fromarray(small.astype(np.uint8) * 255).resize(
                (120, 120), Image.Resampling.NEAREST
            )
        )
        > 0
    )

    def spread(smooth):
        path = mask_path(mask, smooth=smooth)
        assert path is not None
        values = np.array([float(v) for v in re.findall(r"-?\d+\.?\d*", path)])
        points = values.reshape(-1, 2)
        near = points[
            (points[:, 0] > 10)
            & (points[:, 0] < 80)
            & (points[:, 1] > 10)
            & (np.abs(points[:, 1] - points[:, 0]) < 8)
        ]
        return float(np.std(near[:, 1] - near[:, 0]))

    assert spread(3.0) < 0.5 * spread(0.0)


def test_a_tolerance_traces_each_outline_with_the_curves_it_needs():
    yy, xx = np.mgrid[0:120, 0:120]
    circle = (xx - 60) ** 2 + (yy - 60) ** 2 < 45**2
    image = Image.fromarray(
        np.where(circle[..., None], 40, 220).astype(np.uint8).repeat(3, axis=2)
    )
    dense = mask_path(circle, smooth=1.0)
    assert dense is not None
    fitted = _simplified_data(dense, 0.5)

    assert fitted.count("C") < dense.count("C") / 3
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120">'
        f'<rect width="120" height="120" fill="#dcdcdc"/><path d="{fitted}" '
        'fill="#282828"/></svg>'
    )
    png = cairosvg.svg2png(bytestring=svg.encode())
    assert png is not None
    rendered = np.asarray(Image.open(io.BytesIO(png)).convert("L"), dtype=float)
    assert np.abs(rendered - np.asarray(image.convert("L"), dtype=float)).mean() < 2


def test_cubic_fit_reparameterises_nonuniform_curve_samples():
    start = np.array((0.0, 0.0))
    expected_a = np.array((8.0, 20.0))
    expected_b = np.array((22.0, -16.0))
    end = np.array((30.0, 4.0))
    parameters = np.linspace(0.0, 1.0, 25) ** 2
    inverse = 1 - parameters
    points = (
        inverse[:, None] ** 3 * start
        + 3 * inverse[:, None] ** 2 * parameters[:, None] * expected_a
        + 3 * inverse[:, None] * parameters[:, None] ** 2 * expected_b
        + parameters[:, None] ** 3 * end
    )

    uniform = _fit_cubic(points, reparameterize=False)
    refined = _fit_cubic(points)

    assert np.linalg.norm(np.hstack(refined) - np.hstack((expected_a, expected_b))) < (
        np.linalg.norm(np.hstack(uniform) - np.hstack((expected_a, expected_b)))
    )


def test_fixed_cubic_tracing_does_not_duplicate_each_curve_endpoint(monkeypatch):
    loop = [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (2.0, 2.0),
        (1.0, 2.0),
        (0.0, 2.0),
        (0.0, 1.0),
    ]
    samples = []
    monkeypatch.setattr(tracing, "_corners", lambda *_args: [0, 2, 4, 6])

    def fit(pieces):
        samples.extend(points.copy() for points in pieces)
        return np.array([(points[0], points[-1]) for points in pieces])

    monkeypatch.setattr(tracing, "_fit_cubics", fit)

    tracing._cubic_loop(loop, segments=4)

    assert [len(points) for points in samples] == [3, 3, 3, 3]
    assert all(not np.array_equal(points[-1], points[-2]) for points in samples)


def test_cubics_fitted_together_are_exactly_those_fitted_one_by_one():
    rng = np.random.default_rng(4)
    samples = []
    for size in (2, 3, 5, 9, 17, 40):
        t = np.linspace(0, 1, size)[:, None]
        wobble = rng.normal(0, 0.4, (size, 2))
        curve = np.hstack((30 * t, 10 * np.sin(4 * t))) + wobble
        samples.append(curve.astype(np.float32))
    together = tracing._fit_cubics(samples)
    for sample, controls in zip(samples, together, strict=True):
        alone = np.array(_fit_cubic(sample))
        assert np.array_equal(controls, alone)
