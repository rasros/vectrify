"""The soft rasterizer has to draw what cairosvg draws.

It is a second implementation of stroke rendering, and the fit optimises its
output while the run scores cairosvg's. When the two drift, the fit spends its
budget on an image nobody renders: sampling at integer coordinates instead of
pixel centres put every stroke half a pixel off, dropped agreement from 0.888
to 0.573, and made the fitted drawing measurably worse while its own loss fell.
Nothing else in the suite would notice that.
"""

import io
import itertools

import numpy as np
import pytest
from PIL import Image

from tests.helpers import rasterize
from vectrify.refine.paths import (
    _SUPPORTED,
    _TOKEN,
    UnsupportedPathError,
    _as_cubic,
    coverage,
)


# Stroke-path parsing the removed stroke fitter used; kept to feed the live soft
# rasterizer the same cubic chains it has always been checked against.
def parse_cubics(d: str) -> list[list[tuple[float, float]]]:
    """Path data as a list of cubic segments, four control points each.

    Lines become cubics with their controls spaced along them, so the fit has
    one uniform representation to move and a straight edge can bend if the
    target curves.
    """
    groups: list[tuple[str, list[float]]] = []
    numbers: list[float] = []
    for token in _TOKEN.finditer(d):
        if token.group(1):
            command = token.group(1).upper()
            if command not in _SUPPORTED:
                raise UnsupportedPathError(
                    f"unsupported path command {token.group(1)!r}"
                )
            numbers = []
            groups.append((command, numbers))
        elif groups:
            numbers.append(float(token.group(2)))

    segments: list[list[tuple[float, float]]] = []
    current: tuple[float, float] | None = None
    start: tuple[float, float] | None = None
    for command, args in groups:
        if command == "M":
            for index in range(0, len(args) - 1, 2):
                point = (args[index], args[index + 1])
                if index == 0 or current is None:
                    current = start = point
                else:
                    segments.append(_as_cubic(current, point))
                    current = point
        elif command == "L":
            for index in range(0, len(args) - 1, 2):
                if current is None:
                    raise UnsupportedPathError("a lineto before any moveto")
                point = (args[index], args[index + 1])
                segments.append(_as_cubic(current, point))
                current = point
        elif command == "C":
            for index in range(0, len(args) - 5, 6):
                if current is None:
                    raise UnsupportedPathError("a curve before any moveto")
                point = (args[index + 4], args[index + 5])
                segments.append(
                    [
                        current,
                        (args[index], args[index + 1]),
                        (args[index + 2], args[index + 3]),
                        point,
                    ]
                )
                current = point
        elif current is not None and start is not None and current != start:
            segments.append(_as_cubic(current, start))
            current = start
    if not segments:
        raise UnsupportedPathError("path has no drawable segment")
    return segments


def to_knots(segments) -> list[tuple[float, float]]:
    """Flatten segments into one 3n+1 chain so joins share parameters."""
    for before, after in itertools.pairwise(segments):
        if before[3] != after[0]:
            raise UnsupportedPathError("path is not one connected chain")
    points = [segments[0][0]]
    for segment in segments:
        points.extend(segment[1:])
    return points


torch = pytest.importorskip("torch", reason="the fit needs the vision extra")

SIZE = 700
WIDTH = 3.5
CASES = {
    "one gentle curve": "M 150 350 C 250 250 450 250 550 350",
    "an s-curve": "M 150 250 C 250 150 250 450 350 350 C 450 250 500 300 550 320",
    "a closed loop": "M 250 250 C 400 200 450 400 350 450 C 250 480 200 350 250 250 Z",
    "a short stub": "M 300 300 C 320 310 340 330 360 350",
}


def _real_ink(path_d: str) -> np.ndarray:
    """What cairosvg actually paints for this stroke, as ink in [0, 1]."""
    head = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}">'
    drawn = (
        f'{head}<path d="{path_d}" fill="none" stroke="#000000" '
        f'stroke-width="{WIDTH}" stroke-linecap="round" '
        f'stroke-linejoin="round" /></svg>'
    )
    blank = f"{head}</svg>"

    def ink(svg: str) -> np.ndarray:
        png = rasterize(svg, SIZE, SIZE)
        grey = Image.open(io.BytesIO(png)).convert("L")
        return 1.0 - np.asarray(grey, dtype=np.float32) / 255.0

    return ink(drawn) - ink(blank)


@pytest.mark.parametrize("name", list(CASES))
def test_the_soft_rasterizer_agrees_with_the_real_one(name):
    path_d = CASES[name]
    control = torch.tensor(
        [to_knots(parse_cubics(path_d))], dtype=torch.float32
    ).squeeze(0)
    controls = control.unfold(0, 4, 3).permute(0, 2, 1)
    soft = coverage(controls, WIDTH, (0, 0, SIZE, SIZE)).numpy()
    real = _real_ink(path_d)

    overlap = np.minimum(real, soft).sum() / np.maximum(real, soft).sum()
    mass = soft.sum() / max(real.sum(), 1e-6)
    assert overlap > 0.80, f"{name}: only {overlap:.3f} agreement with cairosvg"
    assert 0.85 < mass < 1.30, f"{name}: draws {mass:.2f}x the real ink"


def test_sampling_at_integer_coordinates_would_be_caught():
    """The guard has to be tight enough to fail on the bug it exists for."""
    path_d = CASES["one gentle curve"]
    control = torch.tensor(
        [to_knots(parse_cubics(path_d))], dtype=torch.float32
    ).squeeze(0)
    controls = control.unfold(0, 4, 3).permute(0, 2, 1)
    real = _real_ink(path_d)

    centred = coverage(controls, WIDTH, (0, 0, SIZE, SIZE)).numpy()
    # Half a pixel off, which is what sampling at integer coordinates does.
    shifted = coverage(controls + 0.5, WIDTH, (0, 0, SIZE, SIZE)).numpy()

    def agreement(a):
        return np.minimum(real, a).sum() / np.maximum(real, a).sum()

    assert agreement(centred) > agreement(shifted) + 0.02


_WING = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 700">'
    # A sweep and two feathers meeting it, split across groups the way a model
    # named them: the sweep as "body_outline", the feathers as "tail_feathers".
    '<g id="body_outline" fill="none" stroke="#111111" stroke-width="3.5">'
    '<path d="M 300 455 C 320 490 420 500 470 440" /></g>'
    '<g id="tail_feathers" fill="none" stroke="#111111" stroke-width="3.5">'
    '<path d="M 470 440 C 480 425 470 415 455 410" />'
    '<path d="M 470 440 C 485 435 490 420 480 408" /></g></svg>'
)
