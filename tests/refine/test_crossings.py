"""Outlines are checked for crossing themselves, and the path fit never adds any."""

import math
import xml.etree.ElementTree as ET
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
from PIL import Image, ImageDraw

from vectrify.document import Selection, import_svg
from vectrify.refine.crossings import crossed_nodes, crossings, polyline_crossings
from vectrify.refine.selected import FitOptions, fit_selected_path

SQUARE = "M14 14 L50 14 L50 50 L14 50 Z"
BOW_TIE = "M14 14 L50 14 L14 50 L50 50 Z"
# A tuft of grass: deeply concave, but it never crosses itself.
GRASS = "M5 60 L15 10 L22 60 L32 5 L40 60 L52 15 L60 60 Z"
# One cubic whose handles cross over, tying it into a loop.
LOOP = "M0 0 C40 30 -20 30 20 0"


def geometry(d):
    svg = f'<svg width="64" height="64"><path id="p" d="{d}"/></svg>'
    return import_svg(svg).geometry_for("p")


def test_a_square_and_a_concave_tuft_do_not_cross_themselves():
    assert crossings(geometry(SQUARE)) == 0
    assert crossings(geometry(GRASS)) == 0


def test_a_bow_tie_crosses_itself_once_at_its_waist():
    bow_tie = geometry(BOW_TIE)
    count, nodes = crossed_nodes(bow_tie)
    assert count == 1
    # The two slanted sides cross: the second and the closing one, so every
    # node is at the end of one of them.
    assert nodes == {n.id for s in bow_tie.subpaths for n in s.nodes}
    # Here the first and third sides cross, and the last node is on neither.
    twisted = geometry("M0 0 L10 10 L10 0 L0 10 L-5 5 Z")
    count, nodes = crossed_nodes(twisted)
    assert count == 1
    assert nodes == {n.id for n in twisted.subpaths[0].nodes[:4]}


def test_a_cubic_tied_into_a_loop_crosses_itself():
    assert crossings(geometry(LOOP)) == 1
    # Its handles pulled apart, the same curve is a plain arch.
    assert crossings(geometry("M0 0 C5 30 15 30 20 0")) == 0


def test_separate_contours_and_touching_lines_do_not_count():
    assert crossings(geometry(SQUARE + " M20 20 L60 20 L60 60 Z")) == 0
    # A U whose arms come close, and a line doubling back along itself.
    u = np.array([[0, 0], [0, 10], [4.9, 1], [5.1, 1], [10, 10], [10, 0]])
    assert polyline_crossings(u, True) == 0
    assert polyline_crossings(np.array([[0, 0], [10, 0], [2, 0]]), False) == 0


def fitted(document, result):
    shape = document.geometry_for("p")
    return replace(
        shape,
        subpaths=tuple(
            replace(
                s,
                nodes=tuple(
                    replace(n, values=result.values.get(n.id, n.values))
                    for n in s.nodes
                ),
            )
            for s in shape.subpaths
        ),
    )


def test_the_path_fit_pulls_back_a_candidate_that_crosses_itself(monkeypatch):
    pytest.importorskip("torch")
    import torch

    from vectrify.refine.paths import parse_filled_cubics

    seen = []

    def bow_tie_fit(svg, _target, *, observe, device, **_options):
        """Stand in for the gradient fit: jump straight to a bow-tie."""
        (square,) = parse_filled_cubics(ET.fromstring(svg)[0].get("d", ""))
        corners = torch.tensor([square[i][0] for i in (0, 1, 3, 2)], device=device)
        ends = corners.roll(-1, 0)
        sides = torch.stack(
            (corners, (2 * corners + ends) / 3, (corners + 2 * ends) / 3, ends), 1
        )
        paths = [[sides]]
        observe(10, paths, [torch.zeros(3, device=device)])
        seen.append(paths[0][0].cpu())
        return svg

    monkeypatch.setattr("vectrify.refine.paths.fit_filled_svg", bow_tie_fit)
    svg = (
        '<svg width="64" height="64"><rect width="64" height="64" fill="#fff"/>'
        f'<path id="p" fill="#000000" d="{SQUARE}"/></svg>'
    )
    document = import_svg(svg)
    # The reference is the bow-tie itself, so it would score best.
    reference = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(reference).polygon(
        [(14, 14), (50, 14), (14, 50), (50, 50)], fill="black"
    )
    result = fit_selected_path(
        document,
        Selection(object_ids=frozenset({"p"})),
        reference,
        FitOptions(steps=10, displacement=100, resolution=64),
    )
    assert result.folded == 1
    assert crossings(fitted(document, result)) == 0
    # The fit itself was moved back off the bow-tie before going on.
    assert polyline_crossings(seen[0][:, 0].numpy(), True) == 0


def circle(nodes, r=30.0):
    k = 4 / 3 * math.tan(math.pi / (2 * nodes))
    at = [2 * math.pi * i / nodes for i in range(nodes + 1)]
    d = f"M{64 + r} 64"
    for a0, a1 in pairwise(at):
        d += (
            f" C{64 + r * (math.cos(a0) - k * math.sin(a0))}"
            f" {64 + r * (math.sin(a0) + k * math.cos(a0))}"
            f" {64 + r * (math.cos(a1) + k * math.sin(a1))}"
            f" {64 + r * (math.sin(a1) - k * math.cos(a1))}"
            f" {64 + r * math.cos(a1)} {64 + r * math.sin(a1)}"
        )
    return d + " Z"


def test_a_round_path_pulled_flat_onto_a_bar_does_not_fold():
    """Unguarded, this fit twists the circle's top and bottom into loops."""
    pytest.importorskip("torch")
    svg = (
        '<svg width="128" height="128"><rect width="128" height="128" '
        f'fill="#fff"/><path id="p" fill="#000000" d="{circle(16)}"/></svg>'
    )
    document = import_svg(svg)
    bar = Image.new("RGB", (128, 128), "white")
    ImageDraw.Draw(bar).rectangle((5, 58, 123, 70), fill="black")
    result = fit_selected_path(
        document,
        Selection(object_ids=frozenset({"p"})),
        bar,
        FitOptions(steps=60, displacement=20, resolution=128),
    )
    assert result.after < result.before
    assert crossings(fitted(document, result)) == 0
