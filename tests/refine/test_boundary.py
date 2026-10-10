"""Whole-contour readings smooth dense traces without losing real features."""

from dataclasses import replace

import numpy as np
import pytest

from vectrify.document import export_svg, import_svg
from vectrify.refine.boundary import field_loops, loops, reconstructed
from vectrify.refine.crossings import crossings
from vectrify.refine.snap import _Frame
from vectrify.svg_render import render_image

FRAME = _Frame(np.eye(2), np.zeros(2))


def shape(noisy=False):
    t = np.linspace(0, 2 * np.pi, 129)
    # A broad, genuine inward notch at the top of an otherwise round shape.
    angle = t - np.pi / 2
    radius = 20 - 7 * np.exp(-((angle / 0.3) ** 2))
    derivative = 14 * angle / 0.3**2 * np.exp(-((angle / 0.3) ** 2))
    if noisy:
        radius += 0.8 * np.sin(32 * t)
        derivative += 25.6 * np.cos(32 * t)
    points = 32 + radius[:, None] * np.column_stack((np.cos(t), np.sin(t)))
    tangent = derivative[:, None] * np.column_stack((np.cos(t), np.sin(t)))
    tangent += radius[:, None] * np.column_stack((-np.sin(t), np.cos(t)))
    points[-1] = points[0]
    step = (t[1] - t[0]) / 3
    path = (
        f"M{points[0, 0]} {points[0, 1]} "
        + " ".join(
            "C"
            + " ".join(
                str(v)
                for v in np.r_[
                    points[j - 1] + step * tangent[j - 1],
                    points[j] - step * tangent[j],
                    points[j],
                ]
            )
            for j in range(1, len(t))
        )
        + " Z"
    )
    return import_svg(f'<svg width="64" height="64"><path id="p" d="{path}"/></svg>')


def pixels(document):
    return np.asarray(render_image(export_svg(document)), float) / 255


def reference_contours(reference):
    target = pixels(reference)
    return field_loops(
        2 * target.mean(-1) - 1, np.ones((64, 64)), np.pad(1 - target.mean(-1), 8), 8
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("transformed", [False, True])
def test_reconstruction_removes_dense_wiggles_preserves_notch_ids_and_movement(
    reverse, transformed
):
    source, reference = shape(True), shape()
    geometry = source.geometry_for("p")
    contours = reference_contours(reference)
    frame = (
        _Frame(np.array([[0, -2], [1, 0]]), np.array([120, -10]))
        if transformed
        else FRAME
    )
    contours = [frame.pixels(tuple(p.ravel())) for p in contours]
    if reverse:
        contours = [p[::-1] for p in contours]
    target = pixels(reference)
    before = best = float(np.square(pixels(source) - target).mean())

    def accept(candidate):
        nonlocal best
        actual = float(
            np.square(pixels(source.replace_geometry(candidate)) - target).mean()
        )
        if actual >= best or crossings(candidate):
            return False
        best = actual
        return True

    result = reconstructed(
        geometry, geometry, frame, contours, 1, 2, frozenset(), accept, lambda: False
    )
    old, new = geometry.subpaths[0].nodes, result.subpaths[0].nodes
    assert [n.id for n in new] == [n.id for n in old]
    assert [n.command for n in new] == [n.command for n in old]
    assert new[0].endpoint == new[-1].endpoint
    assert all(
        np.linalg.norm(
            np.asarray(a.values).reshape(-1, 2) - np.asarray(b.values).reshape(-1, 2),
            axis=1,
        ).max()
        <= 2 + 1e-9
        for a, b in zip(old, new, strict=True)
    )
    assert best < before * 0.6
    assert new[32].endpoint[1] < 46  # The reference's deep notch remains.
    assert crossings(result) == 0


@pytest.mark.parametrize("protection", ["pin", "feature", "held", "stopped", "cap"])
def test_reconstruction_respects_protected_nodes_and_stop(protection):
    source, reference = shape(True), shape()
    geometry = source.geometry_for("p")
    node = geometry.subpaths[0].nodes[32]
    if protection == "pin":
        geometry = geometry.replace_node(replace(node, pinned=True))
    elif protection == "feature":
        geometry = geometry.replace_node(replace(node, feature="tip"))
    result = reconstructed(
        geometry,
        geometry,
        FRAME,
        reference_contours(reference),
        1,
        0 if protection == "cap" else 2,
        frozenset({node.id}) if protection == "held" else frozenset(),
        lambda _: True,
        lambda: protection == "stopped",
    )
    assert result == geometry


def test_subpixel_contours_do_not_quantize_diagonal_edges_to_pixels():
    y, x = np.mgrid[:64, :64] + 0.5
    field = np.maximum(np.abs(x - 32) + np.abs(y - 32) - 20.25, y - 51.75)
    contours = loops(field)
    assert len(contours) == 1
    contour = contours[0]
    diagonal = contour[np.abs(contour[:, 1] - 32) < 10]
    assert np.abs(diagonal[:, 0] - 32) + np.abs(diagonal[:, 1] - 32) == pytest.approx(
        20.25
    )


def test_reference_field_preserves_off_crop_closure_and_occluded_edges():
    y, x = np.mgrid[-8:40, -8:40] + 0.5
    coverage = ((x - 30) ** 2 + (y - 16) ** 2 < 9**2).astype(float)
    power = np.ones((32, 32))
    power[:, 24:] = 0
    signal = 1 - 2 * coverage[8:40, 8:40]
    signal[:, 24:] = 1  # An opaque foreground would suggest erasing this area.
    contours = field_loops(signal, power, coverage, 8)
    assert len(contours) == 1
    assert contours[0][:, 0].max() > 38
    assert field_loops(signal, power * 0, coverage, 8) == []
