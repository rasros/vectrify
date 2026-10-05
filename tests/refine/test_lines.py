"""Stroke-centre proposals can resolve errors hidden at a cubic's midpoint."""

from dataclasses import replace

import numpy as np
import pytest

from vectrify.document import export_svg, import_svg
from vectrify.operations.generate import Region
from vectrify.operations.methods.nodes import _pixels, _Scored, _span_lines, _Task
from vectrify.refine.frozen import Paths, frozen
from vectrify.refine.lines import _Reader, fit_lines
from vectrify.svg_render import render_image


def drawing(data, width=2):
    return import_svg(
        '<svg width="64" height="64"><path id="p" fill="none" stroke="black" '
        f'stroke-width="{width}" stroke-linecap="round" d="{data}"/></svg>'
    )


@pytest.mark.parametrize("width", [0.3, 1.1, 2.23, 3.7])
@pytest.mark.parametrize("normal", [(1, 0), (0.6, 0.8)])
def test_reading_the_same_ink_from_the_other_side_preserves_centre_and_width(
    width, normal
):
    from scipy.ndimage import gaussian_filter

    cover = np.zeros((32, 32))
    cover[:, 12:15] = 1
    cover = gaussian_filter(cover, sigma=0.6)
    reader = _Reader(cover, width)
    point = np.array([[13.1, 16]])
    direction = np.asarray([normal])
    shift, measured = reader.across(point, direction)
    reversed_shift, reversed_width = reader.across(point, -direction)
    assert measured[0] > 0
    np.testing.assert_allclose(shift, -reversed_shift, atol=1e-12, rtol=0)
    np.testing.assert_allclose(measured, reversed_width, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "foreground", [(0.02, 0.02, 0.02), (0.9, 0.15, 0.1), (1, 1, 1)]
)
def test_colour_cover_recovers_a_stroke_between_different_backgrounds(foreground):
    # Integrate an analytic straight stroke over 100 samples per pixel.
    # The foreground can be dark, colourful or lighter than both backgrounds.
    width, centre = 2.23, 32.2
    left, right = (0.65, 0.65, 0.65), (0.25, 0.45, 0.7)
    y = np.arange(64)[:, None] + (np.arange(100) + 0.5) / 100
    colours = np.where((y < centre)[..., None], left, right)
    colours = np.where((np.abs(y - centre) < width / 2)[..., None], foreground, colours)
    image = np.broadcast_to(colours.mean(1)[:, None], (64, 64, 3)).copy()
    reader = _Reader(
        np.zeros((64, 64)), width, image=image, foreground=np.asarray(foreground)
    )
    point, normal = np.array([[30, centre]]), np.array([[0, 1]])
    shift, measured = reader.across(point, normal)
    reversed_shift, reversed_width = reader.across(point, -normal)
    assert abs(shift[0]) < 0.1
    assert abs(measured[0] - width) < 0.06
    np.testing.assert_allclose(shift, -reversed_shift, atol=1e-12, rtol=0)
    np.testing.assert_allclose(measured, reversed_width, atol=1e-12, rtol=0)


def problem(width=2):
    target = drawing("M12 40 C24 20 40 50 52 30", width)
    source = drawing("M12 40 C24 24 40 46 52 30", width)
    geometry = source.geometry_for("p")
    sub = geometry.subpaths[0]
    geometry = replace(
        geometry,
        subpaths=(
            replace(sub, nodes=tuple(replace(n, pinned=True) for n in sub.nodes)),
        ),
    )
    source = source.replace_geometry(geometry)
    image = render_image(export_svg(target), (0, 0, 64, 64), (64, 64))
    return source, Region(0, 0, 64, 64, image)


@pytest.mark.parametrize("span", [False, True])
def test_group_opacity_is_not_mistaken_for_a_narrower_stroke(span):
    document = import_svg(
        '<svg width="64" height="64"><g opacity="0.4"><path id="p" fill="none" '
        'stroke="black" stroke-width="2.23" stroke-linecap="round" '
        'd="M32 8 C32 20 32 44 32 56"/></g></svg>'
    )
    image = render_image(export_svg(document), (0, 0, 64, 64), (64, 64))
    region = Region(0, 0, 64, 64, image)
    paths = Paths({"p": document.geometry_for("p")})
    result = fit_lines(document, ["p"], region, frozen(paths), True, span=span)
    assert result.element("p").get("stroke-width") == "2.23"
    assert result.root == document.root


def test_whole_span_fitting_moves_a_coloured_stroke_between_two_backgrounds():
    svg = (
        '<svg width="64" height="64"><rect width="64" height="64" fill="white"/>'
        '<rect x="32" width="32" height="64" fill="#3a80b3"/>'
        '<path id="p" fill="none" stroke="#e6261a" stroke-width="2.23" '
        'stroke-linecap="round" d="M31 8 C31 20 31 44 31 56"/></svg>'
    )
    source = import_svg(svg)
    target = svg.replace("M31 8 C31 20 31 44 31 56", "M32 8 C32 20 32 44 32 56")
    region = Region(0, 0, 64, 64, render_image(target, (0, 0, 64, 64), (64, 64)))
    paths = Paths({"p": source.geometry_for("p")})
    result = fit_lines(source, ["p"], region, frozen(paths), False, span=True)
    before = _Scored.of(_pixels(source, region), region)
    after = _Scored.of(_pixels(result, region), region)
    assert after.difference < before.difference * 0.5
    assert result.root == source.root
    assert [n.id for n in result.geometry_for("p").subpaths[0].nodes] == [
        n.id for n in source.geometry_for("p").subpaths[0].nodes
    ]


def test_colour_reader_recovers_a_thin_ribbon_not_touching_the_current_centre():
    foreground = np.array([0.9, 0.15, 0.1])
    image = np.ones((64, 64, 3))
    image[32] = 0.3 * foreground + 0.7
    reader = _Reader(np.zeros((64, 64)), 0.3, image=image, foreground=foreground)
    point, normal = np.array([[30, 31]]), np.array([[0, 1]])
    shift, measured = reader.across(point, normal)
    reversed_shift, reversed_width = reader.across(point, -normal)
    assert shift[0] == pytest.approx(1.5)
    assert measured[0] == pytest.approx(0.3)
    np.testing.assert_allclose(shift, -reversed_shift, atol=1e-12, rtol=0)
    np.testing.assert_allclose(measured, reversed_width, atol=1e-12, rtol=0)


def test_a_colour_reader_does_not_choose_between_equally_near_separate_ribbons():
    foreground = np.array([0.9, 0.15, 0.1])
    image = np.ones((64, 64, 3))
    image[[30, 32]] = 0.3 * foreground + 0.7
    reader = _Reader(np.zeros((64, 64)), 0.3, image=image, foreground=foreground)
    for normal in ([[0, 1]], [[0, -1]]):
        shift, measured = reader.across(np.array([[30, 31.5]]), np.asarray(normal))
        assert shift[0] == 0
        assert measured[0] == 0


@pytest.mark.parametrize("span", [False, True])
def test_reversing_a_fractional_width_cubic_keeps_the_fitted_curve(span):
    source, region = problem(2.23)
    geometry = source.geometry_for("p")
    sub = geometry.subpaths[0]
    first, curve = sub.nodes
    reverse = source.replace_geometry(
        replace(
            geometry,
            subpaths=(
                replace(
                    sub,
                    nodes=(
                        replace(first, values=curve.endpoint),
                        replace(
                            curve,
                            values=(
                                *curve.values[2:4],
                                *curve.values[:2],
                                *first.endpoint,
                            ),
                        ),
                    ),
                ),
            ),
        )
    )
    result = fit_lines(
        source, ["p"], region, frozen(Paths({"p": geometry})), False, span=span
    )
    reversed_result = fit_lines(
        reverse,
        ["p"],
        region,
        frozen(Paths({"p": reverse.geometry_for("p")})),
        False,
        span=span,
    )
    a, b = result.geometry_for("p").subpaths[0].nodes
    c, d = reversed_result.geometry_for("p").subpaths[0].nodes
    np.testing.assert_allclose(a.endpoint, d.endpoint, atol=1e-12, rtol=0)
    np.testing.assert_allclose(b.endpoint, c.endpoint, atol=1e-12, rtol=0)
    np.testing.assert_allclose(b.values[:2], d.values[2:4], atol=1e-12, rtol=0)
    np.testing.assert_allclose(b.values[2:4], d.values[:2], atol=1e-12, rtol=0)


def test_whole_curve_readings_fix_opposite_handle_errors_with_fixed_ends():
    document, region = problem()
    paths = Paths({"p": document.geometry_for("p")})
    result = fit_lines(document, ["p"], region, frozen(paths), False, span=True)
    before = _Scored.of(_pixels(document, region), region)
    after = _Scored.of(_pixels(result, region), region)
    assert after.difference < 0.4 * before.difference
    old = document.geometry_for("p").subpaths[0]
    new = result.geometry_for("p").subpaths[0]
    assert [n.id for n in new.nodes] == [n.id for n in old.nodes]
    assert [n.endpoint for n in new.nodes] == [n.endpoint for n in old.nodes]
    assert not new.closed
    assert result.element("p") == document.element("p")


@pytest.mark.parametrize("movement", [0, 0.2, 2])
def test_additional_proposal_respects_movement_and_pinned_endpoints(movement):
    document, region = problem()
    current = _Scored.of(_pixels(document, region), region)
    task = _Task(document, region, {"movement": movement}, ("p",), region.image)
    result, after = _span_lines(task, document, current, None)
    assert after.difference <= current.difference
    for old, new in zip(
        document.geometry_for("p").subpaths[0].nodes,
        result.geometry_for("p").subpaths[0].nodes,
        strict=True,
    ):
        delta = np.asarray(new.values).reshape(-1, 2) - np.asarray(old.values).reshape(
            -1, 2
        )
        assert np.linalg.norm(delta, axis=1).max() <= movement + 1e-12
        assert new.endpoint == old.endpoint
    if movement == 0:
        assert result == document
    else:
        assert after.difference < current.difference


def test_a_held_curve_keeps_its_handles_too():
    document, region = problem()
    curve = document.geometry_for("p").subpaths[0].nodes[1]
    task = _Task(
        document,
        region,
        {"movement": 2},
        ("p",),
        region.image,
        held=frozenset({curve.id}),
    )
    current = _Scored.of(_pixels(document, region), region)
    result, _ = _span_lines(task, document, current, None)
    assert result == document


def test_a_worse_whole_curve_proposal_is_discarded(monkeypatch):
    from vectrify.refine import lines

    document, region = problem()
    geometry = document.geometry_for("p")
    sub = geometry.subpaths[0]
    curve = sub.nodes[1]
    values = list(curve.values)
    values[1] += 2
    values[3] -= 2
    bad = document.replace_geometry(
        replace(
            geometry,
            subpaths=(
                replace(
                    sub, nodes=(sub.nodes[0], replace(curve, values=tuple(values)))
                ),
            ),
        )
    )
    monkeypatch.setattr(lines, "fit_lines", lambda *_args, **_kwargs: bad)
    task = _Task(document, region, {"movement": 2}, ("p",), region.image)
    current = _Scored.of(_pixels(document, region), region)
    result, after = _span_lines(task, document, current, None)
    assert result == document
    assert after == current
