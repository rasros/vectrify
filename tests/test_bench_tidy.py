"""The strict geometry checks remain accurate for nearly coincident cubics."""

import numpy as np
import pytest
from shapely.affinity import affine_transform
from shapely.geometry import Polygon

from scripts.bench_tidy import _covered_pixels, _shape, properties
from vectrify.document import import_svg


@pytest.mark.parametrize("covered", [False, True])
def test_visible_transparency_distinguishes_a_gap_covered_by_later_artwork(covered):
    front = '<rect x="0" y="13" width="20" height="7" fill="gold"/>' if covered else ""
    document = import_svg(
        '<svg width="20" height="20"><g id="g">'
        '<path id="fill" fill="red" d="M1 1 L19 1 L19 19 L10 13 L1 19 Z"/>'
        '<path id="outline" fill="none" stroke="black" stroke-width="0.2" '
        f'd="M1 19 L1 1 L19 1 L19 19"/></g>{front}</svg>'
    )
    measured = properties(document, "g", "outline", [(10, 0), (10, 20)])
    assert measured["transparent_pixels"] > 0
    assert measured["gap_area"] > 0
    if covered:
        assert measured["visible_transparent_pixels"] == 0
    else:
        assert measured["visible_transparent_pixels"] == measured["transparent_pixels"]


def test_exact_adjacent_fills_can_have_transparency_without_geometric_gaps():
    document = import_svg(
        '<svg width="48" height="48"><g id="g">'
        '<path fill="white" d="M0 0 L48 0 L48 48 Z"/>'
        '<path fill="white" d="M0 0 L48 48 L0 48 Z"/>'
        '<path id="outline" fill="none" stroke="black" stroke-width="0" '
        'd="M0 48 L0 0 L48 0 L48 48"/></g></svg>'
    )
    measured = properties(document, "g", "outline")
    assert measured["gap_area"] == 0
    assert measured["visible_transparent_pixels"] > 0
    assert (
        measured["visible_transparent_pixels_inside_fills"]
        == measured["visible_transparent_pixels"]
    )


@pytest.mark.parametrize(
    ("aspect", "bounds"),
    [
        ("none", (10, 20, 10.5, 21)),
        ("xMidYMid meet", (5, 20, 6, 21)),
        ("xMaxYMax meet", (0, 20, 1, 21)),
        ("xMidYMid slice", (10, 22.5, 10.5, 23)),
    ],
)
def test_pixel_coverage_maps_svg_viewport(aspect, bounds):
    document = import_svg(
        '<svg width="20" height="10" viewBox="10 20 10 10" '
        f'preserveAspectRatio="{aspect}"></svg>'
    )
    pixels = np.zeros((10, 20), bool)
    pixels[0, 0] = True
    left, top, right, bottom = bounds
    box = Polygon([(left, top), (right, top), (right, bottom), (left, bottom)])
    assert _covered_pixels(document, pixels, box) == 1
    assert _covered_pixels(document, pixels, box.buffer(-0.01)) == 0


def test_completed_symmetry_axis_distinguishes_translation_from_asymmetry():
    document = import_svg(
        '<svg width="40" height="40"><g id="g" transform="translate(4 0)">'
        '<path fill="red" d="M5 35 L10 5 L15 35 Z"/>'
        '<path id="outline" fill="none" stroke="black" stroke-width="0.2" '
        'd="M5 35 L10 5 L15 35"/></g></svg>'
    )
    measured = properties(document, "g", "outline", [(10, 35), (10, 5)])
    assert measured["symmetry_area"] > 0
    assert measured["symmetry_area_completed_axis"] == pytest.approx(0)


def test_transformed_evenodd_fill_area_and_small_spills():
    document = import_svg(
        '<svg width="100" height="100"><g transform="translate(3 4) scale(2 3)">'
        '<path id="p" fill-rule="evenodd" '
        'd="M0 0 L10 0 L10 10 L0 10 Z M2 2 L8 2 L8 8 L2 8 Z"/>'
        '</g><path id="outline" d="M0 0 L10 0 L10 10 L0 10 Z"/>'
        '<path id="spill" d="M0 0 L10.006 0 L10.006 10 L0 10 Z"/></svg>'
    )
    shape = _shape(document, "p")
    assert shape.area == pytest.approx(384)
    assert shape.bounds == (3, 4, 23, 34)
    spill = _shape(document, "spill").difference(_shape(document, "outline")).area
    assert spill == pytest.approx(0.06)
    assert spill > 0.05


def test_nearly_overlapping_cubics_match_an_independent_dense_oracle():
    # The native cubic XOR returned 58,260 here instead of about 2,437.
    controls = np.array(
        [
            [
                [413.7913850377151, 1550.1877421399067],
                [408.87368235367035, 1119.7528425587234],
                [403.0056522353432, 688.0647786191241],
                [395.3617045781167, 257.860629335877],
            ],
            [
                [395.3617045781167, 257.860629335877],
                [393.3505366365732, 194.1675094034897],
                [387.2169959140134, 140.8900429151497],
                [365.0221573329282, 81.7376887708719],
            ],
            [
                [365.0221573329282, 81.7376887708719],
                [337.93149898959757, 138.05711867836465],
                [334.3110828643627, 203.6306464971775],
                [332.5409131831672, 262.05615924322086],
            ],
            [
                [332.5409131831672, 262.05615924322086],
                [324.097613876162, 688.164422897084],
                [320.63386219770393, 1122.151097472574],
                [312.3908807471912, 1556.909849246841],
            ],
        ]
    )
    data = f"M{controls[0, 0, 0]} {controls[0, 0, 1]} " + " ".join(
        "C" + " ".join(str(v) for v in c[1:].ravel()) for c in controls
    )
    document = import_svg(
        f'<svg width="724" height="2172"><path id="p" d="{data}"/></svg>'
    )
    shape = _shape(document, "p")
    t = np.linspace(0, 1, 10001)
    u = 1 - t
    basis = np.column_stack((u**3, 3 * u * u * t, 3 * u * t * t, t**3))
    points = np.concatenate([basis @ c for c in controls])
    axis = np.array([[363.515, 1552.44], [363.57361845498355, 80.9212613608523]])
    direction = axis[1] - axis[0]
    direction /= np.linalg.norm(direction)
    matrix = 2 * np.outer(direction, direction) - np.eye(2)
    offset = axis[0] - matrix @ axis[0]
    oracle = Polygon(points)
    reflected_oracle = Polygon(points @ matrix.T + offset)
    reflected = affine_transform(shape, (*matrix.ravel(), *offset))
    assert oracle.is_valid
    assert reflected_oracle.is_valid
    assert shape.area == pytest.approx(oracle.area, abs=0.001)
    expected = oracle.symmetric_difference(reflected_oracle).area
    assert 2436 < expected < 2438
    assert shape.symmetric_difference(reflected).area == pytest.approx(
        expected, abs=0.001
    )
