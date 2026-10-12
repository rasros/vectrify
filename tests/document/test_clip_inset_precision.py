from __future__ import annotations

import pytest
from shapely import get_coordinates
from shapely.geometry import Point, box

from vectrify.document.clipping import inset_geometry
from vectrify.document.hit_test import _filled, _flatten
from vectrify.document.model import Geometry
from vectrify.document.svg import parse_path


def filled(geometry: Geometry):
    contours = []
    for sub in geometry.subpaths:
        points = [sub.nodes[0].endpoint]
        for node in sub.nodes[1:]:
            v = node.values
            points.extend(
                _flatten(
                    (points[-1], (v[0], v[1]), (v[2], v[3]), node.endpoint),
                    0.0001,
                )
                if node.command == "C"
                else [node.endpoint]
            )
        contours.append((tuple(points), sub.closed))
    return _filled(tuple(contours), "nonzero")


@pytest.mark.parametrize("inset", [0.1, 0.325, 1.0])
def test_thin_curved_inset_stays_at_requested_distance(inset):
    # A tapered curved blade: test both edges away from sharp corners, where
    # the inner stroke perimeter must sit exactly one half-width inward.
    source = parse_path(
        "M162.887 353.507 C166.158 348.818 170.087 343.904 178.530 339.672 "
        "C174.141 348.083 169.191 351.997 169.565 357.249 Z"
    )
    area = filled(source)
    derived = inset_geometry(source, inset, "nonzero")
    edge = filled(derived).boundary.intersection(box(170, 341, 176, 348))
    samples = get_coordinates(edge)
    assert len(samples) > 20
    for point in samples:
        assert Point(point).distance(area.boundary) == pytest.approx(inset, abs=0.002)
    # Precision comes from the derived curve, without polygonising the source
    # or changing the editable geometry that linked clips continue to share.
    assert any(n.command == "C" for s in derived.subpaths for n in s.nodes)
