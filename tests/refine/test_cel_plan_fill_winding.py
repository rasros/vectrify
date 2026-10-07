"""Filled seam correction remains bounded and preserves real holes."""

import numpy as np
import pytest

from vectrify.document import export_svg, import_svg
from vectrify.document.join import transformed_geometry
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.fill_winding import resolved
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.score import render
from vectrify.refine.crossings import crossings


def seam():
    svg = (
        '<svg width="180" height="1510"><path id="fill" fill-rule="nonzero" d="'
        "M160 1497.8643798828125L160 1497L161 1497L161 1496L162 1496"
        "L162 1494L162.6666717529297 1494L162 1496"
        "C161.71609497070312 1496.3134765625 161.3790740966797 1496.65087890625 "
        "161 1496.9996337890625"
        "C160.69239807128906 1497.2825927734375 160.3571014404297 "
        "1497.5731201171875 160 1497.8643798828125Z"
        'M150 1490H158V1500H150Z M152 1492V1498H156V1492Z"/></svg>'
    )
    return import_svg(svg)


@pytest.mark.parametrize("matrix", [(1, 0, 0, 1, 0, 0), (1.2, 0.15, 0.2, 1, 6, -10)])
def test_persistent_filled_seam_is_resolved_in_native_coordinates_with_real_hole(
    matrix,
):
    document = seam()
    geometry = document.geometry_for("fill")
    assert crossings(geometry) == 1
    local = transformed_geometry(geometry, inverse_matrix(matrix))
    corrected = resolved(local, matrix, "nonzero", Work.start(10))
    assert corrected is not None
    assert crossings(corrected) == 0
    native = transformed_geometry(corrected, matrix)
    assert crossings(native) == 0
    changed = document.replace_geometry(type(geometry)(geometry.id, native.subpaths))
    before, after = (render(export_svg(d), (180, 1510)) for d in (document, changed))
    np.testing.assert_array_equal(after[1493:1497, 153:155], before[1493:1497, 153:155])
    assert after[1493:1497, 153:155, 3].max() == 0
    assert np.abs(after - before).max() <= 0.02
    assert np.abs(after[..., 3] - before[..., 3]).sum() <= 0.05


def test_cancelled_winding_resolution_does_not_return_partial_geometry():
    document = seam()
    work = Work.start(10)
    work.stop.set()
    assert (
        resolved(document.geometry_for("fill"), (1, 0, 0, 1, 0, 0), "nonzero", work)
        is None
    )


def test_winding_node_bound_retains_unresolved_input(monkeypatch):
    from vectrify.refine.cel_plan import fill_winding

    document = seam()
    geometry = document.geometry_for("fill")
    before = export_svg(document)
    monkeypatch.setattr(fill_winding, "MAX_NODES", 4)
    assert resolved(geometry, (1, 0, 0, 1, 0, 0), "nonzero", Work.start(10)) is None
    assert export_svg(document) == before
