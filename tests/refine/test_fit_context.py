"""Exact path-fit crops agree with the reference's raster pixel grid."""

import numpy as np
import pytest

from vectrify.document import Selection, import_svg
from vectrify.refine.selected import FitContext, FitOptions
from vectrify.svg_render import render_image


@pytest.mark.parametrize(
    ("size", "origin"), [(64, (0, 0)), (128, (0, 0)), (128, (10, 20))]
)
def test_fractional_path_bounds_do_not_change_the_raster_comparison(size, origin):
    x, y = origin
    svg = (
        f'<svg width="64" height="64" viewBox="{x} {y} 64 64">'
        '<rect x="-100" y="-100" width="200" height="200" fill="white"/>'
        f'<path id="p" fill="black" d="M{x + 12.13} {y + 16.27} '
        f"C{x + 18.31} {y + 9.11} {x + 34.17} {y + 12.93} {x + 40.43} {y + 25.19} "
        f'L{x + 39.71} {y + 42.47} L{x + 14.23} {y + 40.31} Z"/></svg>'
    )
    target = render_image(svg, (x, y, 64, 64), (size, size))
    context = FitContext(
        import_svg(svg),
        Selection(object_ids=frozenset({"p"})),
        target,
        FitOptions(resolution=384),
    )
    assert np.array_equal(np.asarray(context.before_image), np.asarray(context.target))
