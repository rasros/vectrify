"""Exact path-fit crops agree with the reference's raster pixel grid."""

import numpy as np
import pytest

from vectrify.document import Selection, import_svg
from vectrify.refine.selected import FitContext, FitOptions, fit_selected_path
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


def test_tidy_waits_for_the_fit_to_settle_after_an_unchanged_first_check(monkeypatch):
    torch = pytest.importorskip("torch")
    import xml.etree.ElementTree as ET

    from vectrify.refine.paths import parse_filled_cubics

    svg = (
        '<svg width="64" height="64"><path id="p" fill="black" '
        'd="M12 16 L48 16 L48 48 L12 48 Z"/></svg>'
    )
    target = render_image(svg.replace('id="p"', 'id="p" transform="translate(1.5 0)"'))
    seen = []

    def fitting(work, _target, **kwargs):
        d = ET.fromstring(work)[0].get("d")
        assert d is not None
        paths = [[torch.tensor(c, dtype=torch.float32) for c in parse_filled_cubics(d)]]
        colours = torch.zeros((1, 3))
        for step in (10, 20, 30):
            if step == 20:
                for contour in paths[0]:
                    contour[..., 0] += 1.5
                kwargs["project_controls"](paths)
            seen.append(step)
            if not kwargs["observe"](step, paths, colours):
                break
        return work

    monkeypatch.setattr("vectrify.refine.paths.fit_filled_svg", fitting)
    monkeypatch.setattr("vectrify.refine.selected.fit_device", lambda: "cpu")
    result = fit_selected_path(
        import_svg(svg),
        Selection(object_ids=frozenset({"p"})),
        target,
        FitOptions(steps=60, stall=0.001, cleanup=True, snap=False),
    )
    assert seen == [10, 20, 30]
    assert result.changed
    assert result.after < result.before / 10


def test_tidy_reference_profiles_support_transparent_artwork(monkeypatch):
    pytest.importorskip("torch")
    source = (
        '<svg width="64" height="64"><path id="p" fill="#668844" fill-opacity=".6" '
        'd="M8 16 L16 16 L18 18 L20 16 L56 16 L56 56 L8 56 Z"/></svg>'
    )
    target = render_image(source.replace("L18 18", "L18 16"), alpha=True)
    monkeypatch.setattr(
        "vectrify.refine.paths.fit_filled_svg", lambda svg, *_a, **_k: svg
    )
    result = fit_selected_path(
        import_svg(source),
        Selection(object_ids=frozenset({"p"})),
        target,
        FitOptions(cleanup=True, snap=False),
    )
    assert result.changed
    assert result.after < result.before / 10
