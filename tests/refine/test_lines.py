"""Stroke-centre proposals can resolve errors hidden at a cubic's midpoint."""

from dataclasses import replace

import numpy as np
import pytest

from vectrify.document import import_svg
from vectrify.operations.generate import Region
from vectrify.operations.methods.nodes import _pixels, _Scored, _span_lines, _Task
from vectrify.refine.frozen import Paths, frozen
from vectrify.refine.lines import fit_lines
from vectrify.svg_render import render_image


def drawing(data):
    return import_svg(
        '<svg width="64" height="64"><path id="p" fill="none" stroke="black" '
        f'stroke-width="2" stroke-linecap="round" d="{data}"/></svg>'
    )


def problem():
    target = drawing("M12 40 C24 20 40 50 52 30")
    source = drawing("M12 40 C24 24 40 46 52 30")
    geometry = source.geometry_for("p")
    sub = geometry.subpaths[0]
    geometry = replace(
        geometry,
        subpaths=(
            replace(sub, nodes=tuple(replace(n, pinned=True) for n in sub.nodes)),
        ),
    )
    source = source.replace_geometry(geometry)
    from vectrify.document import export_svg

    image = render_image(export_svg(target), (0, 0, 64, 64), (64, 64))
    return source, Region(0, 0, 64, 64, image)


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
