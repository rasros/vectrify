"""Inferred stroke subcurves couple real coordinates and obey edit limits."""

import time
from threading import Event

import numpy as np
import pytest

from vectrify.document import export_svg, import_svg
from vectrify.refine.crossings import bezier
from vectrify.refine.joint import _context, _Coordinates
from vectrify.refine.selected import FitOptions
from vectrify.refine.simplify import curved
from vectrify.refine.stroke_support import StrokeSupports
from vectrify.svg_render import render_image

CONTROL = np.array([[8, 16], [16, 6], [40, 6], [48, 16]], dtype=float)


def coordinates(
    *, reverse=False, later=True, transform="translate(0 0)", held=frozenset()
):
    pytest.importorskip("torch")
    points = bezier(CONTROL, np.linspace(0.2, 0.8, 9))[0]
    if reverse:
        points = points[::-1]
    data = "M" + " L".join(f"{x} {y}" for x, y in points)
    data += " L8 48 L48 48 Z" if reverse else " L48 48 L8 48 Z"
    fill = f'<path id="fill" transform="{transform}" fill="blue" d="{data}"/>'
    ink = (
        '<path id="ink" fill="none" stroke="black" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" d="M8 16 C16 6 40 6 48 16"/>'
    )
    document = import_svg(
        '<svg width="64" height="64"><g>'
        + (fill + ink if later else ink + fill)
        + "</g></svg>"
    )
    ids = ("fill", "ink") if later else ("ink", "fill")
    for oid in ids:
        document = document.replace_geometry(curved(document.geometry_for(oid)))
    target = render_image(export_svg(document), alpha=True)
    options = FitOptions(displacement=4, resolution=64)
    context = _context(document, ids, target, options)
    return document, [
        _Coordinates(document, oid, context, options, held, "cpu") for oid in ids
    ]


@pytest.mark.parametrize("reverse", [False, True])
def test_a_partial_curve_and_its_gradient_follow_the_later_stroke(reverse):
    torch = pytest.importorskip("torch")
    _document, coords = coordinates(reverse=reverse)
    model = StrokeSupports(coords, (1, 1))
    assert len(model.entries) == 8
    values = [p.original.clone().requires_grad_() for p in coords]
    result = model(values)
    first = result[0][coords[0].mapping.gather_index[0]]
    expected = bezier(CONTROL, np.array([0.8 if reverse else 0.2]))[0][0]
    np.testing.assert_allclose(first[0].detach().numpy(), expected, atol=1e-5)
    gradient = torch.autograd.grad(result[0].sum(), values, allow_unused=True)
    assert gradient[1] is not None
    assert gradient[1].abs().sum() > 0
    assert torch.isfinite(gradient[1]).all()
    # Translating the guide translates every copied coordinate identically.
    changed = model([values[0], values[1] + values[1].new_tensor((0.2, 0.3))])
    np.testing.assert_allclose(
        (changed[0][coords[0].mapping.gather_index[0]] - first).detach().numpy(),
        np.tile((0.2, 0.3), (4, 1)),
        atol=1e-5,
    )


def test_proposals_cannot_enlarge_movement_limits():
    _document, coords = coordinates()
    model = StrokeSupports(coords, (1, 1))
    shifted = model([p.original + 100 for p in coords])
    rows = {int(i) for entry in model.entries for i in entry[2]}
    displacement = (shifted[0][list(rows)] - coords[0].original[list(rows)]).norm(
        dim=-1
    )
    assert float(displacement.max()) <= 4 + 1e-5


def test_held_and_explicitly_shared_target_coordinates_are_excluded():
    _document, coords = coordinates()
    indices = coords[0].mapping.gather_index[3]
    coords[0].mapping.movable[indices[-1]] = 0
    model = StrokeSupports(coords, (1, 1))
    assert all(not bool((indices[-1] == entry[2]).any()) for entry in model.entries)
    shared = StrokeSupports(coords, (1, 1), shared_rows={(1, 0)})
    assert not shared.active


def test_a_stroke_below_the_fill_and_expired_inference_are_skipped():
    _document, coords = coordinates(later=False)
    assert not StrokeSupports(coords, (1, 1)).active
    _document, coords = coordinates()
    assert not StrokeSupports(coords, (1, 1), deadline=time.monotonic() - 1).active


def test_reference_coordinates_include_each_paths_transform():
    _document, coords = coordinates(transform="translate(.1 .1)")
    assert StrokeSupports(coords, (1, 1)).active
    # Doubling reference resolution also doubles its distance from the guide.
    # At a high enough scale it is correctly no longer a nearby boundary.
    assert not StrokeSupports(coords, (100, 100)).active


def test_joint_stall_tracks_progress_before_a_new_model_beats_the_retained_fit(
    monkeypatch,
):
    pytest.importorskip("torch")
    from vectrify.refine import joint, paths

    document = import_svg(
        '<svg width="64" height="64"><g><path id="fill" fill="blue" '
        'd="M8 30 L40 30 L40 50 L8 50 Z"/>'
        '<path id="ink" fill="none" stroke="black" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" '
        'd="M10 10 L50 10"/></g></svg>'
    )
    target = render_image(export_svg(document), alpha=True)
    original_class = joint._Coordinates
    coords = []

    def capture(*args, **kwargs):
        result = original_class(*args, **kwargs)
        coords.append(result)
        return result

    monkeypatch.setattr(joint, "_Coordinates", capture)

    def fit(_svg, _target, *, observe, **_kwargs):
        for step, dx in ((10, -0.5), (20, -0.25), (30, 0.5), (40, 0.5)):
            values = [
                p.controls_from_local(p.original + (dx if p.stroke_only else 0))
                for p in coords
            ]
            continuing = observe(step, values, None)
            assert continuing == (step != 40)

    monkeypatch.setattr(paths, "fit_filled_svg", fit)

    def score(doc):
        x = doc.geometry_for("ink").subpaths[0].nodes[0].endpoint[0]
        return (x - 11) ** 2

    result = joint._polish_group(
        document,
        ("fill", "ink"),
        target,
        FitOptions(steps=40, displacement=2, resolution=64, stall=0.005),
        frozenset(),
        (),
        score,
        Event(),
        None,
        frozenset(),
    )
    assert score(result) == pytest.approx(0.25)


def test_an_exact_reference_rejects_the_coupled_proposal():
    pytest.importorskip("torch")
    from vectrify.refine import joint

    document, _coords = coordinates()
    target = render_image(export_svg(document), alpha=True)
    expected = np.asarray(target, dtype=float)

    def score(candidate):
        actual = np.asarray(
            render_image(export_svg(candidate), alpha=True), dtype=float
        )
        return float(np.square(actual - expected).mean())

    result = joint._polish_group(
        document,
        ("fill", "ink"),
        target,
        FitOptions(steps=20, displacement=4, resolution=64),
        frozenset(),
        (),
        score,
        Event(),
        None,
        frozenset(),
        supports=True,
    )
    assert result == document


def test_coupled_bilateral_curve_improves_a_filled_shape_without_edge_labels():
    pytest.importorskip("torch")
    from vectrify.refine import joint
    from vectrify.refine.bilateral import Bilateral

    source = (
        '<svg width="64" height="64"><g>'
        '<path id="fill" fill="#38b" d="M10 50 '
        'C13.5 38 26.5 19 32.2 8 C38.3 19 51.5 36 54 51 Z"/>'
        '<path id="ink" fill="none" stroke="black" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" '
        'd="M9 50 C13 38 26 19 32.2 8 C38.8 19 52 36 54.8 51"/>'
        "</g></svg>"
    )
    target_svg = source.replace(
        "M10 50 C13.5 38 26.5 19 32.2 8 C38.3 19 51.5 36 54 51",
        "M10 50 C10 38 26 18 32 8 C38 18 54 38 54 50",
    ).replace(
        "M9 50 C13 38 26 19 32.2 8 C38.8 19 52 36 54.8 51",
        "M10 50 C10 38 26 18 32 8 C38 18 54 38 54 50",
    )
    document = import_svg(source)
    target = render_image(target_svg, alpha=True)
    expected = np.asarray(target, dtype=float)

    def score(candidate):
        actual = np.asarray(
            render_image(export_svg(candidate), alpha=True), dtype=float
        )
        return float(np.square(actual - expected).mean())

    result = joint._polish_group(
        document,
        ("fill", "ink"),
        target,
        FitOptions(steps=40, displacement=4, resolution=64),
        frozenset(),
        (),
        score,
        Event(),
        None,
        frozenset(),
        bilateral=True,
        supports=True,
    )
    ordinary = joint._polish_group(
        document,
        ("fill", "ink"),
        target,
        FitOptions(steps=40, displacement=4, resolution=64),
        frozenset(),
        (),
        score,
        Event(),
        None,
        frozenset(),
    )
    retained = joint.polish(
        document,
        ("fill", "ink"),
        target,
        FitOptions(steps=40, displacement=4, resolution=64),
        score=score,
    )
    assert score(retained) <= score(result)
    assert score(result) < score(ordinary) * 0.6
    assert score(result) < score(document) * 0.6
    model = Bilateral.infer(result.geometry_for("ink"), frozenset(), 4)
    assert model is not None
    np.testing.assert_allclose(
        model.original, model.average(model.original), atol=1e-10
    )
    for oid in ("fill", "ink"):
        assert result.element(oid) == document.element(oid)
        assert [n.id for s in result.geometry_for(oid).subpaths for n in s.nodes] == [
            n.id for s in document.geometry_for(oid).subpaths for n in s.nodes
        ]


def test_a_fill_cubic_crossing_a_stroke_knot_receives_both_gradients():
    torch = pytest.importorskip("torch")
    first = np.array([[8, 40], [8, 35], [12, 31], [16, 28]], dtype=float)
    second = np.array([[16, 28], [20, 25], [24, 23], [32, 22]], dtype=float)
    parameter = np.linspace(0.4, 1.6, 25)
    points = np.array(
        [
            bezier(first if t <= 1 else second, np.array([t if t <= 1 else t - 1]))[0][
                0
            ]
            for t in parameter
        ]
    )
    t = np.linspace(0, 1, len(parameter))
    basis = np.stack(((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3), -1)
    middle = np.linalg.lstsq(
        basis[:, 1:3],
        points - basis[:, :1] * points[:1] - basis[:, 3:] * points[-1:],
        rcond=None,
    )[0]
    control = np.vstack((points[0], middle, points[-1]))
    data = (
        "M"
        + " ".join(str(v) for v in control[0])
        + " C"
        + " ".join(str(v) for v in control[1:].ravel())
        + " L32 48 L8 48 Z"
    )
    document = import_svg(
        '<svg width="64" height="64"><g>'
        + f'<path id="fill" fill="blue" d="{data}"/>'
        + '<path id="ink" fill="none" stroke="black" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" '
        'd="M8 40 C8 35 12 31 16 28 C20 25 24 23 32 22"/></g></svg>'
    )
    for oid in ("fill", "ink"):
        document = document.replace_geometry(curved(document.geometry_for(oid)))
    target = render_image(export_svg(document), alpha=True)
    options = FitOptions(displacement=4, resolution=64)
    context = _context(document, ("fill", "ink"), target, options)
    coords = [
        _Coordinates(document, oid, context, options, frozenset(), "cpu")
        for oid in ("fill", "ink")
    ]
    model = StrokeSupports(coords, (1, 1))
    assert len(model.entries) == 1
    assert model.entries[0][3].numel() == 8
    values = [p.original.clone().requires_grad_() for p in coords]
    result = model(values)
    gradient = torch.autograd.grad(result[0].sum(), values)[1]
    assert gradient[1:3].abs().sum() > 0
    assert gradient[4:6].abs().sum() > 0
    assert torch.isfinite(gradient).all()


def test_model_comparison_keeps_other_groups_fits_and_uses_the_same_seed(monkeypatch):
    from dataclasses import replace

    from vectrify.refine import joint

    group = (
        '<g><path id="fill_ID" fill="blue" d="M8 50 L32 8 L54 50 Z"/>'
        '<path id="ink_ID" fill="none" stroke="black" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" '
        'd="M9 50 C13 38 26 19 32 8 C38 19 51 38 55 50"/></g>'
    )
    document = import_svg(
        '<svg width="64" height="64">'
        + group.replace("ID", "a")
        + group.replace("ID", "b")
        + "</svg>"
    )
    calls = []

    def propose(initial, group, *_args, bilateral=False, supports=False):
        fill = group[0]
        first = initial.geometry_for(fill).subpaths[0].nodes[0]
        # Every proposal starts from the same group geometry. Fits to other
        # groups must remain in the surrounding artwork when seeds are reset.
        assert first.endpoint[0] == 8
        other = "fill_b" if fill == "fill_a" else "fill_a"
        calls.append(
            (
                fill,
                bilateral,
                initial.geometry_for(other).subpaths[0].nodes[0].endpoint[0],
            )
        )
        x = 11 if supports else 10 if bilateral else 9
        geometry = initial.geometry_for(fill)
        sub = geometry.subpaths[0]
        return initial.replace_geometry(
            replace(
                geometry,
                subpaths=(
                    replace(
                        sub, nodes=(replace(first, values=(x, 50)), *sub.nodes[1:])
                    ),
                ),
            )
        )

    def score(candidate):
        return sum(
            (candidate.geometry_for(oid).subpaths[0].nodes[0].endpoint[0] - 10) ** 2
            for oid in ("fill_a", "fill_b")
        )

    monkeypatch.setattr(joint, "_polish_group", propose)
    result = joint.polish(
        document,
        ("fill_a", "ink_a", "fill_b", "ink_b"),
        render_image(export_svg(document), alpha=True),
        FitOptions(displacement=4),
        score=score,
    )
    assert score(result) == 0
    assert ("fill_a", True, 9) in calls
    assert ("fill_b", True, 10) in calls
