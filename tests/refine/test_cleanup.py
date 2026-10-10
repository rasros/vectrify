"""Local cleanup removes unsupported dents without losing real features."""

from dataclasses import replace

import numpy as np
import pytest

from vectrify.document import export_svg, import_svg
from vectrify.refine.cleanup import cleaned
from vectrify.refine.crossings import bezier, crossings
from vectrify.refine.edge_profiles import EdgeProfiles
from vectrify.refine.snap import _Frame
from vectrify.svg_render import render_image

FRAME = _Frame(np.eye(2), np.zeros(2))


def drawing(path):
    return import_svg(
        '<svg width="64" height="64"><rect width="64" height="64" '
        f'fill="white"/><path id="p" fill="black" d="{path}"/></svg>'
    )


def score(document, target):
    image = render_image(export_svg(document), (0, 0, 64, 64), (128, 128))
    return float(((np.asarray(image, dtype=float) - target) ** 2).mean())


def test_cleanup_removes_false_dents_but_keeps_the_reference_notch():
    source = drawing(
        "M8 16 L16 16 L18 18 L20 16 L28 16 L32 20 L36 16 "
        "L42 16 L44 18 L46 16 L56 16 L56 56 L8 56 Z"
    )
    target = np.asarray(
        render_image(
            export_svg(drawing("M8 16 L28 16 L32 20 L36 16 L56 16 L56 56 L8 56 Z")),
            (0, 0, 64, 64),
            (128, 128),
        ),
        dtype=float,
    )
    geometry = source.geometry_for("p")
    before = best = score(source, target)

    def accept(candidate):
        nonlocal best
        actual = score(source.replace_geometry(candidate), target)
        if actual >= best or crossings(candidate):
            return False
        best = actual
        return True

    result = cleaned(geometry, geometry, FRAME, 2, frozenset(), accept, lambda: False)
    nodes = result.subpaths[0].nodes
    assert [n.id for n in nodes] == [n.id for n in geometry.subpaths[0].nodes]
    assert nodes[2].endpoint[1] == 16
    assert nodes[8].endpoint[1] == 16
    assert nodes[5].endpoint == (32, 20)
    assert best < before * 0.1


def test_dense_knots_do_not_prevent_bridging_a_whole_false_dent():
    dent = np.vstack(
        (
            np.linspace((16, 16), (20, 20), 41)[1:],
            np.linspace((20, 20), (24, 16), 41)[1:],
        )
    )
    path = "M8 16 L16 16 " + " ".join(f"L{x} {y}" for x, y in dent)
    source = drawing(path + " L36 16 L40 24 L44 16 L56 16 L56 56 L8 56 Z")
    target = np.asarray(
        render_image(
            export_svg(drawing("M8 16 L36 16 L40 24 L44 16 L56 16 L56 56 L8 56 Z")),
            (0, 0, 64, 64),
            (128, 128),
        ),
        dtype=float,
    )
    geometry = source.geometry_for("p")
    best = score(source, target)

    def accept(candidate):
        nonlocal best
        actual = score(source.replace_geometry(candidate), target)
        if actual >= best or crossings(candidate):
            return False
        best = actual
        return True

    result = cleaned(geometry, geometry, FRAME, 4.1, frozenset(), accept, lambda: False)
    old, new = geometry.subpaths[0].nodes, result.subpaths[0].nodes
    assert [n.id for n in new] == [n.id for n in old]
    assert max(n.endpoint[1] for n in new[2:82]) < 16.25
    assert new[83].endpoint == (40, 24)


@pytest.mark.parametrize("protection", ["pin", "feature", "held", "stopped", "cap"])
@pytest.mark.parametrize("guided", [False, True])
def test_cleanup_respects_protected_knots_and_the_movement_limit(protection, guided):
    geometry = drawing("M8 16 L10 18 L12 16").geometry_for("p")
    node = geometry.subpaths[0].nodes[1]
    if protection == "pin":
        geometry = geometry.replace_node(replace(node, pinned=True))
    elif protection == "feature":
        geometry = geometry.replace_node(replace(node, feature="tip"))
    held = frozenset({node.id}) if protection == "held" else frozenset()
    result = cleaned(
        geometry,
        geometry,
        FRAME,
        1 if protection == "cap" else 2,
        held,
        lambda _: True,
        lambda: protection == "stopped",
        guide=(
            lambda points, _: np.column_stack((points[:, 0], np.full(len(points), 16)))
        )
        if guided
        else None,
    )
    assert result == geometry


def test_cleanup_removes_a_cubic_hook_without_deleting_its_controls():
    geometry = drawing("M8 16 C9 18 11 18 12 16").geometry_for("p")
    result = cleaned(
        geometry,
        geometry,
        FRAME,
        2.1,
        frozenset(),
        lambda _: True,
        lambda: False,
    )
    node = result.subpaths[0].nodes[1]
    assert node.id == geometry.subpaths[0].nodes[1].id
    assert node.command == "C"
    assert node.values[1::2] == (16, 16, 16)
    assert result.subpaths[0].nodes[0] == geometry.subpaths[0].nodes[0]


def test_reference_fitting_removes_ripples_from_an_inflected_curve():
    control = np.array([[8, 18], [20, 28], [38, 8], [56, 18]])
    t = np.linspace(0, 1, 25)
    points, tangents = bezier(control, t)
    # The same broad curve with small, unsupported waves in its upper edge.
    points[:, 1] += 1.4 * np.sin(12 * np.pi * t) * np.sin(np.pi * t) ** 2
    tangents[:, 1] += 1.4 * (
        12 * np.pi * np.cos(12 * np.pi * t) * np.sin(np.pi * t) ** 2
        + np.pi * np.sin(12 * np.pi * t) * np.sin(2 * np.pi * t)
    )
    path = "M8 18 " + " ".join(
        "C"
        + " ".join(
            str(v)
            for v in np.r_[
                points[j - 1] + tangents[j - 1] / 72,
                points[j] - tangents[j] / 72,
                points[j],
            ]
        )
        for j in range(1, len(t))
    )
    source = drawing(path + " L56 56 L8 56 Z")
    reference = drawing("M8 18 C20 28 38 8 56 18 L56 56 L8 56 Z")
    target = np.asarray(
        render_image(export_svg(reference), (0, 0, 64, 64), (128, 128)), float
    )
    coverage = 1 - np.asarray(render_image(export_svg(source)), float).mean(-1) / 255
    guide = EdgeProfiles(
        np.ones((64, 64, 3)),
        -np.ones((64, 64, 3)),
        np.asarray(render_image(export_svg(reference)), float) / 255,
        coverage,
        FRAME,
        1,
    )
    geometry = source.geometry_for("p")
    scores = []
    for reader in (None, guide):
        best = score(source, target)

        def accept(candidate):
            nonlocal best
            actual = score(source.replace_geometry(candidate), target)
            if actual >= best or crossings(candidate):
                return False
            best = actual
            return True

        result = cleaned(
            geometry,
            geometry,
            FRAME,
            3,
            frozenset(),
            accept,
            lambda: False,
            guide=reader,
        )
        old, new = geometry.subpaths[0].nodes, result.subpaths[0].nodes
        assert [n.id for n in new] == [n.id for n in old]
        assert [n.command for n in new] == [n.command for n in old]
        assert (
            max(
                np.linalg.norm(
                    np.asarray(a.values).reshape(-1, 2)
                    - np.asarray(b.values).reshape(-1, 2),
                    axis=1,
                ).max()
                for a, b in zip(old, new, strict=True)
            )
            <= 3 + 1e-9
        )
        assert crossings(result) == 0
        scores.append(best)
    assert scores[1] < scores[0] / 10
