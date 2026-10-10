"""Local cleanup removes unsupported dents without losing real features."""

from dataclasses import replace

import numpy as np
import pytest

from vectrify.document import export_svg, import_svg
from vectrify.refine.cleanup import cleaned
from vectrify.refine.crossings import crossings
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
def test_cleanup_respects_protected_knots_and_the_movement_limit(protection):
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
