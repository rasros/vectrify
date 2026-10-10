"""Extra fit knots preserve the original curve, topology and editing constraints."""

from dataclasses import replace

import numpy as np

from vectrify.document import import_svg
from vectrify.refine.crossings import bezier, crossings
from vectrify.refine.detail import densified
from vectrify.refine.snap import _Frame


def geometry(d):
    return import_svg(f'<svg><path id="p" d="{d}"/></svg>').geometry_for("p")


def test_subdivision_keeps_all_endpoints_and_follows_the_exact_cubic():
    before = geometry("M0 0 C10 30 30 -10 40 10 L40 40 L0 40 Z")
    after = densified(before, _Frame(np.eye(2), np.zeros(2)))
    old = {n.id: n.endpoint for s in before.subpaths for n in s.nodes}
    new = {n.id: n.endpoint for s in after.subpaths for n in s.nodes}
    assert old.items() <= new.items()
    assert len(new) > len(old)
    assert after.id == before.id
    assert after.subpaths[0].id == before.subpaths[0].id
    nodes = after.subpaths[0].nodes
    end = next(i for i, n in enumerate(nodes) if n.id == before.subpaths[0].nodes[1].id)
    controls = np.array([[0, 0], [10, 30], [30, -10], [40, 10]])
    # Recursive halves end at dyadic parameters; all inserted knots are exactly
    # on the original curve rather than an independently fitted approximation.
    samples = bezier(controls, np.linspace(0, 1, 4097))[0]
    for node in nodes[1 : end + 1]:
        assert np.linalg.norm(samples - node.endpoint, axis=1).min() < 1e-10


def test_density_is_in_reference_pixels_and_balances_long_spans():
    before = geometry("M0 0 L16 0 L16 16 L0 16 Z")
    frame = _Frame(np.eye(2), np.zeros(2))
    after = densified(before, frame)
    moved = replace(
        before,
        subpaths=tuple(
            replace(
                s,
                nodes=tuple(
                    replace(n, values=tuple(v * 100 for v in n.values)) for n in s.nodes
                ),
            )
            for s in before.subpaths
        ),
    )
    scaled = densified(moved, _Frame(np.eye(2) / 100, np.array([50, -10])))
    assert len(after.subpaths[0].nodes) == len(scaled.subpaths[0].nodes)
    assert (
        max(
            np.linalg.norm(
                np.diff([n.endpoint for n in after.subpaths[0].nodes], axis=0), axis=1
            )
        )
        <= 4
    )


def test_held_and_pinned_spans_stay_untouched():
    before = geometry("M0 0 L16 0 L16 16 L0 16 Z")
    nodes = before.subpaths[0].nodes
    pinned = replace(nodes[2], pinned=True)
    before = replace(
        before,
        subpaths=(replace(before.subpaths[0], nodes=(*nodes[:2], pinned, nodes[3])),),
    )
    after = densified(before, _Frame(np.eye(2), np.zeros(2)), frozenset({nodes[0].id}))
    assert after == before


def test_large_inputs_are_not_truncated_or_grown_past_the_ceiling():
    before = geometry("M0 0 " + " ".join(f"L{i} 10" for i in range(1, 514)))
    assert densified(before, _Frame(np.eye(2) * 100, np.zeros(2))) == before


def test_subdivision_retains_a_thin_connection_and_a_separate_hole():
    before = geometry(
        "M0 0 L20 0 L20 9 L30 9 L30 0 L50 0 L50 20 L30 20 "
        "L30 10 L20 10 L20 20 L0 20 Z M4 4 L4 7 L7 7 L7 4 Z"
    )
    after = densified(before, _Frame(np.eye(2), np.zeros(2)))
    assert len(after.subpaths) == 2
    assert all(s.closed for s in after.subpaths)
    assert crossings(after) == 0
    # The two sides of the one-unit connection keep their original endpoints.
    original = {n.id: n.endpoint for s in before.subpaths for n in s.nodes}
    assert (
        original.items()
        <= {n.id: n.endpoint for s in after.subpaths for n in s.nodes}.items()
    )
