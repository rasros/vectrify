"""Ink proofs distinguish a paired dark ridge from shading and blank gaps."""

import numpy as np

from vectrify.refine.cel_plan.ink import measure


def line():
    return np.column_stack((np.arange(10.5, 90), np.full(80, 40.5)))


def test_color_boundary_with_a_dark_ridge_proposes_a_continuous_local_stroke():
    target = np.full((80, 100, 3), (220, 180, 100), dtype=float)
    target[40:] = (160, 100, 70)
    target[39:42, 10:90] = (15, 8, 5)
    proof = measure(line(), target, 2)
    assert proof is not None
    assert proof.support > 0.95
    assert 2 <= proof.width <= 4
    assert np.max(proof.paint) < 30


def test_a_shade_discontinuity_is_not_classified_as_ink():
    target = np.full((80, 100, 3), (220, 180, 100), dtype=float)
    target[40:] = (80, 60, 40)
    assert measure(line(), target, 2) is None


def test_a_large_blank_gap_cannot_be_explained_by_nearby_ink():
    target = np.full((80, 100, 3), 220.0)
    target[39:42, 10:35] = 0
    target[39:42, 65:90] = 0
    assert measure(line(), target, 2) is None


def test_an_antialias_sized_gap_has_evidence_on_both_sides():
    target = np.full((80, 100, 3), 220.0)
    target[39:42, 10:90] = 0
    target[39:42, 47:49] = 220
    proof = measure(line(), target, 2)
    assert proof is not None
    assert 0 < proof.peak_gap <= 4
