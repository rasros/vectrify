"""Soft material boundaries supply finite source votes independent of alpha edges."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from vectrify.refine.cel_plan.facet_lines import (
    MAX_LINES,
    MAX_POINTS,
    SCALES,
    FacetLines,
)
from vectrify.refine.cel_plan.model import StageInterruptedError, Work
from vectrify.refine.cel_plan.surface_splits import lines


def test_soft_low_contrast_material_edge_is_discovered_with_unquantized_direction():
    y, x = np.mgrid[:96, :160]
    normal = np.array((1.0, 0.045))
    normal /= np.linalg.norm(normal)
    rho = 80.0
    side = (x + 0.5) * normal[0] + (y + 0.5) * normal[1] < rho
    gray = gaussian_filter(np.where(side, 180.0, 200.0), 2)
    target = np.repeat(gray[..., None], 3, axis=2)
    own = np.ones(side.shape, bool)
    assert list(lines(target, own, (0, 0), Work.start(10))) == []
    voter = FacetLines(target, own)
    offered = list(voter(own, Work.start(10)))
    assert offered
    assert len(offered) <= MAX_LINES
    assert voter.diagnostics["points"] <= MAX_POINTS * len(SCALES)
    matching = [(n, r) for n, r in offered if abs(n @ normal) > 0.9999]
    assert matching
    assert (
        min(abs(r - rho) if n @ normal > 0 else abs(r + rho) for n, r in matching) < 0.5
    )
    # A current cell cannot claim observations from the other side of the image.
    assert list(voter(own & (x > 120), Work.start(10))) == []


def test_empty_background_and_alpha_hole_do_not_supply_color_edges():
    visible = np.zeros((64, 80), bool)
    visible[8:56, 8:72] = True
    visible[24:40, 30:46] = False
    target = np.zeros((*visible.shape, 3))
    target[visible] = (180, 120, 60)
    voter = FacetLines(target, visible)
    assert list(voter(visible, Work.start(10))) == []
    assert voter.diagnostics["points"] == 0


def test_flat_field_has_no_votes_or_nonfinite_pca():
    visible = np.ones((40, 50), bool)
    target = np.full((*visible.shape, 3), 160.0)
    voter = FacetLines(target, visible)
    assert list(voter(visible, Work.start(10))) == []
    assert np.isfinite(voter.points).all()
    assert np.isfinite(voter.normals).all()


def test_cancelled_preparation_publishes_no_partial_observation_cache():
    visible = np.ones((40, 50), bool)
    target = np.full((*visible.shape, 3), 160.0)
    voter = FacetLines(target, visible)
    work = Work.start(10)
    work.stop.set()
    with pytest.raises(StageInterruptedError):
        list(voter(visible, work))
    assert voter.points is None
    assert voter.normals is None
    assert list(voter(visible, Work.start(10))) == []


def test_source_pixel_exhaustion_yields_no_unbounded_votes(monkeypatch):
    from vectrify.refine.cel_plan import facet_lines

    visible = np.ones((40, 50), bool)
    target = np.full((*visible.shape, 3), 160.0)
    monkeypatch.setattr(facet_lines, "MAX_PIXELS", 16)
    voter = FacetLines(target, visible)
    assert list(voter(visible, Work.start(10))) == []
    assert voter.diagnostics["bounded"] == 1


def test_short_edge_patch_cannot_supply_a_cut_through_a_large_surface():
    y, x = np.mgrid[:64, :160]
    target = np.full((64, 160, 3), 180.0)
    target[(x >= 80) & (y >= 24) & (y < 40)] = 200
    own = np.ones((64, 160), bool)
    voter = FacetLines(target, own)
    offered = list(voter(own, Work.start(10)))

    def vertical(n, r):
        return abs(n[0]) > 0.99 and abs(abs(r) - 80) < 1

    assert any(vertical(n, r) for n, r in offered)
    scope = np.column_stack((x.ravel() + 0.5, y.ravel() + 0.5))
    restricted = list(voter(own, Work.start(10), scope=scope))
    assert not any(vertical(n, r) for n, r in restricted)
