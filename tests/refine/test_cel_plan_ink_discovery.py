"""Reuse complete physical discovery without reusing a fitted interpretation."""

from dataclasses import replace

import numpy as np
import pathops
import pytest

from tests.refine.test_cel_plan_ink_models import drawing
from vectrify.document.join import curve_path
from vectrify.document.svg import parse_path
from vectrify.refine.cel_plan import ink_models
from vectrify.refine.cel_plan.ink_models import InkDiscovery, models
from vectrify.refine.cel_plan.model import Options, Work


def same(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        assert a.geometry == b.geometry
        assert a.footprint.path_data() == b.footprint.path_data()
        assert a.details == b.details
        np.testing.assert_array_equal(a.paint, b.paint)
        np.testing.assert_array_equal(a.selected, b.selected)


def same_profiles(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        assert a.component == b.component
        for field in ("points", "sides", "direction", "tolerance"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
            assert not getattr(a, field).flags.writeable


@pytest.mark.parametrize("gap", [False, True])
@pytest.mark.parametrize("alpha", [255, 128, 64])
def test_original_fitted_original_share_raw_discovery_and_exact_standalone_results(
    gap, alpha
):
    evidence, mask = drawing(alpha, gap=gap)
    options = Options()
    cache = InkDiscovery()
    banks = []
    for fit in (False, True, False):
        independent, bank = [], []
        shared = models(
            mask,
            evidence,
            options,
            Work.start(10),
            source_absence=True,
            source_intervals=True,
            fit_widths=fit,
            fractional_coverage=True,
            prune_spurs=True,
            source_profiles=bank,
            discovery=cache,
        )
        standalone = models(
            mask,
            evidence,
            options,
            Work.start(10),
            source_absence=True,
            source_intervals=True,
            fit_widths=fit,
            fractional_coverage=True,
            prune_spurs=True,
            source_profiles=independent,
        )
        same(shared, standalone)
        same_profiles(bank, independent)
        banks.append(bank)
    same_profiles(banks[0], banks[1])
    assert cache.diagnostics == {"extractions": 1, "reuses": 2}
    cache.clear()
    assert cache._value is None
    assert cache._evidence is None


@pytest.mark.parametrize(
    "change",
    [
        "mask",
        "evidence",
        "options",
        "carrier",
        "fill-rule",
        "cap-flags",
        "coverage",
        "profiles",
        "intervals",
    ],
)
def test_changed_extraction_context_cannot_reuse_stale_discovery(change):
    evidence, mask = drawing(gap=True)
    options, cache = Options(), InkDiscovery()
    carrier = curve_path(parse_path("M8 8H88V88H8Z"))
    kwargs = {
        "carrier": carrier,
        "source_absence": True,
        "source_intervals": True,
        "fractional_coverage": True,
        "boundary_contacts": True,
        "fit_carrier": True,
    }
    models(mask, evidence, options, Work.start(10), discovery=cache, **kwargs)
    if change == "mask":
        mask = mask.copy()
        mask[72, 78] = True
    elif change == "evidence":
        evidence = replace(evidence, offset=(1, 0))
    elif change == "options":
        options = replace(options, line_width=1)
    elif change == "carrier":
        kwargs["carrier"] = curve_path(parse_path("M7 7H89V89H7Z"))
    elif change == "fill-rule":
        kwargs["carrier"] = pathops.Path(carrier)
        kwargs["carrier"].fillType = pathops.FillType.EVEN_ODD
    elif change == "cap-flags":
        kwargs["boundary_contacts"] = False
    elif change == "coverage":
        kwargs["fractional_coverage"] = False
    elif change == "profiles":
        kwargs["source_absence"] = kwargs["source_intervals"] = False
    elif change == "intervals":
        kwargs["source_intervals"] = False
    shared = models(mask, evidence, options, Work.start(10), discovery=cache, **kwargs)
    standalone = models(mask, evidence, options, Work.start(10), **kwargs)
    same(shared, standalone)
    assert cache.diagnostics == {"extractions": 2, "reuses": 0}


def test_failed_physical_extraction_is_never_reused_or_published(monkeypatch):
    evidence, mask = drawing()
    cache = InkDiscovery()
    original = ink_models.measure
    work = Work.start(10)

    def stopped(*args, **kwargs):
        work.stop.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(ink_models, "measure", stopped)
    bank = []
    assert (
        models(mask, evidence, Options(), work, discovery=cache, source_profiles=bank)
        == ()
    )
    assert bank == []
    assert cache._value is None
    monkeypatch.setattr(ink_models, "measure", original)
    assert models(mask, evidence, Options(), Work.start(10), discovery=cache)
    assert cache.diagnostics == {"extractions": 2, "reuses": 0}


def test_cancelled_cache_hit_cannot_publish_models_or_profiles():
    evidence, mask = drawing(gap=True)
    cache = InkDiscovery()
    assert models(mask, evidence, Options(), Work.start(10), discovery=cache)
    work = Work.start(10)
    work.stop.set()
    bank = []
    assert (
        models(mask, evidence, Options(), work, discovery=cache, source_profiles=bank)
        == ()
    )
    assert bank == []
    assert cache.diagnostics["reuses"] == 0
