"""Source residual encoding changes provenance allocation, not drawing semantics."""

from dataclasses import replace

import numpy as np
import pytest

from tests.helpers import required
from tests.refine.test_cel_plan_families import prepared
from tests.refine.test_cel_plan_piecewise_surfaces import step
from tests.refine.test_cel_plan_regional_facets import factory
from vectrify.document import export_svg, load_project, save_project
from vectrify.refine.cel_plan.atoms import ResidualCut
from vectrify.refine.cel_plan.core_cells import CoreCells
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import render


@pytest.mark.parametrize("alpha", [255, 128])
@pytest.mark.parametrize("hole", [False, True])
def test_native_paint_holes_ownership_and_reload_match_retired_encoding(alpha, hole):
    evidence = step(alpha, hole=hole)
    evidence = replace(evidence, opacity=evidence.rgba[..., 3])
    frontier, state, options = prepared(evidence, layers=True)
    before = list(
        factory(evidence, options, facet_fit="regional")(state, Work.start(20))
    )
    after = list(
        factory(evidence, options, facet_fit="regional", atom_layout="residual")(
            state, Work.start(20)
        )
    )
    assert len(before) == len(after)
    assert any(required(required(p.partition).atoms).cuts for p in after)
    ops = Operators(evidence, build(evidence), options)
    for old, new in zip(before, after, strict=True):
        assert old.parameters == new.parameters
        assert new.details is not None
        assert new.details["core_material_cells"]["atom_layout"] == "residual"
        assert all(
            isinstance(c, ResidualCut)
            for c in required(required(new.partition).atoms).cuts
        )
        assert new.partition is not None
        assert old.partition is not None
        assert new.partition.atoms is not None
        assert old.partition.atoms is not None
        assert (
            new.partition.atoms.namespace_count
            == old.partition.atoms.namespace_count - len(old.partition.atoms.cuts)
        )
        ops.validate_partition(new.partition, Work.start(10))
        assert state.partition is not None
        assert new.partition.follows(state.partition)
        assert new.component is not None
        assert new.component.validate(
            state.document,
            new.document,
            state.partition,
            new.partition,
            new.ids,
            new.bounds,
            Work.start(10),
        )
        svg = export_svg(new.document)
        actual = render(svg, evidence.source_size)
        np.testing.assert_array_equal(
            actual, render(export_svg(old.document), evidence.source_size)
        )
        full = frontier.policy.evaluate(svg)
        assert full.valid, full.rejections
        local = LocalPolicy(frontier.policy).update(
            state.snapshot, svg, new.bounds, full.structure
        )
        assert local.canvas.matches(actual)
        assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
        document, _ = load_project(save_project(new.document))
        required(Partition.from_metadata(new.partition.metadata())).validate(document)
        np.testing.assert_array_equal(
            render(export_svg(document), evidence.source_size), actual
        )


@pytest.mark.parametrize(
    "kwargs", [{"atom_layout": "unknown"}, {"joint": False}, {"ink_fit": "source"}]
)
def test_residual_atom_option_rejects_inapplicable_configuration(kwargs):
    evidence = step(128)
    _, _, options = prepared(evidence, layers=True)
    with pytest.raises(ValueError, match=r"joint|Residual"):
        CoreCells(
            Families(evidence, build(evidence), options),
            options,
            **(
                {
                    "joint": True,
                    "ink_fit": "carrier",
                    "ink_support": "connected",
                    "grouping": "ward",
                    "atom_layout": "residual",
                }
                | kwargs
            ),
        )
