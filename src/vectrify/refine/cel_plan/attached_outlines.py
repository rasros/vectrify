"""Atomic complete attached strokes for the explicit experimental planner.

Discovery and fitting use the actual parent and frozen original source bank.
These are proposals: common search and native checkpoints decide acceptance.
"""

from itertools import islice

import numpy as np

from vectrify.refine.cel_plan.attached_spans import AttachedSpans
from vectrify.refine.cel_plan.component_edits import ComponentEdit
from vectrify.refine.cel_plan.constraints import discard
from vectrify.refine.cel_plan.filled_bands import _check
from vectrify.refine.cel_plan.score import representation
from vectrify.refine.cel_plan.search import Proposal
from vectrify.refine.cel_plan.source_family import MAX_FAMILIES, SourceFamily
from vectrify.refine.cel_plan.span_body_fit import SpanBodyFit

MAX_ATTEMPTS = 4


class AttachedOutlines:
    def __init__(self, evidence, options, *, guard):
        self.evidence, self.options, self.guard = evidence, options, guard
        self.diagnostics = dict.fromkeys(
            ("attempts", "fit_exclusions", "seal_exclusions", "proposals"),
            0,
        )

    def __call__(self, state, work):
        _check(work)
        partition = state.partition
        # This fitter infers width. A user-fixed width is never silently changed.
        if (
            partition is None
            or len(partition.families) >= MAX_FAMILIES
            or self.options.line_width > 0
        ):
            return
        guard = self.guard(work)
        discovery = AttachedSpans(self.evidence, guard)
        fitter = SpanBodyFit(self.evidence, guard)
        before = state.document
        old_cost = representation(before).cost
        for seed in islice(discovery(before, partition, work), MAX_ATTEMPTS):
            _check(work)
            self.diagnostics["attempts"] += 1
            paint = "#" + "".join(
                f"{int(v):02x}" for v in np.rint(np.clip(seed.ink.paint, 0, 255))
            )
            fitted = fitter.fit(
                before,
                seed.floor,
                seed.owner,
                seed.centerline,
                seed.ink.width,
                paint,
                seed.lines,
                seed.profile,
                seed.bar,
                work,
            )
            if fitted is None:
                self.diagnostics["fit_exclusions"] += 1
                continue
            document, proof = fitted
            try:
                family = SourceFamily.bind(
                    before,
                    document,
                    seed.owner,
                    seed.span.field,
                    seed.floor.parts,
                    self.evidence.source_size,
                    work,
                    junction=seed.bar,
                )
                planned = partition.with_family(family)
                ids = tuple(sorted(family.dependencies))
                parent = before.ancestry(seed.owner)[-2].id
                component = ComponentEdit.bind(before, partition, parent, work)
                # Local import: the scheduler imports this optional generator.
                from vectrify.refine.cel_plan.proposals import bounds

                box = bounds(before, document, ids)
                component.validate(before, document, partition, planned, ids, box, work)
            except ValueError:
                self.diagnostics["seal_exclusions"] += 1
                continue
            self.diagnostics["proposals"] += 1
            yield Proposal(
                "attached-outline",
                ids,
                (seed.owner, seed.span.field.path_data(), seed.bar, proof["width"]),
                state.key,
                document,
                box,
                estimate=representation(document).cost - old_cost,
                details={
                    "paint_constraints": sorted(
                        set(state.details.get("paint_constraints", ()))
                        | family.dependencies
                    ),
                    "geometry_constraints": sorted(
                        set(state.details.get("geometry_constraints", ()))
                        | family.dependencies
                    ),
                    "chain_constraints": discard(
                        state.details.get("chain_constraints"), family.dependencies
                    ),
                    "attached_outline": {
                        "discovery": seed.proof,
                        "body": proof,
                        "family": family.metadata(),
                        "scope": "complete-attached-proposal-before-common-acceptance",
                        "accepted": False,
                    },
                },
                dependencies=(parent,),
                partition=planned,
                component=component,
            )
