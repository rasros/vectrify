"""Competing complete source ink/material interpretations in bounded search."""

from __future__ import annotations

from dataclasses import replace

import pathops

from vectrify.refine.cel_plan.core_cells import CoreCells
from vectrify.refine.cel_plan.ink_models import InkDiscovery

MAX_PROPOSALS = 64


class JointCells:
    """Preserve the original interpretation while fitting complete bodies.

    Both cursors read the same source and carrier before material budgeting.
    Native search validates proposals independently; this scheduler cannot
    accept an edit or use any redraw geometry. Discovery is retained only for
    this invocation and is released even on cancellation or a failed boolean.
    """

    def __init__(self, families, options, *, minimum_paths=32):
        self.families, self.options = families, options
        self.minimum_paths = minimum_paths
        self.diagnostics = {
            "calls": 0,
            "proposals": 0,
            "bounded": 0,
            "boolean_failures": 0,
            "extractions": 0,
            "reuses": 0,
        }

    def __call__(self, state, work):
        if state.partition is None or work.interrupted:
            return
        self.diagnostics["calls"] += 1
        discovery = InkDiscovery()
        modes = ("source-intervals", "source-widths")
        factories = [
            CoreCells(
                self.families,
                self.options,
                minimum_paths=self.minimum_paths,
                joint=True,
                grouping="ward",
                boundary_fit="anchored",
                ink_support="connected",
                ink_roles="fitted",
                ink_coverage="fractional",
                ink_fit=mode,
                facet_fit="regional",
                atom_layout="residual",
                ink_discovery=discovery,
            )
            for mode in modes
        ]
        cursors = [iter(factory(state, work)) for factory in factories]
        alive = set(range(len(cursors)))
        count = 0
        try:
            while alive and not work.interrupted:
                for index in sorted(alive):
                    if work.interrupted:
                        return
                    if count >= MAX_PROPOSALS:
                        self.diagnostics["bounded"] += 1
                        return
                    try:
                        proposal = next(cursors[index], None)
                    except pathops.PathOpsError:
                        self.diagnostics["boolean_failures"] += 1
                        alive.remove(index)
                        continue
                    if proposal is None:
                        alive.remove(index)
                        continue
                    if work.interrupted:
                        return
                    count += 1
                    self.diagnostics["proposals"] += 1
                    yield replace(
                        proposal,
                        parameters=(*proposal.parameters, ("ink-fit", modes[index])),
                        details={
                            **(proposal.details or {}),
                            "joint_cell_search": {"ink_fit": modes[index]},
                        },
                    )
        finally:
            for cursor in cursors:
                cursor.close()
            self.diagnostics["extractions"] += discovery.diagnostics["extractions"]
            self.diagnostics["reuses"] += discovery.diagnostics["reuses"]
            discovery.clear()
