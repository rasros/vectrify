"""Crossing rollback and independent gradient progress/stall observation."""

from __future__ import annotations

from collections.abc import Callable
from threading import Event
from typing import TYPE_CHECKING

import numpy as np
from cairosvg.colors import color

from vectrify.refine.crossings import crossed_nodes, crossings

if TYPE_CHECKING:
    from vectrify.refine.selected import FitContext, FitOptions
    from vectrify.refine.selected_fit.controls import FitControls
    from vectrify.refine.selected_fit.policy import RetainedFit


class CrossingRollback:
    """Repair new folds against the latest unfolded fixed-topology coordinates."""

    def __init__(self, context: FitContext, controls: FitControls, halvings: int):
        self.context = context
        self.halvings = halvings
        self.folds = crossings(context.geometry)
        self.unfolded = context.local
        self.node_index = {node.id: i for i, node in enumerate(context.nodes)}
        self.owners = np.repeat(
            np.arange(len(context.nodes)),
            [len(node.values) // 2 for node in context.nodes],
        )
        self.coupled = controls.parameters.coupled.cpu().numpy()

    def repair(self, coordinates):
        values, geometry = self.context.reshaped(coordinates)
        for attempt in range(self.halvings + 2):
            count, crossed = crossed_nodes(geometry) if values else (0, set())
            if count <= self.folds:
                break
            back = np.isin(self.owners, [self.node_index[node] for node in crossed])
            # A rollback keeps a smooth knot and both arms together too.
            back = np.isin(self.coupled, self.coupled[back])
            if attempt > self.halvings:
                back[:] = True
            share = 0.5 if attempt < self.halvings else 1.0
            coordinates = coordinates.copy()
            coordinates[back] += (self.unfolded[back] - coordinates[back]) * share
            values, geometry = self.context.reshaped(coordinates)
        return coordinates, values, geometry

    def adopt(self, coordinates, geometry, *, cleanup: bool):
        self.unfolded = coordinates
        if cleanup:
            self.folds = min(self.folds, crossings(geometry))


class FitObserver:
    """The optimizer's observation state stays independent of the edge seed."""

    def __init__(
        self,
        controls: FitControls,
        retained: RetainedFit,
        rollback: CrossingRollback,
        options: FitOptions,
        stop: Event,
        report: Callable[[int, str], None],
    ):
        self.controls, self.retained, self.rollback = controls, retained, rollback
        self.options, self.stop, self.report = options, stop, report
        self.checked = retained.best.quality
        self.completed = self.folded = 0

    def consider_seed(self, seed):
        import torch

        controls, context = self.controls, self.controls.context
        if seed is controls.original:
            return
        with torch.no_grad():
            bounded = controls.local_from_controls([controls.controls_from_local(seed)])
        coordinates = context.local + (bounded - controls.original).cpu().numpy()
        _, _, geometry = self.rollback.repair(coordinates)
        paint = color(
            context.style["stroke"] if context.stroke_only else context.style["fill"]
        )
        self.retained.consider(self.retained.evaluate(geometry, paint[:3]))

    def initialize_from_cleanup(self):
        """Start cleanup-enabled descent from retained post-cleanup geometry."""
        controls = self.controls
        coordinates = controls.coordinate_tensor(self.retained.best.geometry)
        controls.work[0].set("d", controls.path_data(coordinates))
        unfolded = coordinates.detach().cpu().numpy()
        _, geometry = controls.context.reshaped(unfolded)
        self.rollback.adopt(unfolded, geometry, cleanup=True)
        controls.parameters.guard_smooth_joins(
            controls.controls_from_local(coordinates)
        )
        # Fairness guards changed, so refresh the same retained candidate.
        self.retained.rescore_quality(1)
        self.checked = self.retained.best.quality

    def __call__(self, step, paths, colors):
        import torch

        controls, context, options = self.controls, self.controls.context, self.options
        self.completed = step
        self.report(step, f"Fitting path · step {step}/{options.steps}")
        if step and (step % 10 == 0 or step == options.steps or self.stop.is_set()):
            with torch.no_grad():
                shifts = (
                    controls.local_from_controls(paths) - controls.original
                ) * controls.movable
                # Avoid float32 round-tripping untouched or pinned coordinates.
                reached = context.local + shifts.cpu().numpy()
            coordinates, _, geometry = self.rollback.repair(reached)
            if coordinates is not reached:
                self.folded += 1
                with torch.no_grad():
                    back = controls.controls_from_local(
                        controls.original.new_tensor(coordinates)
                    )
                    for dest, source in zip(paths[0], back, strict=True):
                        dest.copy_(source)
            self.rollback.adopt(coordinates, geometry, cleanup=options.cleanup)
            candidate = self.retained.evaluate(
                geometry, colors[0].detach().clamp(0, 1).cpu().numpy()
            )
            # An edge proposal cannot stall descent against its own baseline.
            stalled = candidate.quality > self.checked * (1 - options.stall)
            self.checked = candidate.quality
            self.retained.consider(candidate)
            minimum_steps = min(30, options.steps) if options.cleanup else 10
            if step >= minimum_steps and options.stall and stalled:
                return False
        return not self.stop.is_set()
