"""A bounded bilateral curve family proposed from an open cubic contour.

Reversal pairs its two halves about the tip and midpoint of the ends. This
is an optional fitting proposal, never a drawing constraint: exact reference
scoring decides whether the family fits better than the unrestricted curve.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np


class Bilateral:
    def __init__(self, original, displacement):
        self.original = original
        self.displacement = displacement
        base = (original[0] + original[-1]) / 2
        direction = original[len(original) // 2] - base
        self.direction = direction / np.linalg.norm(direction)
        self.seed = self.average(original)

    @classmethod
    def infer(cls, geometry, held, displacement):
        if len(geometry.subpaths) != 1 or displacement <= 0:
            return None
        subpath = geometry.subpaths[0]
        nodes = subpath.nodes
        if (
            subpath.closed
            or len(nodes) < 3
            or (len(nodes) - 1) % 2
            or any(n.command != "C" for n in nodes[1:])
            or any(n.pinned or n.id in held for n in nodes)
        ):
            return None
        points = np.array(
            [p for n in nodes for p in zip(n.values[::2], n.values[1::2], strict=True)]
        )
        if not np.isfinite(points).all():
            return None
        base = (points[0] + points[-1]) / 2
        if np.linalg.norm(points[len(points) // 2] - base) <= 1e-12:
            return None
        model = cls(points, displacement)
        return model if model.within(model.seed) else None

    def average(self, points):
        base = (points[0] + points[-1]) / 2
        vector = points[len(points) // 2] - base
        length = np.linalg.norm(vector)
        direction = vector / length if length > 1e-12 else self.direction
        reverse = points[::-1] - base
        reflected = 2 * (reverse @ direction)[:, None] * direction - reverse + base
        return (points + reflected) / 2

    def controls(self, points):
        """Differentiable coupling, including a movable center and axis."""
        import torch

        base = (points[0] + points[-1]) / 2
        vector = points[len(points) // 2] - base
        length = vector.norm().clamp_min(1e-12)
        direction = torch.where(
            length > 1e-12, vector / length, points.new_tensor(self.direction)
        )
        reverse = points.flip(0) - base
        reflected = (
            2 * (reverse * direction).sum(-1, keepdim=True) * direction - reverse + base
        )
        return (points + reflected) / 2

    def within(self, points):
        return bool(
            np.isfinite(points).all()
            and np.linalg.norm(points - self.original, axis=1).max()
            <= self.displacement + 1e-9
        )

    def geometry(self, geometry):
        """Materialize paired doubles without exceeding any control's bound."""
        points = np.array(
            [
                p
                for n in geometry.subpaths[0].nodes
                for p in zip(n.values[::2], n.values[1::2], strict=True)
            ]
        )
        for _ in range(12):
            average = self.average(points)
            if self.within(average):
                break
            # Independent point bounds need not preserve a bilateral family.
            # Backtrack toward its feasible seed, then couple again.
            points = (points + self.seed) / 2
        else:
            average = self.seed
        nodes, index = [], 0
        for node in geometry.subpaths[0].nodes:
            count = len(node.values) // 2
            values = tuple(float(v) for v in average[index : index + count].reshape(-1))
            nodes.append(replace(node, values=values))
            index += count
        return replace(
            geometry,
            subpaths=(replace(geometry.subpaths[0], nodes=tuple(nodes)),),
        )
