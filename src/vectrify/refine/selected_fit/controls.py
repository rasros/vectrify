"""Fixed-topology control projection and independent edge proposals."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

from vectrify.image_utils import on_white
from vectrify.refine.parameters import ControlMap

if TYPE_CHECKING:
    from threading import Event

    from vectrify.document import Selection
    from vectrify.refine.selected import FitContext, FitOptions


class FitControls:
    """The coordinate map, movement mask and differentiable optimizer hooks."""

    def __init__(
        self,
        context: FitContext,
        selection: Selection,
        options: FitOptions,
        device: str,
        corners: frozenset[str],
    ):
        import torch

        self.context, self.selection, self.options = context, selection, options
        self.device = device
        self.original = torch.tensor(context.local, dtype=torch.float32, device=device)
        self.linear = self.original.new_tensor(context.linear)
        self.offset = self.original.new_tensor(context.offset)
        self.origin = self.original.new_tensor(context.crop[:2])
        self.scale = self.original.new_tensor(context.scale)
        mask = []
        for node in context.nodes:
            selected = not selection.node_ids or node.id in selection.node_ids
            mask.extend([selected and options.handles] * (len(node.values) // 2 - 1))
            mask.append(selected and options.nodes and not node.pinned)
        self.movable = self.original.new_tensor(mask)[:, None]
        self.parameters = ControlMap(
            context.geometry,
            context.local,
            self.original,
            self.linear,
            self.offset,
            self.origin,
            self.scale,
            self.movable,
            options.displacement,
            stroke_only=context.stroke_only,
            corners=corners,
        )
        if options.cleanup:
            self.parameters.guard_smooth_joins(self.controls_from_local(self.original))
        self.work = self._working_svg()

    @property
    def counts(self):
        return self.parameters.counts

    def controls_from_local(self, coordinates):
        return self.parameters.controls_from_local(coordinates)

    def local_from_controls(self, paths):
        return self.parameters.local_from_controls(paths[0])

    def project(self, paths):
        constrained = self.controls_from_local(self.local_from_controls(paths))
        for dest, source in zip(paths[0], constrained, strict=True):
            dest.copy_(source)

    def transform(self, paths):
        return [self.controls_from_local(self.local_from_controls(paths))]

    def coordinate_tensor(self, geometry):
        return self.original.new_tensor(
            [
                point
                for subpath in geometry.subpaths
                for node in subpath.nodes
                for point in zip(node.values[::2], node.values[1::2], strict=True)
            ]
        )

    def path_data(self, coordinates):
        from vectrify.refine.paths import to_path_d

        return " ".join(
            to_path_d(contour.cpu().tolist(), precision=9) + " Z"
            for contour in self.controls_from_local(coordinates)
        )

    def _working_svg(self):
        context = self.context
        work = ET.Element(
            "svg", width=str(context.size[0]), height=str(context.size[1])
        )
        ET.SubElement(
            work,
            "path",
            {
                "d": self.path_data(self.original),
                "fill": context.style["stroke" if context.stroke_only else "fill"],
                "fill-rule": context.style["fill-rule"],
            },
        )

        return work

    def edge_seed(self, stop: Event):
        """A bounded proposal; gradient descent still starts from the original."""
        context, selection, options = self.context, self.selection, self.options
        document = context.document
        seed = self.original
        if options.snap and (options.nodes or options.handles) and not stop.is_set():
            from vectrify.operations.generate import Region
            from vectrify.refine.frozen import Frozen, Paths, frozen
            from vectrify.refine.lines import fit_lines
            from vectrify.refine.snap import snap

            paths = Paths({context.oid: context.geometry})
            held = frozen(paths).endpoints | frozenset(
                node.id
                for node in context.nodes
                if not options.nodes
                or (selection.node_ids and node.id not in selection.node_ids)
            )
            fixed = Frozen(held)
            left, top, right, bottom = context.crop
            region = Region(
                left, top, right - left, bottom - top, on_white(context.target)
            )
            if context.stroke_only:
                prepared = fit_lines(
                    document, [context.oid], region, fixed, False
                ).geometry_for(context.oid)
            else:
                prepared = snap(
                    document,
                    paths,
                    region,
                    fixed,
                    detail=False,
                    deadline=min(
                        getattr(stop, "deadline", float("inf")), time.monotonic() + 0.25
                    ),
                ).geometries[context.oid]
            by_id = {n.id: n for sub in prepared.subpaths for n in sub.nodes}
            rows = []
            for node in context.nodes:
                values = by_id.get(node.id, node).values
                if len(values) != len(node.values):
                    # Fixed-topology fitting never adds handles to a straight node.
                    values = values[-2:] if len(node.values) == 2 else node.values
                rows.extend(zip(values[::2], values[1::2], strict=True))
            delta = (self.original.new_tensor(rows) - self.original) * self.movable
            distance = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            seed = self.original + delta * (options.displacement / distance).clamp(
                max=1
            )
        return seed
