"""Portable round-stroke and tiled CUDA coverage for a fixed path topology."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from vectrify.document import DocumentError

if TYPE_CHECKING:
    from torch import Tensor

    from vectrify.refine.selected import FitContext, FitOptions
    from vectrify.refine.selected_fit.controls import FitControls


class StrokeCoverage:
    """Conservative fixed tiles cover every permitted node movement."""

    def __init__(self, context: FitContext, options: FitOptions, controls: FitControls):
        import torch

        self.context, self.controls = context, controls
        self.device = controls.device
        self.tiles: dict[tuple[int, int], list[tuple[int, int, int, int, Tensor]]] = {}
        self.width = 0.0
        if context.style["stroke"] != "none":
            mapping = controls.linear * controls.scale[:, None]
            singular = torch.linalg.svdvals(mapping)
            if abs(float(singular[0] / singular[1]) - 1) > 0.02:
                raise DocumentError("Outlined path fitting requires a uniform scale")
            self.width = float(context.style["stroke-width"]) * float(singular.mean())
            miter = context.style["stroke-linejoin"] == "miter"
            limit = float(context.style["stroke-miterlimit"]) if miter else 1
            reach = (
                self.width / 2 * limit + options.displacement * float(singular[0]) + 2
            )
            # Fixed conservative tiles include all permitted movement. Most contours
            # in a landscape are tiny; never evaluate each one over the whole crop.
            for contour_index, contour in (
                enumerate(controls.controls_from_local(controls.original))
                if self.device == "cuda"
                else []
            ):
                contour = contour[: controls.counts[contour_index]]
                for start in range(0, len(contour), 16):
                    chunk = contour[start : start + 16].cpu().numpy().reshape(-1, 2)
                    low = np.floor(chunk.min(0) - reach).astype(int)
                    high = np.ceil(chunk.max(0) + reach).astype(int)
                    left, top = np.maximum(low, (0, 0))
                    right, bottom = np.minimum(high, context.size)
                    if right <= left or bottom <= top:
                        continue
                    width = min(context.size[0], math.ceil((right - left) / 32) * 32)
                    height = min(context.size[1], math.ceil((bottom - top) / 32) * 32)
                    left = min(left, context.size[0] - width)
                    top = min(top, context.size[1] - height)
                    rows = torch.arange(height, device=self.device)[:, None] + int(top)
                    cols = torch.arange(width, device=self.device)[None, :] + int(left)
                    pixels = (rows * context.size[0] + cols).reshape(-1)
                    self.tiles.setdefault((width, height), []).append(
                        (contour_index, start, int(left), int(top), pixels)
                    )

    def __call__(self, paths, alphas):
        import torch

        context, controls = self.context, self.controls
        if self.device == "cpu":
            from vectrify.refine.soft_coverage import soft_stroke_coverage

            contours = [c[:n] for c, n in zip(paths[0], controls.counts, strict=True)]
            stroke = soft_stroke_coverage(contours, (0, 0, *context.size), self.width)
            return (
                stroke[None]
                if context.stroke_only
                else (1 - (1 - alphas[0]) * (1 - stroke))[None]
            )
        from vectrify.refine.cuda_renderer import stroke_coverage
        from vectrify.refine.miter import incoming_directions, miter_stroke_coverage

        miter = context.style["stroke-linejoin"] == "miter"
        incoming = [incoming_directions(c) for c in paths[0]] if miter else []
        samples, indices = [], []
        for (width, height), tiles in self.tiles.items():
            for start in range(0, len(tiles), 8):
                packed, tangents = [], []
                for contour_index, offset, left, top, pixels in tiles[
                    start : start + 8
                ]:
                    chunk = paths[0][contour_index][offset : offset + 16]
                    if miter:
                        direction = incoming[contour_index][offset : offset + 16]
                        tangents.append(
                            torch.cat(
                                (direction, direction.new_zeros((16 - len(chunk), 2)))
                            )
                        )
                    if len(chunk) < 16:
                        chunk = torch.cat(
                            (chunk, chunk[-1:, 3:4].expand(16 - len(chunk), 4, 2))
                        )
                    packed.append(chunk - chunk.new_tensor((left, top)))
                    indices.append(pixels)
                batch = torch.stack(packed)
                widths = batch.new_full((len(batch),), self.width)
                stroke = (
                    miter_stroke_coverage(
                        batch,
                        torch.stack(tangents),
                        widths,
                        (0, 0, width, height),
                        miter_limit=float(context.style["stroke-miterlimit"]),
                    )
                    if miter
                    else stroke_coverage(batch, widths, (0, 0, width, height))
                )
                if stroke is None:
                    raise DocumentError("CUDA stroke coverage is unavailable")
                samples.append(stroke.reshape(-1))
        if not samples:
            return alphas
        stroke = (
            alphas.new_zeros(context.size[0] * context.size[1])
            .scatter_reduce(0, torch.cat(indices), torch.cat(samples), reduce="amax")
            .reshape(context.size[1], context.size[0])
        )
        return (
            stroke[None]
            if context.stroke_only
            else (1 - (1 - alphas[0]) * (1 - stroke))[None]
        )
