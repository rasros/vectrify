"""Exact image error, reference-unit fairness and cleanup edge evidence."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from cairosvg.colors import color
from PIL import Image

from vectrify.refine.enclosure import protected_pixels
from vectrify.refine.frame import Frame
from vectrify.svg_render import render_image

if TYPE_CHECKING:
    from torch import Tensor

    from vectrify.refine.selected import FitContext, FitOptions
    from vectrify.refine.selected_fit.controls import FitControls


class ExactObjective:
    """Exact acceptance scoring and differentiable regularization share units."""

    def __init__(
        self,
        context: FitContext,
        options: FitOptions,
        controls: FitControls,
        target: Image.Image,
    ):
        import torch

        self.context, self.options, self.controls = context, options, controls
        document = context.document
        self.protected = protected_pixels(
            document,
            context.oid,
            context.geometry,
            Frame(
                context.linear * context.scale[:, None],
                (context.offset - context.crop[:2]) * context.scale,
            ),
            context.size,
        )
        self.previous_pixels: Tensor | None = None
        self.loss_transform: (
            Callable[[Tensor, Tensor, int], tuple[Tensor, Tensor]] | None
        ) = None
        if self.protected is not None:
            self.held_pixels = controls.original.new_tensor(self.protected)[
                ..., None
            ].bool()
            self.weights = torch.where(self.held_pixels, 3.0, 1.0)
            self.loss_transform = self.transform_loss
        _, _, vw, vh = document.artboard()
        self.fairness_scale = (
            target.width / vw / context.scale[0],
            target.height / vh / context.scale[1],
        )
        paint = np.asarray(
            color(
                context.style["fill"]
                if not context.stroke_only
                else context.style["stroke"]
            )[:3]
        )
        contrast = float(
            np.square(
                context.delta[..., :3] + context.transmission[..., :3] * paint
            ).mean()
        )
        reference_area = (
            context.size[0] * context.size[1] * np.prod(self.fairness_scale)
        )
        perimeter = float(
            (
                (
                    controls.parameters.initial_controls[:, 3]
                    - controls.parameters.initial_controls[:, 0]
                )
                * controls.original.new_tensor(self.fairness_scale)
            )
            .norm(dim=-1)
            .sum()
        )
        # Scale a turn's cost like a few reference pixels of edge mismatch,
        # independent of the crop resolution and the fill's contrast.
        self.fairness_weight = 4 * contrast * perimeter / reference_area
        self.guide = None
        if (
            options.cleanup
            and options.nodes
            and options.handles
            and not context.stroke_only
        ):
            from vectrify.refine.edge_profiles import EdgeProfiles

            response = context.delta.copy()
            response[..., :3] += context.transmission[..., :3] * paint
            base, guide_target = (
                context.base.copy(),
                context.array(context.target).copy(),
            )
            if context.alpha:
                for field in (response, base, guide_target):
                    field[..., 3] *= np.sqrt(3)
            self.guide = EdgeProfiles(
                base,
                response,
                guide_target,
                self.outline_coverage(context.geometry),
                Frame(
                    context.linear * context.scale[:, None],
                    (context.offset - context.crop[:2]) * context.scale,
                ),
                float(np.mean(self.fairness_scale)),
            )

    def transform_loss(self, rendered, goal, _step):
        import torch

        if self.previous_pixels is None:
            # Compare protected pixels against the renderer's own starting
            # antialiasing, so stationary edges have no approximation bias.
            self.previous_pixels = rendered.detach().clone()
        return rendered * self.weights, torch.where(
            self.held_pixels, self.previous_pixels, goal
        ) * self.weights

    def score(self, image):
        context = self.context
        pixels = context.array(image)
        error = (pixels - context.array(context.target)) ** 2
        if self.protected is not None:
            error += (
                8
                * self.protected[..., None]
                * (pixels - context.array(context.before_image)) ** 2
            )
        if context.alpha:
            # Mean RGB error and opacity error each contribute half the score.
            return float(np.mean((error[:, :, :3].sum(-1) + 3 * error[:, :, 3]) / 6))
        return float(np.mean(error))

    def outline_coverage(self, geometry, padding=0):
        context = self.context
        coordinates = self.controls.coordinate_tensor(geometry)
        silhouette = ET.fromstring(ET.tostring(self.controls.work))
        silhouette[0].set("fill", "black")
        if padding:
            silhouette.set("width", str(context.size[0] + 2 * padding))
            silhouette.set("height", str(context.size[1] + 2 * padding))
            silhouette[0].set("transform", f"translate({padding} {padding})")
        silhouette[0].set("d", self.controls.path_data(coordinates))
        return (
            1
            - np.asarray(render_image(ET.tostring(silhouette)), dtype=float).mean(-1)
            / 255
        )

    def quality(self, geometry, actual, smoothing=1):
        import torch

        if not self.options.cleanup or self.context.stroke_only:
            return actual
        coordinates = self.controls.coordinate_tensor(geometry)
        with torch.no_grad():
            penalty = self.controls.parameters.fairness_loss(
                self.controls.controls_from_local(coordinates), self.fairness_scale
            )
        return float(actual + smoothing * self.fairness_weight * float(penalty))

    def control_loss(self, paths):
        if self.options.cleanup:
            return self.fairness_weight * self.controls.parameters.fairness_loss(
                paths[0], self.fairness_scale
            )
        return 0.001 * self.controls.parameters.bending_loss(paths[0])
