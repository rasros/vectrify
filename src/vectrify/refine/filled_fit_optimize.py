"""Monolithic gradients and bounded painter-order replay for filled fits.

Replay snapshots and tile-local autograd graphs live only for one backward
step; parameters, optimizers and raster caches live for the complete fit.
"""

from __future__ import annotations

from typing import Any

from vectrify.refine import paths
from vectrify.refine.filled_fit_raster import FilledFitRaster
from vectrify.refine.filled_fit_scene import (
    FilledFitHooks,
    FilledFitOptions,
    FilledFitScene,
)


class FilledFitOptimizer:
    """Preserve Adam order and callbacks across the three backward strategies."""

    def __init__(
        self,
        scene: FilledFitScene,
        raster: FilledFitRaster,
        options: FilledFitOptions,
        hooks: FilledFitHooks,
        *,
        point_learning_rate: float,
        color_learning_rate: float,
    ) -> None:
        import torch

        self.scene = scene
        self.raster = raster
        self.options = options
        self.hooks = hooks
        self.point_optimizer = torch.optim.Adam(
            [scene.control_storage],
            lr=point_learning_rate,
            fused=scene.device == "cuda",
        )
        self.colour_optimizer = torch.optim.Adam(
            [
                scene.color_storage,
                *([] if scene.alpha_values is None else [scene.alpha_values]),
            ],
            lr=color_learning_rate,
            fused=scene.device == "cuda",
        )

    def run(self, steps: int) -> None:
        """Observe candidates, backpropagate, then update and project in order."""
        import torch

        completed_steps = 0
        for step in range(steps):
            if self.hooks.observe is not None and not self.hooks.observe(
                step, self.scene.controls, self.scene.color_storage
            ):
                break
            completed_steps = step + 1
            self.point_optimizer.zero_grad()
            self.colour_optimizer.zero_grad()
            all_controls = torch.cat(
                [control for path in self.scene.controls for control in path]
            )
            if self.scene.monolithic:
                self.backward_monolithic(step, all_controls)
            elif self.options.sparse_replay:
                self.backward_sparse(all_controls)
            else:
                self.backward_dense(all_controls)
            self.point_optimizer.step()
            self.colour_optimizer.step()
            self.scene.close_contours(self.options, self.hooks)
            self.raster.refresh_simple_tiles()

        if self.hooks.observe is not None:
            self.hooks.observe(
                completed_steps, self.scene.controls, self.scene.color_storage
            )

    def backward_monolithic(self, _step: int, all_controls: Any) -> None:
        import torch

        simple_groups = self.raster.initial_simple_groups
        alphas: list[Any | None] = [None] * len(self.scene.entries)
        for (
            _shape,
            fill_rule,
            tile_width,
            tile_height,
        ), items in simple_groups.items():
            for index, alpha in self.raster.rasterise_simple(
                fill_rule, tile_width, tile_height, items
            ):
                alphas[index] = alpha
        for (
            fill_rule,
            tile_width,
            tile_height,
        ), items in self.raster.initial_multi_groups.items():
            for index, alpha in self.raster.rasterise_multi_group(
                fill_rule, tile_width, tile_height, items
            ):
                alphas[index] = alpha
        for index, path in enumerate(self.scene.controls):
            if alphas[index] is None:
                alphas[index] = self.raster.rasterise_multi(index, path)
        alpha_stack = torch.stack([alpha for alpha in alphas if alpha is not None])
        if self.hooks.coverage_transform is not None:
            alpha_stack = self.hooks.coverage_transform(
                self.scene.controls, alpha_stack
            )
        if self.scene.alpha_values is not None:
            alpha_stack = (
                alpha_stack * self.scene.alpha_values.clamp(0, 1)[:, None, None]
            )
        if self.scene.context_tensors is not None:
            # Selected-path fitting supplies its own compositing response.
            # Avoid compiling an unused full-document composite for every
            # crop size (which can exhaust Torch's recompilation cache).
            # Sibling layers share clips, group opacity and later artwork.
            # Keep the gradient through their changing overlaps by
            # composing premultiplied paint before the common response.
            rendered = paths.composite_in_context(
                alpha_stack, self.scene.color_storage, self.scene.context_tensors
            )
        else:
            composite = (
                paths.compiled_opaque_fill_composite()
                if self.scene.goal.is_cuda
                else paths.composite_opaque_fills
            )
            rendered = (
                composite(alpha_stack, self.scene.color_storage)
                if self.scene.under is None
                else paths.composite_opaque_fills(
                    alpha_stack, self.scene.color_storage, self.scene.under
                )
            )
        comparison, target_image = (
            (rendered, self.scene.goal)
            if self.hooks.loss_transform is None
            else self.hooks.loss_transform(rendered, self.scene.goal, _step)
        )
        error = (comparison - target_image) ** 2
        loss = (
            (error[..., :3].sum(-1) + 3 * error[..., 3]).mean() / 6
            if self.scene.channels == 4
            else error.mean()
        )
        if self.options.xing_weight:
            loss = loss + self.options.xing_weight * paths.xing_loss(all_controls)
        if self.hooks.control_loss is not None:
            loss = loss + self.hooks.control_loss(self.scene.controls)
        loss.backward()
        if self.hooks.control_transform is not None:
            # The renderer sees duplicated cubic endpoints and derived
            # controls. Sum their derivatives into the editor's actual
            # coordinates instead of discarding them during projection.
            assert self.scene.control_storage.grad is not None
            transformed = self.hooks.control_transform(self.scene.controls)
            outputs = [c for path in transformed for c in path]
            gradients = [
                self.scene.control_storage.grad[i, :n].detach().clone()
                for i, n in enumerate(self.scene.contour_sizes)
            ]
            gradient = torch.autograd.grad(
                outputs, self.scene.control_storage, grad_outputs=gradients
            )[0]
            self.scene.control_storage.grad.copy_(gradient)

    def backward_sparse(self, all_controls: Any) -> None:
        import torch

        simple_groups = self.raster.initial_simple_groups
        # Dense replay previously saved a full alpha, pre-layer canvas and
        # downstream-transparency map for every SVG path.  Painter-order
        # compositing is local to a path's coverage tile, so retain only
        # those slices while keeping the current canvas/transparency as
        # full images. This is algebraically the same replay derivative.
        with torch.no_grad():
            coverages: list[tuple[Any, int, int] | None] = [None] * len(
                self.scene.entries
            )
            for (
                _shape,
                fill_rule,
                tile_width,
                tile_height,
            ), items in simple_groups.items():
                for index, alpha, left, top in self.raster.rasterise_simple_tiles(
                    fill_rule, tile_width, tile_height, items
                ):
                    coverages[index] = (alpha, left, top)
            for index, path in enumerate(self.scene.controls):
                if coverages[index] is None:
                    coverages[index] = (self.raster.rasterise_multi(index, path), 0, 0)

            opacity_values = (
                self.scene.alpha_values.detach().clamp(0, 1)
                if self.scene.alpha_values is not None
                else None
            )
            stored_alphas: list[Any] = []
            before_tiles: list[Any] = []
            rendered = (
                torch.zeros_like(self.scene.goal)
                if self.scene.under is None
                else self.scene.under.clone()
            )
            for index, item in enumerate(coverages):
                assert item is not None
                alpha, left, top = item
                if opacity_values is not None:
                    alpha = alpha * opacity_values[index]
                bottom, right = top + alpha.shape[0], left + alpha.shape[1]
                canvas = rendered[top:bottom, left:right]
                before_tiles.append(canvas.clone())
                rendered[top:bottom, left:right] = (
                    canvas * (1 - alpha[..., None])
                    + self.scene.color_storage[index].detach().clamp(0, 1)
                    * alpha[..., None]
                )
                stored_alphas.append(alpha)

            suffix_tiles: list[Any] = [None] * len(self.scene.entries)
            transparency = torch.ones(
                (self.scene.work_height, self.scene.work_width),
                dtype=self.scene.goal.dtype,
                device=self.scene.device,
            )
            for index in range(len(self.scene.entries) - 1, -1, -1):
                item = coverages[index]
                assert item is not None
                alpha, left, top = item
                bottom, right = top + alpha.shape[0], left + alpha.shape[1]
                suffix = transparency[top:bottom, left:right]
                suffix_tiles[index] = suffix.clone()
                suffix.mul_(1 - stored_alphas[index])
            image_gradient = 2 * (rendered - self.scene.goal) / rendered.numel()

        def sparse_layer_loss(
            index: int,
            alpha: Any,
            left: int,
            top: int,
            *,
            saved_coverages: list[tuple[Any, int, int] | None] = coverages,
            saved_alphas: list[Any] = stored_alphas,
            saved_suffixes: list[Any] = suffix_tiles,
            saved_canvases: list[Any] = before_tiles,
            gradient: Any = image_gradient,
        ) -> Any:
            item = saved_coverages[index]
            assert item is not None
            coverage, _stored_left, _stored_top = item
            stored_alpha = saved_alphas[index]
            suffix = saved_suffixes[index]
            canvas = saved_canvases[index]
            bottom, right = top + alpha.shape[0], left + alpha.shape[1]
            gradient = gradient[top:bottom, left:right]
            colour = self.scene.color_storage[index]
            colour_delta = colour.detach().clamp(0, 1) - canvas
            alpha_gradient = (gradient * suffix[..., None] * colour_delta).sum(dim=-1)
            opacity = (
                self.scene.alpha_values[index].clamp(0, 1)
                if self.scene.alpha_values is not None
                else None
            )
            colour_gradient = (
                gradient * suffix[..., None] * stored_alpha[..., None]
            ).sum(dim=(0, 1))
            geometry_loss = (alpha * alpha_gradient.detach()).sum()
            if opacity is not None:
                geometry_loss = geometry_loss * opacity
                geometry_loss = (
                    geometry_loss + opacity * (coverage * alpha_gradient.detach()).sum()
                )
            return geometry_loss + (colour.clamp(0, 1) * colour_gradient.detach()).sum()

        for (
            _shape,
            fill_rule,
            tile_width,
            tile_height,
        ), items in simple_groups.items():
            # Sparse replay keeps only a tile-local graph, so it can
            # batch more equal-size paths than the legacy dense replay.
            # This reduces native coverage launches without increasing the
            # full-canvas memory footprint.
            batch_size = self.raster.sparse_backward_batch_size(tile_width, tile_height)
            for offset in range(0, len(items), batch_size):
                loss = torch.zeros((), device=self.scene.device)
                for index, alpha, left, top in self.raster.rasterise_simple_tiles(
                    fill_rule,
                    tile_width,
                    tile_height,
                    items[offset : offset + batch_size],
                ):
                    loss = loss + sparse_layer_loss(index, alpha, left, top)
                loss.backward()
        simple_indices = {
            index for group in simple_groups.values() for index, _left, _top in group
        }
        for index, path in enumerate(self.scene.controls):
            if index not in simple_indices:
                alpha = self.raster.rasterise_multi(index, path)
                sparse_layer_loss(index, alpha, 0, 0).backward()
        if self.options.xing_weight:
            (self.options.xing_weight * paths.xing_loss(all_controls)).backward()

    def backward_dense(self, all_controls: Any) -> None:
        import torch

        simple_groups = self.raster.initial_simple_groups
        # First composite the exact same soft fills without recording an
        # autograd graph.  The saved canvases and suffix transparencies are
        # enough to derive the MSE gradient of each layer independently.
        with torch.no_grad():
            initial_alphas: list[Any | None] = [None] * len(self.scene.entries)
            for (
                _shape,
                fill_rule,
                tile_width,
                tile_height,
            ), items in simple_groups.items():
                for index, alpha in self.raster.rasterise_simple(
                    fill_rule, tile_width, tile_height, items
                ):
                    initial_alphas[index] = alpha
            for (
                fill_rule,
                tile_width,
                tile_height,
            ), items in self.raster.initial_multi_groups.items():
                for index, alpha in self.raster.rasterise_multi_group(
                    fill_rule, tile_width, tile_height, items
                ):
                    initial_alphas[index] = alpha
            for index, path in enumerate(self.scene.controls):
                if initial_alphas[index] is None:
                    initial_alphas[index] = self.raster.rasterise_multi(index, path)

            before: list[Any] = []
            rendered = (
                torch.zeros_like(self.scene.goal)
                if self.scene.under is None
                else self.scene.under
            )
            opacity_values = (
                self.scene.alpha_values.detach().clamp(0, 1)
                if self.scene.alpha_values is not None
                else None
            )
            initial_coverages: list[Any | None] = initial_alphas.copy()
            for index, alpha in enumerate(initial_alphas):
                assert alpha is not None
                if opacity_values is not None:
                    alpha = alpha * opacity_values[index]
                    initial_alphas[index] = alpha
                colour = self.scene.color_storage[index]
                before.append(rendered)
                rendered = (
                    rendered * (1 - alpha[..., None])
                    + colour.detach().clamp(0, 1) * alpha[..., None]
                )
            downstream: list[Any | None] = [None] * len(self.scene.entries)
            transparency = torch.ones(
                (self.scene.work_height, self.scene.work_width),
                dtype=self.scene.goal.dtype,
                device=self.scene.device,
            )
            for index in range(len(self.scene.entries) - 1, -1, -1):
                downstream[index] = transparency
                alpha = initial_alphas[index]
                assert alpha is not None
                transparency = transparency * (1 - alpha)
            image_gradient = 2 * (rendered - self.scene.goal) / rendered.numel()

        def layer_loss(
            index: int,
            alpha: Any,
            *,
            alphas: list[Any | None] = initial_alphas,
            suffixes: list[Any | None] = downstream,
            canvases: list[Any] = before,
            coverages: list[Any | None] = initial_coverages,
            gradient: Any = image_gradient,
        ) -> Any:
            stored_alpha = alphas[index]
            suffix = suffixes[index]
            assert stored_alpha is not None
            assert suffix is not None
            colour = self.scene.color_storage[index]
            colour_delta = colour.detach().clamp(0, 1) - canvases[index]
            alpha_gradient = (gradient * suffix[..., None] * colour_delta).sum(dim=-1)
            opacity = (
                self.scene.alpha_values[index].clamp(0, 1)
                if self.scene.alpha_values is not None
                else None
            )
            colour_gradient = (
                gradient * suffix[..., None] * stored_alpha[..., None]
            ).sum(dim=(0, 1))
            geometry_loss = (alpha * alpha_gradient.detach()).sum()
            if opacity is not None:
                geometry_loss = geometry_loss * opacity
                coverage = coverages[index]
                assert coverage is not None
                opacity_gradient = (coverage * alpha_gradient.detach()).sum()
                geometry_loss = geometry_loss + opacity * opacity_gradient
            return geometry_loss + (colour.clamp(0, 1) * colour_gradient.detach()).sum()

        # Backpropagate a bounded batch at a time.  The compositing derivative
        # above accounts for all later opaque layers, so this has the same MSE
        # gradient as one monolithic render without its peak-memory cost.
        for (
            _shape,
            fill_rule,
            tile_width,
            tile_height,
        ), items in simple_groups.items():
            for offset in range(0, len(items), 4):
                batch = items[offset : offset + 4]
                loss = torch.zeros((), device=self.scene.device)
                for index, alpha in self.raster.rasterise_simple(
                    fill_rule, tile_width, tile_height, batch
                ):
                    loss = loss + layer_loss(index, alpha)
                loss.backward()
        simple_indices = {
            index for group in simple_groups.values() for index, _left, _top in group
        }
        multi_indices = {
            index
            for group in self.raster.initial_multi_groups.values()
            for index, _left, _top in group
        }
        for (
            fill_rule,
            tile_width,
            tile_height,
        ), items in self.raster.initial_multi_groups.items():
            for offset in range(0, len(items), 64):
                batch = items[offset : offset + 64]
                loss = torch.zeros((), device=self.scene.device)
                for index, alpha in self.raster.rasterise_multi_group(
                    fill_rule, tile_width, tile_height, batch
                ):
                    loss = loss + layer_loss(index, alpha)
                loss.backward()
        for index, path in enumerate(self.scene.controls):
            if index not in simple_indices and index not in multi_indices:
                layer_loss(
                    index,
                    self.raster.rasterise_multi(index, path),
                ).backward()
        if self.options.xing_weight:
            (self.options.xing_weight * paths.xing_loss(all_controls)).backward()
