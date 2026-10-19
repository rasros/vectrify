"""Tile layouts and native/portable raster preparation for a filled fit."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from vectrify.refine import paths, soft_coverage
from vectrify.refine.filled_fit_scene import FilledFitOptions, FilledFitScene

Tile = tuple[int, int, int, int]
CandidateTile = tuple[int, int, int, int, tuple[int, ...]]
SimpleGroups = dict[tuple[tuple[int, ...], str, int, int], list[tuple[int, int, int]]]
MultiGroups = dict[tuple[str, int, int], list[tuple[int, int, int]]]


class FilledFitRaster:
    """One scene's tile caches; native topology workspace survives Adam steps.

    A scene renders through the native CUDA kernels, or through the portable
    soft coverage when it is ``soft``. Rebuild movement-sensitive caches at the
    original guards.
    """

    scene: FilledFitScene
    options: FilledFitOptions
    initial_simple_groups: SimpleGroups
    simple_tile_reference: Any
    initial_multi_tiles: dict[int, Tile]
    initial_large_multi_tiles: dict[int, list[CandidateTile]]
    large_multi_tile_indices: dict[int, list[Any]]
    large_multi_boundary_indices: dict[int, list[Any]]
    large_multi_topology_workspaces: dict[int, dict[tuple[int, int], Any]]
    large_multi_index_references: dict[int, tuple[Any, ...]]
    initial_multi_groups: MultiGroups

    def __init__(self, scene: FilledFitScene, options: FilledFitOptions) -> None:
        import torch

        self.scene = scene
        self.options = options
        # Tile layout is part of the seed rasterisation setup, not optimisation
        # state.  Re-reading each CUDA control tensor's extrema every Adam step
        # introduces hundreds of device synchronisations on a detailed
        # drawing.  The two-pixel antialias margin already makes these fixed tiles
        # conservative for the local coordinate updates used by the fit.
        self.initial_simple_groups = self.cropped_simple_groups()
        # Simple paths use cropped tiles, so unlike the full-canvas compositor
        # their bounds are optimisation state.  A coordinate update can move
        # a boundary outside its initial two-pixel antialias fringe; continuing to
        # rasterise a stale crop silently clips that fill and creates the holes and
        # spikes visible in long fits.  Keep one packed reference so the movement
        # check is a single device reduction; rebuild the inexpensive Python tile
        # grouping only after a meaningful move.
        self.simple_tile_reference = self.scene.control_storage.detach().clone()

        self.initial_multi_tiles = {
            index: self.tile_for(path)
            for index, path in enumerate(self.scene.controls)
            if len(path) != 1 and len(path) < 16
        }
        self.initial_large_multi_tiles = {
            index: paths.large_path_tile_candidates(
                path, self.scene.work_width, self.scene.work_height
            )
            for index, path in enumerate(self.scene.controls)
            if len(path) >= 16
        }
        self.large_multi_tile_indices = {
            index: [
                torch.tensor(candidates, dtype=torch.long, device=self.scene.device)
                for _left, _top, _width, _height, candidates in tiles
            ]
            for index, tiles in self.initial_large_multi_tiles.items()
        }
        self.large_multi_boundary_indices = {
            index: [
                torch.tensor(
                    [ray_candidates.index(candidate) for candidate in candidates],
                    dtype=torch.long,
                    device=self.scene.device,
                )
                for (_left, _top, _width, _height, ray_candidates), candidates in zip(
                    tiles,
                    paths.large_path_tile_boundary_candidates(
                        self.scene.controls[index], tiles
                    ),
                    strict=True,
                )
            ]
            for index, tiles in self.initial_large_multi_tiles.items()
        }
        self.large_multi_topology_workspaces: dict[int, dict[tuple[int, int], Any]] = {
            index: {} for index in self.initial_large_multi_tiles
        }
        self.large_multi_index_references = {
            index: tuple(control.detach().clone() for control in path)
            for index, path in enumerate(self.scene.controls)
            if len(path) >= 16
        }
        self.initial_multi_groups: dict[
            tuple[str, int, int], list[tuple[int, int, int]]
        ] = defaultdict(list)
        for index, (
            left,
            top,
            tile_width,
            tile_height,
        ) in self.initial_multi_tiles.items():
            self.initial_multi_groups[
                (self.scene.entries[index][3], tile_width, tile_height)
            ].append((index, left, top))

    def tile_for(self, path: list[Any]) -> tuple[int, int, int, int]:
        """A fixed, antialiased raster tile covering a path's control hull."""
        import torch

        points = torch.cat([control.detach().reshape(-1, 2) for control in path])
        # Cubic Beziers lie in their control hull.  Two pixels retain the
        # entire soft edge while avoiding the full-canvas work DiffVG culls.
        left = max(0, math.floor(float(points[:, 0].min())) - 2)
        top = max(0, math.floor(float(points[:, 1].min())) - 2)
        right = min(self.scene.work_width, math.ceil(float(points[:, 0].max())) + 2)
        bottom = min(self.scene.work_height, math.ceil(float(points[:, 1].max())) + 2)

        # Bucket dimensions keep unrelated paths in the same CUDA batch.  A
        # 32px bucket roughly halves the distinct sizes of the 1024px cat
        # seed versus 8px buckets while adding only a small protected fringe
        # to the right/bottom of each tile.  The origin—and therefore every
        # coverage sample belonging to the path—remains unchanged.  Shift a
        # bucket at the canvas edge rather than clipping its antialias margin.
        tile_bucket = 32
        tile_width = min(
            self.scene.work_width,
            tile_bucket * math.ceil(max(right - left, 1) / tile_bucket),
        )
        tile_height = min(
            self.scene.work_height,
            tile_bucket * math.ceil(max(bottom - top, 1) / tile_bucket),
        )
        left = min(left, self.scene.work_width - tile_width)
        top = min(top, self.scene.work_height - tile_height)
        return left, top, tile_width, tile_height

    def restore_tile(self, alpha: Any, left: int, top: int) -> Any:
        import torch

        return torch.nn.functional.pad(
            alpha,
            (
                left,
                self.scene.work_width - left - alpha.shape[1],
                top,
                self.scene.work_height - top - alpha.shape[0],
            ),
        )

    def cropped_simple_groups(self) -> SimpleGroups:
        groups: dict[
            tuple[tuple[int, ...], str, int, int], list[tuple[int, int, int]]
        ] = defaultdict(list)
        for index, path in enumerate(self.scene.controls):
            if len(path) != 1:
                continue
            left, top, tile_width, tile_height = self.tile_for(path)
            groups[
                (
                    tuple(path[0].shape),
                    self.scene.entries[index][3],
                    tile_width,
                    tile_height,
                )
            ].append((index, left, top))
        return groups

    def rasterise_simple_tiles(
        self,
        fill_rule: str,
        tile_width: int,
        tile_height: int,
        items: list[tuple[int, int, int]],
    ) -> list[tuple[int, Any, int, int]]:
        import torch

        if self.scene.soft:
            return [
                (
                    index,
                    soft_coverage.soft_coverage(
                        self.scene.controls[index],
                        (left, top, left + tile_width, top + tile_height),
                        fill_rule=fill_rule,
                    ),
                    left,
                    top,
                )
                for index, left, top in items
            ]
        from vectrify.refine import cuda_renderer

        box = (0, 0, tile_width, tile_height)
        chunked = [
            paths.fused_chunks(
                self.scene.controls[index][0]
                - self.scene.controls[index][0].new_tensor((left, top))
            )
            for index, left, top in items
        ]
        if self.scene.controls[items[0][0]][0].shape[0] > paths.FUSED_CUBICS:
            # A contour longer than the fixed-width primitive is a path of
            # 16-cubic chunks, whose windings add up before the fill rule.
            offsets = [0]
            for chunks in chunked:
                offsets.append(offsets[-1] + len(chunks))
            rasterised = cuda_renderer.multi_coverage(
                torch.cat(chunked),
                offsets,
                box,
                subpixels=self.options.subpixels,
                fill_rule=fill_rule,
            )
        else:
            rasterised = cuda_renderer.coverage(
                torch.cat(chunked),
                box,
                subpixels=self.options.subpixels,
                fill_rule=fill_rule,
            )
        return [
            (index, alpha, left, top)
            for (index, left, top), alpha in zip(
                items, _native(rasterised), strict=True
            )
        ]

    def rasterise_simple(
        self,
        fill_rule: str,
        tile_width: int,
        tile_height: int,
        items: list[tuple[int, int, int]],
    ) -> list[tuple[int, Any]]:
        return [
            (index, self.restore_tile(alpha, left, top))
            for index, alpha, left, top in self.rasterise_simple_tiles(
                fill_rule, tile_width, tile_height, items
            )
        ]

    def sparse_backward_batch_size(self, tile_width: int, tile_height: int) -> int:
        """Bound the live tile-local autograd graph while filling the GPU.

        Sparse replay never needs a full-canvas alpha stack, but it does keep
        the coverage graph for one backward batch alive.  A fixed 16-path
        batch underutilises CUDA for common 32--64px tiles, but a
        recovery pass can create a much larger equal-tile group than the
        initial seed.  Retain the proven 16-path graph cap and apply the
        tile-area budget beneath it.  This bounds peak memory for every
        document without changing the rendered image or its derivative.
        """
        return max(1, min(16, (1 << 20) // max(1, tile_width * tile_height)))

    def soft_multi(self, index: int, path: list[Any]) -> Any:
        # The tile follows the current controls, so movement never clips it.
        left, top, tile_width, tile_height = self.tile_for(path)
        alpha = soft_coverage.soft_coverage(
            path,
            (left, top, left + tile_width, top + tile_height),
            fill_rule=self.scene.entries[index][3],
        )
        return self.restore_tile(alpha, left, top)

    def rasterise_multi(self, index: int, path: list[Any]) -> Any:
        import torch

        if self.scene.soft:
            return self.soft_multi(index, path)
        # Large paths use fixed conservative candidate tiles.  Every tile
        # sees all contours that can cross one of its horizontal rays, rather
        # than every contour at every pixel.
        if len(path) >= 16:
            # The index has a two-pixel conservative guard band.  Rebuild it
            # after local optimisation consumes half that allowance, so a
            # stale index cannot exclude a valid ray crossing.
            reference = self.large_multi_index_references[index]
            movement = max(
                float((control.detach() - saved).abs().amax())
                for control, saved in zip(path, reference, strict=True)
            )
            if movement > 1.0:
                self.initial_large_multi_tiles[index] = (
                    paths.large_path_tile_candidates(
                        path, self.scene.work_width, self.scene.work_height
                    )
                )
                self.large_multi_tile_indices[index] = [
                    torch.tensor(candidates, dtype=torch.long, device=self.scene.device)
                    for _left, _top, _width, _height, candidates in (
                        self.initial_large_multi_tiles[index]
                    )
                ]
                self.large_multi_boundary_indices[index] = [
                    torch.tensor(
                        [ray_candidates.index(candidate) for candidate in candidates],
                        dtype=torch.long,
                        device=self.scene.device,
                    )
                    for (
                        _left,
                        _top,
                        _width,
                        _height,
                        ray_candidates,
                    ), candidates in zip(
                        self.initial_large_multi_tiles[index],
                        paths.large_path_tile_boundary_candidates(
                            path, self.initial_large_multi_tiles[index]
                        ),
                        strict=True,
                    )
                ]
                self.large_multi_index_references[index] = tuple(
                    control.detach().clone() for control in path
                )
                self.large_multi_topology_workspaces[index].clear()
            tiled_alpha = paths.tiled_large_path_coverage(
                path,
                (0, 0, self.scene.work_width, self.scene.work_height),
                self.initial_large_multi_tiles[index],
                fill_rule=self.scene.entries[index][3],
                subpixels=self.options.subpixels,
                packed_contours=self.scene.control_storage[
                    slice(*self.scene.path_storage_spans[index])
                ],
                candidate_indices=self.large_multi_tile_indices[index],
                boundary_candidate_indices=self.large_multi_boundary_indices[index],
                topology_workspaces=self.large_multi_topology_workspaces[index],
            )
            return _native(tiled_alpha)
        left, top, tile_width, tile_height = self.initial_multi_tiles[index]
        offset = path[0].new_tensor((left, top))
        from vectrify.refine.cuda_renderer import multi_coverage

        packed = torch.cat([paths.fused_chunks(control - offset) for control in path])
        analytic = multi_coverage(
            packed,
            [0, len(packed)],
            (0, 0, tile_width, tile_height),
            subpixels=self.options.subpixels,
            fill_rule=self.scene.entries[index][3],
        )
        return self.restore_tile(_native(analytic)[0], left, top)

    def rasterise_multi_group(
        self,
        fill_rule: str,
        tile_width: int,
        tile_height: int,
        items: list[tuple[int, int, int]],
    ) -> list[tuple[int, Any]]:
        """Rasterise equal-sized multi-contour paths in one contour batch."""
        import torch

        if self.scene.soft:
            return [
                (index, self.soft_multi(index, self.scene.controls[index]))
                for index, _left, _top in items
            ]
        chunks = []
        spans = []
        for index, left, top in items:
            offset = self.scene.controls[index][0].new_tensor((left, top))
            start = sum(len(c) for c in chunks)
            chunks.extend(
                paths.fused_chunks(control - offset)
                for control in self.scene.controls[index]
            )
            spans.append((start, sum(len(c) for c in chunks)))
        # Native winding uses one CUDA block per contour chunk.  Combining
        # contours from otherwise independent paths lets its blocks occupy the
        # GPU at once, while summing each recorded span before the fill
        # nonlinearity preserves SVG path semantics (including holes); long
        # contours are split into 16-cubic chunks, whose windings add up.
        packed = torch.cat(chunks)
        from vectrify.refine.cuda_renderer import multi_coverage

        analytic = multi_coverage(
            packed,
            [start for start, _end in spans] + [spans[-1][1]],
            (0, 0, tile_width, tile_height),
            subpixels=self.options.subpixels,
            fill_rule=fill_rule,
        )
        return [
            (index, self.restore_tile(alpha, left, top))
            for (index, left, top), alpha in zip(items, _native(analytic), strict=True)
        ]

    def refresh_simple_tiles(self) -> None:
        movement = (
            (self.scene.control_storage.detach() - self.simple_tile_reference)
            .abs()
            .amax()
        )
        if float(movement) <= 1.0:
            return
        self.initial_simple_groups = self.cropped_simple_groups()
        self.simple_tile_reference = self.scene.control_storage.detach().clone()


def _native(coverage: Any) -> Any:
    """A native kernel's coverage; the scene routes unsupported layouts to soft."""
    if coverage is None:
        raise RuntimeError("the native coverage kernel rejected a supported layout")
    return coverage
