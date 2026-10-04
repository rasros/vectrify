"""Fixed-topology maps between editor coordinates and differentiable cubics."""

from __future__ import annotations

import numpy as np

from vectrify.document import DocumentError, Geometry


class ControlMap:
    """Preserve IDs, straight closures, pins and movement bounds in a fit."""

    def __init__(
        self,
        geometry: Geometry,
        local,
        original,
        linear,
        offset,
        origin,
        scale,
        movable,
        displacement: float,
        *,
        stroke_only: bool = False,
    ):
        import torch

        self.geometry = geometry
        self.original = original
        self.linear, self.offset, self.origin, self.scale = (
            linear,
            offset,
            origin,
            scale,
        )
        self.inverse = torch.linalg.inv(linear)
        self.movable, self.displacement, self.stroke_only = (
            movable,
            displacement,
            stroke_only,
        )
        device = original.device
        # Every original node keeps its ID and command. Straight edges and implicit
        # closures remain straight even though the fitter represents them as cubics.
        definitions = []
        index = 0
        for subpath in geometry.subpaths:
            segments = []
            head = previous = index
            index += 1
            for node in subpath.nodes[1:]:
                length = len(node.values) // 2
                segments.append((previous, tuple(range(index, index + length))))
                previous = index + length - 1
                index += length
            # The implicit closing line, unless the contour already ends where it
            # starts. A zero-length closing cubic is not harmless: it takes
            # 16-cubic contours past the native renderer's width, onto a winding
            # rasteriser whose coverage has no gradient, and nothing moves.
            if (
                (not stroke_only or subpath.closed)
                and previous != head
                and not np.array_equal(local[previous], local[head])
            ):
                segments.append((previous, (head,)))
            if not segments:
                raise DocumentError("This path contains an empty contour")
            definitions.append(segments)

        # Index tables for moving between the coordinates and the cubics in a
        # few whole-tensor operations: a loop of per-point writes costs a kernel
        # launch each on the GPU, most of a fit's time there.
        gather, straight, last_write = [], [], {}
        for segments in definitions:
            for previous, following in segments:
                k = len(gather)
                if len(following) == 1:
                    gather.append((previous, previous, following[0], following[0]))
                    straight.append(True)
                    writes = [(previous, 0), (following[0], 3)]
                else:
                    gather.append((previous, *following))
                    straight.append(False)
                    writes = [
                        (previous, 0),
                        *((f, i + 1) for i, f in enumerate(following)),
                    ]
                # The cubics' points as they are written in turn: a point two
                # segments share takes the later segment's.
                for point, row in writes:
                    last_write.pop(point, None)
                    last_write[point] = 4 * k + row
        gather_index = torch.tensor(gather, dtype=torch.long, device=device)
        straight_mask = torch.tensor(straight, device=device)[:, None]
        written = torch.tensor(list(last_write), dtype=torch.long, device=device)
        sources = torch.tensor(
            list(last_write.values()), dtype=torch.long, device=device
        )
        counts = [len(segments) for segments in definitions]

        self.gather_index = gather_index
        self.straight_mask = straight_mask
        self.written, self.sources, self.counts = written, sources, counts

    def controls_from_local(self, local):
        import torch

        points = (local @ self.linear.T + self.offset - self.origin) * self.scale
        g = points[self.gather_index]
        a, b = g[:, 0], g[:, 3]
        middle = torch.where(
            self.straight_mask[..., None],
            torch.stack(((2 * a + b) / 3, (a + 2 * b) / 3), 1),
            g[:, 1:3],
        )
        cubics = torch.cat((a[:, None], middle, b[:, None]), 1)
        contours = list(torch.split(cubics, self.counts))
        if self.stroke_only:
            # The fill optimizer stores closed contours. An open stroke is
            # represented by its centreline and its reverse, with projection
            # tying the copies. Stroke coverage reads only the forward half.
            contours = [
                contour
                if subpath.closed
                else torch.cat((contour, contour.flip((0, 1))))
                for contour, subpath in zip(
                    contours, self.geometry.subpaths, strict=True
                )
            ]
        return contours

    def local_from_controls(self, contours):
        import torch

        cubics = torch.cat([c[:n] for c, n in zip(contours, self.counts, strict=True)])
        values = (
            (cubics / self.scale + self.origin - self.offset) @ self.inverse.T
        ).reshape(-1, 2)
        local = self.original.clone()
        local[self.written] = values[self.sources]
        delta = (local - self.original) * self.movable
        length = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return self.original + delta * (self.displacement / length).clamp(max=1)
