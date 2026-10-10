"""Fixed-topology maps between editor coordinates and differentiable cubics."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from vectrify.document import DocumentError, Geometry


def line_knots(geometry: Geometry) -> frozenset[str]:
    """Original line junctions are free to develop a new tangent during fitting."""
    knots = set()
    for subpath in geometry.subpaths:
        nodes = subpath.nodes
        for previous, node in pairwise(nodes):
            if node.command == "L":
                knots.update((previous.id, node.id))
        if subpath.closed and nodes[-1].endpoint != nodes[0].endpoint:
            knots.update((nodes[0].id, nodes[-1].id))
    return frozenset(knots)


class ControlMap:
    """Preserve joins, pins and movement bounds without growing handle spikes."""

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
        corners: frozenset[str] = frozenset(),
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
        corner_rows = set()
        for subpath in geometry.subpaths:
            segments = []
            head = previous = index
            if subpath.nodes[0].id in corners:
                corner_rows.add(index)
            index += 1
            for node in subpath.nodes[1:]:
                length = len(node.values) // 2
                segments.append((previous, tuple(range(index, index + length))))
                previous = index + length - 1
                if node.id in corners:
                    corner_rows.add(previous)
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

        # Repeated knots and explicit closures are one point, even when the
        # editor gives them separate IDs. Moving their copies independently
        # opens tiny wedges, and zero-length cubics can sprout loops.
        parents = list(range(len(local)))

        def root(i):
            while parents[i] != i:
                i = parents[i]
            return i

        def join(a, b):
            parents[root(b)] = root(a)

        for segments, subpath in zip(definitions, geometry.subpaths, strict=True):
            for previous, following in segments:
                if all(np.array_equal(local[previous], local[i]) for i in following):
                    for i in following:
                        join(previous, i)
            head, tail = segments[0][0], segments[-1][1][-1]
            if (not stroke_only or subpath.closed) and np.array_equal(
                local[head], local[tail]
            ):
                join(head, tail)
        groups = {}
        for i in range(len(local)):
            groups.setdefault(root(i), []).append(i)
        self.coincident = []
        for group in groups.values():
            if len(group) < 2:
                continue
            indices = torch.tensor(group, dtype=torch.long, device=device)
            self.coincident.append((indices, movable[indices].amin()))
        self.canonical = torch.tensor(
            [root(i) for i in range(len(local))], dtype=torch.long, device=device
        )

        # A smooth knot has one tangent, rather than two unrelated handles.
        # Keep its original arm ratio: exact subdivision then remains smooth
        # while the knot, tangent direction and tangent length can all move.
        # Sharp corners, cusps, held handles and open contour ends are excluded.
        smooth, ratios = [], []
        for segments in definitions:
            curved_segments = [
                (previous, following)
                for previous, following in segments
                if len(following) == 3
                and not all(
                    np.array_equal(local[previous], local[i]) for i in following
                )
            ]
            pairs = list(pairwise(curved_segments))
            if len(curved_segments) > 1:
                pairs.append((curved_segments[-1], curved_segments[0]))
            for (_, incoming), (knot, outgoing) in pairs:
                if root(incoming[-1]) != root(knot):
                    continue
                left, right = incoming[-2], outgoing[0]
                a, b = local[knot] - local[left], local[right] - local[knot]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if (
                    min(na, nb) > 1e-6
                    and knot not in corner_rows
                    and incoming[-1] not in corner_rows
                    and np.dot(a, b) > 0
                    and abs(a[0] * b[1] - a[1] * b[0]) <= 1e-6 * na * nb
                    and bool(movable[left])
                    and bool(movable[right])
                ):
                    smooth.append((root(knot), left, right))
                    ratios.append(nb / na)
        self.smooth = torch.tensor(smooth, dtype=torch.long, device=device).reshape(
            -1, 3
        )
        self.arm_ratio = original.new_tensor(ratios)[:, None]
        coupled = [root(i) for i in range(len(local))]
        for knot, left, right in smooth:
            coupled[left] = coupled[right] = knot
        self.coupled = torch.tensor(coupled, dtype=torch.long, device=device)

        # Bound a handle relative to its own segment, rather than letting a
        # two-unit fit turn a subpixel edge into a long hook. Existing arcs
        # that overhang their chord retain their original freedom; concave
        # contours and inflected curves are still allowed.
        curved = gather_index[~straight_mask[:, 0]]
        self.handle_indices = curved[:, 1:3].reshape(-1)
        self.handle_anchors = curved[:, [0, 3]].reshape(-1)
        self.handle_ends = curved[:, [3, 0]].reshape(-1)
        anchors = original[self.handle_anchors]
        chord = original[self.handle_ends] - anchors
        length = chord.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        handles = original[self.handle_indices] - anchors
        along = (handles * chord).sum(-1, keepdim=True) / length.square()
        self.handle_ratio = (handles.norm(dim=-1, keepdim=True) / length).clamp_min(1)
        self.handle_low = along.clamp_max(0)
        self.handle_high = along.clamp_min(1)
        self.handle_active = (length > 1e-6) & movable[self.handle_indices].bool()
        self.initial_controls = torch.cat(self.controls_from_local(original)).detach()

    def bending_loss(self, contours):
        """Penalize handle changes that do not follow the endpoint displacement.

        This discourages fitting individual noisy pixels with tiny humps,
        while dense knots still allow the outline to bend gradually.
        """
        import torch

        delta = torch.cat(contours) - self.initial_controls
        a, b = delta[:, 0], delta[:, 3]
        expected = torch.stack(((2 * a + b) / 3, (a + 2 * b) / 3), 1)
        return (delta[:, 1:3] - expected).square().mean()

    def fairness_loss(self, contours, pixel_scale=(1.0, 1.0)):
        """Penalize actual outline wiggles, including those in the input."""
        import torch

        basis = self.original.new_tensor(
            [
                [1, 0, 0, 0],
                [0.421875, 0.421875, 0.140625, 0.015625],
                [0.125, 0.375, 0.375, 0.125],
                [0.015625, 0.140625, 0.421875, 0.421875],
            ]
        )
        losses = []
        for contour in contours:
            points = torch.einsum("tk,nkc->ntc", basis, contour).reshape(-1, 2)
            points = points * self.original.new_tensor(pixel_scale)
            edges = points.roll(-1, 0) - points
            lengths = edges.norm(dim=-1).clamp_min(1e-6)
            walked = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
            count = max(4, min(1024, int(float(walked[-1].detach())) + 1))
            distance = (
                torch.arange(count, device=points.device) / count * walked[-1].detach()
            )
            indices = torch.searchsorted(walked.detach(), distance.detach(), right=True)
            indices = (indices - 1).clamp(0, len(points) - 1)
            share = (distance - walked[indices].detach()) / lengths[indices].detach()
            samples = points[indices] + share[:, None] * edges[indices]
            edges = samples.roll(-1, 0) - samples
            lengths = edges.norm(dim=-1)
            following = edges.roll(-1, 0)
            denominator = (lengths * lengths.roll(-1, 0)).clamp_min(0.25**2)
            cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
            dot = (edges * following).sum(-1)
            # A bounded angle penalty ignores point spacing. Tiny/repeated
            # spans cannot create enormous gradients or fake turns.
            turn = (cross / denominator).square() + 4 * (
                (-dot / denominator).clamp_min(0).square()
            )
            losses.append(turn.sum() / lengths.sum().clamp_min(1))
        return torch.stack(losses).mean()

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
        local = self.original + delta * (self.displacement / length).clamp(max=1)
        for indices, free in self.coincident:
            shift = (local[indices] - self.original[indices]).mean(0) * free
            local = local.index_copy(0, indices, self.original[indices] + shift)
        anchors = local[self.handle_anchors]
        chord = local[self.handle_ends] - anchors
        length = chord.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        handles = local[self.handle_indices] - anchors
        along = (handles * chord).sum(-1, keepdim=True) / length.square()
        bounded = along.maximum(self.handle_low).minimum(self.handle_high)
        handles = handles + (bounded - along) * chord
        handles = handles * (
            self.handle_ratio
            * length
            / handles.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        ).clamp_max(1)
        values = torch.where(
            self.handle_active, anchors + handles, local[self.handle_indices]
        )
        # The relative bound must also respect the absolute movement cap.
        delta = values - self.original[self.handle_indices]
        length = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        values = self.original[self.handle_indices] + delta * (
            self.displacement / length
        ).clamp_max(1)
        local = local.index_copy(0, self.handle_indices, values)
        if not len(self.smooth):
            return local
        knot, incoming, outgoing = local[self.smooth].unbind(1)
        arm = ((knot - incoming) + (outgoing - knot) / self.arm_ratio) / 2
        old_arm = self.original[self.smooth[:, 0]] - self.original[self.smooth[:, 1]]
        # Tangents may rotate, but cannot flip or collapse into a cusp.
        along = (arm * old_arm).sum(-1, keepdim=True) / old_arm.square().sum(
            -1, keepdim=True
        )
        arm = arm + (along.clamp_min(0.1) - along) * old_arm
        joined = torch.stack((knot, knot - arm, knot + self.arm_ratio * arm), 1)
        delta = joined - self.original[self.smooth]
        distance = (
            delta.norm(dim=-1, keepdim=True).amax(1, keepdim=True).clamp_min(1e-12)
        )
        # Blend the whole knot and its two arms by one amount. Independent
        # movement clipping here would break their common tangent again.
        joined = self.original[self.smooth] + delta * (
            self.displacement / distance
        ).clamp_max(1)
        local = local.index_copy(0, self.smooth.reshape(-1), joined.reshape(-1, 2))
        local = local[self.canonical]

        # Back off only the groups touching an offending segment. A tiny
        # constrained edge must not stall every other curve in the path.
        proposed = local
        share = local.new_ones((len(local), 1))
        affected = self.coupled[
            torch.stack((self.handle_indices, self.handle_anchors, self.handle_ends), 1)
        ].reshape(-1, 1)
        for _ in range(8):
            bad = (~self._handle_valid(local)).repeat_interleave(3, dim=0)
            factors = torch.where(bad, share[affected[:, 0]] / 2, share[affected[:, 0]])
            share = share.scatter_reduce(
                0, affected, factors, reduce="amin", include_self=True
            )
            local = self.original + share[self.coupled] * (proposed - self.original)
        # Any unresolved bound returns that segment and its tied knots. Repeat
        # to account for the neighbouring chords those knot restorations change.
        for _ in range(4):
            bad = (~self._handle_valid(local)).repeat_interleave(3, dim=0)
            factors = torch.where(bad, 0, share[affected[:, 0]])
            share = share.scatter_reduce(
                0, affected, factors, reduce="amin", include_self=True
            )
            local = self.original + share[self.coupled] * (proposed - self.original)
        return torch.where(self._handle_valid(local).all(), local, self.original)

    def _handle_valid(self, local):
        anchors = local[self.handle_anchors]
        chord = local[self.handle_ends] - anchors
        length = chord.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        handles = local[self.handle_indices] - anchors
        along = (handles * chord).sum(-1, keepdim=True) / length.square()
        allowed = (
            (handles.norm(dim=-1, keepdim=True) <= self.handle_ratio * length + 1e-5)
            & (along >= self.handle_low - 1e-5)
            & (along <= self.handle_high + 1e-5)
        )
        return allowed | ~self.handle_active
