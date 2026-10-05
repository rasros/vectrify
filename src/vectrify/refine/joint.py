"""Joint refinement of selected fills and round strokes in their SVG context.

A run of sibling paths is composited as one premultiplied layer before its
common clipping, opacity and surrounding artwork. Optimizing their changing
overlaps together avoids treating the adjacent selected paths as fixed paint.
"""

from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import replace
from threading import Event

import numpy as np
from cairosvg.colors import color

from vectrify.document import Document, DocumentError, Selection, export_svg, import_svg
from vectrify.document.join import path_style
from vectrify.document.topology import inverse_matrix, mapped_point
from vectrify.document.transforms import root_matrix
from vectrify.refine.crossings import crossings
from vectrify.refine.parameters import ControlMap
from vectrify.refine.selected import (
    FitContext,
    FitOptions,
    fit_device,
    validate_selection,
)
from vectrify.refine.shared import SAME, Link, coordinate_indices, follow, frozen_points
from vectrify.refine.simplify import curved
from vectrify.svg_render import render_image

# Bound the coverage stack and the extra subpixel stroke canvases.
# Larger selections continue through the individual fitter.
MAX_COVERAGE_PIXELS = 16 * 1024 * 1024


def _groups(document: Document, oids, options: FitOptions):
    eligible = set()
    for oid in oids:
        try:
            _, _, style = validate_selection(
                document, Selection(object_ids=frozenset({oid})), options
            )
        except DocumentError:
            continue
        ancestry = document.ancestry(oid)
        a, b, c, d, _e, _f = root_matrix(document, oid)
        if (
            abs(a * d - b * c) >= 1e-12
            and (
                (style["fill"] != "none" and style["stroke"] == "none")
                or (style["fill"] == "none" and style["stroke"] != "none")
            )
            and not ancestry[-1].get("clip-path")
            and not any(a.get("filter") or a.get("mask") for a in ancestry)
        ):
            eligible.add(oid)
    groups = []
    for parent in document.elements():
        run = []
        for child in (*parent.children, None):
            if child is not None and child.id in eligible:
                run.append(child.id)
            else:
                if len(run) > 1:
                    groups.append(tuple(run))
                run = []
    return groups


def _context(document: Document, oids, target, options: FitOptions):
    """Measure the common response by replacing the run with one opaque fill."""
    first = document.element(oids[0])
    first_inverse = inverse_matrix(root_matrix(document, first.id))
    points = []
    for oid in oids:
        matrix = root_matrix(document, oid)
        local_points = np.array(
            [
                mapped_point(mapped_point(point, matrix), first_inverse)
                for sub in document.geometry_for(oid).subpaths
                for node in sub.nodes
                for point in zip(node.values[::2], node.values[1::2], strict=True)
            ]
        )
        if not len(local_points):
            raise DocumentError("This path contains an empty contour")
        points.extend(local_points)
        style = path_style(document, document.element(oid))
        if style["stroke"] != "none":
            a, b, c, d, _e, _f = matrix
            ia, ib, ic, id_, _ie, _if = first_inverse
            relative = np.array([[ia, ic], [ib, id_]]) @ np.array([[a, c], [b, d]])
            radius = float(style["stroke-width"]) / 2 * np.linalg.norm(relative, ord=2)
            points.extend((local_points.min(0) - radius, local_points.max(0) + radius))
    points = np.asarray(points)
    left, top = points.min(0)
    right, bottom = points.max(0)
    # This rectangle is an internal measurement probe, never proposed or saved.
    geometry = import_svg(
        f'<svg><path d="M{left} {top} L{right} {top} L{right} {bottom} '
        f'L{left} {bottom} Z"/></svg>'
    ).geometries[0]
    geometry = replace(geometry, id=document.geometry_for(first.id).id)
    probe = document.replace_geometry(geometry)
    attributes = {
        "fill": "#ffffff",
        "stroke": "none",
        "stroke-width": "0",
        "fill-opacity": "1",
        "opacity": "1",
    }
    if transform := first.get("transform"):
        attributes["transform"] = transform
    probe = probe.replace_element(replace(first, attributes=tuple(attributes.items())))
    parent = document.ancestry(first.id)[-2]
    remove = set(oids[1:])
    probe = probe.replace_element(
        replace(
            probe.element(parent.id),
            children=tuple(
                c for c in probe.element(parent.id).children if c.id not in remove
            ),
        )
    )
    return FitContext(
        probe, Selection(object_ids=frozenset({first.id})), target, options
    )


class _Coordinates:
    def __init__(
        self,
        document: Document,
        oid,
        context,
        options: FitOptions,
        held,
        device,
        endpoint_only=frozenset(),
        bilateral=False,
    ):
        import torch

        self.oid = oid
        self.geometry = document.geometry_for(oid)
        self.nodes = [n for s in self.geometry.subpaths for n in s.nodes]
        self.local = np.array(
            [
                p
                for n in self.nodes
                for p in zip(n.values[::2], n.values[1::2], strict=True)
            ]
        )
        self.original = torch.tensor(self.local, dtype=torch.float32, device=device)
        a, b, c, d, e, f = root_matrix(document, oid)
        style = path_style(document, document.element(oid))
        self.stroke_only = style["fill"] == "none"
        self.stroke_width = 0.0
        if self.stroke_only:
            singular = np.linalg.svd(
                np.array([[a, c], [b, d]]) * context.scale[:, None], compute_uv=False
            )
            if abs(singular[0] / singular[1] - 1) > 0.02:
                raise DocumentError("Round stroke fitting requires a uniform scale")
            self.stroke_width = float(style["stroke-width"]) * float(singular.mean())
        linear = self.original.new_tensor([[a, c], [b, d]])
        mask = []
        for node in self.nodes:
            mask.extend(
                [options.handles and (node.id not in held or node.id in endpoint_only)]
                * (len(node.values) // 2 - 1)
            )
            mask.append(options.nodes and not node.pinned and node.id not in held)
        self.mapping = ControlMap(
            self.geometry,
            self.local,
            self.original,
            linear,
            self.original.new_tensor((e, f)),
            self.original.new_tensor(context.crop[:2]),
            self.original.new_tensor(context.scale),
            self.original.new_tensor(mask)[:, None],
            options.displacement,
            stroke_only=self.stroke_only,
        )
        self.bilateral = None
        if bilateral and self.stroke_only and options.nodes and options.handles:
            from vectrify.refine.bilateral import Bilateral

            self.bilateral = Bilateral.infer(self.geometry, held, options.displacement)

    def controls_from_local(self, local):
        if self.bilateral is not None:
            local = self.bilateral.controls(local)
        return self.mapping.controls_from_local(local)

    def geometry_at(self, local):
        # Restore untouched doubles exactly instead of round-tripping float32.
        coordinates = self.local + (local - self.original).detach().cpu().numpy()
        rows, index = {}, 0
        for node in self.nodes:
            length = len(node.values) // 2
            rows[node.id] = tuple(
                float(v) for v in coordinates[index : index + length].reshape(-1)
            )
            index += length
        geometry = replace(
            self.geometry,
            subpaths=tuple(
                replace(s, nodes=tuple(replace(n, values=rows[n.id]) for n in s.nodes))
                for s in self.geometry.subpaths
            ),
        )
        return (
            self.bilateral.geometry(geometry)
            if self.bilateral is not None
            else geometry
        )

    def local_at(self, geometry):
        by_id = {n.id: n for s in geometry.subpaths for n in s.nodes}
        if set(by_id) != {n.id for n in self.nodes} or any(
            by_id[n.id].command != n.command or len(by_id[n.id].values) != len(n.values)
            for n in self.nodes
        ):
            raise DocumentError("Shared edge changed topology during joint refinement")
        return self.original.new_tensor(
            [
                p
                for n in self.nodes
                for p in zip(
                    by_id[n.id].values[::2], by_id[n.id].values[1::2], strict=True
                )
            ]
        )


class _Within(Event):
    def __init__(self, deadline, stop):
        super().__init__()
        self.deadline, self.stop = deadline, stop

    def is_set(self):
        return (
            super().is_set() or self.stop.is_set() or time.monotonic() >= self.deadline
        )


class _SharedCoordinates:
    """One differentiable coordinate for each explicitly linked point."""

    def __init__(self, coordinates, links):
        import torch

        self.counts = [len(p.original) for p in coordinates]
        offsets = np.cumsum([0, *self.counts])
        by_id = {p.oid: i for i, p in enumerate(coordinates)}
        parents = list(range(int(offsets[-1])))
        self.shared_rows = set()

        def root(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        for link in links:
            if link.path not in by_id or link.neighbour not in by_id:
                continue
            a, b = by_id[link.path], by_id[link.neighbour]
            source = coordinate_indices(
                coordinates[a].geometry, link.subpath, link.start, link.end
            )
            target = coordinate_indices(
                coordinates[b].geometry,
                link.neighbour_subpath,
                link.neighbour_end if link.reversed else link.neighbour_start,
                link.neighbour_start if link.reversed else link.neighbour_end,
            )
            if source is None or target is None:
                continue
            if link.reversed:
                target = [s[::-1] for s in target[::-1]]
            if len(source) != len(target) or any(
                len(x) != len(y) for x, y in zip(source, target, strict=True)
            ):
                continue
            pairs = [
                (x, y)
                for s, t in zip(source, target, strict=True)
                for x, y in zip(s, t, strict=True)
            ]
            if any(
                not np.allclose(
                    coordinates[a].local[x], coordinates[b].local[y], atol=SAME, rtol=0
                )
                for x, y in pairs
            ):
                continue
            for x, y in pairs:
                self.shared_rows.update(((a, x), (b, y)))
                i, j = root(int(offsets[a]) + x), root(int(offsets[b]) + y)
                parents[max(i, j)] = min(i, j)

        self.active = any(root(i) != i for i in range(len(parents)))
        if not self.active:
            return
        self.original = torch.cat([p.original for p in coordinates])
        owners = [root(i) for i in range(len(parents))]
        self.owners = torch.tensor(
            owners, dtype=torch.long, device=self.original.device
        )
        mask = torch.cat([p.mapping.movable for p in coordinates])
        # Every permission on a shared coordinate applies to the whole class.
        class_mask = mask.clone().scatter_reduce(
            0, self.owners[:, None], mask, reduce="amin", include_self=True
        )
        self.movable = class_mask[self.owners]

    def __call__(self, values):
        import torch

        if not self.active:
            return values
        local = torch.cat(values)
        delta = (local - self.original)[self.owners] * self.movable
        return list(torch.split(self.original + delta, self.counts))


def polish(
    document: Document,
    oids,
    target,
    options: FitOptions,
    *,
    held=frozenset(),
    shared: tuple[Link, ...] = (),
    overlaps: bool = False,
    junctions=frozenset(),
    models: bool = True,
    score: Callable[[Document], float] | None = None,
    stop: Event | None = None,
    progress=None,
) -> Document:
    """Refine compatible sibling paths, keeping only exact-render improvements.

    Pinned/held endpoints, point IDs, paint and movement bounds are preserved.
    Unsupported runs retain their individual fit. The common probe is never
    part of a returned document.

    ``overlaps`` additionally permits selected layers to overlap after their
    coordinated fit. Held junctions, pins and shared runs to unselected
    neighbours remain protected. Exact rendering keeps the better candidate
    from either stage, with no change to paint or painter order.

    ``junctions`` identifies inferred holds whose endpoints must stay fixed
    but whose handles may move in that overlapping fit. Explicit region holds
    and junctions linked to unselected paths retain all their controls.

    Remaining time can try a bounded bilateral family for compatible open
    strokes. This is another exact-scored proposal; the unrestricted fit stays
    when the reference does not support the simpler family.
    """
    if not (options.nodes or options.handles) or options.displacement == 0:
        return document
    oids = tuple(oids)
    stop = stop or Event()
    selected = set(oids)
    external = tuple(
        link
        for link in shared
        if link.path not in selected or link.neighbour not in selected
    )
    if overlaps and len(external) < len(shared):
        document = polish(
            document,
            oids,
            target,
            options,
            held=held,
            shared=shared,
            models=False,
            score=score,
            stop=stop,
            progress=progress,
        )
        return polish(
            document,
            oids,
            target,
            options,
            held=held | frozen_points(document, list(external)),
            shared=external,
            junctions=junctions - frozen_points(document, list(external)),
            models=models,
            score=score,
            stop=stop,
            progress=progress,
        )
    groups = []
    for group in _groups(document, oids, options):
        # Preserve the useful fill-only refinement before letting outlines
        # move with it. Changing the group must not discard that candidate.
        if any(
            path_style(document, document.element(oid))["fill"] == "none"
            for oid in group
        ):
            run = []
            for oid in (*group, None):
                if (
                    oid is not None
                    and path_style(document, document.element(oid))["fill"] != "none"
                ):
                    run.append(oid)
                else:
                    if len(run) > 1:
                        groups.append(tuple(run))
                    run = []
        groups.append(group)
    model_seeds = {}
    for index, group in enumerate(groups):
        if stop.is_set():
            break
        now = time.monotonic()
        deadline = getattr(stop, "deadline", math.inf)
        within = _Within(now + (deadline - now) / (len(groups) - index), stop)
        try:
            initial = document
            model_seeds[group] = {oid: initial.geometry_for(oid) for oid in group}
            document = _polish_group(
                document,
                group,
                target,
                options,
                held,
                shared,
                score,
                within,
                progress,
                junctions - frozen_points(document, list(external)),
            )
            if (
                models
                and not within.is_set()
                and any(
                    path_style(initial, initial.element(oid))["fill"] == "none"
                    for oid in group
                )
            ):
                candidate = _polish_group(
                    initial,
                    group,
                    target,
                    options,
                    held,
                    shared,
                    score,
                    within,
                    progress,
                    junctions - frozen_points(initial, list(external)),
                    supports=True,
                )
                context = (
                    _context(initial, group, target, options) if score is None else None
                )
                if _exact_score(candidate, context, score) < _exact_score(
                    document, context, score
                ):
                    document = candidate
        except DocumentError:
            # Unsupported contours leave the individually verified fit intact.
            continue
    if models and options.nodes and options.handles:
        from vectrify.refine.bilateral import Bilateral

        for group in groups:
            if stop.is_set():
                break
            if group not in model_seeds:
                continue
            # Compare model families from the same selected geometry as the
            # free fit. Its endpoint changes can otherwise destroy a useful
            # correspondence before the simpler model is even considered.
            # Keep refinements to other groups in the surrounding document.
            initial = document
            for geometry in model_seeds[group].values():
                initial = initial.replace_geometry(geometry)
            if not any(
                path_style(initial, initial.element(oid))["fill"] == "none"
                and Bilateral.infer(
                    curved(initial.geometry_for(oid)), held, options.displacement
                )
                is not None
                for oid in group
            ):
                continue
            try:
                candidate = _polish_group(
                    initial,
                    group,
                    target,
                    options,
                    held,
                    shared,
                    score,
                    stop,
                    progress,
                    junctions - frozen_points(initial, list(external)),
                    bilateral=True,
                )
                context = (
                    _context(initial, group, target, options) if score is None else None
                )
                if _exact_score(candidate, context, score) < _exact_score(
                    document, context, score
                ):
                    document = candidate
                if not stop.is_set():
                    candidate = _polish_group(
                        initial,
                        group,
                        target,
                        options,
                        held,
                        shared,
                        score,
                        stop,
                        progress,
                        junctions - frozen_points(initial, list(external)),
                        bilateral=True,
                        supports=True,
                    )
                    context = (
                        _context(initial, group, target, options)
                        if score is None
                        else None
                    )
                    if _exact_score(candidate, context, score) < _exact_score(
                        document, context, score
                    ):
                        document = candidate
            except DocumentError:
                continue
    return document


def _exact_score(candidate, context, score):
    if score is not None:
        return score(candidate)
    left, top, right, bottom = context.crop
    image = render_image(
        export_svg(candidate),
        (left, top, right - left, bottom - top),
        context.size,
        alpha=context.alpha,
    )
    error = (context.array(image) - context.array(context.target)) ** 2
    return (
        float(np.mean((error[..., :3].sum(-1) + 3 * error[..., 3]) / 6))
        if context.alpha
        else float(error.mean())
    )


def _polish_group(
    document,
    group,
    target,
    options,
    held,
    shared,
    score,
    within,
    progress,
    endpoint_only,
    bilateral=False,
    supports=False,
):
    import torch

    from vectrify.refine.paths import fit_filled_svg, to_path_d

    before = document
    prepared = document
    for oid in group:
        prepared = prepared.replace_geometry(curved(prepared.geometry_for(oid)))
    try:
        context = _context(prepared, group, target, options)
    except DocumentError:
        return document
    canvases = len(group) + 16 * sum(
        path_style(prepared, prepared.element(oid))["fill"] == "none" for oid in group
    )
    if canvases * context.size[0] * context.size[1] > MAX_COVERAGE_PIXELS:
        return document
    if within.is_set():
        return document
    if progress:
        progress(0, f"Refining {len(group)} adjacent paths together…")
    coordinates = [
        _Coordinates(
            prepared,
            oid,
            context,
            options,
            held,
            fit_device(),
            endpoint_only,
            bilateral,
        )
        for oid in group
    ]
    work = ET.Element("svg", width=str(context.size[0]), height=str(context.size[1]))
    opacity = []
    for p in coordinates:
        style = path_style(prepared, prepared.element(p.oid))
        kind = "stroke" if p.stroke_only else "fill"
        paint = "#" + "".join(f"{round(v * 255):02x}" for v in color(style[kind])[:3])
        opacity.append(float(style[kind + "-opacity"]) * float(style["opacity"]))
        ET.SubElement(
            work,
            "path",
            {
                "d": " ".join(
                    to_path_d(c.cpu().tolist(), precision=9) + " Z"
                    for c in p.controls_from_local(p.original)
                ),
                "fill": paint,
                "fill-rule": style["fill-rule"],
            },
        )
    links = [link for link in shared if link.path in group or link.neighbour in group]
    shared_coordinates = _SharedCoordinates(coordinates, links)
    stroke_supports = None
    if supports:
        from vectrify.refine.stroke_support import StrokeSupports

        _vx, _vy, vw, vh = prepared.artboard()
        stroke_supports = StrokeSupports(
            coordinates,
            (target.width / vw, target.height / vh),
            shared_coordinates.shared_rows,
            getattr(within, "deadline", math.inf),
        )
        if not stroke_supports.active:
            return before

    def transformed(paths):
        local = [
            p.mapping.local_from_controls(path)
            for p, path in zip(coordinates, paths, strict=True)
        ]
        if stroke_supports is not None:
            # Apply the stroke family before its subcurves reach the fills.
            # Applying it only when building stroke coverage would leave the
            # fill copies following an unrelated unrestricted curve.
            local = [
                p.bilateral.controls(values) if p.bilateral is not None else values
                for p, values in zip(coordinates, local, strict=True)
            ]
        local = shared_coordinates(local)
        if stroke_supports is not None:
            local = stroke_supports(local)
        return [
            p.controls_from_local(values)
            for p, values in zip(coordinates, local, strict=True)
        ]

    def exact(candidate):
        return _exact_score(candidate, context, score)

    best, best_score = before, exact(before)
    valid = {p.oid: p.original.clone() for p in coordinates}
    folds = {p.oid: crossings(p.geometry) for p in coordinates}
    watched = set(group) | {link.neighbour for link in links}
    original_folds = {oid: crossings(prepared.geometry_for(oid)) for oid in watched}

    def project(paths):
        current = prepared
        for p, controls in zip(coordinates, transformed(paths), strict=True):
            current = current.replace_geometry(
                p.geometry_at(p.mapping.local_from_controls(controls))
            )
        current, _ = follow(current, links)
        for p, controls in zip(coordinates, paths, strict=True):
            # Reapply masks/bounds after neighbours have followed too.
            local = p.local_at(current.geometry_for(p.oid))
            constrained = p.controls_from_local(local)
            local = p.mapping.local_from_controls(constrained)
            for dest, source in zip(
                controls, p.controls_from_local(local), strict=True
            ):
                dest.copy_(source)

    # A narrow stroke can lie wholly within the wrong pixel. Its exact-area
    # gradient then has no indication which way its ink belongs. Compare
    # nearby pixels together first, then refine at the original resolution.
    coarse_steps = (
        min(10, options.steps // 2) if any(p.stroke_only for p in coordinates) else 0
    )
    blurred_target = None

    def blur(image):
        return torch.nn.functional.avg_pool2d(
            image.permute(2, 0, 1)[None],
            3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )[0].permute(1, 2, 0)

    def loss_transform(image, target_image, step):
        nonlocal blurred_target
        if step < coarse_steps:
            if blurred_target is None:
                blurred_target = blur(target_image)
            image, target_image = blur(image), blurred_target
        if image.shape[-1] == 4:
            # Reweighted least squares retains a useful opacity derivative
            # at narrow seams, where squared error otherwise loses it. The
            # bounded, detached weights approximate a smooth absolute loss;
            # the unchanged exact Cairo score still judges every candidate.
            # Strengthen only the reference's opaque interior: boundary
            # antialiasing and more translucent regions retain the normal loss.
            alpha_error = (image[..., 3:] - target_image[..., 3:]).detach().abs()
            uncertain = (target_image[..., 3] < 0.95).to(image.dtype)
            interior = (
                torch.nn.functional.max_pool2d(
                    uncertain[None, None], 3, stride=1, padding=1
                )[0, 0]
                == 0
            )
            weight = torch.where(
                interior[..., None],
                (0.05 + alpha_error).rsqrt(),
                1,
            )
            image = torch.cat((image[..., :3], image[..., 3:] * weight), -1)
            target_image = torch.cat(
                (target_image[..., :3], target_image[..., 3:] * weight), -1
            )
        return image, target_image

    last_score = best_score

    def observe(step, paths, _colours):
        nonlocal best, best_score, last_score
        if step and (step % 10 == 0 or step == options.steps or within.is_set()):
            current = prepared
            for p, controls in zip(coordinates, paths, strict=True):
                local = p.mapping.local_from_controls(controls)
                geometry = p.geometry_at(local)
                for _ in range(4):
                    if crossings(geometry) <= folds[p.oid]:
                        break
                    local = (local + valid[p.oid]) / 2
                    geometry = p.geometry_at(local)
                if crossings(geometry) > folds[p.oid]:
                    local = valid[p.oid]
                    geometry = p.geometry_at(local)
                valid[p.oid] = local.detach().clone()
                current = current.replace_geometry(geometry)
                with torch.no_grad():
                    for dest, source in zip(
                        controls, p.controls_from_local(local), strict=True
                    ):
                        dest.copy_(source)
            current, _ = follow(current, links)
            if all(
                crossings(current.geometry_for(oid)) <= original_folds[oid]
                for oid in watched
            ):
                actual = exact(current)
                if actual < best_score:
                    best, best_score = current, actual
                # A model can be improving through initialization while still
                # worse than the retained unrestricted fit. Judge its progress
                # independently; the best exact render still decides retention.
                change = (last_score - actual) / max(last_score, 1e-12)
                last_score = actual
                if options.stall and step > coarse_steps and change < options.stall:
                    return False
        return not within.is_set()

    def coverage(paths, alphas):
        from vectrify.refine.soft_coverage import soft_stroke_coverage

        painted = []
        for p, path, alpha, strength in zip(
            coordinates, paths, alphas, opacity, strict=True
        ):
            if p.stroke_only:
                forward = [c[:n] for c, n in zip(path, p.mapping.counts, strict=True)]
                alpha = soft_stroke_coverage(
                    forward, (0, 0, *context.size), p.stroke_width
                )
            painted.append(alpha * strength)
        return torch.stack(painted)

    try:
        fit_filled_svg(
            ET.tostring(work, encoding="unicode"),
            context.target,
            steps=options.steps,
            point_learning_rate=0.25,
            color_learning_rate=0,
            xing_weight=0,
            monolithic=True,
            fit_context=(context.base, context.delta, context.transmission),
            project_controls=project,
            control_transform=transformed,
            observe=observe,
            coverage_transform=coverage,
            loss_transform=loss_transform,
            device=fit_device(),
        )
    except DocumentError:
        # A shared run can change topology during following. The individual
        # fit remains valid; keep any already verified joint improvement.
        return best
    return best
