"""Bounded source-only fitting of an assembled editable band interpretation.

Material decomposition, original-source caps and underpaint are assembled before
fitting, so every crop evaluation sees their complete painted context. Geometry
moves normal to the physical port chord; original ports and supported corners
stay fixed. The original painted/gap bank and actual native stroke body decide
whether the result can become a competitor. Ownership and the common objective
still require independent validation by the enclosing planner.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from xml.etree import ElementTree as ET

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

from vectrify.document import Editor, Selection, export_svg
from vectrify.document.join import path_style, transformed_geometry
from vectrify.document.redraw import root_matrix
from vectrify.document.svg import parse_path
from vectrify.document.topology import inverse_matrix
from vectrify.refine import cel
from vectrify.refine.cel_plan.filled_bands import MAX_WIDTH, _check
from vectrify.refine.cel_plan.ink_replace import identified
from vectrify.refine.cel_plan.local import Box, _native_raster
from vectrify.refine.cel_plan.refine import _bounds
from vectrify.refine.cel_plan.score import composite, render
from vectrify.refine.cel_plan.source_absence import SourceAbsence
from vectrify.refine.crossings import crossings
from vectrify.refine.tracing import _loops

MAX_EVALUATIONS = 600
MAX_PARAMETERS = 24
MAX_MOVEMENT = 2
MAX_CORE_NODES = 1024


def crop(svg, bounds):
    root = ET.fromstring(svg)
    x, y, right, bottom = bounds
    root.set("viewBox", f"{x} {y} {right - x} {bottom - y}")
    root.set("width", str(right - x))
    root.set("height", str(bottom - y))
    return root


def painted_context(document, bounds, work, *, keep=()):
    """Prune only entire independent paths whose paint cannot enter the crop.

    Clip/reference/effect dependencies retain the whole context. Every candidate
    crop is independently checked against full native pixels before publication.
    Nodes, gradient bounds, relative paint order and parent opacity stay exact.
    """
    remove = set()
    for element in document.elements():
        _check(work)
        if (
            element.tag != "path"
            or element.id in keep
            or any(
                a.tag == "defs"
                or any(
                    a.get(key, "none") != "none"
                    for key in ("clip-path", "filter", "mask")
                )
                for a in document.ancestry(element.id)
            )
            or document.dependents({element.id}) != frozenset((element.id,))
        ):
            continue
        style = path_style(document, element)
        if style["stroke"] != "none" and float(style["stroke-miterlimit"]) > 10:
            continue
        x, y, right, bottom = _bounds(document, element.id)
        if x > bounds[2] or y > bounds[3] or right < bounds[0] or bottom < bounds[1]:
            remove.add(element.id)
    root = ET.fromstring(export_svg(document))
    for parent in root.iter():
        for child in tuple(parent):
            if child.get("id") in remove:
                parent.remove(child)
    _check(work)
    return root


def opaque_core(document, oid, footprint, size, work):
    """A conservative native material core with the replacement stroke hidden.

    Native pixel cells at the common parent's opacity bound added paint. This
    is only a proposal bound; complete native alpha must still be checked.
    """
    _check(work)
    frame = root_matrix(document, oid)
    from vectrify.document.join import curve_path

    nb = np.asarray(curve_path(transformed_geometry(footprint, frame)).bounds)
    lo = np.maximum(0, np.floor(nb[:2]).astype(int) - 2)
    hi = np.minimum(size, np.ceil(nb[2:]).astype(int) + 2)
    bounds = tuple(map(int, (*lo, *hi)))
    if np.any(hi <= lo) or np.prod(hi - lo) > 16_384:
        return None
    root = ET.fromstring(export_svg(document))
    stroke = next(element for element in root.iter() if element.get("id") == oid)
    stroke.set("stroke", "none")
    actual = _native_raster(root, size).crop(Box(*bounds))
    # Cairo rounds opacity at each group boundary. An analytic product can
    # exceed every fully covered native pixel (0.75 becomes 191/255), or
    # disagree across nested groups. Measure the identical opacity stack on
    # one fully covered pixel; this bounds proposals, not alpha acceptance.
    swatch = ET.Element("svg", {"width": "1", "height": "1"})
    parent = swatch
    for ancestor in document.ancestry(oid)[:-1]:
        parent = ET.SubElement(parent, "g", {"opacity": ancestor.get("opacity", "1")})
    ET.SubElement(parent, "rect", {"width": "1", "height": "1", "fill": "white"})
    opacity = float(_native_raster(swatch, (1, 1)).root[0, 0, 3]) / 255
    if opacity <= 1 / 255:
        return None
    loops = _loops(actual[..., 3] >= opacity - 1e-7)
    _check(work)
    if not loops or sum(len(loop) for loop in loops) > MAX_CORE_NODES:
        return None
    data = " ".join(
        "M" + "L".join(f"{float(x + lo[0])} {float(y + lo[1])}" for x, y in loop) + "Z"
        for loop in loops
    )
    return transformed_geometry(parse_path(data), inverse_matrix(frame))


class BandFit:
    def __init__(self, evidence, guard):
        self.evidence, self.guard = evidence, guard
        # One parameter vector only. No rasters, documents or source graphs are
        # retained. Complete native/body proofs run again for every sibling.
        self._cached = None

    def fit(
        self,
        before,
        assembled,
        oid,
        seed,
        work,
        *,
        width_fixed=False,
        preserve_alpha=False,
    ):
        """Fit one complete candidate atomically; never publish a partial fit.

        Crop penalties propose useful parameters. Complete native painted and
        white-body source checks independently validate the final interpretation.
        All retained material geometry and paint remain outside parameter space.
        """
        _check(work)
        frame = root_matrix(assembled, oid)
        linear = np.array(frame[:4]).reshape(2, 2).T
        gram = linear.T @ linear
        scale = float(np.sqrt(gram[0, 0]))
        if scale <= 1e-12 or not np.allclose(
            gram, np.eye(2) * scale**2, rtol=1e-8, atol=1e-10
        ):
            return None
        original = seed.band.geometry
        if len(original.subpaths) != 1 or original.subpaths[0].closed:
            return None
        nodes = original.subpaths[0].nodes
        delta = np.asarray(nodes[-1].endpoint) - nodes[0].endpoint
        length = float(np.linalg.norm(delta))
        if length < 8 or len(nodes) > 8:
            return None
        normal = np.array((-delta[1], delta[0])) / length
        corners = {tuple(seed.anchors[i]) for i in cel.run_corners(seed.anchors, False)}
        variables = []
        for i, node in enumerate(nodes):
            for j in range(0, len(node.values), 2):
                endpoint = j == len(node.values) - 2
                if i == 0 or (
                    endpoint
                    and (
                        i == len(nodes) - 1
                        or node.command == "L"
                        or node.endpoint in corners
                    )
                ):
                    continue
                variables.append((i, j))
        if len(variables) + int(not width_fixed) > MAX_PARAMETERS:
            return None
        bounds = seed.bounds
        size = (bounds[2] - bounds[0], bounds[3] - bounds[1])
        source = self.evidence.rgba[bounds[1] : bounds[3], bounds[0] : bounds[2]]
        lines = self.guard.fitting_crop(bounds, work=work)
        if lines is None:
            return None
        baseline = _native_raster(
            painted_context(before, bounds, work), self.evidence.source_size
        ).crop(Box(*bounds))
        original_baseline = _native_raster(
            ET.fromstring(export_svg(before)), self.evidence.source_size
        ).crop(Box(*bounds))
        if not np.array_equal(baseline, original_baseline):
            return None
        seen = lines.observe(baseline, work=work)
        root = painted_context(assembled, bounds, work, keep=(oid,))
        assembled_pixels = _native_raster(root, self.evidence.source_size).crop(
            Box(*bounds)
        )
        if not np.array_equal(
            assembled_pixels,
            _native_raster(
                ET.fromstring(export_svg(assembled)), self.evidence.source_size
            ).crop(Box(*bounds)),
        ):
            return None
        stroke = next(element for element in root.iter() if element.get("id") == oid)
        gaps = self.guard.gap_centres(limit=4096, work=work)
        inside = ((gaps >= bounds[:2]) & (gaps < bounds[2:])).all(axis=1)
        queries = gaps[inside] - bounds[:2] - 0.5
        visible = source[..., 3] > 1 / 255
        if not visible.any():
            return None
        target = composite(source)
        key = hashlib.sha256()
        for value in (baseline, assembled_pixels, source, seed.anchors, gaps):
            key.update(value.tobytes())
        key.update(
            repr(
                (
                    seed.profile,
                    original.path_data(),
                    seed.band.width,
                    width_fixed,
                    preserve_alpha,
                    tuple(frame),
                    sorted(corners),
                )
            ).encode()
        )
        cache_key = key.digest()
        best_value = float("inf")
        best_parameters: np.ndarray | None = None
        evaluations = 0

        def shape(values):
            updated = [list(node.values) for node in nodes]
            for amount, (i, j) in zip(values[: len(variables)], variables, strict=True):
                updated[i][j : j + 2] = (
                    np.asarray(nodes[i].values[j : j + 2]) + amount * normal
                )
            return replace(
                original,
                subpaths=(
                    replace(
                        original.subpaths[0],
                        nodes=tuple(
                            replace(node, values=tuple(map(float, values)))
                            for node, values in zip(nodes, updated, strict=True)
                        ),
                    ),
                ),
            )

        def score(values):
            nonlocal evaluations, best_value, best_parameters
            _check(work)
            evaluations += 1
            geometry = shape(values)
            width = seed.band.width if width_fixed else float(values[-1])
            stroke.set(
                "d", transformed_geometry(geometry, inverse_matrix(frame)).path_data()
            )
            stroke.set("stroke-width", str(width / scale))
            actual = _native_raster(root, self.evidence.source_size).crop(Box(*bounds))
            _check(work)
            penalty = lines.penalty(seen, actual, work=work)
            body = render(
                f'<svg width="{size[0]}" height="{size[1]}" '
                f'viewBox="{bounds[0]} {bounds[1]} {size[0]} {size[1]}">'
                f'<path d="{geometry.path_data()}" fill="none" stroke="white" '
                f'stroke-width="{width}" '
                'stroke-linecap="butt" stroke-linejoin="round"/></svg>',
                size,
            )[..., 3]
            alpha = map_coordinates(
                body, [queries[:, 1], queries[:, 0]], order=1, mode="constant", cval=0
            )
            penalty += float(np.maximum(0, alpha - 1 / 255).sum()) * 1000
            value = (
                float(((composite(actual) - target)[visible] ** 2).mean() * 255**2)
                + penalty
            )
            alpha_exact = np.array_equal(actual[..., 3], baseline[..., 3])
            if preserve_alpha:
                # Guide the bounded search toward the parent's native silhouette,
                # but only retain exactly feasible vectors. A penalty alone can
                # still choose an attractive stroke spilling past the silhouette.
                value += float(np.abs(actual[..., 3] - baseline[..., 3]).sum()) * 255e6
            if (not preserve_alpha or alpha_exact) and value < best_value:
                best_value = value
                best_parameters = np.array(values, copy=True)
            _check(work)
            return value

        values = np.r_[
            np.zeros(len(variables)), [] if width_fixed else [seed.band.width]
        ]
        limits: list[tuple[float, float]] = [
            (-float(MAX_MOVEMENT), float(MAX_MOVEMENT))
        ] * len(variables)
        if not width_fixed:
            limits.append(
                (max(0.8, seed.band.width * 0.5), min(MAX_WIDTH, seed.band.width * 1.5))
            )
        reused = self._cached is not None and self._cached[0] == cache_key
        if reused:
            assert self._cached is not None
            _, initial, best_value, best_parameters = self._cached
        else:
            initial = score(values)
        if len(values) and not reused:
            minimize(
                score,
                values,
                method="Powell",
                bounds=limits,
                options={
                    "maxfev": MAX_EVALUATIONS - 1,
                    "maxiter": 12,
                    "xtol": 0.05,
                    "ftol": 1e-4,
                },
            )
        _check(work)
        if best_parameters is None:
            return None
        fitted = shape(best_parameters)
        width = seed.band.width if width_fixed else float(best_parameters[-1])
        if crossings(fitted):
            return None
        absence = SourceAbsence(self.evidence, (), work, guard=self.guard)
        if not absence.permits(fitted, width, "butt", work):
            return None
        editor = Editor(assembled, selection=Selection(whole_document=True))
        with editor.transaction("Fit source-supported editable outline") as tx:
            _check(work)
            tx.replace_geometry(
                oid,
                identified(transformed_geometry(fitted, inverse_matrix(frame)), oid),
            )
            tx.set_attributes(oid, {"stroke-width": repr(width / scale)})
        _check(work)
        document = editor.snapshot.document
        native = render(export_svg(document), self.evidence.source_size)
        if preserve_alpha and not np.array_equal(
            native[..., 3],
            render(export_svg(before), self.evidence.source_size)[..., 3],
        ):
            return None
        stroke.set("d", document.geometry_for(oid).path_data())
        stroke.set("stroke-width", repr(width / scale))
        if not np.array_equal(
            native[bounds[1] : bounds[3], bounds[0] : bounds[2]],
            _native_raster(root, self.evidence.source_size).crop(Box(*bounds)),
        ):
            return None
        comparison = self.guard.compare(
            render(export_svg(before), self.evidence.source_size),
            native,
            work=work,
        )
        if not comparison["qualified_samples"] or comparison["rejections"]:
            return None
        _check(work)
        stored = np.array(best_parameters, copy=True)
        stored.flags.writeable = False
        self._cached = (cache_key, initial, best_value, stored)
        return document, {
            "profile": seed.profile,
            "evaluations": evaluations,
            "reused_fit": reused,
            "initial_loss": initial,
            "fitted_loss": best_value,
            "width": width,
            "parameters": best_parameters.tolist(),
            "movement": MAX_MOVEMENT,
            "native_body_absence": True,
            **({"native_alpha_exact": True} if preserve_alpha else {}),
            "source_line_comparison": comparison,
            "source_ports": [list(nodes[0].endpoint), list(nodes[-1].endpoint)],
        }
