"""Fixed-topology, single-path GPU fitting in its frozen SVG context.

Cairo supplies the exact clipping, group opacity and painter-order context.
Only selected path coverage is differentiable. Candidates are accepted using
Cairo again, so renderer approximation cannot turn a worse fit into a result.
"""

from __future__ import annotations

import base64
import io
import math
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Event

import cairosvg
import numpy as np
from cairosvg.colors import color
from PIL import Image

from vectrify.document import Document, DocumentError, Selection, export_svg
from vectrify.document.editor import Transaction
from vectrify.document.hit_test import IDENTITY, multiply, transform
from vectrify.document.join import path_style


@dataclass(frozen=True)
class FitOptions:
    nodes: bool = True
    handles: bool = True
    color: bool = False
    steps: int = 8
    displacement: float = 2.0
    resolution: int = 768

    def __post_init__(self):
        if any(type(v) is not bool for v in (self.nodes, self.handles, self.color)):
            raise DocumentError("Edit permissions must be on or off")
        if not (self.nodes or self.handles or self.color):
            raise DocumentError("Allow nodes, handles or fill color to change")
        if type(self.steps) is not int or not 1 <= self.steps <= 1000:
            raise DocumentError("Choose 1 to 1000 fitting steps")
        if type(self.resolution) is not int or not 64 <= self.resolution <= 2048:
            raise DocumentError("Choose a preview resolution from 64 to 2048")
        if not math.isfinite(self.displacement) or not 0 <= self.displacement <= 100:
            raise DocumentError("Maximum movement must be between 0 and 100 SVG units")


@dataclass(frozen=True)
class FitResult:
    object_id: str
    values: dict[str, tuple[float, ...]]
    fill: str | None
    before: float
    after: float
    previews: dict[str, str]
    steps: int
    size: tuple[int, int]
    stroke: str | None = None

    @property
    def changed(self) -> bool:
        return bool(self.values or self.fill is not None)

    def write(self, tx: Transaction) -> None:
        """Record the fitted values in *tx*; committing it applies them."""
        if self.values:
            tx.update_nodes(self.object_id, self.values)
        if self.fill is not None:
            tx.set_attributes(self.object_id, {"fill": self.fill})
        if self.stroke is not None:
            tx.set_attributes(self.object_id, {"stroke": self.stroke})


def validate_selection(document: Document, selection: Selection, options: FitOptions):
    if len(selection.object_ids) != 1 or selection.whole_document:
        raise DocumentError("Select one path to optimize")
    oid = next(iter(selection.object_ids))
    element = document.element(oid)
    ancestry = document.ancestry(oid)
    if element.tag != "path" or any(a.tag in {"defs", "clipPath"} for a in ancestry):
        raise DocumentError(
            "Select a visible path; detach an instance to edit its path"
        )
    style = path_style(document, element)
    if style["fill"] == "none":
        raise DocumentError("This editor path fitter currently requires a filled path")
    if style["stroke"] != "none" and (
        color(style["stroke"]) != color(style["fill"])
        or float(style["fill-opacity"]) != 1
        or float(style["stroke-opacity"]) != 1
        or style["stroke-linejoin"] not in {"round", "miter"}
        or any(not s.closed for s in document.geometry_for(oid).subpaths)
    ):
        raise DocumentError(
            "Outlined fills need closed contours, round or miter joins, "
            "and matching opaque fill/stroke colors"
        )
    rgba = color(style["fill"])
    if rgba[3] != 1:
        raise DocumentError("Use a solid fill with fill opacity for GPU fitting")
    geometry = document.geometry_for(oid)
    if document.dependents({oid}) != {oid}:
        raise DocumentError(
            "This path is referenced elsewhere; detach it before fitting"
        )
    if options.nodes or options.handles:
        if document.geometry_users(geometry.id) != {oid}:
            raise DocumentError("Detach shared geometry before fitting its nodes")
        if any(
            m.geometry_id == geometry.id for b in document.boundaries for m in b.members
        ):
            raise DocumentError(
                "Linked boundaries must stay fixed; enable only fill color"
            )
    for ancestor in ancestry:
        if (options.nodes or options.handles) and "geometry" in ancestor.locks:
            raise DocumentError("Geometry is locked; enable only fill color")
        paint_locks = (
            {"paint", "fill", "stroke"}
            if style["stroke"] != "none"
            else {"paint", "fill"}
        )
        if options.color and (paint_locks & ancestor.locks):
            raise DocumentError("Fill color is locked")
        clip = ancestor.get("clip-path")
        if (
            clip
            and clip.startswith("url(#")
            and document.element(clip[5:-1]).get("clipPathUnits") == "objectBoundingBox"
        ):
            raise DocumentError("Fitting object-bounds clipping is not supported yet")
    return oid, geometry, style


def png_url(image: Image.Image) -> str:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


class FitContext:
    """A bounded crop and the affine compositing response of a selected fill."""

    def __init__(
        self,
        document: Document,
        selection: Selection,
        target: Image.Image,
        options: FitOptions,
    ):
        self.oid, self.geometry, self.style = validate_selection(
            document, selection, options
        )
        self.document = document
        self.root = ET.fromstring(export_svg(document))
        self.path = next(e for e in self.root.iter() if e.get("id") == self.oid)
        vx, vy, vw, vh = document.artboard()
        matrix = IDENTITY
        for ancestor in document.ancestry(self.oid):
            matrix = multiply(matrix, transform(ancestor.get("transform")))
        a, b, c, d, e, f = matrix
        self.linear = np.array([[a, c], [b, d]], dtype=np.float64)
        self.offset = np.array([e, f])
        if abs(np.linalg.det(self.linear)) < 1e-12:
            raise DocumentError("Cannot fit a path with a singular transform")
        self.nodes = [n for s in self.geometry.subpaths for n in s.nodes]
        self.local = np.array(
            [
                p
                for n in self.nodes
                for p in zip(n.values[::2], n.values[1::2], strict=True)
            ]
        )
        world = self.local @ self.linear.T + self.offset
        margin = (
            options.displacement
            + float(self.style["stroke-width"])
            / 2
            * (
                float(self.style["stroke-miterlimit"])
                if self.style["stroke-linejoin"] == "miter"
                else 1
            )
        ) * np.linalg.norm(self.linear, ord=2) + 3 * max(
            vw / target.width, vh / target.height
        )
        left, top = np.maximum(world.min(0) - margin, (vx, vy))
        right, bottom = np.minimum(world.max(0) + margin, (vx + vw, vy + vh))
        if right <= left or bottom <= top:
            raise DocumentError("The selected path is outside the artboard")
        self.crop = (left, top, right, bottom)
        width = max(1, math.ceil((right - left) * target.width / vw))
        height = max(1, math.ceil((bottom - top) * target.height / vh))
        scale = min(1, options.resolution / max(width, height))
        self.size = (max(1, round(width * scale)), max(1, round(height * scale)))
        self.scale = np.array(
            [self.size[0] / (right - left), self.size[1] / (bottom - top)]
        )
        # The UI stretches the reference over the artboard, including
        # nonzero viewBox origins.
        opaque_target = Image.alpha_composite(
            Image.new("RGBA", target.size, "white"), target.convert("RGBA")
        ).convert("RGB")
        self.target = opaque_target.resize(
            self.size,
            Image.Resampling.BICUBIC,
            box=(
                (left - vx) * target.width / vw,
                (top - vy) * target.height / vh,
                (right - vx) * target.width / vw,
                (bottom - vy) * target.height / vh,
            ),
        )
        self.root.set("viewBox", f"{left} {top} {right - left} {bottom - top}")
        self.root.set("width", str(self.size[0]))
        self.root.set("height", str(self.size[1]))
        self.root.set("preserveAspectRatio", "none")
        self.before_image = self.render()
        self.original_attrs = dict(self.path.attrib)
        self.path.set("fill", "none")
        self.path.set("stroke", "none")
        self.base = self.array(self.render())
        # Cover the whole crop in the path's local frame; keep ancestor clips,
        # opacity, isolated groups, and all later objects exactly as exported.
        corners = np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
        corners = (corners - self.offset) @ np.linalg.inv(self.linear).T
        self.path.set("d", "M" + " L".join(f"{x} {y}" for x, y in corners) + " Z")
        self.path.set("fill", "black")
        black = self.array(self.render())
        self.path.set("fill", "white")
        white = self.array(self.render())
        self.delta = black - self.base
        self.transmission = white - black
        self.path.attrib.clear()
        self.path.attrib.update(self.original_attrs)

    @staticmethod
    def array(image: Image.Image) -> np.ndarray:
        return np.asarray(image, dtype=np.float32) / 255

    def render(self) -> Image.Image:
        data = cairosvg.svg2png(
            bytestring=ET.tostring(self.root), background_color="white"
        )
        assert data is not None
        return Image.open(io.BytesIO(data)).convert("RGB")

    def candidate(self, coordinates: np.ndarray, rgb: np.ndarray, options: FitOptions):
        changes = {}
        start = 0
        for node in self.nodes:
            length = len(node.values) // 2
            values = tuple(
                float(v) for v in coordinates[start : start + length].reshape(-1)
            )
            if values != node.values:
                changes[node.id] = values
            start += length
        geometry = replace(
            self.geometry,
            subpaths=tuple(
                replace(
                    s,
                    nodes=tuple(
                        replace(n, values=changes.get(n.id, n.values)) for n in s.nodes
                    ),
                )
                for s in self.geometry.subpaths
            ),
        )
        self.path.set("d", geometry.path_data())
        fill = (
            "#" + "".join(f"{round(float(v) * 255):02x}" for v in rgb)
            if options.color
            else None
        )
        if fill is not None:
            self.path.set("fill", fill)
            if self.style["stroke"] != "none":
                self.path.set("stroke", fill)
        return changes, fill, self.render()


def fit_selected_path(
    document: Document,
    selection: Selection,
    target: Image.Image,
    options: FitOptions,
    *,
    stop: Event | None = None,
    progress: Callable[[int, str], None] | None = None,
) -> FitResult:
    """Expose the path-fit mutator's filled-path optimizer for an explicit selection."""
    import torch

    from vectrify.refine.cuda_renderer import available
    from vectrify.refine.paths import fit_filled_svg, to_path_d

    if not torch.cuda.is_available() or not available():
        raise DocumentError("GPU fitting needs CUDA and the Vectrify CUDA extension")
    stop = stop or Event()
    report = progress or (lambda _step, _message: None)
    report(0, "Preparing clipping and surrounding artwork…")
    context = FitContext(document, selection, target, options)
    original = torch.tensor(context.local, dtype=torch.float32, device="cuda")
    linear = original.new_tensor(context.linear)
    inverse = torch.linalg.inv(linear)
    offset = original.new_tensor(context.offset)
    origin = original.new_tensor(context.crop[:2])
    scale = original.new_tensor(context.scale)
    mask = []
    for node in context.nodes:
        selected = not selection.node_ids or node.id in selection.node_ids
        mask.extend([selected and options.handles] * (len(node.values) // 2 - 1))
        mask.append(selected and options.nodes and not node.pinned)
    movable = original.new_tensor(mask)[:, None]
    # Every original node keeps its ID and command. Straight edges and implicit
    # closures remain straight even though the fitter represents them as cubics.
    definitions = []
    index = 0
    for subpath in context.geometry.subpaths:
        segments = []
        head = previous = index
        index += 1
        for node in subpath.nodes[1:]:
            length = len(node.values) // 2
            segments.append((previous, tuple(range(index, index + length))))
            previous = index + length - 1
            index += length
        if previous != head:
            segments.append((previous, (head,)))
        if not segments:
            raise DocumentError("This path contains an empty contour")
        definitions.append(segments)

    def controls_from_local(local):
        points = (local @ linear.T + offset - origin) * scale
        result = []
        for segments in definitions:
            cubics = []
            for previous, following in segments:
                a = points[previous]
                if len(following) == 1:
                    b = points[following[0]]
                    cubics.append(torch.stack((a, (2 * a + b) / 3, (a + 2 * b) / 3, b)))
                else:
                    cubics.append(torch.stack((a, *(points[i] for i in following))))
            result.append(torch.stack(cubics))
        return result

    def local_from_controls(paths):
        local = original.clone()
        for segments, contour in zip(definitions, paths[0], strict=True):
            for (previous, following), cubic in zip(segments, contour, strict=True):
                values = (cubic / scale + origin - offset) @ inverse.T
                local[previous] = values[0]
                for i, point_index in enumerate(following):
                    local[point_index] = (
                        values[i + 1] if len(following) == 3 else values[3]
                    )
        delta = (local - original) * movable
        length = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return original + delta * (options.displacement / length).clamp(max=1)

    def project(paths):
        constrained = controls_from_local(local_from_controls(paths))
        for dest, source in zip(paths[0], constrained, strict=True):
            dest.copy_(source)

    work = ET.Element("svg", width=str(context.size[0]), height=str(context.size[1]))
    ET.SubElement(
        work,
        "path",
        {
            "d": " ".join(
                to_path_d(c.cpu().tolist(), precision=9) + " Z"
                for c in controls_from_local(original)
            ),
            "fill": context.style["fill"],
            "fill-rule": context.style["fill-rule"],
        },
    )

    def score(image):
        return float(
            np.mean((context.array(image) - context.array(context.target)) ** 2)
        )

    before = best = score(context.before_image)
    best_values, best_fill, best_image = {}, None, context.before_image
    completed = 0

    def observe(step, paths, colors):
        nonlocal completed, best, best_values, best_fill, best_image
        completed = step
        report(step, f"Path-fit mutator · {step}/{options.steps} GPU steps")
        if step and (step % 10 == 0 or step == options.steps or stop.is_set()):
            with torch.no_grad():
                shifts = (local_from_controls(paths) - original) * movable
                # Avoid float32 round-tripping untouched or pinned coordinates.
                coordinates = context.local + shifts.cpu().numpy()
            values, fill, image = context.candidate(
                coordinates, colors[0].detach().clamp(0, 1).cpu().numpy(), options
            )
            actual = score(image)
            if actual < best:
                best, best_values, best_fill, best_image = actual, values, fill, image
        return not stop.is_set()

    stroke_tiles = {}
    stroke_width = 0.0
    if context.style["stroke"] != "none":
        mapping = linear * scale[:, None]
        singular = torch.linalg.svdvals(mapping)
        if abs(float(singular[0] / singular[1]) - 1) > 0.02:
            raise DocumentError("Outlined path fitting requires a uniform scale")
        stroke_width = float(context.style["stroke-width"]) * float(singular.mean())
        miter = context.style["stroke-linejoin"] == "miter"
        limit = float(context.style["stroke-miterlimit"]) if miter else 1
        reach = stroke_width / 2 * limit + options.displacement * float(singular[0]) + 2
        # Fixed conservative tiles include all permitted movement. Most contours
        # in a landscape are tiny; never evaluate each one over the whole crop.
        for contour_index, contour in enumerate(controls_from_local(original)):
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
                rows = torch.arange(height, device="cuda")[:, None] + int(top)
                cols = torch.arange(width, device="cuda")[None, :] + int(left)
                pixels = (rows * context.size[0] + cols).reshape(-1)
                stroke_tiles.setdefault((width, height), []).append(
                    (contour_index, start, int(left), int(top), pixels)
                )

    def include_stroke(paths, alphas):
        from vectrify.refine.cuda_renderer import stroke_coverage
        from vectrify.refine.miter import incoming_directions, miter_stroke_coverage

        miter = context.style["stroke-linejoin"] == "miter"
        incoming = [incoming_directions(c) for c in paths[0]] if miter else []
        samples, indices = [], []
        for (width, height), tiles in stroke_tiles.items():
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
                widths = batch.new_full((len(batch),), stroke_width)
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
        return (1 - (1 - alphas[0]) * (1 - stroke))[None]

    fit_filled_svg(
        ET.tostring(work, encoding="unicode"),
        context.target,
        steps=options.steps,
        point_learning_rate=0.25 if options.nodes or options.handles else 0,
        color_learning_rate=0.01 if options.color else 0,
        xing_weight=0,
        monolithic=True,
        fit_context=(context.base, context.delta, context.transmission),
        project_controls=project,
        observe=observe,
        coverage_transform=include_stroke
        if context.style["stroke"] != "none"
        else None,
    )
    return FitResult(
        context.oid,
        best_values,
        best_fill,
        before,
        best,
        {
            "reference": png_url(context.target),
            "before": png_url(context.before_image),
            "after": png_url(best_image),
        },
        completed,
        context.size,
        stroke=best_fill if context.style["stroke"] != "none" else None,
    )
