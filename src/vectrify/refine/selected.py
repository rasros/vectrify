"""Fixed-topology, single-path gradient fitting in its frozen SVG context.

Cairo supplies the exact clipping, group opacity and painter-order context.
Only selected path coverage is differentiable. Candidates are accepted using
Cairo again, so renderer approximation cannot turn a worse fit into a result.
No candidate crosses itself more than the path did: where the fit folds the
outline, the nodes of the crossing segments are pulled back and it goes on.
The fit runs on CUDA with the native analytic kernel when it can, and otherwise
on the CPU with portable polyline coverage for fills and round strokes.
Miter outlines require CUDA.
"""

from __future__ import annotations

import functools
import math
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Event

import numpy as np
from cairosvg.colors import color
from PIL import Image

from vectrify.document import (
    Document,
    DocumentError,
    Geometry,
    Selection,
    export_svg,
)
from vectrify.document.editor import Transaction
from vectrify.document.join import path_style
from vectrify.document.transforms import ancestry_matrix, root_matrix
from vectrify.image_utils import on_white, preview_urls
from vectrify.refine.crossings import crossed_nodes, crossings
from vectrify.svg_render import cached_rendering, render_image

# How many times the nodes of crossing segments are moved halfway back before
# they go all the way back, and then the whole outline does.
HALVINGS = 2


@dataclass(frozen=True)
class FitOptions:
    nodes: bool = True
    handles: bool = True
    color: bool = False
    steps: int = 8
    displacement: float = 2.0
    resolution: int = 768
    # Stop once a check (every tenth step) improves the previous check by less
    # than this share of it: the fit has stalled. 0 runs every step.
    stall: float = 0.0
    # Compare a bounded edge-seeking proposal alongside gradient fitting.
    snap: bool = True
    # Tidy additionally tests short bridges against the exact reference.
    cleanup: bool = False

    def __post_init__(self):
        if any(
            type(v) is not bool
            for v in (self.nodes, self.handles, self.color, self.snap, self.cleanup)
        ):
            raise DocumentError("Edit permissions must be on or off")
        if not (self.nodes or self.handles or self.color):
            raise DocumentError("Allow nodes, handles or fill color to change")
        if type(self.steps) is not int or not 1 <= self.steps <= 1000:
            raise DocumentError("Choose 1 to 1000 fitting steps")
        if type(self.resolution) is not int or not 64 <= self.resolution <= 2048:
            raise DocumentError("Choose a preview resolution from 64 to 2048")
        if not math.isfinite(self.stall) or not 0 <= self.stall < 1:
            raise DocumentError("The stall share must be from 0 to below 1")
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
    # How many times the outline had crossed itself more than it started and
    # the fit pulled it back.
    folded: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.values or self.fill is not None or self.stroke is not None)

    def write(self, tx: Transaction) -> None:
        """Record the fitted values in *tx*; committing it applies them."""
        if self.values:
            tx.update_nodes(self.object_id, self.values)
        if self.fill is not None:
            tx.set_attributes(self.object_id, {"fill": self.fill})
        if self.stroke is not None:
            tx.set_attributes(self.object_id, {"stroke": self.stroke})


@functools.cache
def gpu_problem() -> str | None:
    """Why GPU fitting cannot run on this machine, or None when it can."""
    try:
        import torch
    except ImportError:
        return "GPU fitting needs PyTorch with CUDA and the Vectrify CUDA extension"
    from vectrify.refine.cuda_renderer import available

    if not torch.cuda.is_available():
        return "GPU fitting needs an NVIDIA GPU with CUDA"
    if not available():
        return "GPU fitting needs the Vectrify CUDA extension"
    return None


@functools.cache
def fit_problem() -> str | None:
    """Why path fitting cannot run at all here, or None when it can."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return "Path fitting needs PyTorch"
    return None


def fit_device() -> str:
    """CUDA when the native kernel can run there, otherwise the CPU."""
    return "cuda" if gpu_problem() is None else "cpu"


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
    stroke_only = style["fill"] == "none"
    if stroke_only and style["stroke"] == "none":
        raise DocumentError("This editor path fitter currently requires a filled path")
    if options.color and "url(" in style["fill"] + style["stroke"]:
        raise DocumentError("Gradient fills are fitted with Fit colours")
    if stroke_only and (
        style["stroke-linejoin"] != "round" or style["stroke-linecap"] != "round"
    ):
        raise DocumentError("Stroke-only fitting currently needs round caps and joins")
    if (
        not stroke_only
        and style["stroke"] != "none"
        and (
            color(style["stroke"]) != color(style["fill"])
            or float(style["fill-opacity"]) != 1
            or float(style["stroke-opacity"]) != 1
            or style["stroke-linejoin"] not in {"round", "miter"}
            or any(not s.closed for s in document.geometry_for(oid).subpaths)
        )
    ):
        raise DocumentError(
            "Outlined fills need closed contours, round or miter joins, "
            "and matching opaque fill/stroke colors"
        )
    if (
        style["stroke"] != "none"
        and style["stroke-linejoin"] == "miter"
        and gpu_problem()
    ):
        raise DocumentError("Fitting an outlined shape needs an NVIDIA GPU")
    rgba = color(style["stroke"] if stroke_only else style["fill"])
    if rgba[3] != 1:
        raise DocumentError("Use a solid fill with fill opacity for path fitting")
    geometry = document.geometry_for(oid)
    if document.dependents({oid}) != {oid}:
        raise DocumentError(
            "This path is referenced elsewhere; detach it before fitting"
        )
    if (options.nodes or options.handles) and document.geometry_users(geometry.id) != {
        oid
    }:
        raise DocumentError("Detach shared geometry before fitting its nodes")
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


class FitContext:
    """A bounded crop and the compositing response of a selected path.

    Transparent references include opacity alongside colour over white.
    """

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
        self.stroke_only = self.style["fill"] == "none"
        self.document = document
        self.root = ET.fromstring(export_svg(document))
        self.path = next(e for e in self.root.iter() if e.get("id") == self.oid)
        vx, vy, vw, vh = document.artboard()
        matrix = root_matrix(document, self.oid)
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
        self.alpha = target.has_transparency_data and bool(
            np.asarray(target.convert("RGBA").getchannel("A")).min() < 255
        )
        self.target = (
            target.convert("RGBA") if self.alpha else on_white(target)
        ).resize(
            self.size,
            Image.Resampling.BICUBIC,
            box=(
                (left - vx) * target.width / vw,
                (top - vy) * target.height / vh,
                (right - vx) * target.width / vw,
                (bottom - vy) * target.height / vh,
            ),
        )
        # Paths that paint nowhere near the crop draw nothing in it: left
        # out, each of the many renders below costs a fraction.
        pruned(self.root, document, self.crop, keep=self.oid)
        self.root.set("viewBox", f"{left} {top} {right - left} {bottom - top}")
        self.root.set("width", str(self.size[0]))
        self.root.set("height", str(self.size[1]))
        self.root.set("preserveAspectRatio", "none")
        self.before_image = self.render()
        self.original_attrs = dict(self.path.attrib)
        self.path.set("fill", "none")
        self.path.set("stroke", "none")
        self.base = self.array(self.render())
        if self.stroke_only:
            self.path.set("fill-opacity", self.style["stroke-opacity"])
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

    def array(self, image: Image.Image) -> np.ndarray:
        if not self.alpha:
            return np.asarray(image, dtype=np.float32) / 255
        rgb = np.asarray(on_white(image), dtype=np.float32) / 255
        opacity = np.asarray(image.convert("RGBA").getchannel("A"), dtype=np.float32)
        return np.dstack((rgb, opacity / 255))

    def render(self) -> Image.Image:
        return render_image(ET.tostring(self.root), alpha=self.alpha)

    def reshaped(self, coordinates: np.ndarray):
        """(the changed node values, the whole geometry) for *coordinates*."""
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
        return changes, geometry

    def candidate(self, geometry: Geometry, rgb: np.ndarray, options: FitOptions):
        self.path.set("d", geometry.path_data())
        fill = (
            "#" + "".join(f"{round(float(v) * 255):02x}" for v in rgb)
            if options.color
            else None
        )
        if fill is not None:
            if not self.stroke_only:
                self.path.set("fill", fill)
            if self.style["stroke"] != "none":
                self.path.set("stroke", fill)
        return fill, self.render()


# Elements that draw what others refer to, or change how what is in them is
# drawn beyond their own outline: nothing in or under them is left out.
_KEPT = frozenset({"defs", "clipPath", "mask", "pattern", "symbol", "marker"})


def pruned(root: ET.Element, document: Document, crop, keep: str) -> None:
    """Take out of the exported *root* every path of *document* (but
    *keep*) whose painted bounds, its controls' hull widened by its stroke,
    miss the box *crop* (left, top, right, bottom, in root units): it draws
    nothing there. Referenced paths, those under a filter or with markers,
    and anything in defs, clips, masks or patterns stay."""
    left, top, right, bottom = crop
    text = ET.tostring(root, encoding="unicode")
    referenced = set(re.findall(r"url\(#([^)]+)\)", text)) | set(
        re.findall(r'href="#([^"]+)"', text)
    )
    gone = set()
    for element in document.elements():
        if element.tag != "path" or element.id == keep or element.id in referenced:
            continue
        ancestry = document.ancestry(element.id)
        if any(
            a.tag in _KEPT
            or a.get("filter")
            or a.get("marker-start")
            or a.get("marker-mid")
            or a.get("marker-end")
            for a in ancestry
        ):
            continue
        values = [
            v
            for sub in document.geometry_for(element.id).subpaths
            for n in sub.nodes
            for v in n.values
        ]
        if not values:
            continue
        matrix = ancestry_matrix(ancestry)
        a, b, c, d, e, f = matrix
        x, y = np.asarray(values[::2], float), np.asarray(values[1::2], float)
        wx, wy = a * x + c * y + e, b * x + d * y + f
        style = path_style(document, element)
        reach = 0.0
        if style["stroke"] != "none":
            try:
                width = float(style["stroke-width"])
                limit = float(style["stroke-miterlimit"])
            except ValueError:
                continue
            reach = (
                width / 2 * max(1.0, limit) * float(np.linalg.norm([[a, c], [b, d]], 2))
            )
        reach += 1.0
        if (
            wx.max() + reach < left
            or wx.min() - reach > right
            or wy.max() + reach < top
            or wy.min() - reach > bottom
        ):
            gone.add(element.id)
    if not gone:
        return
    for parent in list(root.iter()):
        for child in list(parent):
            if child.get("id") in gone:
                parent.remove(child)


@cached_rendering()
def fit_selected_path(
    document: Document,
    selection: Selection,
    target: Image.Image,
    options: FitOptions,
    *,
    stop: Event | None = None,
    progress: Callable[[int, str], None] | None = None,
    corners: frozenset[str] = frozenset(),
) -> FitResult:
    """Expose the path-fit mutator's filled-path optimizer for an explicit selection."""
    problem = fit_problem()
    if problem:
        raise DocumentError(problem)
    import torch

    from vectrify.refine.paths import fit_filled_svg, to_path_d

    stop = stop or Event()
    report = progress or (lambda _step, _message: None)
    report(0, "Preparing clipping and surrounding artwork…")
    context = FitContext(document, selection, target, options)
    device = fit_device()
    original = torch.tensor(context.local, dtype=torch.float32, device=device)
    linear = original.new_tensor(context.linear)
    offset = original.new_tensor(context.offset)
    origin = original.new_tensor(context.crop[:2])
    scale = original.new_tensor(context.scale)
    mask = []
    for node in context.nodes:
        selected = not selection.node_ids or node.id in selection.node_ids
        mask.extend([selected and options.handles] * (len(node.values) // 2 - 1))
        mask.append(selected and options.nodes and not node.pinned)
    movable = original.new_tensor(mask)[:, None]
    seed = original
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
        region = Region(left, top, right - left, bottom - top, on_white(context.target))
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
        delta = (original.new_tensor(rows) - original) * movable
        distance = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        seed = original + delta * (options.displacement / distance).clamp(max=1)
    from vectrify.refine.parameters import ControlMap

    parameters = ControlMap(
        context.geometry,
        context.local,
        original,
        linear,
        offset,
        origin,
        scale,
        movable,
        options.displacement,
        stroke_only=context.stroke_only,
        corners=corners,
    )
    counts = parameters.counts
    controls_from_local = parameters.controls_from_local

    def local_from_controls(paths):
        return parameters.local_from_controls(paths[0])

    def project(paths):
        constrained = controls_from_local(local_from_controls(paths))
        for dest, source in zip(paths[0], constrained, strict=True):
            dest.copy_(source)

    work = ET.Element("svg", width=str(context.size[0]), height=str(context.size[1]))
    # The low-level fitter reads six-digit colours. Editor paths can also use
    # CSS names, short hex or rgb(), all already validated as solid paint.
    working_fill = "#" + "".join(
        f"{round(channel * 255):02x}"
        for channel in color(
            context.style["stroke"] if context.stroke_only else context.style["fill"]
        )[:3]
    )
    ET.SubElement(
        work,
        "path",
        {
            "d": " ".join(
                to_path_d(c.cpu().tolist(), precision=9) + " Z"
                for c in controls_from_local(original)
            ),
            "fill": working_fill,
            "fill-rule": context.style["fill-rule"],
        },
    )

    def score(image):
        pixels = context.array(image)
        error = (pixels - context.array(context.target)) ** 2
        if protected is not None:
            error += (
                8
                * protected[..., None]
                * (pixels - context.array(context.before_image)) ** 2
            )
        if context.alpha:
            # Mean RGB error and opacity error each contribute half the score.
            return float(np.mean((error[:, :, :3].sum(-1) + 3 * error[:, :, 3]) / 6))
        return float(np.mean(error))

    from vectrify.refine.enclosure import protected_pixels
    from vectrify.refine.snap import _Frame

    protected = protected_pixels(
        document,
        context.oid,
        context.geometry,
        _Frame(
            context.linear * context.scale[:, None],
            (context.offset - context.crop[:2]) * context.scale,
        ),
        context.size,
    )
    loss_transform = None
    if protected is not None:
        held_pixels = original.new_tensor(protected)[..., None].bool()
        weights = torch.where(held_pixels, 3.0, 1.0)
        previous_pixels = None

        def loss_transform(rendered, goal, _step):
            nonlocal previous_pixels
            if previous_pixels is None:
                # Keep the renderer's own starting coverage. Comparing its
                # soft antialiasing to Cairo here would bias a stationary edge.
                previous_pixels = rendered.detach().clone()
            return rendered * weights, torch.where(
                held_pixels, previous_pixels, goal
            ) * weights

    before = best = checked = score(context.before_image)
    best_values, best_fill, best_image = {}, None, context.before_image
    completed = folded = 0
    folds = crossings(context.geometry)
    node_index = {node.id: i for i, node in enumerate(context.nodes)}
    # The node each row of coordinates belongs to.
    owners = np.repeat(
        np.arange(len(context.nodes)), [len(n.values) // 2 for n in context.nodes]
    )
    coupled = parameters.coupled.cpu().numpy()
    # Where the outline last stood without crossing itself more than at first.
    unfolded = context.local

    def unfold(coordinates):
        """*coordinates* with the nodes of any new crossing moved back towards
        where they last were without it; (coordinates, values, geometry)."""
        values, geometry = context.reshaped(coordinates)
        for attempt in range(HALVINGS + 2):
            count, crossed = crossed_nodes(geometry) if values else (0, set())
            if count <= folds:
                break
            back = np.isin(owners, [node_index[i] for i in crossed])
            # A rollback must keep a smooth knot and both arms together too.
            back = np.isin(coupled, coupled[back])
            if attempt > HALVINGS:
                back[:] = True
            share = 0.5 if attempt < HALVINGS else 1.0
            coordinates = coordinates.copy()
            coordinates[back] += (unfolded[back] - coordinates[back]) * share
            values, geometry = context.reshaped(coordinates)
        return coordinates, values, geometry

    # Keep edge-seeking as a candidate within this fit. Gradient descent starts
    # from the original geometry so a poor heuristic cannot trap it in a worse
    # basin; exact rendering chooses between both approaches.
    if seed is not original:
        with torch.no_grad():
            bounded_seed = local_from_controls([controls_from_local(seed)])
        coordinates = context.local + (bounded_seed - original).cpu().numpy()
        _, values, geometry = unfold(coordinates)
        paint = color(
            context.style["stroke"] if context.stroke_only else context.style["fill"]
        )
        fill, image = context.candidate(geometry, paint[:3], options)
        actual = score(image)
        if actual < best:
            best, best_values, best_fill, best_image = actual, values, fill, image

    def observe(step, paths, colors):
        nonlocal completed, folded, best, best_values, best_fill, best_image
        nonlocal unfolded, checked
        completed = step
        report(step, f"Fitting path · step {step}/{options.steps}")
        if step and (step % 10 == 0 or step == options.steps or stop.is_set()):
            with torch.no_grad():
                shifts = (local_from_controls(paths) - original) * movable
                # Avoid float32 round-tripping untouched or pinned coordinates.
                reached = context.local + shifts.cpu().numpy()
            # An outline folding over itself more than it started is no fit,
            # however well it covers the reference. The nodes of the segments
            # that cross go back, and the fit carries on from there with the
            # rest of the outline where it got to.
            coordinates, values, geometry = unfold(reached)
            if coordinates is not reached:
                folded += 1
                with torch.no_grad():
                    back = controls_from_local(original.new_tensor(coordinates))
                    for dest, source in zip(paths[0], back, strict=True):
                        dest.copy_(source)
            unfolded = coordinates
            fill, image = context.candidate(
                geometry, colors[0].detach().clamp(0, 1).cpu().numpy(), options
            )
            actual = score(image)
            # A strong edge proposal must not stop the gradient fit before it
            # has had time to improve on its own starting geometry.
            stalled = actual > checked * (1 - options.stall)
            checked = actual
            if actual < best:
                best, best_values, best_fill, best_image = actual, values, fill, image
            if step and options.stall and stalled:
                return False
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
        for contour_index, contour in (
            enumerate(controls_from_local(original)) if device == "cuda" else []
        ):
            contour = contour[: counts[contour_index]]
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
                rows = torch.arange(height, device=device)[:, None] + int(top)
                cols = torch.arange(width, device=device)[None, :] + int(left)
                pixels = (rows * context.size[0] + cols).reshape(-1)
                stroke_tiles.setdefault((width, height), []).append(
                    (contour_index, start, int(left), int(top), pixels)
                )

    def include_stroke(paths, alphas):
        if device == "cpu":
            from vectrify.refine.soft_coverage import soft_stroke_coverage

            contours = [c[:n] for c, n in zip(paths[0], counts, strict=True)]
            stroke = soft_stroke_coverage(contours, (0, 0, *context.size), stroke_width)
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
        return (
            stroke[None]
            if context.stroke_only
            else (1 - (1 - alphas[0]) * (1 - stroke))[None]
        )

    fit_filled_svg(
        ET.tostring(work, encoding="unicode"),
        context.target,
        steps=options.steps,
        point_learning_rate=0.25 if options.nodes or options.handles else 0,
        color_learning_rate=0.01 if options.color else 0,
        # The Xing term only sees a cubic's own handles crossing; the
        # folds a fit makes are mostly neighbouring segments crossing at a
        # node, which observe undoes, and the term did not reduce them.
        xing_weight=0,
        monolithic=True,
        fit_context=(context.base, context.delta, context.transmission),
        project_controls=project,
        control_transform=lambda paths: [
            controls_from_local(local_from_controls(paths))
        ],
        control_loss=(lambda paths: 0.001 * parameters.bending_loss(paths[0]))
        if not context.stroke_only
        else None,
        loss_transform=loss_transform,
        observe=observe,
        coverage_transform=include_stroke
        if context.style["stroke"] != "none"
        else None,
        device=device,
    )
    if (
        options.cleanup
        and options.nodes
        and options.handles
        and not context.stroke_only
    ):
        from vectrify.refine.cleanup import cleaned
        from vectrify.refine.snap import _Frame

        held = frozenset(
            n.id
            for n in context.nodes
            if (selection.node_ids and n.id not in selection.node_ids)
            or n.pinned
            or n.feature is not None
        )
        geometry = replace(
            context.geometry,
            subpaths=tuple(
                replace(
                    s,
                    nodes=tuple(
                        replace(n, values=best_values.get(n.id, n.values))
                        for n in s.nodes
                    ),
                )
                for s in context.geometry.subpaths
            ),
        )
        paint = color(context.style["fill"])
        bridge_folds = crossings(geometry)
        original_values = {n.id: n.values for n in context.nodes}

        def accept_bridge(candidate):
            nonlocal best, best_values, best_image, bridge_folds
            count = crossings(candidate)
            if count > bridge_folds:
                return False
            rgb = color(best_fill)[:3] if best_fill is not None else paint[:3]
            _, image = context.candidate(candidate, np.array(rgb), options)
            actual = score(image)
            if actual >= best:
                return False
            best, best_image = actual, image
            bridge_folds = count
            best_values = {
                n.id: n.values
                for s in candidate.subpaths
                for n in s.nodes
                if n.values != original_values[n.id]
            }
            return True

        vx, vy, vw, vh = document.artboard()
        pixels = np.diag([target.width / vw, target.height / vh])
        cleaned(
            geometry,
            context.geometry,
            _Frame(pixels @ context.linear, pixels @ (context.offset - [vx, vy])),
            options.displacement,
            held,
            accept_bridge,
            stop.is_set,
        )
    return FitResult(
        context.oid,
        best_values,
        None if context.stroke_only else best_fill,
        before,
        best,
        preview_urls(context.target, context.before_image, best_image),
        completed,
        context.size,
        stroke=best_fill if context.style["stroke"] != "none" else None,
        folded=folded,
    )
