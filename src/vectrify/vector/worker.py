"""Worker processes for the node search: apply one move and render the result.

A worker keeps the region's SVG parsed and, per task, only rewrites the
selected paths' contours and stroke widths before rendering. A task is seeded
by the main process, so a run repeats exactly. Scoring happens in the main
process, so every score in a run comes from one scorer.
"""

from __future__ import annotations

import copy
import io
import logging
import random
import signal
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from vectrify.vector.nodes import MOVES, Frozen, Paths

log = logging.getLogger(__name__)
SVG_NS = "http://www.w3.org/2000/svg"

_DRAWABLE = {
    "path",
    "rect",
    "circle",
    "ellipse",
    "line",
    "polyline",
    "polygon",
    "use",
    "text",
    "image",
}
# What paints only where something else refers to it.
_NOT_PAINTED = {"defs", "clipPath", "mask", "pattern", "marker", "symbol"}
# How many unchanging shapes a drawing needs before rendering it in slices
# pays: on 9 shapes a render took 2.0 ms whole and 2.5 ms sliced, on 533
# shapes 154 ms whole and 1.0 ms sliced.
SLICE_ABOVE = 24
# Group effects that apply to their contents as one: a layer rendered on its
# own would come out differently.
_GROUP_EFFECTS = {"opacity", "clip-path", "mask", "filter"}


@dataclass(frozen=True)
class WorkerContext:
    """What a worker needs, sent once when it starts."""

    # The drawing with its viewBox on the region, as region_svg writes it.
    svg: str
    size: tuple[int, int]
    fixed: Frozen


@dataclass(frozen=True)
class Mutant:
    """One task's outcome. No state means the move found nothing to change."""

    move: str
    state: Paths | None = None
    # The child's render: RGB, uint8, height x width x 3.
    image: np.ndarray | None = None


def render_rgba(svg: str, size: tuple[int, int]) -> np.ndarray:
    """Render *svg* to premultiplied RGBA floats in [0, 1], without a PNG.

    Encoding and decoding PNGs took most of a render's time: this reads
    cairo's surface directly, which comes out identical.
    """
    from cairosvg.parser import Tree
    from cairosvg.surface import PNGSurface

    width, height = size
    surface = PNGSurface(
        Tree(bytestring=svg.encode("utf-8")),
        io.BytesIO(),
        96,
        output_width=width,
        output_height=height,
    )
    cairo = surface.cairo
    assert cairo is not None
    cairo.flush()
    pixels = np.frombuffer(cairo.get_data(), np.uint8).reshape(
        cairo.get_height(), cairo.get_stride() // 4, 4
    )[:height, :width]
    # Cairo stores BGRA in native (little-endian) order.
    return pixels[..., [2, 1, 0, 3]].astype(np.float32) / 255.0


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


class Renderer:
    """Renders a state by rewriting only the selected paths of one SVG.

    Everything that does not change is rendered once, in slices between the
    selected paths, and each render then paints only the selected paths and
    composites the slices around them. A drawing with hundreds of shapes then
    costs about as much per render as the selected paths do. Where a selected
    path sits inside a group whose opacity, clip, mask or filter applies to
    the group as a whole, or is drawn again elsewhere by a <use>, slicing
    would change the picture, so the whole drawing is rendered instead.
    """

    def __init__(self, context: WorkerContext):
        ET.register_namespace("", SVG_NS)
        self.size = context.size
        self.root = ET.fromstring(context.svg)
        self.selected = set(context.fixed.step)
        self.elements = self._by_id(self.root)
        self.layers: list[np.ndarray | str] | None = None
        self.solo: dict[str, tuple[ET.Element, ET.Element]] = {}
        # Below a few dozen other shapes compositing costs more than it saves.
        others = len(self._painted(self.root)) - len(self.selected)
        if others > SLICE_ABOVE and self._can_slice():
            self._slice()

    def _by_id(self, root: ET.Element) -> dict[str, ET.Element]:
        return {e.get("id", ""): e for e in root.iter() if e.get("id") in self.selected}

    def _painted(self, root: ET.Element) -> list[ET.Element]:
        """Drawables in paint order, leaving out definitions."""
        out: list[ET.Element] = []

        def walk(element: ET.Element) -> None:
            for child in element:
                tag = _local(child.tag)
                if tag in _NOT_PAINTED:
                    continue
                if tag in _DRAWABLE:
                    out.append(child)
                else:
                    walk(child)

        walk(root)
        return out

    def _can_slice(self) -> bool:
        if set(self.elements) != self.selected:
            return False
        parents = {child: parent for parent in self.root.iter() for child in parent}
        for element in self.elements.values():
            ancestor = parents.get(element)
            while ancestor is not None and ancestor is not self.root:
                style = ancestor.get("style", "")
                if any(ancestor.get(a) for a in _GROUP_EFFECTS) or any(
                    f"{a}:" in style.replace(" ", "") for a in _GROUP_EFFECTS
                ):
                    return False
                ancestor = parents.get(ancestor)
        hrefs = {
            (e.get("href") or e.get("{http://www.w3.org/1999/xlink}href") or "")
            for e in self.root.iter()
        }
        return not any(f"#{oid}" in hrefs for oid in self.selected)

    def _keeping(self, keep: set[int]) -> ET.Element:
        """A copy of the drawing painting only the drawables at *keep*.

        The others are removed rather than hidden: cairosvg parses hidden
        elements too, and parsing is most of what a large drawing costs.
        """
        root = copy.deepcopy(self.root)
        parents = {child: parent for parent in root.iter() for child in parent}
        for index, element in enumerate(self._painted(root)):
            if index not in keep:
                parents[element].remove(element)
        return root

    def _slice(self) -> None:
        painted = self._painted(self.root)
        layers: list[np.ndarray | str] = []
        pending: set[int] = set()

        def flush() -> None:
            if pending:
                svg = ET.tostring(self._keeping(set(pending)), encoding="unicode")
                layers.append(render_rgba(svg, self.size))
                pending.clear()

        for index, element in enumerate(painted):
            oid = element.get("id", "")
            if oid in self.selected:
                flush()
                solo = self._keeping({index})
                self.solo[oid] = (solo, self._by_id(solo)[oid])
                layers.append(oid)
            else:
                pending.add(index)
        flush()
        self.layers = layers

    def __call__(self, state: Paths) -> np.ndarray:
        if self.layers is None:
            for oid, geometry in state.geometries.items():
                self.elements[oid].set("d", geometry.path_data())
            for oid, width in state.strokes.items():
                self.elements[oid].set("stroke-width", repr(width))
            rgba = render_rgba(ET.tostring(self.root, encoding="unicode"), self.size)
            canvas = rgba[..., :3] + (1.0 - rgba[..., 3:4])
        else:
            width, height = self.size
            canvas = np.ones((height, width, 3), dtype=np.float32)
            for layer in self.layers:
                if isinstance(layer, str):
                    root, element = self.solo[layer]
                    element.set("d", state.geometries[layer].path_data())
                    if layer in state.strokes:
                        element.set("stroke-width", repr(state.strokes[layer]))
                    svg = ET.tostring(root, encoding="unicode")
                    layer = render_rgba(svg, self.size)
                canvas = layer[..., :3] + canvas * (1.0 - layer[..., 3:4])
        return np.clip(np.rint(canvas * 255.0), 0, 255).astype(np.uint8)


_context: WorkerContext | None = None
_render: Renderer | None = None


def init_worker(context: WorkerContext) -> None:
    global _context, _render
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _context, _render = context, Renderer(context)


def mutate(parent: Paths, move: str, seed: int) -> Mutant:
    """Apply *move* to *parent* and render the child."""
    assert _context is not None, "init_worker has not run"
    assert _render is not None
    try:
        child = MOVES[move](parent, random.Random(seed), _context.fixed)
        if child is None or child.key() == parent.key():
            return Mutant(move)
        return Mutant(move, child, _render(child))
    except Exception as exc:
        log.debug(f"Move {move} failed: {exc!r}")
        return Mutant(move)
