"""Worker processes for the node search: apply one move and render the result.

A worker keeps the region's SVG parsed and, per task, only rewrites the
selected paths' contours and stroke widths before rendering. A task is seeded
by the main process, so a one-worker run repeats exactly. Scoring happens in
the main process, so every score in a run comes from one scorer.
"""

from __future__ import annotations

import logging
import random
import signal
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from vectrify.image_utils import rasterize_svg_to_png_bytes
from vectrify.vector.nodes import MOVES, Frozen, Paths

log = logging.getLogger(__name__)
SVG_NS = "http://www.w3.org/2000/svg"


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
    png: bytes | None = None


class Renderer:
    """Renders a state by rewriting only the selected paths of one SVG."""

    def __init__(self, context: WorkerContext):
        ET.register_namespace("", SVG_NS)
        self.root = ET.fromstring(context.svg)
        self.size = context.size
        wanted = set(context.fixed.step)
        self.elements = {
            element.get("id"): element
            for element in self.root.iter()
            if element.get("id") in wanted
        }

    def __call__(self, state: Paths) -> bytes:
        for oid, geometry in state.geometries.items():
            self.elements[oid].set("d", geometry.path_data())
        for oid, width in state.strokes.items():
            self.elements[oid].set("stroke-width", repr(width))
        return rasterize_svg_to_png_bytes(
            ET.tostring(self.root, encoding="unicode"),
            out_w=self.size[0],
            out_h=self.size[1],
        )


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
