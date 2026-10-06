"""Bounded CPU fitting with whole-drawing acceptance and retained checkpoints.

Small reference crops generate proposals; only the frontier's fixed native
policy can accept them. This is the CPU foundation, not the later local beam
or optional differentiable joint fitter. No Torch import is needed here.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np
from PIL import Image

from vectrify.document import (
    Document,
    Editor,
    Geometry,
    Selection,
    export_svg,
    import_svg,
)
from vectrify.document.join import path_style
from vectrify.document.paint import LinearGradient
from vectrify.document.redraw import root_matrix
from vectrify.operations.generate import Region
from vectrify.operations.methods.colours import (
    _flat,
    _terms,
    fit_ramp,
    local_gradient,
)
from vectrify.refine import shared
from vectrify.refine.cel_plan.frontier import Entry, Frontier
from vectrify.refine.cel_plan.model import Evidence, Options, Work
from vectrify.refine.cel_plan.score import composite
from vectrify.refine.frozen import Frozen, Paths
from vectrify.refine.simplify import simplify
from vectrify.refine.snap import snap

MAX_PATH_NODES = 512
MAX_GEOMETRY_NODES = 32_000
PAINT_SIDE = 64


def _paint(document: Document, oid: str, fill: str | LinearGradient) -> Document:
    """Use normal paint ownership and remove replaced private definitions."""
    editor = Editor(document, selection=Selection(whole_document=True))
    with editor.transaction("Fit planned paint") as transaction:
        transaction.set_fill(oid, fill)
    return editor.snapshot.document


def _paths(document: Document) -> list[str]:
    return [
        element.id
        for element in document.elements()
        if element.tag == "path"
        and not any(a.tag == "defs" for a in document.ancestry(element.id))
    ]


def _bounds(document: Document, oid: str) -> tuple[float, float, float, float]:
    """Conservative control-hull bounds, including transformed stroke width."""
    a, b, c, d, e, f = root_matrix(document, oid)
    points = np.array(
        [
            node.values[i : i + 2]
            for subpath in document.geometry_for(oid).subpaths
            for node in subpath.nodes
            for i in range(0, len(node.values), 2)
        ]
    ).reshape(-1, 2)
    if not len(points):
        return (0, 0, 0, 0)
    linear = np.array([[a, c], [b, d]])
    points = points @ linear.T + (e, f)
    style = path_style(document, document.element(oid))
    width = float(style["stroke-width"]) if style["stroke"] != "none" else 0
    # Include miter paint in the proposal crop, even at acute corners. Exact
    # full validation later includes all layers and all native pixels.
    pad = 4 + width * float(np.linalg.norm(linear, ord=2)) * 5
    low, high = points.min(axis=0) - pad, points.max(axis=0) + pad
    return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))


def _region(
    document: Document, oid: str, image: Image.Image, *, downsample: bool = False
) -> Region | None:
    left, top, right, bottom = _bounds(document, oid)
    x, y = max(0, math.floor(left)), max(0, math.floor(top))
    end_x, end_y = (
        min(image.width, math.ceil(right)),
        min(image.height, math.ceil(bottom)),
    )
    if end_x <= x or end_y <= y:
        return None
    crop = image.crop((x, y, end_x, end_y))
    if downsample:
        crop.thumbnail((PAINT_SIDE, PAINT_SIDE), Image.Resampling.LANCZOS)
    return Region(x, y, end_x - x, end_y - y, crop)


def _anchors(document: Document, oid: str) -> frozenset[str]:
    """Open endpoints, pins and supported sharp joins in native coordinates."""
    a, b, c, d, _e, _f = root_matrix(document, oid)
    linear = np.array([[a, c], [b, d]])
    held: set[str] = set()
    for subpath in document.geometry_for(oid).subpaths:
        nodes = list(subpath.nodes)
        held.update(node.id for node in nodes if node.pinned)
        if not nodes:
            continue
        if not subpath.closed:
            held.update((nodes[0].id, nodes[-1].id))
        for i, node in enumerate(nodes):
            if not subpath.closed and i in (0, len(nodes) - 1):
                continue
            previous, following = nodes[i - 1], nodes[(i + 1) % len(nodes)]
            point = np.array(node.endpoint)
            incoming = point - (
                np.array(node.values[-4:-2])
                if node.command == "C"
                else np.array(previous.endpoint)
            )
            outgoing = (
                np.array(following.values[:2])
                if following.command == "C"
                else np.array(following.endpoint)
            ) - point
            incoming, outgoing = linear @ incoming, linear @ outgoing
            lengths = np.linalg.norm(incoming) * np.linalg.norm(outgoing)
            if lengths > 1e-8 and float(incoming @ outgoing) / lengths < math.cos(
                math.pi / 4
            ):
                held.add(node.id)
    return frozenset(held)


def _held_intact(before: Document, after: Document, held: Frozen, oids) -> bool:
    for oid in oids:
        old = {
            node.id: node.endpoint
            for subpath in before.geometry_for(oid).subpaths
            for node in subpath.nodes
            if node.id in held.endpoints
        }
        new = {
            node.id: node.endpoint
            for subpath in after.geometry_for(oid).subpaths
            for node in subpath.nodes
        }
        if any(new.get(key) != point for key, point in old.items()):
            return False
    return True


def _geometry_proposal(
    document: Document,
    oid: str,
    image: Image.Image,
    options: Options,
    work: Work,
    constraints: frozenset[str],
    *,
    move: bool,
) -> Document | None:
    if oid in constraints:
        return None
    region = _region(document, oid, image)
    if region is None:
        return None
    oids = _paths(document)
    links = shared.links(document, [oid], oids)
    neighbours = {link.neighbour for link in links}
    held = Frozen(
        shared.frozen_points(document, links)
        | frozenset(
            node for other in {oid, *neighbours} for node in _anchors(document, other)
        )
    )
    paths = Paths({oid: document.geometry_for(oid)})
    # Existing CPU kernels poll deadlines inside their node loops. Limit each
    # spatial opportunity so a compound path cannot consume all fitting time.
    deadline = min(work.deadline, time.monotonic() + 0.075)
    if move:
        fitted = snap(document, paths, region, held, deadline=deadline, long_side=128)
    else:
        fitted = simplify(
            document, paths, region, held, options.boundary_tolerance, deadline
        )
    geometry: Geometry = fitted.geometries[oid]
    if geometry == paths.geometries[oid]:
        return None
    proposed, changed = shared.follow(document.replace_geometry(geometry), links)
    if changed & constraints:
        return None
    if not _held_intact(document, proposed, held, {oid, *changed}):
        return None
    if not all(shared.intact(proposed, link) for link in links):
        return None
    return proposed


class _Session:
    def __init__(
        self, frontier: Frontier, entry: Entry, complexity: int, work: Work, limit: int
    ):
        self.frontier, self.entry = frontier, entry
        self.document = import_svg(entry.svg)
        self.complexity, self.work, self.limit = complexity, work, limit
        self.attempts = self.accepted = 0
        self.validation_seconds = 0.0
        self.last_validation = 0.0
        self.stages: dict[str, dict[str, int]] = {}

    @property
    def available(self) -> bool:
        return (
            not self.work.interrupted
            and self.attempts < self.limit
            and self.work.remaining > max(0.02, self.last_validation * 1.5)
        )

    def accept(self, proposed: Document | None, stage: str, oid: str) -> None:
        if proposed is None or proposed == self.document or not self.available:
            return
        svg = export_svg(proposed)
        if not self.available:
            return
        self.attempts += 1
        counts = self.stages.setdefault(stage, {"attempted": 0, "accepted": 0})
        counts["attempted"] += 1
        before = self.entry.evaluation.objective(
            self.complexity,
            self.frontier.normalizer,
            self.frontier.policy.weights.detail,
        )
        started = time.monotonic()
        accepted = self.frontier.refine(
            svg,
            f"CPU {stage} at complexity {self.complexity}",
            {
                **self.entry.details,
                "refined": True,
                "refinement_stage": stage,
                "refinement_object": oid,
            },
            complexity=self.complexity,
            before=before,
        )
        self.last_validation = time.monotonic() - started
        self.validation_seconds += self.last_validation
        if accepted:
            self.entry = next(
                entry for entry in self.frontier.entries if entry.svg == svg
            )
            self.document = proposed
            self.accepted += 1
            counts["accepted"] += 1

    def paint(self, oid: str, image: Image.Image, options: Options, stage: str) -> None:
        if not self.available:
            return
        style = path_style(self.document, self.document.element(oid))
        if style["fill"] == "none":
            return
        region = _region(self.document, oid, image, downsample=True)
        if region is None:
            return
        # Black/white render probes temporarily detach this fill's private
        # ownership. Retained proposals use normal paint transactions.
        probe = self.document
        for element in self.document.elements():
            if element.paint_owner == oid:
                probe = probe.replace_element(replace(element, paint_owner=None))
        terms = _terms(probe, oid, region)
        if terms is None or not self.available:
            return
        dark, coverage = terms
        target = np.asarray(region.image, dtype=np.float64) / 255
        self.accept(
            _paint(self.document, oid, _flat(dark, coverage, target)), stage, oid
        )
        if options.gradients and self.available:
            # Terms describe the same geometry and context even if its flat
            # paint changed. Ramp fitting is bounded by a 64px proposal crop.
            ramp = fit_ramp(dark, coverage, target, region)
            if ramp is not None and not ramp.flat() and self.available:
                self.accept(
                    _paint(
                        self.document, oid, local_gradient(self.document, oid, ramp)
                    ),
                    stage,
                    oid,
                )


def refine(
    frontier: Frontier, evidence: Evidence, options: Options, work: Work
) -> dict:
    """Refine fixed anchors into the common frontier, with CPU only dependencies."""
    started = time.monotonic()
    if not options.refine:
        return {"status": "disabled", "attempted": 0, "accepted": 0, "seconds": 0.0}
    if work.interrupted:
        return {"status": "interrupted", "attempted": 0, "accepted": 0, "seconds": 0.0}
    image = Image.fromarray(
        (composite(evidence.rgba) * 255).round().clip(0, 255).astype(np.uint8)
    )
    limit = {"fast": 16, "balanced": 48, "high": 128}[options.quality]
    seeds = frontier.seeds({"fast": 1, "balanced": 2, "high": 3}[options.quality])
    sessions = []
    visited = 0
    bounded = 0
    complete = True
    for seed_index, (seed, complexity) in enumerate(seeds):
        if work.interrupted:
            complete = False
            break
        # Seed choice and objective are independent of the requested slider.
        # Later selection on the common refined frontier stays monotonic.
        allowance = max(
            1, (limit - sum(s.attempts for s in sessions)) // (len(seeds) - seed_index)
        )
        seed_work = Work(
            min(
                work.deadline,
                time.monotonic() + work.remaining / (len(seeds) - seed_index),
            ),
            work.stop,
            work.timings,
        )
        session = _Session(frontier, seed, complexity, seed_work, allowance)
        fitting_options = replace(options, complexity=complexity)
        sessions.append(session)
        constraints = frozenset(seed.details.get("geometry_constraints", ()))
        oids = _paths(session.document)
        total_nodes = sum(
            len(s.nodes)
            for oid in oids
            for s in session.document.geometry_for(oid).subpaths
        )
        # Deterministic spatial order gives separate components an opportunity.
        oids.sort(key=lambda oid: (*_bounds(session.document, oid)[:2], oid))
        stages = ("simplify", "paint", "geometry", "width", "paint-refit")
        for stage_index, stage in enumerate(stages):
            # Give paint refitting and width their own opportunities even if
            # one difficult geometry stage exhausts its slice.
            session.work = Work(
                min(
                    seed_work.deadline,
                    time.monotonic()
                    + seed_work.remaining / (len(stages) - stage_index),
                ),
                work.stop,
                work.timings,
            )
            session.limit = min(
                allowance,
                session.attempts
                + max(1, (allowance - session.attempts) // (len(stages) - stage_index)),
            )
            for oid in oids:
                if not session.available:
                    complete = False
                    break
                visited += 1
                if stage in {"paint", "paint-refit"}:
                    session.paint(oid, image, fitting_options, stage)
                elif stage == "width":
                    # Explicit widths remain fixed throughout fitting.
                    if options.line_width > 0:
                        continue
                    style = path_style(session.document, session.document.element(oid))
                    if style["stroke"] == "none":
                        continue
                    original_width = float(style["stroke-width"])
                    for ratio in (0.95, 1.05):
                        element = session.document.element(oid)
                        attrs = dict(element.attributes)
                        attrs["stroke-width"] = str(original_width * ratio)
                        session.accept(
                            session.document.replace_element(
                                replace(element, attributes=tuple(attrs.items()))
                            ),
                            stage,
                            oid,
                        )
                else:
                    nodes = sum(
                        len(s.nodes)
                        for s in session.document.geometry_for(oid).subpaths
                    )
                    if total_nodes > MAX_GEOMETRY_NODES or nodes > MAX_PATH_NODES:
                        bounded += 1
                        complete = False
                        continue
                    proposed = _geometry_proposal(
                        session.document,
                        oid,
                        image,
                        fitting_options,
                        session.work,
                        constraints,
                        move=stage == "geometry",
                    )
                    session.accept(proposed, stage, oid)
            if seed_work.interrupted or session.attempts >= allowance:
                complete = False
                break
        if sum(s.attempts for s in sessions) >= limit:
            complete = False
            break
    elapsed = time.monotonic() - started
    work.timings["refinement"] = elapsed
    return {
        "status": "interrupted"
        if work.interrupted
        else "completed"
        if complete
        else "bounded",
        "backend": "cpu",
        "attempted": sum(s.attempts for s in sessions),
        "accepted": sum(s.accepted for s in sessions),
        "visited": visited,
        "bounded_geometry": bounded,
        "anchors": len(sessions),
        "seconds": elapsed,
        "validation_seconds": sum(s.validation_seconds for s in sessions),
        "stages": [s.stages for s in sessions],
    }
