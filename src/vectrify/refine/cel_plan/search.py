"""Bounded beam edits, fixed native crop scores and independent checkpoints."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from vectrify.document import Document, export_svg, import_svg
from vectrify.document.join import path_style
from vectrify.document.model import paint_server
from vectrify.refine.cel_plan.frontier import Entry, Frontier
from vectrify.refine.cel_plan.local import (
    HALO,
    MAX_CROP_PIXELS,
    MAX_TILES,
    Box,
    LocalLimitError,
    LocalPolicy,
    Snapshot,
    tile_boxes,
)
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.refine import _bounds
from vectrify.refine.cel_plan.score import SCORE_VERSION, representation

LIMITS = {"fast": (1, 16), "balanced": (4, 48), "high": (8, 128)}
EXPANSIONS = {"fast": 4, "balanced": 8, "high": 16}
ANCHORS = (50, 0, 100, 25, 75)
MAX_BYTES = 64 * 1024 * 1024
MAX_REJECTIONS = 256
MAX_SEED_NODES = 32_000
MAX_SEED_OBJECTS = 8_192
MAX_EDIT_OBJECTS = 256


@dataclass(frozen=True)
class State:
    document: Document
    svg: str
    snapshot: Snapshot
    key: str
    details: dict
    edits: tuple[dict, ...] = ()
    partition: Partition | None = None


@dataclass(frozen=True)
class Proposal:
    operator: str
    ids: tuple[str, ...]
    parameters: tuple
    parent: str
    document: Document
    bounds: Box
    estimate: float = 0
    details: dict | None = None
    dependencies: tuple[str, ...] = ()
    partition: Partition | None = None


def _revision(document: Document, oid: str) -> str:
    parts: list[object] = []
    try:
        for element in document.ancestry(oid):
            parts.append((element.id, element.attributes, element.geometry_id))
            if element.geometry_id is not None:
                parts.append(document.geometry_for(element.id))
            for attribute in ("fill", "stroke", "clip-path"):
                server = paint_server(element.get(attribute))
                if server is not None:
                    parts.append(document.element(server))
                    parts.extend(
                        document.geometry_for(child.id)
                        for child in Document(document.element(server)).elements()
                        if child.geometry_id is not None
                    )
    except ValueError:
        parts.append((oid, "absent"))
    return hashlib.sha256(repr(parts).encode()).hexdigest()


class Rejections:
    """Bounded proofs keyed by dependencies, visible context and score state.

    Independent color edits need not invalidate a rejection. Global feature,
    alpha and edge-count aggregates do affect acceptance and remain in its key.
    No mutable geometry, canvas or human target is retained in this cache.
    """

    def __init__(self):
        self.values: OrderedDict[str, tuple[str, ...]] = OrderedDict()
        self.indices: OrderedDict[int, tuple[Document, list]] = OrderedDict()

    def dependencies(self, state: State, box: Box):
        # Include occluded shapes too: an alpha/geometry edit can reveal paint
        # that is absent from the current visible pixel hash. Document order
        # among intersecting objects is a visibility dependency.
        document = state.document
        key = id(document)
        if key not in self.indices:
            records = []
            for element in document.elements():
                if element.tag not in {
                    "path",
                    "use",
                    "rect",
                    "ellipse",
                    "circle",
                    "line",
                } or any(
                    a.tag in {"defs", "clipPath"} for a in document.ancestry(element.id)
                ):
                    continue
                style = path_style(document, element)
                if element.tag == "path" and float(style["stroke-miterlimit"]) <= 5:
                    left, top, right, bottom = _bounds(document, element.id)
                    native = Box(
                        int(left // 1),
                        int(top // 1),
                        int(-(-right // 1)),
                        int(-(-bottom // 1)),
                    )
                else:
                    # The generator emits ordinary paths. Imported primitives,
                    # uses and long miters keep conservative visibility support.
                    native = Box(
                        0,
                        0,
                        state.snapshot.canvas.root.shape[1],
                        state.snapshot.canvas.root.shape[0],
                    )
                records.append((element.id, native, _revision(document, element.id)))
            self.indices[key] = (document, records)
        self.indices.move_to_end(key)
        while len(self.indices) > 8:
            self.indices.popitem(last=False)
        return tuple(
            (oid, revision)
            for oid, native, revision in self.indices[key][1]
            if native.intersection(box).area
        )

    def key(self, state: State, proposal: Proposal, delta: float) -> str:
        snapshot = state.snapshot
        context = proposal.bounds.expand(2 * HALO, snapshot.canvas.root.shape)
        signature = (
            SCORE_VERSION,
            proposal.operator,
            proposal.ids,
            proposal.parameters,
            proposal.bounds,
            tuple(
                _revision(state.document, oid)
                for oid in (*proposal.ids, *proposal.dependencies)
            ),
            self.dependencies(
                state, proposal.bounds.expand(2 * HALO, snapshot.canvas.root.shape)
            ),
            delta,
            snapshot.features,
            snapshot.holes,
            snapshot.retained,
            snapshot.opacity,
            bool(snapshot.observed),
            tuple(
                snapshot.evaluation.terms[key]
                for key in (
                    "interior_missing_pixels",
                    "outside_spill_pixels",
                    "opacity_missing_pixels",
                    "opacity_excess_pixels",
                )
            ),
        )
        digest = hashlib.sha256(repr(signature).encode())
        for chunk in context.chunks(MAX_CROP_PIXELS):
            digest.update(snapshot.canvas.values(chunk).tobytes())
        return digest.hexdigest()

    def visible_ids(self, state: State, proposal: Proposal) -> frozenset[str]:
        box = proposal.bounds.expand(2 * HALO, state.snapshot.canvas.root.shape)
        # Existing operators edit only their declared paths. Newly changed
        # paths stay eligible even when their old hull lay outside the crop.
        visible = {oid for oid, _revision in self.dependencies(state, box)}
        visible.update(proposal.ids)
        return frozenset(visible)

    def get(self, key):
        if key not in self.values:
            return None
        self.values.move_to_end(key)
        return self.values[key]

    def put(self, key, reasons):
        self.values[key] = tuple(reasons)
        self.values.move_to_end(key)
        while len(self.values) > MAX_REJECTIONS:
            self.values.popitem(last=False)


def _bound(states: list[State], frontier: Frontier, width: int) -> list[State]:
    chosen = []
    keys = set()
    for complexity in ANCHORS:
        best = min(
            states,
            key=lambda state: (
                state.snapshot.evaluation.objective(
                    complexity, frontier.normalizer, frontier.policy.weights.detail
                ),
                state.snapshot.evaluation.cost,
                tuple(
                    (edit["operator"], tuple(edit["ids"]), repr(edit["parameters"]))
                    for edit in state.edits
                ),
                state.key,
            ),
        )
        if best.key not in keys:
            chosen.append(best)
            keys.add(best.key)
        if len(chosen) == width:
            break
    for state in sorted(
        states,
        key=lambda state: (
            state.snapshot.evaluation.cost,
            state.snapshot.evaluation.visual,
            state.key,
        ),
    ):
        if len(chosen) == width:
            break
        if state.key not in keys:
            chosen.append(state)
            keys.add(state.key)
    return chosen


def _bytes(states: list[State]) -> int:
    arrays = {}
    for state in states:
        canvas = state.snapshot.canvas
        arrays[id(canvas.root)] = canvas.root.nbytes
        for patch in canvas.patches:
            arrays[id(patch.pixels)] = patch.pixels.nbytes
    return sum(arrays.values()) + sum(len(state.svg.encode()) for state in states)


def search(
    frontier: Frontier,
    options: Options,
    work: Work,
    proposals: Callable[[State, Work], Iterator[Proposal]],
) -> dict:
    """Accept individual local improvements; publish only complete checkpoints."""
    started = time.monotonic()
    width, limit = LIMITS[options.quality]
    decisions = []
    attempted = accepted = cached = bounded = scanned = bounded_expansions = 0
    if work.interrupted:
        return {"status": "interrupted", "attempted": 0, "accepted": 0, "seconds": 0.0}
    if not frontier.normalizer_fixed:
        raise ValueError("Freeze the detailed cost scale before local search")
    entry: Entry = frontier.seeds(1)[0][0]
    if (
        entry.evaluation.structure["nodes"] > MAX_SEED_NODES
        or entry.evaluation.structure["paths"]
        + entry.evaluation.structure.get("primitive_objects", 0)
        > MAX_SEED_OBJECTS
        or len(entry.svg.encode())
        + frontier.policy.truth.shape[0] * frontier.policy.truth.shape[1] * 4
        > MAX_BYTES
    ):
        return {
            "status": "bounded",
            "attempted": 0,
            "accepted": 0,
            "bounded_seeds": 1,
            "seconds": time.monotonic() - started,
        }
    evaluator = LocalPolicy(frontier.policy)
    try:
        initial = State(
            import_svg(entry.svg),
            entry.svg,
            evaluator.start(entry.svg, entry.evaluation),
            entry.key,
            {
                **entry.details,
                "search_budget": {
                    "anchor": 50,
                    "representation_target": frontier.budget(50)["nominal_target"],
                    "node_target": options.node_budget,
                    "normalizer": frontier.normalizer,
                },
            },
            partition=Partition.from_metadata(entry.details.get("planning_surfaces")),
        )
        if initial.partition is not None:
            initial.partition.validate(initial.document)
    except LocalLimitError:
        return {
            "status": "bounded",
            "attempted": 0,
            "accepted": 0,
            "bounded_seeds": 1,
            "seconds": time.monotonic() - started,
        }
    states = [initial]
    cache = Rejections()
    finished = set()
    cursors: dict[str, Iterator[Proposal]] = {}
    cursor_peak = resumed = 0
    peak = _bytes(states)
    # Full validation has its own slice, in addition to the pipeline reserve.
    local_deadline = work.deadline - max(0.05, work.remaining * 0.25)
    local_work = Work(local_deadline, work.stop, work.timings)
    while not local_work.interrupted and attempted < limit and scanned < limit * 8:
        additions = []
        for state in states:
            if state.key in finished:
                continue
            if state.key not in cursors:
                cursors[state.key] = iter(proposals(state, local_work))
                cursor_peak = max(cursor_peak, len(cursors))
            else:
                resumed += 1
            expansion_start = attempted
            while True:
                if local_work.interrupted or attempted >= limit or scanned >= limit * 8:
                    break
                if attempted - expansion_start >= EXPANSIONS[options.quality]:
                    # Preserve depth as well as alternative first edits. Cached
                    # and invalid/stale scans retain their separate global cap.
                    bounded_expansions += 1
                    break
                proposal = next(cursors[state.key], None)
                if local_work.interrupted:
                    break
                if proposal is None:
                    finished.add(state.key)
                    break
                scanned += 1
                decision: dict = {
                    "operator": proposal.operator,
                    "ids": list(proposal.ids[:MAX_EDIT_OBJECTS]),
                    "estimated_cost_delta": proposal.estimate,
                    "parent_revision": state.key,
                    "parameters": list(proposal.parameters),
                    "bounds": [
                        proposal.bounds.x,
                        proposal.bounds.y,
                        proposal.bounds.right,
                        proposal.bounds.bottom,
                    ],
                    "accepted": False,
                }
                if proposal.parent != state.key:
                    decision["rejections"] = ["stale-proposal"]
                    decisions.append(decision)
                    continue
                partition = proposal.partition or state.partition
                if partition is not None:
                    try:
                        partition.validate(proposal.document)
                        if state.partition is not None and (
                            partition.owners.keys() != state.partition.owners.keys()
                        ):
                            raise ValueError("Structural edit lost source regions")
                    except ValueError as exc:
                        decision.update(
                            rejections=["invalid-surface-ownership"], detail=str(exc)
                        )
                        decisions.append(decision)
                        continue
                if len(proposal.ids) + len(proposal.dependencies) > MAX_EDIT_OBJECTS:
                    bounded += 1
                    decision.update(
                        rejections=["local-dependency-limit"],
                        ids_omitted=max(0, len(proposal.ids) - MAX_EDIT_OBJECTS),
                    )
                    decisions.append(decision)
                    continue
                decision["dependency_revisions"] = [
                    [oid, _revision(state.document, oid)]
                    for oid in (*proposal.ids, *proposal.dependencies)
                ]
                decision["feature_supports"] = [
                    index
                    for index, feature in enumerate(frontier.policy.features)
                    if proposal.bounds.intersection(
                        Box(
                            feature.box[0],
                            feature.box[1],
                            feature.box[0] + feature.box[2],
                            feature.box[1] + feature.box[3],
                        )
                    ).area
                ]
                try:
                    decision["tiles"] = len(
                        tile_boxes(proposal.bounds, state.snapshot.canvas.root.shape)
                    )
                except LocalLimitError:
                    bounded += 1
                    decision["rejections"] = ["local-buffer-limit"]
                    decisions.append(decision)
                    continue
                except ValueError as exc:
                    decision.update(
                        rejections=["invalid-local-bounds"], detail=str(exc)
                    )
                    decisions.append(decision)
                    continue
                structure = representation(proposal.document).metrics()
                if (
                    structure["nodes"] > MAX_SEED_NODES
                    or structure["paths"] + structure.get("primitive_objects", 0)
                    > MAX_SEED_OBJECTS
                ):
                    bounded += 1
                    decision["rejections"] = ["local-graph-limit"]
                    decisions.append(decision)
                    continue
                delta = (
                    structure["representation_cost"] - state.snapshot.evaluation.cost
                )
                decision["representation_delta"] = delta
                key = cache.key(state, proposal, delta)
                reasons = cache.get(key)
                if reasons is not None:
                    cached += 1
                    decision.update(rejections=list(reasons), cached=True)
                    decisions.append(decision)
                    continue
                svg = export_svg(proposal.document)
                attempted += 1
                visible_ids = cache.visible_ids(state, proposal)
                decision["context_objects"] = len(visible_ids)
                try:
                    updated = evaluator.update(
                        state.snapshot,
                        svg,
                        proposal.bounds,
                        structure,
                        visible_ids=visible_ids,
                        work=local_work,
                    )
                except StageInterruptedError:
                    decision["rejections"] = ["local-search-interrupted"]
                    decisions.append(decision)
                    break
                except LocalLimitError:
                    bounded += 1
                    decision["rejections"] = ["local-buffer-limit"]
                    decisions.append(decision)
                    cache.put(key, decision["rejections"])
                    continue
                except ValueError as exc:
                    decision.update(
                        rejections=["invalid-local-proposal"], detail=str(exc)
                    )
                    decisions.append(decision)
                    cache.put(key, decision["rejections"])
                    continue
                if local_work.interrupted:
                    decision["rejections"] = ["local-search-interrupted"]
                    decisions.append(decision)
                    break
                decision["visual_delta"] = (
                    updated.evaluation.visual - state.snapshot.evaluation.visual
                )
                decision["score_term_deltas"] = {
                    term: value - state.snapshot.evaluation.terms[term]
                    for term, value in updated.evaluation.terms.items()
                    if term in state.snapshot.evaluation.terms
                }
                improvements = [
                    complexity
                    for complexity in ANCHORS
                    if (
                        updated.evaluation.objective(
                            complexity,
                            frontier.normalizer,
                            frontier.policy.weights.detail,
                        )
                        < state.snapshot.evaluation.objective(
                            complexity,
                            frontier.normalizer,
                            frontier.policy.weights.detail,
                        )
                        - 1e-9
                    )
                ]
                reasons = list(updated.evaluation.rejections)
                if not improvements:
                    reasons.append("local-objective-regression")
                if reasons:
                    decision["rejections"] = reasons
                    decisions.append(decision)
                    cache.put(key, reasons)
                    continue
                child = State(
                    proposal.document,
                    svg,
                    updated,
                    hashlib.sha256(svg.encode()).hexdigest(),
                    {
                        **state.details,
                        **(proposal.details or {}),
                        **(
                            {"planning_surfaces": partition.metadata()}
                            if partition
                            else {}
                        ),
                    },
                    (*state.edits, decision),
                    partition,
                )
                if _bytes([*states, *additions, child]) > MAX_BYTES:
                    bounded += 1
                    decision["rejections"] = ["local-state-memory-limit"]
                    decisions.append(decision)
                    continue
                decision.update(
                    accepted=True,
                    rejections=[],
                    improved_at=improvements,
                )
                decisions.append(decision)
                additions.append(child)
                accepted += 1
                peak = max(peak, _bytes([*states, *additions]))
                # Bound live states during expansion, not just at round end.
                additions = _bound(additions, frontier, width)
            if local_work.interrupted or attempted >= limit or scanned >= limit * 8:
                break
        states = _bound([*states, *additions], frontier, width)
        retained = {state.key for state in states} - finished
        for key in list(cursors):
            if key not in retained:
                cursor = cursors.pop(key)
                close = getattr(cursor, "close", None)
                if close is not None:
                    close()
        if not retained:
            break
    for cursor in cursors.values():
        close = getattr(cursor, "close", None)
        if close is not None:
            close()
    validation_seconds = 0.0
    checkpoints = 0
    disagreements = 0
    for state in states:
        if not state.edits or work.interrupted:
            continue
        began = time.monotonic()
        frontier.checkpoint(
            state.svg,
            "Local structural search",
            {
                **state.details,
                "local_edits": list(state.edits),
                "local_search": True,
                "local_revision": state.key,
            },
            state.snapshot.evaluation,
            raster=state.snapshot.canvas,
        )
        validation_seconds += time.monotonic() - began
        checkpoints += 1
        if frontier.decisions[-1].get("rejections") in (
            ["local-score-disagreement"],
            ["local-raster-disagreement"],
        ):
            disagreements += 1
        if work.remaining <= validation_seconds / checkpoints:
            break
    return {
        "status": "interrupted"
        if local_work.interrupted
        else "bounded"
        if attempted >= limit or scanned >= limit * 8 or bounded or bounded_expansions
        else "complete",
        "attempted": attempted,
        "accepted": accepted,
        "cached_rejections": cached,
        "bounded_proposals": bounded,
        "bounded_expansions": bounded_expansions,
        "resumed_expansions": resumed,
        "proposal_cursor_peak": cursor_peak,
        "search_budget": initial.details["search_budget"],
        "expansion_evaluation_limit": EXPANSIONS[options.quality],
        "scanned": scanned,
        "beam_limit": width,
        "evaluation_limit": limit,
        "beam_states": len(states),
        "checkpointed": checkpoints,
        "score_disagreements": disagreements,
        "rejection_cache_entries": len(cache.values),
        "dependency_index_entries": len(cache.indices),
        "peak_retained_bytes": peak,
        "native_context_renders": evaluator.native_context_renders,
        "tiles_scored": evaluator.tiles_scored,
        "tile_limit": MAX_TILES,
        "seconds": time.monotonic() - started,
        "validation_seconds": validation_seconds,
        "decisions": decisions,
    }
