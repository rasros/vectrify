"""Bounded beam edits, fixed native crop scores and independent checkpoints."""

from __future__ import annotations

import hashlib
import sys
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields, is_dataclass, replace

from vectrify.document import Document, export_svg, import_svg
from vectrify.document.join import path_style
from vectrify.document.model import paint_server
from vectrify.refine.cel_plan.component_edits import (
    MAX_OBJECTS as MAX_COMPONENT_OBJECTS,
)
from vectrify.refine.cel_plan.component_edits import ComponentEdit
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
    component: ComponentEdit | None = None


def identity(svg: str, partition: Partition | None) -> str:
    digest = hashlib.sha256(svg.encode())
    if partition is not None and partition.atoms is not None:
        digest.update(partition.atoms.key.encode())
    if partition is not None and partition.families:
        digest.update(repr(tuple(f.metadata() for f in partition.families)).encode())
    return digest.hexdigest()


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

    def key(
        self, state: State, proposal: Proposal, delta: float, *, component_revision=None
    ) -> str:
        snapshot = state.snapshot
        if proposal.component is not None and component_revision is None:
            raise ValueError("Component rejections require a validated target revision")
        context = proposal.bounds.expand(2 * HALO, snapshot.canvas.root.shape)
        signature = (
            SCORE_VERSION,
            state.partition.atoms.key
            if state.partition is not None and state.partition.atoms is not None
            else None,
            proposal.partition.atoms.key
            if proposal.partition is not None and proposal.partition.atoms is not None
            else None,
            proposal.operator,
            proposal.ids,
            proposal.parameters,
            proposal.bounds,
            proposal.component,
            component_revision,
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
    records = set()

    def size(value):
        if id(value) in records:
            return 0
        records.add(id(value))
        total = sys.getsizeof(value)
        if isinstance(value, dict):
            return total + sum(size(k) + size(v) for k, v in value.items())
        if isinstance(value, (tuple, list)):
            return total + sum(size(v) for v in value)
        if is_dataclass(value):
            return total + sum(size(getattr(value, f.name)) for f in fields(value))
        return total

    atom_bytes = 0
    for state in states:
        canvas = state.snapshot.canvas
        arrays[id(canvas.root)] = canvas.root.nbytes
        for patch in canvas.patches:
            arrays[id(patch.pixels)] = patch.pixels.nbytes
        if state.partition is not None and state.partition.atoms is not None:
            atoms = state.partition.atoms
            atom_bytes += size(atoms)
            atom_bytes += size(
                state.details.get("planning_surfaces", {}).get("source_atoms")
            )
    return (
        atom_bytes
        + sum(arrays.values())
        + sum(len(state.svg.encode()) for state in states)
    )


def search(
    frontier: Frontier,
    options: Options,
    work: Work,
    proposals: Callable[[State, Work], Iterator[Proposal]],
    *,
    checkpoint_work: Work | None = None,
    minimum_checkpoint_seconds: float = 0,
    seed_document: Document | None = None,
) -> dict:
    """Accept local edits; publish full checkpoints within their live reserve.

    A pipeline can supply its remaining global search time for checkpoints
    after the bounded local phase expires. Cancellation still prevents publish.
    """
    started = time.monotonic()
    validation_work = checkpoint_work or work
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
        # Native families seal editor identities as well as geometry/paint.
        # A caller resuming a saved native candidate can preserve those IDs;
        # its exported drawing must still be exactly the selected scored seed.
        if seed_document is not None and export_svg(seed_document) != entry.svg:
            raise ValueError("Native search seed does not match the selected drawing")
        initial = State(
            seed_document if seed_document is not None else import_svg(entry.svg),
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
            validator = getattr(proposals, "validate_partition", None)
            if validator is not None:
                validator(initial.partition, work)
            elif initial.partition.atoms is not None:
                raise ValueError(
                    "Source atom states require an original graph validator"
                )
            initial = replace(initial, key=identity(initial.svg, initial.partition))
    except (LocalLimitError, StageInterruptedError):
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
    if peak > MAX_BYTES:
        return {
            "status": "bounded",
            "attempted": 0,
            "accepted": 0,
            "bounded_seeds": 1,
            "seconds": time.monotonic() - started,
        }
    # Standalone search reserves its own checkpoints. A pipeline supplies a
    # separate validation deadline; discovery may use its entire local slice
    # only while that deadline still leaves the measured full-check duration.
    local_deadline = (
        min(work.deadline, validation_work.deadline - minimum_checkpoint_seconds)
        if checkpoint_work is not None
        else work.deadline - max(0.05, work.remaining * 0.25)
    )
    local_work = Work(local_deadline, work.stop, work.timings)
    checkpoint_guard = 0.0
    proposal_started = None

    def reserve_checkpoint():
        nonlocal checkpoint_guard
        if checkpoint_work is None or minimum_checkpoint_seconds <= 0:
            return
        # A renderer/boolean can finish after its last deadline check. Leave
        # the longest observed proposal opportunity as well as full-check time.
        # This changes the time bound, not candidate priority or acceptance.
        checkpoint_guard = max(
            checkpoint_guard,
            0.05,
            time.monotonic() - proposal_started if proposal_started is not None else 0,
        )
        local_work.deadline = min(
            local_work.deadline,
            validation_work.deadline - minimum_checkpoint_seconds - checkpoint_guard,
        )

    reserve_checkpoint()
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
                reserve_checkpoint()
                if local_work.interrupted or attempted >= limit or scanned >= limit * 8:
                    break
                if attempted - expansion_start >= EXPANSIONS[options.quality]:
                    # Preserve depth as well as alternative first edits. Cached
                    # and invalid/stale scans retain their separate global cap.
                    bounded_expansions += 1
                    break
                proposal_started = time.monotonic()
                try:
                    proposal = next(cursors[state.key], None)
                except StageInterruptedError:
                    finished.add(state.key)
                    break
                if local_work.interrupted:
                    break
                if proposal is None:
                    finished.add(state.key)
                    break
                scanned += 1
                decision: dict = {
                    "operator": proposal.operator,
                    "ids": list(
                        proposal.ids[: 2 * MAX_COMPONENT_OBJECTS]
                        if proposal.component is not None
                        else proposal.ids[:MAX_EDIT_OBJECTS]
                    ),
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
                        previous_families = (
                            state.partition.families
                            if state.partition is not None
                            else ()
                        )
                        if proposal.component is None and any(
                            f not in previous_families for f in partition.families
                        ):
                            raise ValueError(
                                "New physical source families require a sealed "
                                "component replacement"
                            )
                        if state.partition is not None and not partition.follows(
                            state.partition
                        ):
                            raise ValueError("Structural edit lost source regions")
                        validator = getattr(proposals, "validate_partition", None)
                        if validator is not None:
                            validator(partition, local_work)
                        elif partition.atoms is not None:
                            raise ValueError(
                                "Source splits require an original graph validator"
                            )
                    except StageInterruptedError:
                        break
                    except ValueError as exc:
                        decision.update(
                            rejections=["invalid-surface-ownership"], detail=str(exc)
                        )
                        decisions.append(decision)
                        continue
                component_revision = None
                if proposal.component is not None:
                    try:
                        if (
                            len(proposal.dependencies) > MAX_EDIT_OBJECTS
                            or proposal.component.parent not in proposal.dependencies
                        ):
                            raise ValueError(
                                "Component replacement lacks its parent dependency"
                            )
                        component_revision = proposal.component.validate(
                            state.document,
                            proposal.document,
                            state.partition,
                            partition,
                            proposal.ids,
                            proposal.bounds,
                            local_work,
                        )
                        decision["component_dependency"] = {
                            "parent": proposal.component.parent,
                            "source": proposal.component.source,
                            "target": component_revision,
                            "declared_objects": len(proposal.ids),
                        }
                    except StageInterruptedError:
                        break
                    except ValueError as exc:
                        bounded += 1
                        decision.update(
                            rejections=["invalid-component-dependencies"],
                            detail=str(exc),
                        )
                        decisions.append(decision)
                        continue
                elif len(proposal.ids) + len(proposal.dependencies) > MAX_EDIT_OBJECTS:
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
                key = cache.key(
                    state, proposal, delta, component_revision=component_revision
                )
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
                    identity(svg, partition),
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
    reserve_checkpoint()
    for cursor in cursors.values():
        close = getattr(cursor, "close", None)
        if close is not None:
            close()
    validation_seconds = 0.0
    checkpoints = 0
    disagreements = 0
    for state in states:
        if (
            not state.edits
            or work.stop.is_set()
            or validation_work.interrupted
            or validation_work.remaining < minimum_checkpoint_seconds
        ):
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
        if validation_work.remaining <= max(
            minimum_checkpoint_seconds, validation_seconds / checkpoints
        ):
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
        "checkpoint_guard_seconds": checkpoint_guard,
        "checkpoint_scope": "shared-search"
        if checkpoint_work is not None
        else "local-phase",
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
