"""Bounded beam edits, fixed native crop scores and independent checkpoints."""

from __future__ import annotations

import hashlib
import sys
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, fields, is_dataclass, replace

from vectrify.document import Document, export_svg, import_svg
from vectrify.document.join import path_style
from vectrify.document.model import paint_server
from vectrify.refine.cel_plan.bounds import conservative_bounds
from vectrify.refine.cel_plan.component_edits import (
    MAX_OBJECTS as MAX_COMPONENT_OBJECTS,
)

# Deliberate compatibility reexports; contracts and limits own these values.
from vectrify.refine.cel_plan.contracts import Proposal as Proposal
from vectrify.refine.cel_plan.contracts import State as State
from vectrify.refine.cel_plan.frontier import Entry, Frontier
from vectrify.refine.cel_plan.limits import EXPANSIONS as EXPANSIONS
from vectrify.refine.cel_plan.limits import MAX_EDIT_OBJECTS as MAX_EDIT_OBJECTS
from vectrify.refine.cel_plan.limits import MAX_SEED_NODES as MAX_SEED_NODES
from vectrify.refine.cel_plan.limits import MAX_SEED_OBJECTS as MAX_SEED_OBJECTS
from vectrify.refine.cel_plan.local import (
    HALO,
    MAX_CROP_PIXELS,
    MAX_TILES,
    Box,
    LocalLimitError,
    LocalPolicy,
    tile_boxes,
)
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.score import SCORE_VERSION, representation

LIMITS = {"fast": (1, 16), "balanced": (4, 48), "high": (8, 128)}
ANCHORS = (50, 0, 100, 25, 75)
MAX_BYTES = 64 * 1024 * 1024
MAX_REJECTIONS = 256


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
                    left, top, right, bottom = conservative_bounds(document, element.id)
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


@dataclass(frozen=True)
class _SearchSeed:
    state: State
    evaluator: LocalPolicy
    retained_bytes: int


def _search_seed(
    frontier: Frontier,
    options: Options,
    work: Work,
    proposals: Callable[[State, Work], Iterator[Proposal]],
    seed_document: Document | None,
) -> _SearchSeed | None:
    """Admit a bounded native seed while preserving caller-owned identities."""
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
        return None
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
        return None
    peak = _bytes([initial])
    if peak > MAX_BYTES:
        return None
    return _SearchSeed(initial, evaluator, peak)


@dataclass
class _CheckpointBudget:
    """Discovery can shrink; publication keeps its separate live deadline."""

    local_work: Work
    validation_work: Work
    shared: bool
    minimum_seconds: float
    guard: float = 0.0
    proposal_started: float | None = None

    @classmethod
    def allocate(
        cls,
        work: Work,
        checkpoint_work: Work | None,
        minimum_checkpoint_seconds: float,
    ) -> _CheckpointBudget:
        validation_work = checkpoint_work or work
        # Standalone search reserves its own checkpoints. A pipeline supplies a
        # separate validation deadline; discovery may use its entire local slice
        # only while that deadline still leaves the measured full-check duration.
        local_deadline = (
            min(work.deadline, validation_work.deadline - minimum_checkpoint_seconds)
            if checkpoint_work is not None
            else work.deadline - max(0.05, work.remaining * 0.25)
        )
        local_work = Work(local_deadline, work.stop, work.timings)
        return cls(
            local_work,
            validation_work,
            checkpoint_work is not None,
            minimum_checkpoint_seconds,
        )

    def reserve(self) -> None:
        if not self.shared or self.minimum_seconds <= 0:
            return
        # A renderer/boolean can finish after its last deadline check. Leave
        # the longest observed proposal opportunity as well as full-check time.
        # This changes the time bound, not candidate priority or acceptance.
        self.guard = max(
            self.guard,
            0.05,
            time.monotonic() - self.proposal_started
            if self.proposal_started is not None
            else 0,
        )
        self.local_work.deadline = min(
            self.local_work.deadline,
            self.validation_work.deadline - self.minimum_seconds - self.guard,
        )


@dataclass
class _Beam:
    """Own retained states and suspended proposal cursors for one search."""

    states: list[State]
    peak_bytes: int
    additions: list[State] = field(default_factory=list)
    finished: set[str] = field(default_factory=set)
    cursors: dict[str, Iterator[Proposal]] = field(default_factory=dict)
    cursor_peak: int = 0
    resumed: int = 0

    def cursor(
        self,
        state: State,
        proposals: Callable[[State, Work], Iterator[Proposal]],
        work: Work,
    ) -> Iterator[Proposal]:
        if state.key not in self.cursors:
            self.cursors[state.key] = iter(proposals(state, work))
            self.cursor_peak = max(self.cursor_peak, len(self.cursors))
        else:
            self.resumed += 1
        return self.cursors[state.key]

    def retain(self, frontier: Frontier, width: int) -> bool:
        self.states = _bound([*self.states, *self.additions], frontier, width)
        retained = {state.key for state in self.states} - self.finished
        for key in list(self.cursors):
            if key not in retained:
                self._close(self.cursors.pop(key))
        return bool(retained)

    @staticmethod
    def _close(cursor: Iterator[Proposal]) -> None:
        close = getattr(cursor, "close", None)
        if close is not None:
            close()

    def close(self) -> None:
        for cursor in self.cursors.values():
            self._close(cursor)


@dataclass
class _SearchCounts:
    attempted: int = 0
    accepted: int = 0
    cached: int = 0
    bounded: int = 0
    scanned: int = 0
    bounded_expansions: int = 0
    decisions: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class _PreparedProposal:
    decision: dict
    partition: Partition | None
    structure: dict
    rejection_key: str


@dataclass
class _CheckpointReport:
    seconds: float = 0.0
    count: int = 0
    disagreements: int = 0


@dataclass
class _SearchRun:
    """One local search owns admission, beam expansion and publication reports."""

    frontier: Frontier
    options: Options
    work: Work
    proposals: Callable[[State, Work], Iterator[Proposal]]
    started: float
    initial: State
    evaluator: LocalPolicy
    beam: _Beam
    budget: _CheckpointBudget
    width: int
    limit: int
    counts: _SearchCounts = field(default_factory=_SearchCounts)
    cache: Rejections = field(default_factory=Rejections)

    @property
    def exhausted(self) -> bool:
        return (
            self.budget.local_work.interrupted
            or self.counts.attempted >= self.limit
            or self.counts.scanned >= self.limit * 8
        )

    def validate_ownership(
        self, state: State, proposal: Proposal, decision: dict
    ) -> tuple[Partition | None, str | None] | None:
        """Check source ownership and sealed component dependencies before scoring."""
        partition = proposal.partition or state.partition
        if partition is not None:
            try:
                partition.validate(proposal.document)
                previous_families = (
                    state.partition.families if state.partition is not None else ()
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
                validator = getattr(self.proposals, "validate_partition", None)
                if validator is not None:
                    validator(partition, self.budget.local_work)
                elif partition.atoms is not None:
                    raise ValueError(
                        "Source splits require an original graph validator"
                    )
            except ValueError as exc:
                decision.update(
                    rejections=["invalid-surface-ownership"], detail=str(exc)
                )
                self.counts.decisions.append(decision)
                return None
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
                    self.budget.local_work,
                )
                decision["component_dependency"] = {
                    "parent": proposal.component.parent,
                    "source": proposal.component.source,
                    "target": component_revision,
                    "declared_objects": len(proposal.ids),
                }
            except ValueError as exc:
                self.counts.bounded += 1
                decision.update(
                    rejections=["invalid-component-dependencies"],
                    detail=str(exc),
                )
                self.counts.decisions.append(decision)
                return None
        elif len(proposal.ids) + len(proposal.dependencies) > MAX_EDIT_OBJECTS:
            self.counts.bounded += 1
            decision.update(
                rejections=["local-dependency-limit"],
                ids_omitted=max(0, len(proposal.ids) - MAX_EDIT_OBJECTS),
            )
            self.counts.decisions.append(decision)
            return None
        return partition, component_revision

    def prepare(self, state: State, proposal: Proposal) -> _PreparedProposal | None:
        """Reject stale, oversized or cached proposals before native evaluation."""
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
            self.counts.decisions.append(decision)
            return None
        ownership = self.validate_ownership(state, proposal, decision)
        if ownership is None:
            return None
        partition, component_revision = ownership
        decision["dependency_revisions"] = [
            [oid, _revision(state.document, oid)]
            for oid in (*proposal.ids, *proposal.dependencies)
        ]
        decision["feature_supports"] = [
            index
            for index, feature in enumerate(self.frontier.policy.features)
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
            self.counts.bounded += 1
            decision["rejections"] = ["local-buffer-limit"]
            self.counts.decisions.append(decision)
            return None
        except ValueError as exc:
            decision.update(rejections=["invalid-local-bounds"], detail=str(exc))
            self.counts.decisions.append(decision)
            return None
        structure = representation(proposal.document).metrics()
        if (
            structure["nodes"] > MAX_SEED_NODES
            or structure["paths"] + structure.get("primitive_objects", 0)
            > MAX_SEED_OBJECTS
        ):
            self.counts.bounded += 1
            decision["rejections"] = ["local-graph-limit"]
            self.counts.decisions.append(decision)
            return None
        delta = structure["representation_cost"] - state.snapshot.evaluation.cost
        decision["representation_delta"] = delta
        key = self.cache.key(
            state, proposal, delta, component_revision=component_revision
        )
        reasons = self.cache.get(key)
        if reasons is not None:
            self.counts.cached += 1
            decision.update(rejections=list(reasons), cached=True)
            self.counts.decisions.append(decision)
            return None
        return _PreparedProposal(decision, partition, structure, key)

    def evaluate(self, state: State, proposal: Proposal) -> None:
        """Accept only safe anchor improvements that fit retained memory."""
        prepared = self.prepare(state, proposal)
        if prepared is None:
            return
        decision = prepared.decision
        partition, structure, key = (
            prepared.partition,
            prepared.structure,
            prepared.rejection_key,
        )
        svg = export_svg(proposal.document)
        self.counts.attempted += 1
        visible_ids = self.cache.visible_ids(state, proposal)
        decision["context_objects"] = len(visible_ids)
        try:
            updated = self.evaluator.update(
                state.snapshot,
                svg,
                proposal.bounds,
                structure,
                visible_ids=visible_ids,
                work=self.budget.local_work,
                **(
                    {"document": proposal.document, "partition": partition}
                    if self.frontier.policy.editable_ink is not None
                    else {}
                ),
            )
        except StageInterruptedError:
            decision["rejections"] = ["local-search-interrupted"]
            self.counts.decisions.append(decision)
            raise
        except LocalLimitError:
            self.counts.bounded += 1
            decision["rejections"] = ["local-buffer-limit"]
            self.counts.decisions.append(decision)
            self.cache.put(key, decision["rejections"])
            return
        except ValueError as exc:
            decision.update(rejections=["invalid-local-proposal"], detail=str(exc))
            self.counts.decisions.append(decision)
            self.cache.put(key, decision["rejections"])
            return
        if self.budget.local_work.interrupted:
            decision["rejections"] = ["local-search-interrupted"]
            self.counts.decisions.append(decision)
            raise StageInterruptedError
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
                    self.frontier.normalizer,
                    self.frontier.policy.weights.detail,
                )
                < state.snapshot.evaluation.objective(
                    complexity,
                    self.frontier.normalizer,
                    self.frontier.policy.weights.detail,
                )
                - 1e-9
            )
        ]
        reasons = list(updated.evaluation.rejections)
        if not improvements:
            reasons.append("local-objective-regression")
        if reasons:
            decision["rejections"] = reasons
            self.counts.decisions.append(decision)
            self.cache.put(key, reasons)
            return
        child = State(
            proposal.document,
            svg,
            updated,
            identity(svg, partition),
            {
                **state.details,
                **(proposal.details or {}),
                **({"planning_surfaces": partition.metadata()} if partition else {}),
            },
            (*state.edits, decision),
            partition,
        )
        if _bytes([*self.beam.states, *self.beam.additions, child]) > MAX_BYTES:
            self.counts.bounded += 1
            decision["rejections"] = ["local-state-memory-limit"]
            self.counts.decisions.append(decision)
            return
        decision.update(
            accepted=True,
            rejections=[],
            improved_at=improvements,
        )
        self.counts.decisions.append(decision)
        self.beam.additions.append(child)
        self.counts.accepted += 1
        self.beam.peak_bytes = max(
            self.beam.peak_bytes, _bytes([*self.beam.states, *self.beam.additions])
        )
        # Bound live states during expansion, not just at round end.
        self.beam.additions = _bound(self.beam.additions, self.frontier, self.width)

    def expand_state(self, state: State) -> None:
        """Resume one cursor within its per-round evaluation allowance."""
        cursor = self.beam.cursor(state, self.proposals, self.budget.local_work)
        expansion_start = self.counts.attempted
        while True:
            self.budget.reserve()
            if (
                self.budget.local_work.interrupted
                or self.counts.attempted >= self.limit
                or self.counts.scanned >= self.limit * 8
            ):
                break
            if (
                self.counts.attempted - expansion_start
                >= EXPANSIONS[self.options.quality]
            ):
                # Preserve depth as well as alternative first edits. Cached
                # and invalid/stale scans retain their separate global cap.
                self.counts.bounded_expansions += 1
                break
            self.budget.proposal_started = time.monotonic()
            try:
                proposal = next(cursor, None)
            except StageInterruptedError:
                self.beam.finished.add(state.key)
                break
            if self.budget.local_work.interrupted:
                break
            if proposal is None:
                self.beam.finished.add(state.key)
                break
            self.counts.scanned += 1
            try:
                self.evaluate(state, proposal)
            except StageInterruptedError:
                break

    def expand(self) -> None:
        """Expand and prune the bounded beam while retaining live cursors."""
        self.budget.reserve()
        while not self.exhausted:
            self.beam.additions = []
            for state in self.beam.states:
                if state.key in self.beam.finished:
                    continue
                self.expand_state(state)
                if self.exhausted:
                    break
            if not self.beam.retain(self.frontier, self.width):
                break
        self.budget.reserve()
        self.beam.close()

    def publish_checkpoints(self) -> _CheckpointReport:
        """Independently validate edited states before publishing to the frontier."""
        report = _CheckpointReport()
        for state in self.beam.states:
            if (
                not state.edits
                or self.work.stop.is_set()
                or self.budget.validation_work.interrupted
                or self.budget.validation_work.remaining < self.budget.minimum_seconds
            ):
                continue
            began = time.monotonic()
            try:
                self.frontier.checkpoint(
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
                    **(
                        {
                            "document": state.document,
                            "partition": state.partition,
                            "work": self.budget.validation_work,
                        }
                        if self.frontier.policy.editable_ink is not None
                        else {}
                    ),
                )
            except StageInterruptedError:
                break
            report.seconds += time.monotonic() - began
            report.count += 1
            if self.frontier.decisions[-1].get("rejections") in (
                ["local-score-disagreement"],
                ["local-raster-disagreement"],
            ):
                report.disagreements += 1
            if self.budget.validation_work.remaining <= max(
                self.budget.minimum_seconds, report.seconds / report.count
            ):
                break
        return report

    def report(self, checkpoints: _CheckpointReport) -> dict:
        """Serialize the existing search counters and decision schema."""
        return {
            "status": "interrupted"
            if self.budget.local_work.interrupted
            else "bounded"
            if self.counts.attempted >= self.limit
            or self.counts.scanned >= self.limit * 8
            or self.counts.bounded
            or self.counts.bounded_expansions
            else "complete",
            "attempted": self.counts.attempted,
            "accepted": self.counts.accepted,
            "cached_rejections": self.counts.cached,
            "bounded_proposals": self.counts.bounded,
            "bounded_expansions": self.counts.bounded_expansions,
            "resumed_expansions": self.beam.resumed,
            "proposal_cursor_peak": self.beam.cursor_peak,
            "search_budget": self.initial.details["search_budget"],
            "expansion_evaluation_limit": EXPANSIONS[self.options.quality],
            "scanned": self.counts.scanned,
            "beam_limit": self.width,
            "evaluation_limit": self.limit,
            "beam_states": len(self.beam.states),
            "checkpointed": checkpoints.count,
            "checkpoint_guard_seconds": self.budget.guard,
            "checkpoint_scope": "shared-search"
            if self.budget.shared
            else "local-phase",
            "score_disagreements": checkpoints.disagreements,
            "rejection_cache_entries": len(self.cache.values),
            "dependency_index_entries": len(self.cache.indices),
            "peak_retained_bytes": self.beam.peak_bytes,
            "native_context_renders": self.evaluator.native_context_renders,
            "tiles_scored": self.evaluator.tiles_scored,
            "tile_limit": MAX_TILES,
            "seconds": time.monotonic() - self.started,
            "validation_seconds": checkpoints.seconds,
            "decisions": self.counts.decisions,
        }


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
    width, limit = LIMITS[options.quality]
    if work.interrupted:
        return {"status": "interrupted", "attempted": 0, "accepted": 0, "seconds": 0.0}
    if not frontier.normalizer_fixed:
        raise ValueError("Freeze the detailed cost scale before local search")
    seed = _search_seed(frontier, options, work, proposals, seed_document)
    if seed is None:
        return {
            "status": "bounded",
            "attempted": 0,
            "accepted": 0,
            "bounded_seeds": 1,
            "seconds": time.monotonic() - started,
        }
    run = _SearchRun(
        frontier=frontier,
        options=options,
        work=work,
        proposals=proposals,
        started=started,
        initial=seed.state,
        evaluator=seed.evaluator,
        beam=_Beam([seed.state], seed.retained_bytes),
        budget=_CheckpointBudget.allocate(
            work, checkpoint_work, minimum_checkpoint_seconds
        ),
        width=width,
        limit=limit,
    )
    run.expand()
    checkpoints = run.publish_checkpoints()
    return run.report(checkpoints)
