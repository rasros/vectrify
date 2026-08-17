import dataclasses
from typing import Generic, TypeVar

TState = TypeVar("TState")
TResultPayload = TypeVar("TResultPayload")


@dataclasses.dataclass
class ChainState(Generic[TState]):
    payload: TState


@dataclasses.dataclass
class SearchNode(Generic[TState]):
    # Whether this candidate was measured: its markup parsed, it rasterized,
    # and the measures computed. Not a quality judgement and not orderable --
    # the measures are traded off by dominance, so no single number orders
    # candidates, and the run's only score is the evaluator's, in
    # metrics[FRONT_SCORE] on the nodes it has actually seen.
    #
    # This was a float called `score` holding one of two sentinel values, which
    # read as a quality anyone could sort by. Three separate defects came from
    # something sorting by it after it had stopped meaning anything.
    valid: bool
    id: int
    parent_id: int
    state: ChainState[TState]
    secondary_parent_id: int | None = None
    # Registered metrics, keyed by name (see score.metrics.METRIC_NAMES).
    # A dict rather than named fields so adding a metric does not ripple through
    # every constructor call between the worker and the objective vector.
    metrics: dict[str, float] = dataclasses.field(default_factory=dict, repr=False)
    signature: int | None = None
    epoch: int = 0
    # The seed this node descends from. Crossover between two nodes of the same
    # lineage recombines a candidate with itself, so selection uses this to pair
    # only across lineages.
    root_id: int = 0
    # Which mutation operator produced this node, so the policy that picked it
    # can be told whether it survived. None for seeds and crossover children.
    operator: str | None = None


@dataclasses.dataclass
class Task(Generic[TState]):
    task_id: int
    parent_id: int
    parent_state: ChainState[TState]
    secondary_parent_id: int | None = None
    secondary_parent_state: ChainState[TState] | None = None
    force_llm: bool = False
    # The mutation operator to apply. The engine picks it so one policy sees
    # every outcome; None lets the backend choose for itself.
    operator: str | None = None


@dataclasses.dataclass
class Result(Generic[TResultPayload]):
    task_id: int
    parent_id: int
    valid: bool
    # Whether the measures have been taken yet. The worker produces candidates
    # and a scorer thread measures them, so a result crosses the queue once
    # before anything is known about it.
    measured: bool
    payload: TResultPayload
    invalid_msg: str | None = None
    secondary_parent_id: int | None = None
    metrics: dict[str, float] = dataclasses.field(default_factory=dict)
    signature: int | None = None
    llm_type: str | None = None
    # Echoed back from the task: results arrive out of order and some never
    # arrive at all, so carrying it beats a pending-task map in the engine.
    operator: str | None = None
