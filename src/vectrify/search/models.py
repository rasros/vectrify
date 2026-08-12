import dataclasses
from typing import Generic, TypeVar

INVALID_SCORE = float("inf")

TState = TypeVar("TState")
TResultPayload = TypeVar("TResultPayload")


def valid_scores(nodes: "list[SearchNode]") -> list[float]:
    """Scores of nodes that were successfully evaluated."""
    return [n.score for n in nodes if n.score < INVALID_SCORE]


@dataclasses.dataclass
class ChainState(Generic[TState]):
    score: float | None
    payload: TState


@dataclasses.dataclass(order=True)
class SearchNode(Generic[TState]):
    score: float
    id: int = dataclasses.field(compare=False)
    parent_id: int = dataclasses.field(compare=False)
    state: ChainState[TState] = dataclasses.field(compare=False)
    secondary_parent_id: int | None = dataclasses.field(default=None, compare=False)
    # Registered complexity metrics, keyed by name (see score.complexity.METRICS).
    # A dict rather than named fields so adding a metric does not ripple through
    # every constructor call between the worker and the objective vector.
    metrics: dict[str, float] = dataclasses.field(
        default_factory=dict, compare=False, repr=False
    )
    signature: int | None = dataclasses.field(default=None, compare=False)
    epoch: int = dataclasses.field(default=0, compare=False)
    # The seed this node descends from. Crossover between two nodes of the same
    # lineage recombines a candidate with itself, so selection uses this to pair
    # only across lineages.
    root_id: int = dataclasses.field(default=0, compare=False)
    # Which mutation operator produced this node, so the policy that picked it
    # can be told whether it survived. None for seeds and crossover children.
    operator: str | None = dataclasses.field(default=None, compare=False)


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
    score: float | None
    payload: TResultPayload
    invalid_msg: str | None = None
    secondary_parent_id: int | None = None
    metrics: dict[str, float] = dataclasses.field(default_factory=dict)
    signature: int | None = None
    llm_type: str | None = None
    # Echoed back from the task: results arrive out of order and some never
    # arrive at all, so carrying it beats a pending-task map in the engine.
    operator: str | None = None
