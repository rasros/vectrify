import dataclasses
from typing import Generic, TypeVar

INVALID_SCORE = float("inf")

TState = TypeVar("TState")
TResultPayload = TypeVar("TResultPayload")


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
    complexity: float = dataclasses.field(default=0.0, compare=False)
    signature: tuple[int, ...] | None = dataclasses.field(default=None, compare=False)
    epoch: int = dataclasses.field(default=0, compare=False)


@dataclasses.dataclass
class Task(Generic[TState]):
    task_id: int
    parent_id: int
    parent_state: ChainState[TState]
    worker_slot: int
    secondary_parent_id: int | None = None
    secondary_parent_state: ChainState[TState] | None = None
    force_llm: bool = False


@dataclasses.dataclass
class Result(Generic[TResultPayload]):
    task_id: int
    parent_id: int
    worker_slot: int
    valid: bool
    score: float | None
    payload: TResultPayload
    invalid_msg: str | None = None
    secondary_parent_id: int | None = None
    complexity: float = 0.0
    content: str | None = None
    signature: tuple[int, ...] | None = None
    llm_type: str | None = None
