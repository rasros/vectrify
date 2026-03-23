import dataclasses
from typing import Any, Generic, TypeVar

INVALID_SCORE = 1e9

TState = TypeVar("TState")
TResultPayload = TypeVar("TResultPayload")


@dataclasses.dataclass
class ChainState(Generic[TState]):
    score: float
    model_temperature: float
    stale_hits: int
    payload: TState


@dataclasses.dataclass(order=True)
class SearchNode(Generic[TState]):
    score: float
    id: int = dataclasses.field(compare=False)
    parent_id: int = dataclasses.field(compare=False)
    state: ChainState[TState] = dataclasses.field(compare=False)


@dataclasses.dataclass
class Task(Generic[TState]):
    task_id: int
    parent_id: int
    parent_state: ChainState[TState]
    worker_slot: int
    secondary_parent_id: int | None = None
    secondary_parent_state: ChainState[TState] | None = None


@dataclasses.dataclass
class Result(Generic[TResultPayload]):
    task_id: int
    parent_id: int
    worker_slot: int
    valid: bool
    score: float
    used_temperature: float
    payload: TResultPayload
    invalid_msg: str | None = None
    secondary_parent_id: int | None = None