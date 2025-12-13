from __future__ import annotations

import dataclasses
from typing import Optional


INVALID_SCORE = 1e9


@dataclasses.dataclass
class ChainState:
    """
    The state we refine over time.
    - model_temperature: prompt sampling temperature for OpenAI generation
    - stale_hits: used to bump model_temperature if proposals are near-identical
    """
    svg: Optional[str]
    raster_data_url: Optional[str]  # rasterized svg as png data-url
    score: float
    model_temperature: float
    stale_hits: int
    invalid_msg: Optional[str]


@dataclasses.dataclass(order=True)
class SearchNode:
    score: float
    id: int = dataclasses.field(compare=False)
    state: ChainState = dataclasses.field(compare=False)


@dataclasses.dataclass
class Task:
    task_id: int
    parent_id: int
    parent_state: ChainState
    proposal_index: int  # just for diversity / jitter


@dataclasses.dataclass
class Result:
    task_id: int
    parent_id: int
    proposal_index: int
    svg: Optional[str]
    valid: bool
    invalid_msg: Optional[str]
    raster_png: Optional[bytes]
    score: float
    used_temperature: float
    change_summary: Optional[str]
