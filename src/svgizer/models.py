from __future__ import annotations

import dataclasses
from typing import Optional


INVALID_SCORE = 1e9


@dataclasses.dataclass
class ChainState:
    svg: Optional[str]
    raster_data_url: Optional[str]          # FULL-res (for lineage / inspection)
    raster_preview_data_url: Optional[str]  # DOWNSCALED (for OpenAI)
    score: float
    model_temperature: float
    stale_hits: int
    invalid_msg: Optional[str]
    change_summary: Optional[str] = None


@dataclasses.dataclass(order=True)
class SearchNode:
    score: float
    id: int = dataclasses.field(compare=False)
    parent_id: int = dataclasses.field(compare=False)
    state: ChainState = dataclasses.field(compare=False)


@dataclasses.dataclass
class Task:
    task_id: int
    parent_id: int
    parent_state: ChainState
    worker_slot: int  # used only for diversity/jitter


@dataclasses.dataclass
class Result:
    task_id: int
    parent_id: int
    worker_slot: int
    svg: Optional[str]
    valid: bool
    invalid_msg: Optional[str]
    raster_png: Optional[bytes]
    score: float
    used_temperature: float
    change_summary: Optional[str]
