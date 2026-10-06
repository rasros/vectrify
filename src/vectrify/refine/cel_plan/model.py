"""Temporary immutable values used while planning a cel drawing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Event

import numpy as np


@dataclass(frozen=True)
class Options:
    complexity: int = 50
    quality: str = "balanced"
    refine: bool = True
    gradients: bool = True
    line_width: float = 0
    palette: int = 0
    tolerance: float = 0
    protection: float = 1
    node_budget: int = 0

    @property
    def seconds(self) -> float:
        return {"fast": 5.0, "balanced": 20.0, "high": 60.0}[self.quality]

    @property
    def boundary_tolerance(self) -> float:
        return self.tolerance or (3.5 - 2.75 * self.complexity / 100)

    @property
    def palette_size(self) -> int:
        return self.palette or 32

    @property
    def detail_cost(self) -> float:
        return 0.04 * 2 ** ((50 - self.complexity) / 25)


@dataclass
class Work:
    """One deadline and cancellation signal shared by all planning stages."""

    deadline: float
    stop: Event = field(default_factory=Event)
    timings: dict[str, float] = field(default_factory=dict)

    @classmethod
    def start(cls, seconds: float, stop: Event | None = None) -> Work:
        return cls(time.monotonic() + seconds, stop if stop is not None else Event())

    @property
    def interrupted(self) -> bool:
        return self.stop.is_set() or time.monotonic() >= self.deadline

    @property
    def remaining(self) -> float:
        return max(0, self.deadline - time.monotonic())


@dataclass(frozen=True)
class Evidence:
    rgba: np.ndarray
    target: np.ndarray
    smooth: np.ndarray
    coarse: np.ndarray
    empty: np.ndarray
    foreground: np.ndarray
    line: np.ndarray
    drawn: np.ndarray
    darkness: np.ndarray
    texture: np.ndarray
    labels: np.ndarray
    source_size: tuple[int, int]
    offset: tuple[int, int]
    scale: tuple[float, float]
    background: tuple[float, float, float] | None
    grainy: bool


@dataclass(frozen=True)
class Region:
    id: int
    area: int
    paint: tuple[float, float, float]
    texture: float
    feature: float
    component: int


@dataclass(frozen=True)
class Boundary:
    id: int
    left: int
    right: int
    points: np.ndarray
    line_support: float


@dataclass(frozen=True)
class Graph:
    labels: np.ndarray
    regions: tuple[Region, ...]
    boundaries: tuple[Boundary, ...]
    junctions: tuple[tuple[float, float], ...]
    hidden: frozenset[int]


@dataclass(frozen=True)
class Candidate:
    svg: str
    label: str
    complexity: int
    metrics: dict
    decisions: tuple[dict, ...] = ()
    alternatives: tuple[Candidate, ...] = ()


class PlanningStoppedError(RuntimeError):
    """Cancellation before a fully validated drawing exists."""


class StageInterruptedError(RuntimeError):
    """Discard an unfinished planning stage at its next bounded checkpoint."""
