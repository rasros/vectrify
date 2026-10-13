"""Shared planner state and proposal records, independent of search execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectrify.document.model import Document
    from vectrify.refine.cel_plan.component_edits import ComponentEdit
    from vectrify.refine.cel_plan.local import Box, Snapshot
    from vectrify.refine.cel_plan.ownership import Partition


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
