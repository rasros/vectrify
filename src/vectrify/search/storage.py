"""A StorageAdapter that keeps nothing on disk, for searches run in-process."""

from __future__ import annotations

from pathlib import Path
from typing import Generic

from vectrify.search.base import TState
from vectrify.search.models import SearchNode


class MemoryStorage(Generic[TState]):
    """Remembers the best node and the highest id; writes no files."""

    current_run_dir: Path | None = None

    def __init__(self) -> None:
        self.best: SearchNode[TState] | None = None
        self._max_id = 0

    def initialize(self) -> None:
        pass

    def save_node(
        self,
        node: SearchNode[TState],
        tasks_completed: int = 0,
        keep_content: bool = True,
    ) -> None:
        del tasks_completed, keep_content
        self._max_id = max(self._max_id, node.id)

    def save_best(self, node: SearchNode[TState]) -> None:
        self.best = node

    def record_eviction(self, node_id: int, tasks_completed: int) -> None:
        del node_id, tasks_completed

    def load_resume_nodes(self) -> list[tuple[int, str]]:
        return []

    @property
    def max_node_id(self) -> int:
        return self._max_id
