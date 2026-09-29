"""The one GPU admission gate shared by editor jobs, searches and workers.

GPU fitting and the vision evaluator each fit comfortably on the device alone
but not together, so every user of the GPU in this process holds this gate.
It is a spawn-context semaphore rather than a thread lock so the same object
can be handed to search worker processes, which then queue on it too.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

_GATE: Any = None


def gpu_gate() -> Any:
    """The process-wide GPU semaphore, created on first use."""
    global _GATE
    if _GATE is None:
        _GATE = mp.get_context("spawn").Semaphore(1)
    return _GATE
