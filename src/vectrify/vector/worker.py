"""Worker processes for the search: mutate a drawing and render the result.

A task is one mutation of the drawing the main process hands over, seeded by
that process so a one-worker run repeats exactly. Rendering happens here too,
since it is most of a task's cost; scoring does not, so every score in a run
comes from one scorer.
"""

from __future__ import annotations

import hashlib
import logging
import random
import signal
from dataclasses import dataclass

from vectrify.image_utils import rasterize_svg_to_png_bytes
from vectrify.svg.operations import apply_mutation
from vectrify.svg.prompts import is_valid_svg
from vectrify.svg.selection import MutationScope
from vectrify.svg.targets import element_targets

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerContext:
    """What a worker process needs, sent once when it starts."""

    # The reference at render size: candidates are rendered at original_w x
    # original_h and error is attributed against these bytes.
    original_png_bytes: bytes
    original_w: int
    original_h: int
    # Set by an editor operation: limits every mutation to these elements and
    # edit kinds.
    scope: MutationScope | None = None


@dataclass(frozen=True)
class Mutant:
    """One task's outcome. No content means nothing new came out of it."""

    operator: str | None
    content: str | None = None
    png: bytes | None = None


_context: WorkerContext | None = None
# Attribution costs two renders, and the parent stays the same until a child
# replaces it, so it is computed once per parent rather than once per task.
_targets: dict[str, dict[int, float]] = {}


def init_worker(context: WorkerContext) -> None:
    global _context
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _context = context
    _targets.clear()


def _targets_for(source: str, context: WorkerContext) -> dict[int, float]:
    key = hashlib.blake2b(source.encode(), digest_size=16).hexdigest()
    if key not in _targets:
        if len(_targets) > 64:
            _targets.clear()
        try:
            _targets[key] = element_targets(source, context.original_png_bytes)
        except Exception as exc:
            log.debug(f"Error attribution failed: {exc}")
            _targets[key] = {}
    return _targets[key]


def mutate(parent: str, operator: str | None, seed: int) -> Mutant:
    """Apply *operator* to *parent* and render the child."""
    context = _context
    assert context is not None, "init_worker has not run in this process"
    random.seed(seed)
    try:
        content, applied = apply_mutation(
            parent, operator, _targets_for(parent, context), context.scope
        )
        # An operator that found nothing to change hands back its parent. That
        # is no candidate, and it has to reach the policy as a blank draw
        # rather than as a child scoring exactly what its parent did.
        if content == parent:
            return Mutant(applied)
        valid, err = is_valid_svg(content)
        if not valid:
            raise ValueError(err)
        png = rasterize_svg_to_png_bytes(
            content, out_w=context.original_w, out_h=context.original_h
        )
        return Mutant(applied, content, png)
    except Exception as exc:
        log.debug(f"Mutation {operator} failed: {exc!r}")
        return Mutant(operator)
