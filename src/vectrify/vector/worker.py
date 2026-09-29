import contextlib
import dataclasses
import hashlib
import logging
import random
import signal
from typing import Any, Protocol

from vectrify.formats.svg.operations import apply_crossover, apply_mutation
from vectrify.formats.svg.prompts import is_valid_svg
from vectrify.formats.svg.selection import MutationScope
from vectrify.formats.svg.targets import element_targets
from vectrify.image_utils import (
    rasterize_svg_to_png_bytes,
)
from vectrify.search import Result
from vectrify.search.diversity import simhash
from vectrify.utils import setup_worker_logger
from vectrify.vector.payloads import VectorResultPayload


class NoChangeError(Exception):
    """An operator handed back the content it was given.

    Distinct from an invalid candidate: nothing is wrong with the markup, there
    is simply no new candidate to score. The operator that drew the blank still
    spent a draw, so it carries its own name -- the one that actually ran, not
    the one the task asked for -- for the policy to charge.
    """

    def __init__(self, operator: str) -> None:
        super().__init__(f"{operator} left the candidate unchanged")
        self.operator = operator


@dataclasses.dataclass
class WorkerContext:
    """All configuration a worker process needs to handle tasks."""

    # The reference at render size: candidates are rendered at original_w x
    # original_h and attributed against these bytes.
    original_png_bytes: bytes
    original_w: int
    original_h: int
    log_level: str = "WARNING"
    # Set it and a single-worker run repeats exactly.
    random_seed: int | None = None
    # Set by an editor operation: limits every mutation to these elements and
    # edit kinds, and turns crossover off.
    scope: MutationScope | None = None
    worker_index: int = 0
    log_queue: Any = None


class MessageQueue(Protocol):
    """The queue surface worker_loop needs.

    Declared as a protocol rather than mp.Queue because that is more than the
    loop uses: it only gets tasks and puts results. Production passes a
    multiprocessing queue and the tests pass a queue.Queue, and those two share
    no base class.
    """

    # Positional-only: the two queue classes name this argument differently
    # (item vs obj), which would otherwise fail protocol matching.
    def get(self) -> Any: ...

    def put(self, obj: Any, /) -> None: ...


def _build_local_contents(
    ctx: WorkerContext,
    task: Any,
    target_cache: dict[str, dict[int, float]],
    log: logging.Logger,
) -> tuple[str, str]:
    """Apply crossover when available, otherwise a targeted local mutation."""
    parent = task.parent_state
    if task.secondary_parent_state and task.secondary_parent_state.payload.content:
        content, origin = apply_crossover(
            parent.payload.content,
            task.secondary_parent_state.payload.content,
            ctx.scope,
        )
        return content, origin
    source = parent.payload.content
    key = hashlib.blake2b(source.encode(), digest_size=16).hexdigest()
    if key not in target_cache:
        if len(target_cache) > 64:
            target_cache.clear()
        try:
            target_cache[key] = element_targets(source, ctx.original_png_bytes)
        except Exception as exc:
            log.debug(f"Error attribution failed: {exc}")
            target_cache[key] = {}
    return apply_mutation(source, task.operator, target_cache[key], ctx.scope)


def _materialize(ctx: WorkerContext, task: Any, content: str, origin: str) -> Result:
    """Validate and render one candidate for the scorer."""
    valid, err = is_valid_svg(content)
    if not valid:
        raise ValueError(err)
    return Result(
        task_id=task.task_id,
        parent_id=task.parent_id,
        valid=True,
        measured=False,
        payload=VectorResultPayload(
            content=content,
            raster_png=rasterize_svg_to_png_bytes(
                content, out_w=ctx.original_w, out_h=ctx.original_h
            ),
            origin=origin,
        ),
        secondary_parent_id=task.secondary_parent_id,
        signature=simhash(content),
        operator=origin,
    )


def worker_loop(task_q: MessageQueue, result_q: MessageQueue, ctx: WorkerContext):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    setup_worker_logger(ctx.log_level, ctx.log_queue)
    log = logging.getLogger("worker")
    if ctx.random_seed is not None:
        random.seed(ctx.random_seed + ctx.worker_index)
    # Attribution costs two renders, and a parent is reused across many tasks,
    # so it is computed once per parent rather than once per task.
    target_cache: dict[str, dict[int, float]] = {}

    while True:
        try:
            task = task_q.get()
        except (OSError, EOFError, BrokenPipeError):
            break  # queue torn down during shutdown
        if task is None:
            break

        try:
            content, origin = _build_local_contents(ctx, task, target_cache, log)
            # An operator that could not find anything to change hands back the
            # parent it was given, and nothing downstream can tell that apart
            # from a real edit: the clone is rasterized, scored, stored, and
            # admitted to the pool wherever its parent sits, which reports back
            # to the operator policy as a success. Catch it here, where the
            # parent is still in hand, so the draw resolves as a failure
            # instead of as free reward.
            if content == task.parent_state.payload.content:
                raise NoChangeError(origin)
            result_q.put(_materialize(ctx, task, content, origin))
        except Exception as e:
            if isinstance(e, NoChangeError):
                log.debug(f"Task {task.task_id} produced no change: {e}")
            else:
                log.error(f"Task {task.task_id} failed: {e!r}")
            with contextlib.suppress(OSError, EOFError, BrokenPipeError):
                result_q.put(
                    Result(
                        task_id=task.task_id,
                        parent_id=task.parent_id,
                        valid=False,
                        measured=True,
                        payload=VectorResultPayload(None, None, None),
                        invalid_msg=repr(e),
                        secondary_parent_id=task.secondary_parent_id,
                        signature=None,
                        operator=e.operator if isinstance(e, NoChangeError) else None,
                    )
                )
