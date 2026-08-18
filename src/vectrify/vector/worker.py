import contextlib
import dataclasses
import hashlib
import io
import logging
import random
import signal
from typing import Any, Protocol

from PIL import Image

from vectrify.formats.models import VectorResultPayload
from vectrify.image_utils import (
    png_bytes_to_data_url,
    resize_long_side,
)
from vectrify.llm import LLMConfig, get_provider
from vectrify.search import Result
from vectrify.search.diversity import simhash
from vectrify.utils import setup_worker_logger


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

    format_plugin: Any
    image_data_url: str
    original_png_bytes: bytes
    original_w: int
    original_h: int
    resolution_llm: int
    log_level: str
    log_file: str | None
    goal: str | None
    # The input file's name. Often the only place the subject is stated, and
    # the model is otherwise working from the picture alone.
    source_name: str | None
    llm_provider: str
    llm_model: str
    reasoning: str
    api_key: str | None
    # Set it and a single-worker run repeats exactly.
    random_seed: int | None = None
    worker_index: int = 0
    log_queue: Any = None
    llm_in_flight: Any = None


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


def worker_loop(task_q: MessageQueue, result_q: MessageQueue, ctx: WorkerContext):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    setup_worker_logger(ctx.log_level, ctx.log_queue)
    log = logging.getLogger("worker")

    try:
        plugin = ctx.format_plugin
        # Attribution costs two renders, and a parent is reused across many
        # tasks, so it is computed once per parent rather than once per task.
        target_cache: dict[str, dict[int, float]] = {}

        if ctx.random_seed is not None:
            random.seed(ctx.random_seed + ctx.worker_index)

    except Exception as e:
        log.critical(f"Worker failed initialization: {e!r}")
        return

    # Built on first use: constructing one resolves an API key, which a run that
    # never calls the LLM should not need.
    client: Any = None

    while True:
        try:
            task = task_q.get()
        except (OSError, EOFError, BrokenPipeError):
            break  # queue torn down during shutdown
        if task is None:
            break

        parent = task.parent_state
        has_content = bool(parent.payload.content)

        # LLM calls happen only in an epoch's seed batch; the engine decides.
        use_llm = task.force_llm
        llm_type = None

        try:
            if use_llm:
                llm_type = "llm-generate"
                if client is None:
                    client = get_provider(ctx.llm_provider, ctx.api_key)
                if ctx.llm_in_flight is not None:
                    with ctx.llm_in_flight.get_lock():
                        ctx.llm_in_flight.value += 1
                try:
                    parent_preview = (
                        parent.payload.raster_preview_data_url
                        or parent.payload.raster_data_url
                    )

                    gen_config = LLMConfig(model=ctx.llm_model, reasoning=ctx.reasoning)
                    gen_prompt = plugin.build_generate_prompt(
                        ctx.image_data_url,
                        task.parent_id,
                        content_prev=parent.payload.content,
                        raster_preview_url=parent_preview if has_content else None,
                        goal=ctx.goal,
                        canvas=(ctx.original_w, ctx.original_h),
                        source_name=ctx.source_name,
                    )
                    log.debug(
                        f"LLM call [generate] task={task.task_id} "
                        f"parent={task.parent_id} model={ctx.llm_model}"
                    )
                    raw = client.generate(gen_prompt, gen_config)
                    content = (
                        plugin.apply_edit(parent.payload.content, raw)
                        if has_content
                        else plugin.extract_from_llm(raw)
                    )
                    origin = "llm edit"
                finally:
                    if ctx.llm_in_flight is not None:
                        with ctx.llm_in_flight.get_lock():
                            ctx.llm_in_flight.value -= 1

            elif (
                task.secondary_parent_state
                and task.secondary_parent_state.payload.content
            ):
                secondary_content = task.secondary_parent_state.payload.content
                content, origin = plugin.crossover(
                    parent.payload.content,
                    secondary_content,
                )

            else:
                source = parent.payload.content
                key = hashlib.blake2b(source.encode(), digest_size=16).hexdigest()
                if key not in target_cache:
                    if len(target_cache) > 64:
                        target_cache.clear()
                    try:
                        target_cache[key] = plugin.element_targets(
                            source, ctx.original_png_bytes
                        )
                    except Exception as exc:
                        log.debug(f"Error attribution failed: {exc}")
                        target_cache[key] = {}
                content, origin = plugin.mutate(
                    source, task.operator, target_cache[key]
                )

            # An operator that could not find anything to change hands back the
            # parent it was given, and nothing downstream can tell that apart
            # from a real edit: the clone is rasterized, scored, stored, and
            # admitted to the pool wherever its parent sits, which reports back
            # to the operator policy as a success. Catch it here, where the
            # parent is still in hand, so the draw resolves as a failure
            # instead of as free reward.
            if not use_llm and content == parent.payload.content:
                raise NoChangeError(origin)

            valid, err = plugin.validate(content)
            if not valid:
                raise ValueError(err)

            png = plugin.rasterize(
                content,
                out_w=ctx.original_w,
                out_h=ctx.original_h,
            )
            signature = simhash(content)

            full_img = Image.open(io.BytesIO(png)).convert("RGB")
            preview_img = resize_long_side(full_img, ctx.resolution_llm)
            preview_buf = io.BytesIO()
            preview_img.save(preview_buf, format="PNG")
            preview_data_url = png_bytes_to_data_url(preview_buf.getvalue())

            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    valid=True,
                    measured=False,
                    payload=VectorResultPayload(
                        content=content,
                        raster_png=png,
                        origin=origin,
                        raster_preview_data_url=preview_data_url,
                    ),
                    secondary_parent_id=task.secondary_parent_id,
                    metrics={},
                    signature=signature,
                    llm_type=llm_type,
                    # What actually ran, not what was asked for: crossover can
                    # fall back to mutation, and a task can name an operator
                    # this backend does not have.
                    operator=None if use_llm else origin,
                )
            )

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
                        llm_type=llm_type,
                        operator=e.operator if isinstance(e, NoChangeError) else None,
                    )
                )
