import io
import logging
import multiprocessing as mp
import os
from typing import Any

from openai import OpenAI
from PIL import Image

from svgizer.diff import get_scorer
from svgizer.image_utils import rasterize_svg_to_png_bytes
from svgizer.search import INVALID_SCORE, Result, Task
from svgizer.svg_adapter import SvgResultPayload
from svgizer.openai_iface import (
    call_openai_for_crossover,
    call_openai_for_svg,
    extract_svg_fragment,
    is_valid_svg,
    summarize_changes,
)
from svgizer.utils import setup_logger

MAX_TEMP = 1.6


def worker_loop(
    task_q: mp.Queue,
    result_q: mp.Queue,
    worker_params: dict,
):
    openai_original_data_url = worker_params["openai_original_data_url"]
    original_png_bytes = worker_params["original_png_bytes"]
    original_w = worker_params["original_w"]
    original_h = worker_params["original_h"]
    openai_image_long_side = worker_params["openai_image_long_side"]
    log_level = worker_params["log_level"]
    scorer_type = worker_params["scorer_type"]
    goal = worker_params["goal"]

    setup_logger(log_level)
    log = logging.getLogger("worker")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY environment variable is not set.")
        return

    client = OpenAI(api_key=api_key)

    scorer = get_scorer(scorer_type)
    original_rgb = Image.open(io.BytesIO(original_png_bytes)).convert("RGB")
    scoring_ref = scorer.prepare_reference(original_rgb)

    while True:
        task: Any = task_q.get()
        if task is None:
            return
        assert isinstance(task, Task)

        parent = task.parent_state
        temperature = min(
            MAX_TEMP, max(0.0, parent.model_temperature + (task.worker_slot * 0.07))
        )

        is_crossover = (
            task.secondary_parent_state is not None
            and task.secondary_parent_state.payload.svg is not None
        )
        change_summary = None

        if is_crossover:
            try:
                raw = call_openai_for_crossover(
                    client=client,
                    original_data_url=openai_original_data_url,
                    temperature=temperature,
                    svg_a=parent.payload.svg,
                    svg_b=task.secondary_parent_state.payload.svg,
                )
                svg = extract_svg_fragment(raw)
            except Exception as e:
                result_q.put(
                    Result(
                        task_id=task.task_id,
                        parent_id=task.parent_id,
                        worker_slot=task.worker_slot,
                        valid=False,
                        score=INVALID_SCORE,
                        used_temperature=temperature,
                        payload=SvgResultPayload(svg=None, raster_png=None, change_summary=None),
                        invalid_msg=f"FATAL: OpenAI Crossover call failed: {e}",
                        secondary_parent_id=task.secondary_parent_id,
                    )
                )
                continue
        else:
            parent_preview_data_url = (
                parent.payload.raster_preview_data_url or parent.payload.raster_data_url
            )

            if parent.payload.svg:
                try:
                    change_summary = summarize_changes(
                        client=client,
                        original_data_url=openai_original_data_url,
                        iter_index=task.parent_id,
                        rasterized_svg_data_url=parent_preview_data_url,
                    )
                except Exception as e:
                    log.debug(f"Summary failed task={task.task_id}: {e}")

            try:
                raw = call_openai_for_svg(
                    client=client,
                    original_data_url=openai_original_data_url,
                    iter_index=task.parent_id,
                    temperature=temperature,
                    svg_prev=parent.payload.svg,
                    svg_prev_invalid_msg=parent.payload.invalid_msg,
                    rasterized_svg_data_url=parent_preview_data_url,
                    change_summary=change_summary,
                    diversity_hint=f"parent={task.parent_id} worker={task.worker_slot}",
                    custom_goal=goal,
                )
                svg = extract_svg_fragment(raw)
            except Exception as e:
                result_q.put(
                    Result(
                        task_id=task.task_id,
                        parent_id=task.parent_id,
                        worker_slot=task.worker_slot,
                        valid=False,
                        score=INVALID_SCORE,
                        used_temperature=temperature,
                        payload=SvgResultPayload(svg=None, raster_png=None, change_summary=change_summary),
                        invalid_msg=f"FATAL: OpenAI call failed: {e}",
                        secondary_parent_id=task.secondary_parent_id,
                    )
                )
                continue

        valid, err = is_valid_svg(svg)
        if not valid:
            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    worker_slot=task.worker_slot,
                    valid=False,
                    score=INVALID_SCORE,
                    used_temperature=temperature,
                    payload=SvgResultPayload(svg=svg, raster_png=None, change_summary=change_summary),
                    invalid_msg=err,
                    secondary_parent_id=task.secondary_parent_id,
                )
            )
            continue

        try:
            png = rasterize_svg_to_png_bytes(svg, out_w=original_w, out_h=original_h)
            score = scorer.score(scoring_ref, png)
        except Exception as e:
            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    worker_slot=task.worker_slot,
                    valid=False,
                    score=INVALID_SCORE,
                    used_temperature=temperature,
                    payload=SvgResultPayload(svg=svg, raster_png=None, change_summary=change_summary),
                    invalid_msg=f"FATAL: Rasterize/score failed: {e}",
                    secondary_parent_id=task.secondary_parent_id,
                )
            )
            continue

        result_q.put(
            Result(
                task_id=task.task_id,
                parent_id=task.parent_id,
                worker_slot=task.worker_slot,
                valid=True,
                score=score,
                used_temperature=temperature,
                payload=SvgResultPayload(svg=svg, raster_png=png, change_summary=change_summary),
                invalid_msg=None,
                secondary_parent_id=task.secondary_parent_id,
            )
        )