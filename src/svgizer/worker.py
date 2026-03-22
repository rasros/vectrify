import io
import logging
import multiprocessing as mp
import os
from typing import Any

from openai import OpenAI
from PIL import Image

from svgizer.diff_scores import get_default_scorer
from svgizer.image_utils import rasterize_svg_to_png_bytes
from svgizer.models import Result, Task, INVALID_SCORE
from svgizer.openai_iface import (
    summarize_changes,
    call_openai_for_svg,
    extract_svg_fragment,
    is_valid_svg,
)
from svgizer.utils import setup_logger

MAX_TEMP = 1.6


def worker_loop(
    task_q: mp.Queue,
    result_q: mp.Queue,
    openai_original_data_url: str,
    original_png_bytes: bytes,
    original_w: int,
    original_h: int,
    openai_image_long_side: int,
    log_level: str,
):
    setup_logger(log_level)
    log = logging.getLogger("worker")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY environment variable is not set.")
        return

    client = OpenAI(api_key=api_key)

    # Prepare scoring reference ONCE per worker using the new diff scorer factory.
    scorer = get_default_scorer()
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

        # Prefer downscaled preview; fallback to full-res data-url.
        parent_preview_data_url = (
            parent.raster_preview_data_url or parent.raster_data_url
        )

        change_summary = None
        if parent.svg:
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
                svg_prev=parent.svg,
                svg_prev_invalid_msg=parent.invalid_msg,
                rasterized_svg_data_url=parent_preview_data_url,
                change_summary=change_summary,
                diversity_hint=f"parent={task.parent_id} worker={task.worker_slot}",
            )
            svg = extract_svg_fragment(raw)
        except Exception as e:
            result_q.put(
                Result(
                    task.task_id,
                    task.parent_id,
                    task.worker_slot,
                    None,
                    False,
                    f"FATAL: OpenAI call failed: {e}",
                    None,
                    INVALID_SCORE,
                    temperature,
                    change_summary,
                )
            )
            continue

        valid, err = is_valid_svg(svg)
        if not valid:
            result_q.put(
                Result(
                    task.task_id,
                    task.parent_id,
                    task.worker_slot,
                    svg,
                    False,
                    err,
                    None,
                    INVALID_SCORE,
                    temperature,
                    change_summary,
                )
            )
            continue

        try:
            png = rasterize_svg_to_png_bytes(svg, out_w=original_w, out_h=original_h)
            # Use the injected scorer instead of the hardcoded pixel_diff_score
            score = scorer.score(scoring_ref, png)
        except Exception as e:
            result_q.put(
                Result(
                    task.task_id,
                    task.parent_id,
                    task.worker_slot,
                    svg,
                    False,
                    f"FATAL: Rasterize/score failed: {e}",
                    None,
                    INVALID_SCORE,
                    temperature,
                    change_summary,
                )
            )
            continue

        result_q.put(
            Result(
                task.task_id,
                task.parent_id,
                task.worker_slot,
                svg,
                True,
                None,
                png,
                score,
                temperature,
                change_summary,
            )
        )
