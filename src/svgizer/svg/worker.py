import io
import logging
import multiprocessing as mp

from PIL import Image

from svgizer.diff import get_scorer
from svgizer.image_utils import rasterize_svg_to_png_bytes
from svgizer.llm import LLMConfig, get_provider
from svgizer.search import INVALID_SCORE, Result
from svgizer.svg.adapter import SvgResultPayload
from svgizer.svg.prompts import (
    build_crossover_prompt,
    build_summarize_prompt,
    build_svg_gen_prompt,
    extract_svg_fragment,
    is_valid_svg,
)
from svgizer.utils import setup_logger

_SUMMARY_TEMP_MULTIPLIER = 1.2


def worker_loop(task_q: mp.Queue, result_q: mp.Queue, worker_params: dict):
    setup_logger(worker_params["log_level"])
    log = logging.getLogger("worker")

    try:
        provider_name = worker_params.get("llm_provider", "openai")
        api_key = worker_params.get("api_key")
        model_name = worker_params.get("llm_model", "gpt-5.4")
        reasoning = worker_params.get("reasoning", "medium")

        client = get_provider(provider_name, api_key)
        scorer = get_scorer(worker_params["scorer_type"])

        original_rgb = Image.open(
            io.BytesIO(worker_params["original_png_bytes"])
        ).convert("RGB")
        scoring_ref = scorer.prepare_reference(original_rgb)

        worker_max_temp = worker_params.get("worker_max_temp", 1.6)
        worker_temp_step = worker_params.get("worker_temp_step", 0.07)

    except Exception as e:
        log.critical(f"Worker failed initialization: {e!r}")
        return

    while True:
        task = task_q.get()
        if task is None:
            break

        parent = task.parent_state
        # Base slot offset logic
        temp = min(
            worker_max_temp,
            max(0.0, parent.model_temperature + (task.worker_slot * worker_temp_step)),
        )

        try:
            if task.secondary_parent_state:
                config = LLMConfig(
                    model=model_name, temperature=temp, reasoning=reasoning
                )
                prompt = build_crossover_prompt(
                    worker_params["image_data_url"],
                    parent.payload.svg,
                    task.secondary_parent_state.payload.svg,
                )
                raw = client.generate(prompt, config)
                change_summary = "Genetic Crossover"
            else:
                parent_preview = (
                    parent.payload.raster_preview_data_url
                    or parent.payload.raster_data_url
                )
                change_summary = None

                if parent.payload.svg:
                    # Bump summary temperature
                    sum_temp = min(worker_max_temp, temp * _SUMMARY_TEMP_MULTIPLIER)
                    sum_prompt = build_summarize_prompt(
                        worker_params["image_data_url"], parent_preview
                    )
                    sum_config = LLMConfig(
                        model=model_name, temperature=sum_temp, reasoning=reasoning
                    )
                    change_summary = client.generate(sum_prompt, sum_config)

                gen_config = LLMConfig(
                    model=model_name, temperature=temp, reasoning=reasoning
                )
                gen_prompt = build_svg_gen_prompt(
                    worker_params["image_data_url"],
                    task.parent_id,
                    svg_prev=parent.payload.svg,
                    change_summary=change_summary,
                    custom_goal=worker_params["goal"],
                )
                raw = client.generate(gen_prompt, gen_config)

            svg = extract_svg_fragment(raw)
            valid, err = is_valid_svg(svg)
            if not valid:
                raise ValueError(err)

            png = rasterize_svg_to_png_bytes(
                svg,
                out_w=worker_params["original_w"],
                out_h=worker_params["original_h"],
            )
            score = scorer.score(scoring_ref, png)

            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    worker_slot=task.worker_slot,
                    valid=True,
                    score=score,
                    used_temperature=temp,
                    payload=SvgResultPayload(
                        svg=svg, raster_png=png, change_summary=change_summary
                    ),
                    secondary_parent_id=task.secondary_parent_id,
                )
            )

        except Exception as e:
            log.error(f"Task {task.task_id} failed: {e!r}")
            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    worker_slot=task.worker_slot,
                    valid=False,
                    score=INVALID_SCORE,
                    used_temperature=temp,
                    payload=SvgResultPayload(None, None, None),
                    invalid_msg=repr(e),
                    secondary_parent_id=task.secondary_parent_id,
                )
            )
