import io
import logging
import multiprocessing as mp

from PIL import Image

from svgizer.diff import get_scorer
from svgizer.image_utils import rasterize_svg_to_png_bytes
from svgizer.llm import LLMClient, LLMConfig
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


def worker_loop(task_q: mp.Queue, result_q: mp.Queue, worker_params: dict):
    setup_logger(worker_params["log_level"])
    log = logging.getLogger("worker")

    # Use the refactored LLM Client
    client = LLMClient()
    scorer = get_scorer(worker_params["scorer_type"])

    original_rgb = Image.open(io.BytesIO(worker_params["original_png_bytes"])).convert(
        "RGB"
    )
    scoring_ref = scorer.prepare_reference(original_rgb)

    while True:
        task = task_q.get()
        if task is None:
            break

        parent = task.parent_state
        temp = min(1.6, max(0.0, parent.model_temperature + (task.worker_slot * 0.07)))
        config = LLMConfig(temperature=temp)

        try:
            if task.secondary_parent_state:
                prompt = build_crossover_prompt(
                    worker_params["openai_original_data_url"],
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
                    sum_prompt = build_summarize_prompt(
                        worker_params["openai_original_data_url"], parent_preview
                    )
                    change_summary = client.generate(
                        sum_prompt, LLMConfig(temperature=1.0)
                    )

                gen_prompt = build_svg_gen_prompt(
                    worker_params["openai_original_data_url"],
                    task.parent_id,
                    svg_prev=parent.payload.svg,
                    change_summary=change_summary,
                    custom_goal=worker_params["goal"],
                )
                raw = client.generate(gen_prompt, config)

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
            log.error(f"Task {task.task_id} failed: {e}")
            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    worker_slot=task.worker_slot,
                    valid=False,
                    score=INVALID_SCORE,
                    used_temperature=temp,
                    payload=SvgResultPayload(None, None, None),
                    invalid_msg=str(e),
                    secondary_parent_id=task.secondary_parent_id,
                )
            )
