import base64
import io
import logging
import multiprocessing as mp

from PIL import Image

from svgizer.image_utils import generate_diff_data_url, rasterize_svg_to_png_bytes
from svgizer.llm import LLMConfig, get_provider
from svgizer.score import get_scorer
from svgizer.score.complexity import svg_complexity
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

    except Exception as e:
        log.critical(f"Worker failed initialization: {e!r}")
        return

    while True:
        task = task_q.get()
        if task is None:
            break

        parent = task.parent_state

        try:
            if task.secondary_parent_state:
                config = LLMConfig(model=model_name, reasoning=reasoning)
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

                change_summary = worker_params.get("goal")

                if parent.payload.svg:
                    sum_prompt = build_summarize_prompt(
                        worker_params["image_data_url"],
                        parent_preview,
                        custom_goal=worker_params.get("goal"),
                        previous_summary=parent.payload.change_summary,
                    )
                    sum_config = LLMConfig(model=model_name, reasoning=reasoning)
                    change_summary = client.generate(sum_prompt, sum_config)

                diff_data_url = None
                if parent.payload.raster_data_url:
                    _, encoded = parent.payload.raster_data_url.split(",", 1)
                    cand_bytes = base64.b64decode(encoded)
                    diff_data_url = generate_diff_data_url(
                        worker_params["original_png_bytes"],
                        cand_bytes,
                        worker_params.get("image_long_side", 512),
                    )
                elif parent.payload.svg:
                    # Fallback for when write_lineage is False in tests
                    cand_bytes = rasterize_svg_to_png_bytes(
                        parent.payload.svg,
                        out_w=worker_params["original_w"],
                        out_h=worker_params["original_h"],
                    )
                    diff_data_url = generate_diff_data_url(
                        worker_params["original_png_bytes"],
                        cand_bytes,
                        worker_params.get("image_long_side", 512),
                    )

                gen_config = LLMConfig(model=model_name, reasoning=reasoning)
                gen_prompt = build_svg_gen_prompt(
                    worker_params["image_data_url"],
                    task.parent_id,
                    svg_prev=parent.payload.svg,
                    rasterized_svg_data_url=parent_preview
                    if parent.payload.svg
                    else None,
                    change_summary=change_summary,
                    diff_data_url=diff_data_url,
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
            complexity = svg_complexity(svg)

            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    worker_slot=task.worker_slot,
                    valid=True,
                    score=score,
                    payload=SvgResultPayload(
                        svg=svg, raster_png=png, change_summary=change_summary
                    ),
                    secondary_parent_id=task.secondary_parent_id,
                    complexity=complexity,
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
                    payload=SvgResultPayload(None, None, None),
                    invalid_msg=repr(e),
                    secondary_parent_id=task.secondary_parent_id,
                )
            )
