import base64
import io
import logging
import multiprocessing as mp
import random

from PIL import Image

from svgizer.image_utils import (
    generate_diff_data_url,
    rasterize_svg_to_png_bytes,
    resize_long_side,
)
from svgizer.llm import LLMConfig, get_provider
from svgizer.score.complexity import svg_complexity
from svgizer.score.utils import lab_l1
from svgizer.search import INVALID_SCORE, Result
from svgizer.svg.adapter import SvgResultPayload
from svgizer.svg.operations import (
    crossover,
    mutate_drop_style_property,
    mutate_numeric,
    mutate_remove_node,
    with_retries,
)
from svgizer.svg.prompts import (
    build_summarize_prompt,
    build_svg_gen_prompt,
    extract_svg_fragment,
    is_valid_svg,
)
from svgizer.utils import setup_logger


def _use_llm(has_svg: bool, llm_rate: float) -> bool:
    """Return True if this task should call the LLM."""
    return not has_svg or random.random() < llm_rate


def worker_loop(task_q: mp.Queue, result_q: mp.Queue, worker_params: dict):
    setup_logger(worker_params["log_level"])
    log = logging.getLogger("worker")

    try:
        provider_name = worker_params["llm_provider"]
        api_key = worker_params["api_key"]
        model_name = worker_params["llm_model"]
        reasoning = worker_params["reasoning"]
        llm_rate = float(worker_params["llm_rate"])

        client = get_provider(provider_name, api_key)

        # Setup fast local reference for CPU micro-search
        orig_img = Image.open(io.BytesIO(worker_params["original_png_bytes"])).convert(
            "RGB"
        )
        fast_eval_side = 128
        orig_img_fast = resize_long_side(orig_img, fast_eval_side)
        fast_w, fast_h = orig_img_fast.size

    except Exception as e:
        log.critical(f"Worker failed initialization: {e!r}")
        return

    while True:
        task = task_q.get()
        if task is None:
            break

        parent = task.parent_state
        has_svg = bool(parent.payload.svg)
        use_llm = task.force_llm or task.force_diverse or _use_llm(has_svg, llm_rate)

        try:
            if task.secondary_parent_state and task.secondary_parent_state.payload.svg:
                secondary_svg = task.secondary_parent_state.payload.svg
                svg = with_retries(
                    lambda a=parent.payload.svg, b=secondary_svg: crossover(a, b),
                    fallback=parent.payload.svg,
                )
                change_summary = "Local crossover"

            elif use_llm:
                # force_diverse: generate from scratch to seed a new lineage.
                # Normal LLM: summarise parent then refine.
                if task.force_diverse:
                    change_summary = "Diversity seed: fresh generation"
                    gen_config = LLMConfig(model=model_name, reasoning=reasoning)
                    gen_prompt = build_svg_gen_prompt(
                        worker_params["image_data_url"],
                        task.parent_id,
                        svg_prev=None,
                        force_diverse=True,
                    )
                    log.info(f"LLM call [diverse-seed] task={task.task_id}")
                    raw = client.generate(gen_prompt, gen_config)
                    svg = extract_svg_fragment(raw)
                else:
                    parent_preview = (
                        parent.payload.raster_preview_data_url
                        or parent.payload.raster_data_url
                    )
                    change_summary = worker_params.get("goal")

                    if has_svg:
                        sum_prompt = build_summarize_prompt(
                            worker_params["image_data_url"],
                            parent_preview,
                            custom_goal=worker_params.get("goal"),
                            previous_summary=parent.payload.change_summary,
                        )
                        sum_config = LLMConfig(model=model_name, reasoning=reasoning)
                        log.info(f"LLM call [summarize] task={task.task_id}")
                        change_summary = client.generate(sum_prompt, sum_config)

                    diff_data_url = None
                    if parent.payload.raster_data_url:
                        _, encoded = parent.payload.raster_data_url.split(",", 1)
                        cand_bytes = base64.b64decode(encoded)
                        diff_data_url = generate_diff_data_url(
                            worker_params["original_png_bytes"],
                            cand_bytes,
                            worker_params["image_long_side"],
                        )
                    elif has_svg:
                        cand_bytes = rasterize_svg_to_png_bytes(
                            parent.payload.svg,
                            out_w=worker_params["original_w"],
                            out_h=worker_params["original_h"],
                        )
                        diff_data_url = generate_diff_data_url(
                            worker_params["original_png_bytes"],
                            cand_bytes,
                            worker_params["image_long_side"],
                        )

                    gen_config = LLMConfig(model=model_name, reasoning=reasoning)
                    gen_prompt = build_svg_gen_prompt(
                        worker_params["image_data_url"],
                        task.parent_id,
                        svg_prev=parent.payload.svg,
                        rasterized_svg_data_url=parent_preview if has_svg else None,
                        change_summary=change_summary,
                        diff_data_url=diff_data_url,
                    )
                    log.info(
                        f"LLM call [generate] task={task.task_id} "
                        f"parent={task.parent_id} model={model_name}"
                    )
                    raw = client.generate(gen_prompt, gen_config)
                    svg = extract_svg_fragment(raw)

            else:
                # Local mutation: Micro-search via fast CPU diff
                parent_svg = parent.payload.svg
                best_svg = parent_svg
                best_fast_score = float("inf")
                best_summary = "Mutation: no improvement"

                # 1. Evaluate parent's baseline fast score
                try:
                    parent_png = rasterize_svg_to_png_bytes(
                        parent_svg, out_w=fast_w, out_h=fast_h
                    )
                    parent_img = Image.open(io.BytesIO(parent_png)).convert("RGB")
                    best_fast_score = lab_l1(orig_img_fast, parent_img)
                except Exception:
                    pass

                # 2. Try N rapid mutations, keep the one that improves the L1 score the most
                num_micro_mutations = 15

                for _ in range(num_micro_mutations):
                    roll = random.random()
                    if roll < 0.33:
                        cand_svg = with_retries(
                            lambda s=parent_svg: mutate_remove_node(s),
                            fallback=parent_svg,
                        )
                        summary = "Mutation: removed node"
                    elif roll < 0.66:
                        cand_svg = with_retries(
                            lambda s=parent_svg: mutate_numeric(s),
                            fallback=parent_svg,
                        )
                        summary = "Mutation: tweaked value"
                    else:
                        cand_svg = with_retries(
                            lambda s=parent_svg: mutate_drop_style_property(s),
                            fallback=parent_svg,
                        )
                        summary = "Mutation: dropped style property"

                    if cand_svg == parent_svg:
                        continue

                    try:
                        cand_png = rasterize_svg_to_png_bytes(
                            cand_svg, out_w=fast_w, out_h=fast_h
                        )
                        cand_img = Image.open(io.BytesIO(cand_png)).convert("RGB")
                        score = lab_l1(orig_img_fast, cand_img)

                        if score < best_fast_score:
                            best_fast_score = score
                            best_svg = cand_svg
                            best_summary = summary
                    except Exception:
                        continue

                svg = best_svg
                change_summary = best_summary

            valid, err = is_valid_svg(svg)
            if not valid:
                raise ValueError(err)

            png = rasterize_svg_to_png_bytes(
                svg,
                out_w=worker_params["original_w"],
                out_h=worker_params["original_h"],
            )
            complexity = svg_complexity(svg)

            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    worker_slot=task.worker_slot,
                    valid=True,
                    score=None,  # scored by main process via score_fn
                    payload=SvgResultPayload(
                        svg=svg, raster_png=png, change_summary=change_summary
                    ),
                    secondary_parent_id=task.secondary_parent_id,
                    complexity=complexity,
                    content=svg,
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
