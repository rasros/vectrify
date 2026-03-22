import io
import logging
import multiprocessing as mp
import os
import queue
import random
import time
from typing import Dict, List, Optional, Tuple

from PIL import Image

from svgizer.diff import get_scorer
from svgizer.image_utils import (
    downscale_png_bytes,
    png_bytes_to_data_url,
    rasterize_svg_to_png_bytes,
    make_preview_data_url,
)
from svgizer.models import ChainState, SearchNode, Task, Result, INVALID_SCORE
from svgizer.openai_iface import is_valid_svg
from svgizer.storage import StorageAdapter
from svgizer.utils import (
    setup_logger,
    is_stale,
    calculate_elite_prob,
    choose_from_top_k_weighted,
)
from svgizer.worker import worker_loop

TEMP_STEP = 0.3
MAX_TEMP = 1.6
STALE_HITS_BEFORE_BUMP = 1

# Internal (non-CLI) safety / tuning constants
MAX_TOTAL_TASKS = 10_000
TOP_K = 3
ELITE_PROB_START = 0.70
ELITE_PROB_END = 0.10


def run_search(
    image_path: str,
    storage: StorageAdapter,
    seed_svg_path: Optional[str],
    max_accepts: int,
    workers: int,
    base_model_temperature: float,
    openai_image_long_side: int,
    max_wall_seconds: Optional[float],
    log_level: str,
    scorer_type: str,
) -> None:
    setup_logger(log_level)
    log = logging.getLogger("main")

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY environment variable is not set.")
    if not os.path.isfile(image_path):
        raise SystemExit(f"Input image '{image_path}' does not exist.")

    # Initialize storage layer
    storage.initialize()

    original_img = Image.open(image_path).convert("RGB")
    original_w, original_h = original_img.size

    buf = io.BytesIO()
    original_img.save(buf, format="PNG")
    original_png_bytes = buf.getvalue()

    model_png_bytes = (
        downscale_png_bytes(original_png_bytes, openai_image_long_side)
        if openai_image_long_side and openai_image_long_side > 0
        else original_png_bytes
    )
    openai_original_data_url = png_bytes_to_data_url(model_png_bytes)

    start_time = time.monotonic()

    ctx = mp.get_context("spawn")
    task_q: mp.Queue = ctx.Queue(maxsize=max(64, workers * 8))
    result_q: mp.Queue = ctx.Queue()

    procs: List[mp.Process] = []
    for _ in range(max(1, workers)):
        p = ctx.Process(
            target=worker_loop,
            args=(
                task_q,
                result_q,
                openai_original_data_url,
                original_png_bytes,
                original_w,
                original_h,
                openai_image_long_side,
                log_level,
                scorer_type,
            ),
            daemon=True,
        )
        p.start()
        procs.append(p)

    def shutdown_all(reason: str, terminate: bool = True) -> None:
        (log.error if terminate else log.info)(
            f"{'Canceling run' if terminate else 'Shutting down'}: {reason}"
        )
        for _ in procs:
            try:
                task_q.put_nowait(None)
            except Exception:
                pass
        if terminate:
            for p in procs:
                try:
                    p.terminate()
                except Exception:
                    pass
        for p in procs:
            try:
                p.join(timeout=1.0)
            except Exception:
                pass

    root_state = ChainState(
        None, None, None, INVALID_SCORE, base_model_temperature, 0, None
    )
    node_states: Dict[int, ChainState] = {0: root_state}
    node_info: Dict[int, Tuple[int, float, str, Optional[str]]] = {}
    accepted_nodes: List[SearchNode] = []
    best_node: Optional[SearchNode] = None
    next_node_id = 0
    best_k: List[SearchNode] = []

    # Initialize the diff scorer
    scorer = get_scorer(scorer_type)
    scoring_ref = scorer.prepare_reference(original_img)

    # Attempt to load resume state via the storage adapter
    prior_nodes, _, max_id = storage.load_resume_nodes(
        log,
        base_model_temperature,
        original_w,
        original_h,
        openai_image_long_side,
    )

    if prior_nodes:
        log.info(
            f"Resuming from {len(prior_nodes)} prior nodes. Recalculating scores..."
        )

        # Recalculate scores for all loaded nodes because the scoring method might have changed
        for n in prior_nodes:
            if n.state.svg:
                full_png = rasterize_svg_to_png_bytes(
                    n.state.svg, out_w=original_w, out_h=original_h
                )
                new_score = scorer.score(scoring_ref, full_png)
                n.score = new_score
                n.state.score = new_score

        # Re-evaluate best_node based on the fresh scores
        best_node = min(prior_nodes, key=lambda n: n.score)

        accepted_nodes.extend(prior_nodes)
        next_node_id = max_id
        for n in prior_nodes:
            node_states[n.id] = n.state

            # Since these are loaded, we can just log their paths without summaries for now
            # (Lineage will be rewritten based on this if enabled)
            node_info[n.id] = (
                n.parent_id,
                n.score,
                f"resumed_node_{n.id}.svg",
                n.state.change_summary,
            )

        best_k = sorted(accepted_nodes, key=lambda n: n.score)[: max(1, TOP_K)]
        log.info(f"Resume scoring complete. Current best score: {best_node.score:.6f}")

    if seed_svg_path:
        try:
            seed_svg = storage.load_seed_svg(seed_svg_path)
            valid, err = is_valid_svg(seed_svg)
            if not valid:
                raise ValueError(err)

            full_png = rasterize_svg_to_png_bytes(
                seed_svg, out_w=original_w, out_h=original_h
            )
            seed_score = scorer.score(scoring_ref, full_png)

            seed_state = ChainState(
                svg=seed_svg,
                raster_data_url=png_bytes_to_data_url(full_png)
                if storage.write_lineage_enabled
                else None,
                raster_preview_data_url=make_preview_data_url(
                    full_png, openai_image_long_side
                ),
                score=seed_score,
                model_temperature=base_model_temperature,
                stale_hits=0,
                invalid_msg=None,
                change_summary=None,
            )
            next_node_id += 1
            seed_node = SearchNode(
                score=seed_score, id=next_node_id, parent_id=0, state=seed_state
            )
            accepted_nodes.append(seed_node)
            node_states[seed_node.id] = seed_state

            iter_path = storage.save_node(seed_node)
            node_info[seed_node.id] = (
                0,
                seed_score,
                iter_path,
                seed_state.change_summary,
            )

            if best_node is None or seed_score < best_node.score:
                best_node = seed_node
            best_k = sorted(accepted_nodes, key=lambda n: n.score)[: max(1, TOP_K)]
        except Exception as e:
            raise SystemExit(f"--seed-svg error: {e}")

    next_task_id = 1
    tasks_completed = 0
    accepted_valid = 0
    in_flight = 0

    while True:
        if (
            (max_wall_seconds and (time.monotonic() - start_time) >= max_wall_seconds)
            or (accepted_valid >= max_accepts)
            or (tasks_completed >= MAX_TOTAL_TASKS)
            or (in_flight == 0 and next_task_id > MAX_TOTAL_TASKS)
        ):
            break

        while in_flight < workers and next_task_id <= MAX_TOTAL_TASKS:
            pid = 0
            if accepted_nodes:
                progress = (
                    accepted_valid / float(max_accepts) if max_accepts > 0 else 0.0
                )
                if best_k and random.random() < calculate_elite_prob(
                    progress, ELITE_PROB_START, ELITE_PROB_END
                ):
                    pid = choose_from_top_k_weighted(best_k)
                elif best_node:
                    pid = best_node.id
                else:
                    pid = best_k[0].id

            worker_slot = in_flight % workers
            task_q.put(
                Task(next_task_id, pid, node_states.get(pid, root_state), worker_slot)
            )
            next_task_id += 1
            in_flight += 1

        try:
            res: Result = result_q.get(timeout=0.2)
        except queue.Empty:
            continue

        in_flight -= 1
        tasks_completed += 1

        if res.invalid_msg and res.invalid_msg.startswith("FATAL:"):
            shutdown_all(res.invalid_msg, terminate=True)
            raise SystemExit(res.invalid_msg)

        if not res.valid or res.score >= INVALID_SCORE:
            continue

        next_node_id += 1
        p_state = node_states.get(res.parent_id, root_state)
        next_temp = p_state.model_temperature
        stale_hits = p_state.stale_hits

        if p_state.svg and is_stale(p_state.svg, res.svg):
            stale_hits += 1
            if stale_hits >= STALE_HITS_BEFORE_BUMP and next_temp < MAX_TEMP:
                next_temp = min(MAX_TEMP, next_temp + TEMP_STEP)
                stale_hits = 0
        else:
            stale_hits = 0

        full_png = res.raster_png
        new_state = ChainState(
            svg=res.svg,
            raster_data_url=png_bytes_to_data_url(full_png)
            if storage.write_lineage_enabled
            else None,
            raster_preview_data_url=make_preview_data_url(
                full_png, openai_image_long_side
            ),
            score=res.score,
            model_temperature=next_temp,
            stale_hits=stale_hits,
            invalid_msg=None,
            change_summary=res.change_summary,
        )
        node = SearchNode(
            score=res.score,
            id=next_node_id,
            parent_id=res.parent_id,
            state=new_state,
        )
        accepted_nodes.append(node)
        node_states[node.id] = new_state
        accepted_valid += 1

        iter_path = storage.save_node(node)
        node_info[node.id] = (
            node.parent_id,
            node.score,
            iter_path,
            new_state.change_summary,
        )

        if res.change_summary:
            one_line = " | ".join(
                s.strip() for s in res.change_summary.splitlines() if s.strip()
            )
            log.info(
                f"CHANGE_SUMMARY node={node.id} parent={node.parent_id} "
                f"score={node.score:.6f}: {one_line}"
            )

        if best_node is None or node.score < best_node.score:
            best_node = node
            log.info(f"NEW BEST node={node.id} score={node.score:.6f}")

        best_k = sorted(accepted_nodes, key=lambda n: n.score)[: max(1, TOP_K)]

        if storage.write_lineage_enabled and (accepted_valid % 10 == 0):
            storage.write_lineage(node_info)

    if best_node is None or not best_node.state.svg:
        shutdown_all("No valid SVG produced.", terminate=True)
        raise SystemExit("No valid SVG produced.")

    storage.save_final_svg(best_node.state.svg)

    if storage.write_lineage_enabled:
        storage.write_lineage(node_info)

    shutdown_all("completed", terminate=False)
