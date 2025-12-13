from __future__ import annotations

import base64
import difflib
import io
import logging
import math
import os
import queue
import random
import re
import time
from collections import deque
from typing import Optional, List, Any, Tuple, Dict

import multiprocessing as mp

import cairosvg
from openai import OpenAI
from PIL import Image

from svgizer.models import ChainState, SearchNode, Task, Result, INVALID_SCORE
from svgizer.diff_scores import pixel_diff_score
from svgizer.openai_iface import (
    summarize_changes,
    call_openai_for_svg,
    extract_svg_fragment,
    is_valid_svg,
)

# Prompt sampling temperature exploration
TEMP_STEP = 0.3
MAX_TEMP = 1.6

# If SVG text is too similar, bump temperature to avoid repeated generations
STALENESS_THRESHOLD = 0.995
STALE_HITS_BEFORE_BUMP = 1

# Resume file pattern written by this tool:
#   {base_name}_node{node_id:05d}_score{score:.6f}.svg
NODE_FILE_RE = re.compile(r"_node(\d+)_score([0-9.]+)\.svg$")


def guess_mime_type(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    return "image/png"


def encode_image_to_data_url(path: str) -> str:
    mime = guess_mime_type(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def png_bytes_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def rasterize_svg_to_png_bytes(svg_text: str) -> bytes:
    return cairosvg.svg2png(bytestring=svg_text.encode("utf-8"))


def is_stale(prev_svg: Optional[str], new_svg: str) -> bool:
    if prev_svg is None:
        return False
    ratio = difflib.SequenceMatcher(None, prev_svg, new_svg).ratio()
    return ratio >= STALENESS_THRESHOLD


def setup_logger(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(processName)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _fatal(msg: str) -> str:
    return "FATAL: " + msg


def worker_loop(
    task_q: mp.Queue,
    result_q: mp.Queue,
    original_data_url: str,
    original_png_bytes: bytes,
    log_level: str,
):
    setup_logger(log_level)
    log = logging.getLogger("worker")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY environment variable is not set.")
        return

    client = OpenAI(api_key=api_key)
    original_rgb = Image.open(io.BytesIO(original_png_bytes)).convert("RGB")

    while True:
        task: Any = task_q.get()
        if task is None:
            log.info("Worker received shutdown signal.")
            return

        assert isinstance(task, Task)
        parent = task.parent_state

        # Jitter around parent's prompt sampling temperature
        temperature = min(MAX_TEMP, max(0.0, parent.model_temperature + (task.proposal_index * 0.05)))
        log.info(
            f"Start task={task.task_id} parent={task.parent_id} proposal={task.proposal_index} temp={temperature:.2f}"
        )

        change_summary = None
        if parent.svg and parent.raster_data_url:
            try:
                change_summary = summarize_changes(
                    client=client,
                    original_data_url=original_data_url,
                    iter_index=task.parent_id,
                    rasterized_svg_data_url=parent.raster_data_url,
                )
                log.info(f"Summary task={task.task_id}:\n{change_summary}")
            except Exception as e:
                log.warning(f"Summary failed task={task.task_id}: {e}")

        # OpenAI call
        try:
            raw = call_openai_for_svg(
                client=client,
                original_data_url=original_data_url,
                iter_index=task.parent_id,
                temperature=temperature,
                svg_prev=parent.svg,
                svg_prev_invalid_msg=parent.invalid_msg,
                rasterized_svg_data_url=parent.raster_data_url,
                change_summary=change_summary,
                diversity_hint=f"parent={task.parent_id} proposal={task.proposal_index}",
            )
            svg = extract_svg_fragment(raw)
            log.info(f"OpenAI returned task={task.task_id} svg_len={len(svg)}")
        except Exception as e:
            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    proposal_index=task.proposal_index,
                    svg=None,
                    valid=False,
                    invalid_msg=_fatal(f"OpenAI call failed: {e}"),
                    raster_png=None,
                    score=INVALID_SCORE,
                    used_temperature=temperature,
                    change_summary=change_summary,
                )
            )
            continue

        # Validate SVG (invalid candidates are expected sometimes)
        valid, err = is_valid_svg(svg)
        if not valid:
            log.warning(f"Invalid SVG task={task.task_id}: {err}")
            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    proposal_index=task.proposal_index,
                    svg=svg,
                    valid=False,
                    invalid_msg=err,
                    raster_png=None,
                    score=INVALID_SCORE,
                    used_temperature=temperature,
                    change_summary=change_summary,
                )
            )
            continue

        # Rasterize and score
        try:
            png = rasterize_svg_to_png_bytes(svg)
            score = pixel_diff_score(original_rgb, png)
            log.info(f"Scored task={task.task_id} score={score:.6f}")
        except Exception as e:
            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    proposal_index=task.proposal_index,
                    svg=svg,
                    valid=False,
                    invalid_msg=_fatal(f"Rasterize/score failed: {e}"),
                    raster_png=None,
                    score=INVALID_SCORE,
                    used_temperature=temperature,
                    change_summary=change_summary,
                )
            )
            continue

        result_q.put(
            Result(
                task_id=task.task_id,
                parent_id=task.parent_id,
                proposal_index=task.proposal_index,
                svg=svg,
                valid=True,
                invalid_msg=None,
                raster_png=png,
                score=score,
                used_temperature=temperature,
                change_summary=change_summary,
            )
        )


def _load_resume_nodes(
    base_name: str,
    ext: str,
    log: logging.Logger,
    base_model_temperature: float,
) -> Tuple[List[SearchNode], Optional[SearchNode], int]:
    directory = os.path.dirname(base_name) or "."
    prefix = os.path.basename(base_name) + "_node"

    accepted: List[SearchNode] = []
    best_seen: Optional[SearchNode] = None
    max_id = 0

    for fn in os.listdir(directory):
        if not fn.startswith(prefix) or not fn.endswith(ext):
            continue
        m = NODE_FILE_RE.search(fn)
        if not m:
            continue

        node_id = int(m.group(1))
        score = float(m.group(2))
        path = os.path.join(directory, fn)

        try:
            with open(path, "r", encoding="utf-8") as f:
                svg = f.read()
            png = rasterize_svg_to_png_bytes(svg)
            raster_data_url = png_bytes_to_data_url(png)
        except Exception as e:
            log.warning(f"Resume: failed to load {fn}: {e}")
            continue

        state = ChainState(
            svg=svg,
            raster_data_url=raster_data_url,
            score=score,
            model_temperature=base_model_temperature,
            stale_hits=0,
            invalid_msg=None,
        )
        node = SearchNode(score=score, id=node_id, state=state)
        accepted.append(node)

        if best_seen is None or score < best_seen.score:
            best_seen = node
        max_id = max(max_id, node_id)

    accepted.sort(key=lambda n: n.id)
    if accepted:
        log.info(f"Resume: loaded {len(accepted)} prior nodes (max_id={max_id}).")
        if best_seen:
            log.info(f"Resume: best prior node={best_seen.id} score={best_seen.score:.6f}")

    return accepted, best_seen, max_id


def run_search(
    image_path: str,
    output_svg_path: str,
    max_accepts: int,
    proposals_per_step: int,
    base_model_temperature: float,
    workers: int,
    log_level: str,
    top_k: int,
    write_top_k_each: int,
    *,
    # Limits
    max_total_tasks: int = 10_000,
    max_wall_seconds: Optional[float] = None,
    # Resume
    resume: bool = True,
    # Simulated annealing (score-space)
    anneal_t0: float = 0.03,
    anneal_alpha: float = 0.995,
    anneal_min_t: float = 0.002,
    # Occasionally propose from best-ever rather than current
    propose_from_best_prob: float = 0.2,
) -> None:
    setup_logger(log_level)
    log = logging.getLogger("main")

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY environment variable is not set.")
    if not os.path.isfile(image_path):
        raise SystemExit(f"Input image '{image_path}' does not exist.")
    if max_accepts <= 0:
        raise SystemExit("max_accepts must be > 0")
    if proposals_per_step <= 0:
        raise SystemExit("proposals_per_step must be > 0")
    if workers <= 0:
        raise SystemExit("workers must be > 0")
    if top_k <= 0:
        raise SystemExit("top_k must be > 0")
    if max_total_tasks <= 0:
        raise SystemExit("max_total_tasks must be > 0")

    original_data_url = encode_image_to_data_url(image_path)

    original_img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    original_img.save(buf, format="PNG")
    original_png_bytes = buf.getvalue()

    base_name, ext = os.path.splitext(output_svg_path)
    if not ext:
        ext = ".svg"

    start_time = time.monotonic()

    def time_up() -> bool:
        return max_wall_seconds is not None and (time.monotonic() - start_time) >= max_wall_seconds

    # Multiprocessing
    ctx = mp.get_context("spawn")
    task_q: mp.Queue = ctx.Queue(maxsize=max(512, workers * 256))
    result_q: mp.Queue = ctx.Queue()

    procs: List[mp.Process] = []
    for _ in range(max(1, workers)):
        p = ctx.Process(
            target=worker_loop,
            args=(task_q, result_q, original_data_url, original_png_bytes, log_level),
            daemon=True,
        )
        p.start()
        procs.append(p)

    def shutdown_all(reason: str, terminate: bool = True) -> None:
        log.error(f"Canceling run: {reason}")
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

    # Root chain state
    root_state = ChainState(
        svg=None,
        raster_data_url=None,
        score=INVALID_SCORE,
        model_temperature=base_model_temperature,
        stale_hits=0,
        invalid_msg=None,
    )

    # Registry for looking up parent states by id (current/best + accepted history)
    node_states: Dict[int, ChainState] = {0: root_state}
    accepted_nodes: List[SearchNode] = []
    best_seen: Optional[SearchNode] = None
    next_node_id = 0

    # Resume: load prior nodes so we can continue from them
    if resume:
        prior_nodes, prior_best, max_id = _load_resume_nodes(base_name, ext, log, base_model_temperature)
        if prior_nodes:
            accepted_nodes.extend(prior_nodes)
            best_seen = prior_best
            next_node_id = max_id
            for n in prior_nodes:
                node_states[n.id] = n.state

    # SA chain:
    # - current: proposals usually refine from here
    # - best: best-ever seen (final output)
    if best_seen is not None:
        current_id = best_seen.id
        current_state = best_seen.state
        current_score = best_seen.score
        best_id = best_seen.id
        best_state = best_seen.state
        best_score = best_seen.score
    else:
        current_id = 0
        current_state = root_state
        current_score = INVALID_SCORE
        best_id = 0
        best_state = root_state
        best_score = INVALID_SCORE

    anneal_T = max(0.0, float(anneal_t0))

    # Track top-K for snapshot writing
    top_list: List[SearchNode] = []
    if best_seen is not None:
        top_list = sorted(accepted_nodes, key=lambda n: n.score)[: top_k]

    next_task_id = 1
    pending_tasks: deque[Task] = deque()
    global_proposal_counter = 0

    def accept_move(new_score: float, cur_score: float, T: float) -> bool:
        if cur_score >= INVALID_SCORE:
            return True
        if new_score <= cur_score:
            return True
        if T <= 0:
            return False
        d = new_score - cur_score
        try:
            p = math.exp(-d / T)
        except OverflowError:
            p = 0.0
        return random.random() < p

    def choose_parent_id() -> int:
        if best_id != 0 and random.random() < propose_from_best_prob:
            return best_id
        return current_id

    def try_enqueue_task(task: Task) -> bool:
        try:
            task_q.put_nowait(task)
            log.info(
                f"Enqueue task={task.task_id} parent={task.parent_id} proposal={task.proposal_index} "
                f"temp≈{min(MAX_TEMP, task.parent_state.model_temperature + (task.proposal_index*0.05)):.2f}"
            )
            return True
        except queue.Full:
            return False

    def pump_enqueues(budget: int = 8192) -> None:
        nonlocal next_task_id, global_proposal_counter
        if time_up():
            return

        # 1) Flush pending first
        for _ in range(budget):
            if not pending_tasks:
                break
            if next_task_id > max_total_tasks:
                return
            t = pending_tasks[0]
            if try_enqueue_task(t):
                pending_tasks.popleft()
                next_task_id += 1
            else:
                return

        # 2) Generate new tasks (pipelined)
        for _ in range(budget):
            if next_task_id > max_total_tasks:
                return

            parent_id = choose_parent_id()
            parent_state = node_states.get(parent_id, root_state)

            # proposal_index is small and reused to control jitter; global counter ensures diversity_hint changes
            proposal_index = global_proposal_counter % max(1, proposals_per_step)
            global_proposal_counter += 1

            task = Task(
                task_id=next_task_id,
                parent_id=parent_id,
                parent_state=parent_state,
                proposal_index=proposal_index,
            )

            if try_enqueue_task(task):
                next_task_id += 1
            else:
                pending_tasks.append(task)
                return

    def handle_result(res: Result) -> bool:
        nonlocal next_node_id
        nonlocal current_id, current_state, current_score
        nonlocal best_id, best_state, best_score
        nonlocal anneal_T
        nonlocal best_seen, top_list

        # Fatal worker errors cancel the run
        if res.invalid_msg and res.invalid_msg.startswith("FATAL:"):
            shutdown_all(res.invalid_msg, terminate=True)
            raise SystemExit(res.invalid_msg)

        # Reject invalid / unscored proposals
        if (not res.valid) or (res.svg is None) or (res.raster_png is None) or (res.score >= INVALID_SCORE):
            log.warning(
                f"Reject task={res.task_id} parent={res.parent_id} proposal={res.proposal_index} "
                f"reason={res.invalid_msg}"
            )
            return False

        # Parent state for staleness tracking / temp bumping
        parent_state = node_states.get(res.parent_id, root_state)
        next_temp = parent_state.model_temperature
        stale_hits = parent_state.stale_hits

        if parent_state.svg and is_stale(parent_state.svg, res.svg):
            stale_hits += 1
            if stale_hits >= STALE_HITS_BEFORE_BUMP and next_temp < MAX_TEMP:
                next_temp = min(MAX_TEMP, next_temp + TEMP_STEP)
                stale_hits = 0
                log.info(f"Staleness: bump model temp for parent={res.parent_id} -> {next_temp:.2f}")
        else:
            stale_hits = 0

        # Create a new accepted node (accepted in the sense "valid & scored"; SA decision is separate)
        next_node_id += 1
        state = ChainState(
            svg=res.svg,
            raster_data_url=png_bytes_to_data_url(res.raster_png),
            score=res.score,
            model_temperature=next_temp,
            stale_hits=stale_hits,
            invalid_msg=None,
        )
        node = SearchNode(score=res.score, id=next_node_id, state=state)
        accepted_nodes.append(node)
        node_states[node.id] = node.state

        # Write intermediate for resume/debugging
        iter_path = f"{base_name}_node{node.id:05d}_score{node.score:.6f}{ext}"
        try:
            with open(iter_path, "w", encoding="utf-8") as f:
                f.write(node.state.svg or "")
            log.info(f"Wrote {iter_path}")
        except Exception as e:
            shutdown_all(f"Failed to write {iter_path}: {e}", terminate=True)
            raise SystemExit(1)

        # Best-ever tracking
        if best_score >= INVALID_SCORE or node.score < best_score:
            best_id = node.id
            best_state = node.state
            best_score = node.score
            best_seen = node
            log.info(f"NEW BEST node={node.id} score={node.score:.6f}")

        # Maintain top-K list for snapshots only
        top_list.append(node)
        top_list.sort(key=lambda n: n.score)
        if len(top_list) > top_k:
            top_list[:] = top_list[:top_k]

        # SA decision: whether this node becomes the new "current"
        moved = accept_move(node.score, current_score, anneal_T)
        if moved:
            prev = current_score
            current_id = node.id
            current_state = node.state
            current_score = node.score
            anneal_T = max(float(anneal_min_t), anneal_T * float(anneal_alpha))
            log.info(
                f"ACCEPT move -> current=node{node.id} score={node.score:.6f} "
                f"(prev={prev:.6f}) T={anneal_T:.6f}"
            )
        else:
            log.info(
                f"REJECT move node{node.id} score={node.score:.6f} "
                f"(current={current_score:.6f}) T={anneal_T:.6f}"
            )

        # Periodic TOP-K snapshot writing
        if write_top_k_each > 0 and (node.id % write_top_k_each == 0):
            snap = sorted(top_list, key=lambda x: x.score)
            for rank, bn in enumerate(snap, start=1):
                pth = f"{base_name}_TOP_rank{rank:02d}_node{bn.id:05d}_score{bn.score:.6f}{ext}"
                try:
                    with open(pth, "w", encoding="utf-8") as f:
                        f.write(bn.state.svg or "")
                except Exception:
                    pass
            log.info(f"Wrote TOP snapshot (every {write_top_k_each} accepts)")

        return True

    # Initial fill
    pump_enqueues(budget=50_000)

    accepted_valid = 0
    while True:
        if time_up():
            log.warning("Stopping: wall-clock limit reached.")
            break
        if next_task_id > max_total_tasks:
            log.warning("Stopping: max_total_tasks reached.")
            break
        if accepted_valid >= max_accepts:
            break

        pump_enqueues(budget=8192)

        try:
            res: Result = result_q.get(timeout=0.2)
        except queue.Empty:
            if not pending_tasks and next_task_id > max_total_tasks:
                log.warning("Stopping: task budget exhausted and no pending tasks.")
                break
            continue

        if handle_result(res):
            accepted_valid += 1
            log.info(f"Accepted valid nodes: {accepted_valid}/{max_accepts} (best={best_score:.6f})")

    # Final output = best-ever
    if best_score >= INVALID_SCORE or best_state.svg is None:
        shutdown_all("No valid SVG produced.", terminate=True)
        raise SystemExit("No valid SVG produced.")

    try:
        with open(output_svg_path, "w", encoding="utf-8") as f:
            f.write(best_state.svg)
        log.info(f"Final SVG written to: {output_svg_path} (best score={best_score:.6f})")
    except Exception as e:
        shutdown_all(f"Failed to write final SVG '{output_svg_path}': {e}", terminate=True)
        raise SystemExit(1)

    shutdown_all("completed", terminate=False)
