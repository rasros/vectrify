from __future__ import annotations

import base64
import difflib
import io
import logging
import os
import queue
import random
import re
import time
from typing import Optional, List, Any, Tuple, Dict

import multiprocessing as mp

import cairosvg
from openai import OpenAI
from PIL import Image

from svgizer.models import ChainState, SearchNode, Task, Result, INVALID_SCORE
from svgizer.diff_scores import pixel_diff_score, get_scoring_reference
from svgizer.openai_iface import summarize_changes, call_openai_for_svg, extract_svg_fragment, is_valid_svg

TEMP_STEP = 0.3
MAX_TEMP = 1.6
STALENESS_THRESHOLD = 0.995
STALE_HITS_BEFORE_BUMP = 1

# Only used for legacy file loading
NODE_FILE_RE_NEW = re.compile(r"^score([0-9.]+)_node(\d+)_parent(\d+)\.svg$")
NODE_FILE_RE_OLD = re.compile(r"_node(\d+)_score([0-9.]+)\.svg$")


def png_bytes_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def downscale_png_bytes(png_bytes: bytes, long_side: int) -> bytes:
    if long_side <= 0:
        return png_bytes

    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    w, h = im.size
    if max(w, h) <= long_side:
        # If already small, return original bytes to avoid re-compression artifacts/time
        return png_bytes

    if w >= h:
        new_w = long_side
        new_h = int(round(h * (long_side / float(w))))
    else:
        new_h = long_side
        new_w = int(round(w * (long_side / float(h))))

    new_w = max(1, new_w)
    new_h = max(1, new_h)

    im2 = im.resize((new_w, new_h), resample=Image.BILINEAR)
    out = io.BytesIO()
    im2.save(out, format="PNG")
    return out.getvalue()


def rasterize_svg_to_png_bytes(svg_text: str, *, out_w: int, out_h: int) -> bytes:
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"Invalid raster target size: {out_w}x{out_h}")
    return cairosvg.svg2png(
        bytestring=svg_text.encode("utf-8"),
        output_width=out_w,
        output_height=out_h,
    )


def is_stale(prev_svg: Optional[str], new_svg: str) -> bool:
    if prev_svg is None:
        return False
    if prev_svg == new_svg:
        return True
    ratio = difflib.SequenceMatcher(None, prev_svg, new_svg).ratio()
    return ratio >= STALENESS_THRESHOLD


def setup_logger(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(processName)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _elite_prob(progress01: float, p_start: float, p_end: float) -> float:
    progress01 = max(0.0, min(1.0, progress01))
    return float(p_start + (p_end - p_start) * progress01)


def _choose_from_top_k_weighted(best_k: List[SearchNode]) -> int:
    if not best_k:
        return 0
    weights = [1.0 / (i + 1.0) for i in range(len(best_k))]
    node = random.choices(best_k, weights=weights, k=1)[0]
    return node.id


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

    # Prepare scoring reference ONCE per worker.
    original_rgb = Image.open(io.BytesIO(original_png_bytes)).convert("RGB")
    scoring_ref = get_scoring_reference(original_rgb)

    while True:
        task: Any = task_q.get()
        if task is None:
            return
        assert isinstance(task, Task)

        parent = task.parent_state
        temperature = min(MAX_TEMP, max(0.0, parent.model_temperature + (task.worker_slot * 0.07)))

        # Prefer downscaled preview; fallback to full-res data-url.
        parent_preview_data_url = parent.raster_preview_data_url or parent.raster_data_url

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
            result_q.put(Result(task.task_id, task.parent_id, task.worker_slot, None, False,
                               f"FATAL: OpenAI call failed: {e}", None, INVALID_SCORE, temperature, change_summary))
            continue

        valid, err = is_valid_svg(svg)
        if not valid:
            result_q.put(Result(task.task_id, task.parent_id, task.worker_slot, svg, False,
                               err, None, INVALID_SCORE, temperature, change_summary))
            continue

        try:
            png = rasterize_svg_to_png_bytes(svg, out_w=original_w, out_h=original_h)
            score = pixel_diff_score(scoring_ref, png)
        except Exception as e:
            result_q.put(Result(task.task_id, task.parent_id, task.worker_slot, svg, False,
                               f"FATAL: Rasterize/score failed: {e}", None, INVALID_SCORE, temperature, change_summary))
            continue

        result_q.put(Result(task.task_id, task.parent_id, task.worker_slot, svg, True,
                           None, png, score, temperature, change_summary))


def _make_preview_data_url(full_png: bytes, openai_image_long_side: int) -> str:
    preview_png = downscale_png_bytes(full_png, openai_image_long_side)
    return png_bytes_to_data_url(preview_png)


def _load_resume_nodes(
    nodes_dir: str,
    base_name: str,
    ext: str,
    log: logging.Logger,
    base_model_temperature: float,
    original_w: int,
    original_h: int,
    openai_image_long_side: int,
) -> Tuple[List[SearchNode], Optional[SearchNode], int]:
    accepted: List[SearchNode] = []
    best_seen: Optional[SearchNode] = None
    max_id = 0

    scan_paths: List[Tuple[str, str]] = []
    if os.path.isdir(nodes_dir):
        scan_paths.append((nodes_dir, "new"))
    out_dir = os.path.dirname(base_name) or "."
    scan_paths.append((out_dir, "old"))

    for directory, mode in scan_paths:
        try:
            filenames = os.listdir(directory)
        except Exception:
            continue

        for fn in filenames:
            if not fn.endswith(ext):
                continue

            path = os.path.join(directory, fn)
            if mode == "new":
                m = NODE_FILE_RE_NEW.match(fn)
                if not m:
                    continue
                score = float(m.group(1))
                node_id = int(m.group(2))
                parent_id = int(m.group(3))
            else:
                m = NODE_FILE_RE_OLD.search(fn)
                if not m:
                    continue
                node_id = int(m.group(1))
                score = float(m.group(2))
                parent_id = 0

            try:
                with open(path, "r", encoding="utf-8") as f:
                    svg = f.read()
                full_png = rasterize_svg_to_png_bytes(svg, out_w=original_w, out_h=original_h)
                raster_preview_data_url = _make_preview_data_url(full_png, openai_image_long_side)
            except Exception as e:
                log.warning(f"Resume: failed to load {path}: {e}")
                continue

            state = ChainState(
                svg=svg,
                raster_data_url=None,  # avoid big RAM usage for resume nodes
                raster_preview_data_url=raster_preview_data_url,
                score=score,
                model_temperature=base_model_temperature,
                stale_hits=0,
                invalid_msg=None,
            )
            node = SearchNode(score=score, id=node_id, parent_id=parent_id, state=state)
            accepted.append(node)

            if best_seen is None or node.score < best_seen.score:
                best_seen = node
            max_id = max(max_id, node_id)

    accepted.sort(key=lambda n: n.id)
    return accepted, best_seen, max_id


def run_search(
    image_path: str,
    output_svg_path: str,
    seed_svg_path: Optional[str],
    max_accepts: int,
    workers: int,
    base_model_temperature: float,
    openai_image_long_side: int,
    max_total_tasks: int,
    max_wall_seconds: Optional[float],
    resume: bool,
    top_k: int,
    elite_prob_start: float,
    elite_prob_end: float,
    write_lineage: bool,
    log_level: str,
) -> None:
    setup_logger(log_level)
    log = logging.getLogger("main")

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY environment variable is not set.")
    if not os.path.isfile(image_path):
        raise SystemExit(f"Input image '{image_path}' does not exist.")

    original_img = Image.open(image_path).convert("RGB")
    original_w, original_h = original_img.size

    buf = io.BytesIO()
    original_img.save(buf, format="PNG")
    original_png_bytes = buf.getvalue()

    # Always give OpenAI a PNG data URL; optionally downscale first.
    model_png_bytes = (
        downscale_png_bytes(original_png_bytes, openai_image_long_side)
        if openai_image_long_side and openai_image_long_side > 0
        else original_png_bytes
    )
    openai_original_data_url = png_bytes_to_data_url(model_png_bytes)

    base_name, ext = os.path.splitext(output_svg_path)
    if not ext:
        ext = ".svg"

    out_dir = os.path.dirname(base_name) or "."
    nodes_dir = os.path.join(out_dir, os.path.basename(base_name) + "_nodes")
    os.makedirs(nodes_dir, exist_ok=True)
    lineage_csv_path = os.path.join(out_dir, os.path.basename(base_name) + "_lineage.csv")

    start_time = time.monotonic()

    ctx = mp.get_context("spawn")
    task_q: mp.Queue = ctx.Queue(maxsize=max(64, workers * 8))
    result_q: mp.Queue = ctx.Queue()

    procs: List[mp.Process] = []
    for _ in range(max(1, workers)):
        p = ctx.Process(
            target=worker_loop,
            args=(
                task_q, result_q, openai_original_data_url, original_png_bytes,
                original_w, original_h, openai_image_long_side, log_level
            ),
            daemon=True,
        )
        p.start()
        procs.append(p)

    def shutdown_all(reason: str, terminate: bool = True) -> None:
        (log.error if terminate else log.info)(f"{'Canceling run' if terminate else 'Shutting down'}: {reason}")
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

    def write_lineage_files(node_info: Dict[int, Tuple[int, float, str]]) -> None:
        if not write_lineage:
            return
        try:
            with open(lineage_csv_path, "w", encoding="utf-8") as csv_file:
                csv_file.write("node_id,parent_id,score,path\n")
                for nid in sorted(node_info.keys()):
                    pid, sc, pth = node_info[nid]
                    csv_file.write(f"{nid},{pid},{sc:.6f},{pth}\n")
        except Exception as e:
            log.warning(f"Failed writing lineage files: {e}")

    root_state = ChainState(None, None, None, INVALID_SCORE, base_model_temperature, 0, None)
    node_states: Dict[int, ChainState] = {0: root_state}
    node_info: Dict[int, Tuple[int, float, str]] = {}
    accepted_nodes: List[SearchNode] = []
    best_node: Optional[SearchNode] = None
    next_node_id = 0
    best_k: List[SearchNode] = []

    if resume:
        prior_nodes, prior_best, max_id = _load_resume_nodes(
            nodes_dir, base_name, ext, log, base_model_temperature,
            original_w, original_h, openai_image_long_side
        )
        if prior_nodes:
            accepted_nodes.extend(prior_nodes)
            best_node = prior_best
            next_node_id = max_id
            for n in prior_nodes:
                node_states[n.id] = n.state
            best_k = sorted(accepted_nodes, key=lambda n: n.score)[: max(1, top_k)]

    if seed_svg_path:
        try:
            with open(seed_svg_path, "r", encoding="utf-8") as f:
                seed_svg = f.read()
            valid, err = is_valid_svg(seed_svg)
            if not valid:
                raise ValueError(err)

            full_png = rasterize_svg_to_png_bytes(seed_svg, out_w=original_w, out_h=original_h)
            seed_score = pixel_diff_score(get_scoring_reference(original_img), full_png)

            seed_state = ChainState(
                svg=seed_svg,
                raster_data_url=png_bytes_to_data_url(full_png) if write_lineage else None,
                raster_preview_data_url=_make_preview_data_url(full_png, openai_image_long_side),
                score=seed_score,
                model_temperature=base_model_temperature,
                stale_hits=0,
                invalid_msg=None,
            )
            next_node_id += 1
            seed_node = SearchNode(score=seed_score, id=next_node_id, parent_id=0, state=seed_state)
            accepted_nodes.append(seed_node)
            node_states[seed_node.id] = seed_state

            fn = f"score{seed_score:012.6f}_node{seed_node.id:05d}_parent00000{ext}"
            iter_path = os.path.join(nodes_dir, fn)
            with open(iter_path, "w", encoding="utf-8") as f:
                f.write(seed_svg)
            node_info[seed_node.id] = (0, seed_score, iter_path)

            if best_node is None or seed_score < best_node.score:
                best_node = seed_node
            best_k = sorted(accepted_nodes, key=lambda n: n.score)[: max(1, top_k)]
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
            or (tasks_completed >= max_total_tasks)
            or (in_flight == 0 and next_task_id > max_total_tasks)
        ):
            break

        while in_flight < workers and next_task_id <= max_total_tasks:
            pid = 0
            if accepted_nodes:
                progress = accepted_valid / float(max_accepts) if max_accepts > 0 else 0.0
                if best_k and random.random() < _elite_prob(progress, elite_prob_start, elite_prob_end):
                    pid = _choose_from_top_k_weighted(best_k)
                elif best_node:
                    pid = best_node.id
                else:
                    pid = best_k[0].id

            worker_slot = in_flight % workers
            task_q.put(Task(next_task_id, pid, node_states.get(pid, root_state), worker_slot))
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
            raster_data_url=png_bytes_to_data_url(full_png) if write_lineage else None,
            raster_preview_data_url=_make_preview_data_url(full_png, openai_image_long_side),
            score=res.score,
            model_temperature=next_temp,
            stale_hits=stale_hits,
            invalid_msg=None,
        )
        node = SearchNode(score=res.score, id=next_node_id, parent_id=res.parent_id, state=new_state)
        accepted_nodes.append(node)
        node_states[node.id] = new_state
        accepted_valid += 1

        fn = f"score{node.score:012.6f}_node{node.id:05d}_parent{node.parent_id:05d}{ext}"
        iter_path = os.path.join(nodes_dir, fn)
        with open(iter_path, "w", encoding="utf-8") as f:
            f.write(res.svg)
        node_info[node.id] = (node.parent_id, node.score, iter_path)

        if best_node is None or node.score < best_node.score:
            best_node = node
            log.info(f"NEW BEST node={node.id} score={node.score:.6f}")

        best_k = sorted(accepted_nodes, key=lambda n: n.score)[: max(1, top_k)]
        if write_lineage and (accepted_valid % 10 == 0):
            write_lineage_files(node_info)

    if best_node is None or not best_node.state.svg:
        shutdown_all("No valid SVG produced.", terminate=True)
        raise SystemExit("No valid SVG produced.")

    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write(best_node.state.svg)

    if write_lineage:
        write_lineage_files(node_info)

    shutdown_all("completed", terminate=False)