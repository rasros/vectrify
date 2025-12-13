from __future__ import annotations

import base64
import difflib
import io
import logging
import os
import queue
import re
import time
from typing import Optional, List, Any, Tuple, Dict

import multiprocessing as mp

import cairosvg
from openai import OpenAI
from PIL import Image

from svgizer.models import ChainState, SearchNode, Task, Result, INVALID_SCORE
from svgizer.diff_scores import pixel_diff_score
from svgizer.openai_iface import summarize_changes, call_openai_for_svg, extract_svg_fragment, is_valid_svg

TEMP_STEP = 0.3
MAX_TEMP = 1.6

STALENESS_THRESHOLD = 0.995
STALE_HITS_BEFORE_BUMP = 1

NODE_FILE_RE_NEW = re.compile(r"^score([0-9.]+)_node(\d+)_parent(\d+)\.svg$")
NODE_FILE_RE_OLD = re.compile(r"_node(\d+)_score([0-9.]+)\.svg$")


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


def _score_key(score: float) -> str:
    return f"{score:012.6f}"


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
            return

        assert isinstance(task, Task)
        parent = task.parent_state

        temperature = min(MAX_TEMP, max(0.0, parent.model_temperature + (task.worker_slot * 0.07)))

        change_summary = None
        if parent.svg and parent.raster_data_url:
            try:
                change_summary = summarize_changes(
                    client=client,
                    original_data_url=original_data_url,
                    iter_index=task.parent_id,
                    rasterized_svg_data_url=parent.raster_data_url,
                )
            except Exception as e:
                log.debug(f"Summary failed task={task.task_id}: {e}")

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
                diversity_hint=f"parent={task.parent_id} worker={task.worker_slot}",
            )
            svg = extract_svg_fragment(raw)
        except Exception as e:
            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    worker_slot=task.worker_slot,
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

        valid, err = is_valid_svg(svg)
        if not valid:
            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    worker_slot=task.worker_slot,
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

        try:
            png = rasterize_svg_to_png_bytes(svg)
            score = pixel_diff_score(original_rgb, png)
        except Exception as e:
            result_q.put(
                Result(
                    task_id=task.task_id,
                    parent_id=task.parent_id,
                    worker_slot=task.worker_slot,
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
                worker_slot=task.worker_slot,
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
    nodes_dir: str,
    base_name: str,
    ext: str,
    log: logging.Logger,
    base_model_temperature: float,
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
                png = rasterize_svg_to_png_bytes(svg)
                raster_data_url = png_bytes_to_data_url(png)
            except Exception as e:
                log.warning(f"Resume: failed to load {path}: {e}")
                continue

            state = ChainState(
                svg=svg,
                raster_data_url=raster_data_url,
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


def _load_seed_svg(
    seed_svg_path: str,
    original_rgb: Image.Image,
    base_model_temperature: float,
    log: logging.Logger,
) -> SearchNode:
    if not os.path.isfile(seed_svg_path):
        raise SystemExit(f"--seed-svg file does not exist: {seed_svg_path}")

    with open(seed_svg_path, "r", encoding="utf-8") as f:
        svg_text = f.read()

    valid, err = is_valid_svg(svg_text)
    if not valid:
        raise SystemExit(f"--seed-svg is not a valid SVG: {err}")

    try:
        png = rasterize_svg_to_png_bytes(svg_text)
        score = pixel_diff_score(original_rgb, png)
        raster_data_url = png_bytes_to_data_url(png)
    except Exception as e:
        raise SystemExit(f"--seed-svg could not be rasterized/scored: {e}")

    log.info(f"Seed SVG scored {score:.6f}: {seed_svg_path}")

    state = ChainState(
        svg=svg_text,
        raster_data_url=raster_data_url,
        score=score,
        model_temperature=base_model_temperature,
        stale_hits=0,
        invalid_msg=None,
    )

    # id=1 placeholder; caller will re-id if needed. parent_id=0 for seed.
    return SearchNode(score=score, id=1, parent_id=0, state=state)


def run_search(
    image_path: str,
    output_svg_path: str,
    seed_svg_path: Optional[str],
    max_accepts: int,
    workers: int,
    base_model_temperature: float,
    max_total_tasks: int,
    max_wall_seconds: Optional[float],
    resume: bool,
    top_k: int,
    write_lineage: bool,
    log_level: str,
) -> None:
    setup_logger(log_level)
    log = logging.getLogger("main")

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY environment variable is not set.")
    if not os.path.isfile(image_path):
        raise SystemExit(f"Input image '{image_path}' does not exist.")

    original_data_url = encode_image_to_data_url(image_path)
    original_img = Image.open(image_path).convert("RGB")

    # Keep both RGB image and PNG bytes (for workers)
    original_rgb = original_img
    buf = io.BytesIO()
    original_img.save(buf, format="PNG")
    original_png_bytes = buf.getvalue()

    base_name, ext = os.path.splitext(output_svg_path)
    if not ext:
        ext = ".svg"

    out_dir = os.path.dirname(base_name) or "."
    nodes_dir = os.path.join(out_dir, os.path.basename(base_name) + "_nodes")
    os.makedirs(nodes_dir, exist_ok=True)

    lineage_tsv_path = os.path.join(out_dir, os.path.basename(base_name) + "_lineage.tsv")
    lineage_dot_path = os.path.join(out_dir, os.path.basename(base_name) + "_lineage.dot")

    start_time = time.monotonic()

    def time_up() -> bool:
        return max_wall_seconds is not None and (time.monotonic() - start_time) >= max_wall_seconds

    ctx = mp.get_context("spawn")
    task_q: mp.Queue = ctx.Queue(maxsize=max(64, workers * 8))
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

    def write_lineage_files(node_info: Dict[int, Tuple[int, float, str]]) -> None:
        if not write_lineage:
            return
        try:
            with open(lineage_tsv_path, "w", encoding="utf-8") as f:
                f.write("node_id\tparent_id\tscore\tpath\n")
                for nid in sorted(node_info.keys()):
                    pid, sc, pth = node_info[nid]
                    f.write(f"{nid}\t{pid}\t{sc:.6f}\t{pth}\n")

            with open(lineage_dot_path, "w", encoding="utf-8") as f:
                f.write("digraph lineage {\n")
                f.write('  rankdir="LR";\n')
                for nid in sorted(node_info.keys()):
                    pid, sc, _ = node_info[nid]
                    f.write(f'  n{nid} [label="{nid}\\n{sc:.6f}"];\n')
                    if pid != 0:
                        f.write(f"  n{pid} -> n{nid};\n")
                f.write("}\n")
        except Exception as e:
            log.warning(f"Failed writing lineage files: {e}")

    root_state = ChainState(
        svg=None,
        raster_data_url=None,
        score=INVALID_SCORE,
        model_temperature=base_model_temperature,
        stale_hits=0,
        invalid_msg=None,
    )

    node_states: Dict[int, ChainState] = {0: root_state}
    node_info: Dict[int, Tuple[int, float, str]] = {}

    accepted_nodes: List[SearchNode] = []
    best_node: Optional[SearchNode] = None
    next_node_id = 0

    # 1) Optional resume from prior nodes
    if resume:
        prior_nodes, prior_best, max_id = _load_resume_nodes(nodes_dir, base_name, ext, log, base_model_temperature)
        if prior_nodes:
            accepted_nodes.extend(prior_nodes)
            best_node = prior_best
            next_node_id = max_id
            for n in prior_nodes:
                node_states[n.id] = n.state

    # 2) Optional seed SVG overrides/augments resume best:
    # If seed is better than current best (or no best), use it as best_node.
    if seed_svg_path:
        seed = _load_seed_svg(seed_svg_path, original_rgb, base_model_temperature, log)
        next_node_id += 1
        seed = SearchNode(score=seed.score, id=next_node_id, parent_id=0, state=seed.state)
        accepted_nodes.append(seed)
        node_states[seed.id] = seed.state

        fn = f"score{_score_key(seed.score)}_node{seed.id:05d}_parent00000{ext}"
        iter_path = os.path.join(nodes_dir, fn)
        with open(iter_path, "w", encoding="utf-8") as f:
            f.write(seed.state.svg or "")
        node_info[seed.id] = (0, seed.score, iter_path)

        if best_node is None or seed.score < best_node.score:
            best_node = seed
            log.info(f"Seed SVG set as best node={seed.id} score={seed.score:.6f}")

    def choose_parent_id() -> int:
        if best_node is not None:
            return best_node.id
        return 0

    def next_lineage_temp(parent_id: int, parent_svg: Optional[str], child_svg: str) -> Tuple[float, int]:
        parent_state = node_states.get(parent_id, root_state)
        next_temp = parent_state.model_temperature
        stale_hits = parent_state.stale_hits

        if parent_svg and is_stale(parent_svg, child_svg):
            stale_hits += 1
            if stale_hits >= STALE_HITS_BEFORE_BUMP and next_temp < MAX_TEMP:
                next_temp = min(MAX_TEMP, next_temp + TEMP_STEP)
                stale_hits = 0
        else:
            stale_hits = 0
        return next_temp, stale_hits

    next_task_id = 1
    tasks_completed = 0
    accepted_valid = 0
    in_flight = 0

    def enqueue_one(worker_slot: int) -> bool:
        nonlocal next_task_id, in_flight
        if next_task_id > max_total_tasks:
            return False

        pid = choose_parent_id()
        pstate = node_states.get(pid, root_state)

        t = Task(
            task_id=next_task_id,
            parent_id=pid,
            parent_state=pstate,
            worker_slot=worker_slot,
        )

        try:
            task_q.put_nowait(t)
        except queue.Full:
            return False

        next_task_id += 1
        in_flight += 1
        return True

    # Start: exactly `workers` initial tasks, from best if seed/resume best exists, else from root.
    for slot in range(workers):
        if not enqueue_one(worker_slot=slot):
            break

    while True:
        if time_up():
            log.warning("Stopping: wall-clock limit reached.")
            break
        if accepted_valid >= max_accepts:
            break
        if tasks_completed >= max_total_tasks:
            break
        if in_flight == 0:
            break

        try:
            res: Result = result_q.get(timeout=0.2)
        except queue.Empty:
            continue

        in_flight = max(0, in_flight - 1)
        tasks_completed += 1

        if res.invalid_msg and res.invalid_msg.startswith("FATAL:"):
            shutdown_all(res.invalid_msg, terminate=True)
            raise SystemExit(res.invalid_msg)

        # enqueue replacement immediately
        if (accepted_valid < max_accepts) and (next_task_id <= max_total_tasks) and (not time_up()):
            enqueue_one(worker_slot=res.worker_slot)

        if (not res.valid) or (res.svg is None) or (res.raster_png is None) or (res.score >= INVALID_SCORE):
            continue

        next_node_id += 1

        parent_state = node_states.get(res.parent_id, root_state)
        next_temp, stale_hits = next_lineage_temp(res.parent_id, parent_state.svg, res.svg)

        state = ChainState(
            svg=res.svg,
            raster_data_url=png_bytes_to_data_url(res.raster_png),
            score=res.score,
            model_temperature=next_temp,
            stale_hits=stale_hits,
            invalid_msg=None,
        )
        node = SearchNode(score=res.score, id=next_node_id, parent_id=res.parent_id, state=state)
        accepted_nodes.append(node)
        node_states[node.id] = node.state

        fn = f"score{_score_key(node.score)}_node{node.id:05d}_parent{node.parent_id:05d}{ext}"
        iter_path = os.path.join(nodes_dir, fn)
        with open(iter_path, "w", encoding="utf-8") as f:
            f.write(node.state.svg or "")
        node_info[node.id] = (node.parent_id, node.score, iter_path)

        if best_node is None or node.score < best_node.score:
            best_node = node
            log.info(f"NEW BEST node={node.id} score={node.score:.6f}")

        accepted_valid += 1

        if write_lineage and (accepted_valid % 10 == 0):
            write_lineage_files(node_info)

    if best_node is None or best_node.state.svg is None or best_node.score >= INVALID_SCORE:
        shutdown_all("No valid SVG produced.", terminate=True)
        raise SystemExit("No valid SVG produced.")

    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write(best_node.state.svg)

    if write_lineage:
        write_lineage_files(node_info)

    shutdown_all("completed", terminate=False)
