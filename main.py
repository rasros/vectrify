#!/usr/bin/env python3
import argparse
import base64
import dataclasses
import difflib
import heapq
import io
import logging
import os
import queue
import sys
import threading
import xml.etree.ElementTree as ET
from typing import Optional, List, Tuple, Any

import multiprocessing as mp

from openai import OpenAI

try:
    import cairosvg
except ImportError:
    print("cairosvg is required. Install with: pip install cairosvg", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Pillow is required. Install with: pip install pillow", file=sys.stderr)
    sys.exit(1)

MODEL_NAME = "gpt-5.1"

# Iterations are now an "expansion budget" for pipelined search
DEFAULT_MAX_ITER = 8
DEFAULT_BASE_TEMP = 0.2

# Beam search defaults
DEFAULT_NUM_BEAMS = 4
DEFAULT_CANDIDATES_PER_NODE = 4
DEFAULT_WORKERS = 4

# Temperature / staleness knobs
TEMP_STEP = 0.3
MAX_TEMP = 1.6
STALENESS_THRESHOLD = 0.995
STALENESS_HITS_BEFORE_TEMP_INCREASE = 1

# Candidate scoring
INVALID_SCORE = 1e9


def setup_logger(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(processName)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


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


def extract_svg_fragment(raw: str) -> str:
    lower = raw.lower()
    start_idx = lower.find("<svg")
    end_idx = lower.rfind("</svg>")
    if start_idx != -1 and end_idx != -1:
        end_idx += len("</svg>")
        return raw[start_idx:end_idx].strip()
    return raw.strip()


def is_valid_svg(svg_text: str) -> Tuple[bool, Optional[str]]:
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as e:
        return False, f"XML parse error: {e}"
    if root.tag.lower().endswith("svg"):
        return True, None
    return False, f"Root tag is not <svg>: got <{root.tag}>"


def is_stale(prev_svg: Optional[str], new_svg: str) -> bool:
    if prev_svg is None:
        return False
    ratio = difflib.SequenceMatcher(None, prev_svg, new_svg).ratio()
    return ratio >= STALENESS_THRESHOLD


def pixel_diff_score(original_rgb: Image.Image, candidate_png: bytes) -> float:
    """
    Mean absolute RGB diff normalized to [0, 1]. Lower is better.
    Simple baseline scorer.
    """
    cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")
    if cand.size != original_rgb.size:
        cand = cand.resize(original_rgb.size, resample=Image.BILINEAR)

    a = original_rgb.tobytes()
    b = cand.tobytes()

    total = 0
    for x, y in zip(a, b):
        total += abs(x - y)

    mean = total / (len(a) * 255.0)
    return float(mean)


def summarize_changes(
    client: OpenAI,
    original_data_url: str,
    iter_index: int,
    rasterized_svg_data_url: Optional[str],
) -> str:
    lines = [
        "You compare an original raster image and a current SVG rendering (shown as a rasterized image).",
        "Describe only the FEW MOST IMPORTANT changes needed to improve overall likeness.",
        "The priority is to fix overall shapes first, then position of items, and then details like arrows, icons and text.",
        "Ignore noise and artifacts typical of AI-generated images (random speckles, smeared or melting details, ghost edges, meaningless text blobs, watermarks).",
        "Keep the list short (3–6 bullets). Output plain text only.",
        f"Iteration #{iter_index}.",
    ]

    content: List[dict] = [
        {"type": "input_text", "text": "\n".join(lines)},
        {"type": "input_image", "image_url": original_data_url},
    ]

    if rasterized_svg_data_url:
        content.append({"type": "input_image", "image_url": rasterized_svg_data_url})

    resp = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": content}],
        temperature=0.1,
        text={"format": {"type": "text"}},
    )
    return resp.output_text


def call_openai_for_svg(
    client: OpenAI,
    original_data_url: str,
    iter_index: int,
    temperature: float,
    svg_prev: Optional[str] = None,
    svg_prev_invalid_msg: Optional[str] = None,
    rasterized_svg_data_url: Optional[str] = None,
    change_summary: Optional[str] = None,
    diversity_hint: Optional[str] = None,
) -> str:
    lines = [
        "You convert a raster input image into a clean, valid SVG.",
        "Output ONLY a complete <svg>...</svg> with no commentary or backticks.",
        "Prioritize matching the overall likeness of the image: silhouette, composition, large shapes, major color blocks.",
        "Do NOT reproduce noise or artifacts typical of AI-generated images (random speckles, glitchy edges, smeared details, meaningless text or watermark-like blobs). Clean them up in the SVG.",
        "Aim for structural fidelity and clear vector geometry, not pixel-level accuracy.",
        "Ensure the SVG is valid XML with a single <svg> root.",
        f"Iteration #{iter_index}.",
    ]

    if diversity_hint:
        lines.append(f"Diversity hint (to avoid producing the exact same SVG): {diversity_hint}")

    if svg_prev is None:
        lines.append("This is the first attempt. Produce your best high-level SVG approximation.")
    else:
        lines.append("Refine the previous SVG by addressing the differences.")

    if svg_prev_invalid_msg:
        lines.append(
            f"The previous SVG was INVALID:\n{svg_prev_invalid_msg}\n"
            "Return a corrected, valid SVG."
        )

    if change_summary:
        lines.append(
            "Here is a summary of the MOST IMPORTANT changes needed to improve likeness. Use these as priorities:\n"
            + change_summary
        )

    if rasterized_svg_data_url:
        lines.append(
            "You are given the original raster and a rasterized rendering of your current SVG. "
            "Use this to improve the likeness as much as possible. "
            "The priority is to fix overall shapes first, then position of items, and then details like arrows, icons and text."
        )

    if svg_prev:
        lines.append("Here is the previous SVG to refine:\n" + svg_prev)

    content: List[dict] = [
        {"type": "input_text", "text": "\n".join(lines)},
        {"type": "input_image", "image_url": original_data_url},
    ]

    if rasterized_svg_data_url:
        content.append({"type": "input_image", "image_url": rasterized_svg_data_url})

    response = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": content}],
        temperature=temperature,
        text={"format": {"type": "text"}},
    )

    return response.output_text


@dataclasses.dataclass
class BeamState:
    svg: Optional[str]
    raster_data_url: Optional[str]  # rasterized svg as png data-url
    score: float
    temperature: float
    stale_hits: int
    invalid_msg: Optional[str]


@dataclasses.dataclass(order=True)
class SearchNode:
    score: float
    id: int = dataclasses.field(compare=False)
    state: BeamState = dataclasses.field(compare=False)


@dataclasses.dataclass
class Task:
    task_id: int
    parent_id: int
    parent_state: BeamState
    candidate_index: int


@dataclasses.dataclass
class Result:
    task_id: int
    parent_id: int
    candidate_index: int
    svg: Optional[str]
    valid: bool
    invalid_msg: Optional[str]
    raster_png: Optional[bytes]
    score: float
    used_temperature: float
    change_summary: Optional[str]


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

        # Temperature jitter per candidate for diversity, plus staleness-driven temperature
        temperature = min(MAX_TEMP, max(0.0, parent.temperature + (task.candidate_index * 0.05)))

        log.info(
            f"Start task={task.task_id} parent={task.parent_id} cand={task.candidate_index} temp={temperature:.2f}"
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
                diversity_hint=f"parent={task.parent_id} cand={task.candidate_index}",
            )
            svg = extract_svg_fragment(raw)
            log.info(f"OpenAI returned task={task.task_id} svg_len={len(svg)}")
        except Exception as e:
            result_q.put(Result(
                task_id=task.task_id,
                parent_id=task.parent_id,
                candidate_index=task.candidate_index,
                svg=None,
                valid=False,
                invalid_msg=f"OpenAI call failed: {e}",
                raster_png=None,
                score=INVALID_SCORE,
                used_temperature=temperature,
                change_summary=change_summary,
            ))
            continue

        valid, err = is_valid_svg(svg)
        if not valid:
            log.warning(f"Invalid SVG task={task.task_id}: {err}")
            result_q.put(Result(
                task_id=task.task_id,
                parent_id=task.parent_id,
                candidate_index=task.candidate_index,
                svg=svg,
                valid=False,
                invalid_msg=err,
                raster_png=None,
                score=INVALID_SCORE,
                used_temperature=temperature,
                change_summary=change_summary,
            ))
            continue

        try:
            png = rasterize_svg_to_png_bytes(svg)
            score = pixel_diff_score(original_rgb, png)
            log.info(f"Scored task={task.task_id} score={score:.6f}")
        except Exception as e:
            log.warning(f"Raster/score failed task={task.task_id}: {e}")
            result_q.put(Result(
                task_id=task.task_id,
                parent_id=task.parent_id,
                candidate_index=task.candidate_index,
                svg=svg,
                valid=False,
                invalid_msg=f"Rasterize/score failed: {e}",
                raster_png=None,
                score=INVALID_SCORE,
                used_temperature=temperature,
                change_summary=change_summary,
            ))
            continue

        result_q.put(Result(
            task_id=task.task_id,
            parent_id=task.parent_id,
            candidate_index=task.candidate_index,
            svg=svg,
            valid=True,
            invalid_msg=None,
            raster_png=png,
            score=score,
            used_temperature=temperature,
            change_summary=change_summary,
        ))


def run(
    image_path: str,
    output_svg_path: str,
    max_iter: int,
    base_temperature: float,
    num_beams: int,
    candidates_per_node: int,
    workers: int,
    log_level: str,
    write_top_k_each: int,
):
    setup_logger(log_level)
    log = logging.getLogger("main")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(image_path):
        print(f"Input image '{image_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    original_data_url = encode_image_to_data_url(image_path)

    # Normalize original to PNG bytes for workers.
    original_img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    original_img.save(buf, format="PNG")
    original_png_bytes = buf.getvalue()

    base_name, ext = os.path.splitext(output_svg_path)
    if not ext:
        ext = ".svg"

    ctx = mp.get_context("spawn")
    task_q: mp.Queue = ctx.Queue(maxsize=max(16, workers * 8))
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

    # Beam maintained as a max-set, implemented via min-heap but we evict worst manually.
    # We'll keep a separate list of best nodes (heap by score).
    best_heap: List[SearchNode] = []

    # Global task bookkeeping
    next_task_id = 1
    next_node_id = 0

    # Expansion budget: number of accepted nodes to expand (approx).
    # Total tasks will be up to budget * candidates_per_node, but invalids do not contribute nodes.
    target_accepts = max_iter * num_beams

    # Root state: no svg yet, just a starting temperature
    root_state = BeamState(
        svg=None,
        raster_data_url=None,
        score=INVALID_SCORE,
        temperature=base_temperature,
        stale_hits=0,
        invalid_msg=None,
    )
    root_node = SearchNode(score=INVALID_SCORE, id=0, state=root_state)
    heapq.heappush(best_heap, root_node)

    # Nodes that are eligible for expansion are those we accept (valid scored nodes).
    # Fully pipelined: as soon as we accept a node, we enqueue its children immediately.
    def enqueue_children(parent_id: int, parent_state: BeamState):
        nonlocal next_task_id

        for c in range(candidates_per_node):
            task = Task(
                task_id=next_task_id,
                parent_id=parent_id,
                parent_state=parent_state,
                candidate_index=c,
            )

            # Non-blocking-ish enqueue: if full, briefly drain a result and retry.
            while True:
                try:
                    task_q.put(task, timeout=0.05)
                    log.info(
                        f"Enqueue task={task.task_id} parent={task.parent_id} cand={c} temp≈{min(MAX_TEMP, parent_state.temperature + (c*0.05)):.2f}"
                    )
                    next_task_id += 1
                    break
                except queue.Full:
                    try:
                        r = result_q.get_nowait()
                        handle_result(r)
                    except queue.Empty:
                        continue

    # Track best nodes (for periodic writing)
    accepted_nodes: List[SearchNode] = []
    best_seen: Optional[SearchNode] = None

    # Result handler needs to be defined before enqueue_children uses it.
    def handle_result(res: Result):
        nonlocal next_node_id, best_seen

        if not res.valid or res.svg is None or res.raster_png is None or res.score >= INVALID_SCORE:
            log.warning(
                f"Reject task={res.task_id} parent={res.parent_id} cand={res.candidate_index} reason={res.invalid_msg}"
            )
            return

        # Parent staleness/temperature update (lineage-based)
        parent_state = None
        # Look up parent from accepted nodes list / heap contents.
        # Fast path: parent likely recently accepted. We'll scan accepted_nodes (small).
        for n in reversed(accepted_nodes[-200:]):
            if n.id == res.parent_id:
                parent_state = n.state
                break
        if parent_state is None and res.parent_id == 0:
            parent_state = root_state

        stale_hits = parent_state.stale_hits if parent_state else 0
        next_temp = parent_state.temperature if parent_state else base_temperature

        if parent_state and is_stale(parent_state.svg, res.svg):
            stale_hits += 1
            if stale_hits >= STALENESS_HITS_BEFORE_TEMP_INCREASE and next_temp < MAX_TEMP:
                next_temp = min(MAX_TEMP, next_temp + TEMP_STEP)
                stale_hits = 0
                log.info(f"Staleness: increasing temp for lineage parent={res.parent_id} -> {next_temp:.2f}")
        else:
            stale_hits = 0

        next_node_id += 1
        state = BeamState(
            svg=res.svg,
            raster_data_url=png_bytes_to_data_url(res.raster_png),
            score=res.score,
            temperature=next_temp,
            stale_hits=stale_hits,
            invalid_msg=None,
        )
        node = SearchNode(score=res.score, id=next_node_id, state=state)

        # Insert into best set (beam) and accepted list
        accepted_nodes.append(node)

        # Maintain a top-num_beams "best set" using a heap of all candidates is fine,
        # but we want "current beams" of best states. We'll keep best_heap and evict worst manually.
        heapq.heappush(best_heap, node)
        # Evict worst until <= num_beams by removing via sort-and-trim occasionally.
        # Since heapq is min-heap, easiest is keep a separate trimmed list periodically.
        # For simplicity: rebuild every time size exceeds 4*num_beams.
        if len(best_heap) > max(num_beams * 4, 32):
            best_heap.sort(key=lambda x: x.score)
            best_heap[:] = best_heap[:num_beams]

        # Track best
        if best_seen is None or node.score < best_seen.score:
            best_seen = node
            log.info(f"NEW BEST node={node.id} score={node.score:.6f} (from task={res.task_id})")

        # Write candidate SVG for visibility
        iter_path = f"{base_name}_node{node.id:05d}_score{node.score:.6f}{ext}"
        try:
            with open(iter_path, "w", encoding="utf-8") as f:
                f.write(node.state.svg or "")
            log.info(f"Wrote {iter_path}")
        except Exception as e:
            log.warning(f"Failed to write {iter_path}: {e}")

        # Enqueue children immediately (pipelined)
        enqueue_children(parent_id=node.id, parent_state=node.state)

        # Optional: also write current top-K snapshot occasionally
        if write_top_k_each > 0 and (node.id % write_top_k_each == 0):
            snap = sorted(best_heap, key=lambda x: x.score)[: min(num_beams, len(best_heap))]
            for rank, bn in enumerate(snap, start=1):
                pth = f"{base_name}_TOP_rank{rank:02d}_node{bn.id:05d}_score{bn.score:.6f}{ext}"
                try:
                    with open(pth, "w", encoding="utf-8") as f:
                        f.write(bn.state.svg or "")
                except Exception:
                    pass
            log.info(f"Wrote TOP snapshot (every {write_top_k_each} accepts)")

    # Seed initial expansions from root (no barrier)
    enqueue_children(parent_id=0, parent_state=root_state)

    # Main loop: consume results and accept until budget met
    accepted_valid = 0
    while accepted_valid < target_accepts:
        res: Result = result_q.get()
        before = len(accepted_nodes)
        handle_result(res)
        after = len(accepted_nodes)
        if after > before:
            accepted_valid += 1
            log.info(f"Accepted valid nodes: {accepted_valid}/{target_accepts}")

    # Choose best in current beam set
    if best_seen is None or best_seen.state.svg is None:
        print("No valid SVG produced.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(output_svg_path, "w", encoding="utf-8") as f:
            f.write(best_seen.state.svg)
        log.info(f"Final SVG written to: {output_svg_path} (best score={best_seen.score:.6f})")
    except Exception as e:
        print(f"Failed to write final SVG '{output_svg_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # Shutdown workers
    for _ in procs:
        try:
            task_q.put(None, timeout=0.1)
        except Exception:
            pass
    for p in procs:
        p.join(timeout=1.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fully pipelined beam/best-first SVG approximation of an image using OpenAI."
    )
    parser.add_argument("image", help="Path to input raster image (PNG/JPEG/WEBP/GIF).")
    parser.add_argument(
        "--output",
        "-o",
        default="output.svg",
        help="Final SVG path (default: output.svg).",
    )
    parser.add_argument(
        "--max-iter",
        "-n",
        type=int,
        default=DEFAULT_MAX_ITER,
        help=f"Expansion budget factor (default: {DEFAULT_MAX_ITER}). Total accepts ~= max_iter * num_beams.",
    )
    parser.add_argument(
        "--base-temp",
        type=float,
        default=DEFAULT_BASE_TEMP,
        help=f"Base sampling temperature (default: {DEFAULT_BASE_TEMP}).",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=DEFAULT_NUM_BEAMS,
        help=f"Beam width (default: {DEFAULT_NUM_BEAMS}).",
    )
    parser.add_argument(
        "--candidates-per-node",
        type=int,
        default=DEFAULT_CANDIDATES_PER_NODE,
        help=f"Number of candidates sampled per accepted node (default: {DEFAULT_CANDIDATES_PER_NODE}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of multiprocessing workers (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )
    parser.add_argument(
        "--write-top-k-each",
        type=int,
        default=10,
        help="Write TOP snapshot every N accepted nodes (0 disables). Default: 10.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        image_path=args.image,
        output_svg_path=args.output,
        max_iter=args.max_iter,
        base_temperature=args.base_temp,
        num_beams=args.num_beams,
        candidates_per_node=args.candidates_per_node,
        workers=args.workers,
        log_level=args.log_level,
        write_top_k_each=args.write_top_k_each,
    )

