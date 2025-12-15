import csv
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from svgizer.image_utils import make_preview_data_url, rasterize_svg_to_png_bytes
from svgizer.models import ChainState, SearchNode

# Only used for legacy file loading
NODE_FILE_RE_NEW = re.compile(r"^score([0-9.]+)_node(\d+)_parent(\d+)\.svg$")
NODE_FILE_RE_OLD = re.compile(r"_node(\d+)_score([0-9.]+)\.svg$")


def save_node_to_disk(
    nodes_dir: str,
    node: SearchNode,
    ext: str = ".svg"
) -> str:
    fn = f"score{node.score:012.6f}_node{node.id:05d}_parent{node.parent_id:05d}{ext}"
    iter_path = os.path.join(nodes_dir, fn)
    if node.state.svg:
        with open(iter_path, "w", encoding="utf-8") as f:
            f.write(node.state.svg)
    return iter_path



def write_lineage_csv(
    csv_path: str,
    node_info: Dict[int, Tuple[int, float, str, Optional[str]]]
) -> None:
    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["node_id", "parent_id", "score", "path", "change_summary"])
            for nid in sorted(node_info.keys()):
                pid, sc, pth, summ = node_info[nid]
                w.writerow([nid, pid, f"{sc:.6f}", pth, summ or ""])
    except Exception as e:
        logging.getLogger("main").warning(f"Failed writing lineage files: {e}")


def load_resume_nodes(
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
                raster_preview_data_url = make_preview_data_url(full_png, openai_image_long_side)
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
                change_summary=None,
            )
            node = SearchNode(score=score, id=node_id, parent_id=parent_id, state=state)
            accepted.append(node)

            if best_seen is None or node.score < best_seen.score:
                best_seen = node
            max_id = max(max_id, node_id)

    accepted.sort(key=lambda n: n.id)
    return accepted, best_seen, max_id