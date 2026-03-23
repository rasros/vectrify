import csv
import logging
import os
import re
from typing import Protocol

from svgizer.image_utils import make_preview_data_url, rasterize_svg_to_png_bytes
from svgizer.search import ChainState, SearchNode
from svgizer.svg_adapter import SvgStatePayload

NODE_FILE_RE_NEW = re.compile(r"^score([0-9.]+)_node(\d+)_parent(\d+)\.svg$")
NODE_FILE_RE_OLD = re.compile(r"_node(\d+)_score([0-9.]+)\.svg$")


class StorageAdapter(Protocol):
    """Protocol defining how the search algorithm interacts with storage."""

    @property
    def write_lineage_enabled(self) -> bool: ...

    def initialize(self) -> None: ...

    def save_node(self, node: SearchNode) -> str: ...

    def write_lineage(
        self, node_info: dict[int, tuple[int, float, str, str | None]]
    ) -> None: ...

    def load_resume_nodes(
        self,
        log: logging.Logger,
        base_model_temperature: float,
        original_w: int,
        original_h: int,
        openai_image_long_side: int,
    ) -> tuple[list[SearchNode], SearchNode | None, int]: ...

    def save_final_svg(self, svg_content: str) -> None: ...

    def load_seed_svg(self, seed_path: str) -> str: ...


class FileStorageAdapter:
    """Concrete implementation for standard local file system I/O."""

    def __init__(
        self, output_svg_path: str, write_lineage: bool = True, resume: bool = False
    ):
        self.output_svg_path = output_svg_path
        self._write_lineage_enabled = write_lineage
        self.resume = resume

        base_name, ext = os.path.splitext(output_svg_path)
        if not ext:
            ext = ".svg"

        self.base_name = os.path.basename(base_name)
        self.ext = ext
        self.out_dir = os.path.dirname(base_name) or "."
        self.nodes_dir = os.path.join(self.out_dir, self.base_name + "_nodes")
        self.lineage_csv_path = os.path.join(
            self.out_dir, self.base_name + "_lineage.csv"
        )

    @property
    def write_lineage_enabled(self) -> bool:
        return self._write_lineage_enabled

    def initialize(self) -> None:
        os.makedirs(self.nodes_dir, exist_ok=True)

    def save_node(self, node: SearchNode) -> str:
        fn = f"score{node.score:012.6f}_node{node.id:05d}_parent{node.parent_id:05d}{self.ext}"
        iter_path = os.path.join(self.nodes_dir, fn)
        if node.state.payload.svg:
            with open(iter_path, "w", encoding="utf-8") as f:
                f.write(node.state.payload.svg)
        return iter_path

    def write_lineage(
        self, node_info: dict[int, tuple[int, float, str, str | None]]
    ) -> None:
        if not self._write_lineage_enabled:
            return

        try:
            with open(self.lineage_csv_path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["node_id", "parent_id", "score", "path", "change_summary"])
                for nid in sorted(node_info.keys()):
                    pid, sc, pth, summ = node_info[nid]
                    w.writerow([nid, pid, f"{sc:.6f}", pth, summ or ""])
        except Exception as e:
            logging.getLogger("main").warning(f"Failed writing lineage files: {e}")

    def load_resume_nodes(
        self,
        log: logging.Logger,
        base_model_temperature: float,
        original_w: int,
        original_h: int,
        openai_image_long_side: int,
    ) -> tuple[list[SearchNode], SearchNode | None, int]:
        if not self.resume:
            return [], None, 0

        accepted: list[SearchNode] = []
        best_seen: SearchNode | None = None
        max_id = 0

        scan_paths: list[tuple[str, str]] = []
        if os.path.isdir(self.nodes_dir):
            scan_paths.append((self.nodes_dir, "new"))
        scan_paths.append((self.out_dir, "old"))

        for directory, mode in scan_paths:
            try:
                filenames = os.listdir(directory)
            except Exception:
                continue

            for fn in filenames:
                if not fn.endswith(self.ext):
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
                    with open(path, encoding="utf-8") as f:
                        svg = f.read()
                    full_png = rasterize_svg_to_png_bytes(
                        svg, out_w=original_w, out_h=original_h
                    )
                    raster_preview_data_url = make_preview_data_url(
                        full_png, openai_image_long_side
                    )
                except Exception as e:
                    log.warning(f"Resume: failed to load {path}: {e}")
                    continue

                payload = SvgStatePayload(
                    svg=svg,
                    raster_data_url=None,  # avoid big RAM usage for resume nodes
                    raster_preview_data_url=raster_preview_data_url,
                    change_summary=None,
                    invalid_msg=None,
                )

                state = ChainState(
                    score=score,
                    model_temperature=base_model_temperature,
                    stale_hits=0,
                    payload=payload,
                )
                node = SearchNode(
                    score=score, id=node_id, parent_id=parent_id, state=state
                )
                accepted.append(node)

                if best_seen is None or node.score < best_seen.score:
                    best_seen = node
                max_id = max(max_id, node_id)

        accepted.sort(key=lambda n: n.id)
        return accepted, best_seen, max_id

    def save_final_svg(self, svg_content: str) -> None:
        with open(self.output_svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)

    def load_seed_svg(self, seed_path: str) -> str:
        with open(seed_path, encoding="utf-8") as f:
            return f.read()