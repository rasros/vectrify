import base64
import csv
import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path

from svgizer.formats.models import VectorStatePayload
from svgizer.search import SearchNode

log = logging.getLogger(__name__)


class FileStorageAdapter:
    def __init__(
        self,
        output_path: str,
        file_extension: str = ".svg",
        resume: bool = False,
        img_dims: tuple[int, int] = (512, 512),
        image_long_side: int = 512,
        save_raster: bool = False,
        save_heatmap: bool = False,
    ):
        self.output_path = Path(output_path)
        self.file_extension = file_extension
        self.resume = resume
        self.img_dims = img_dims
        self.image_long_side = image_long_side
        self.save_raster = save_raster
        self.save_heatmap = save_heatmap
        self._max_id = 0

        self.base_name = self.output_path.stem
        self.project_dir = self.output_path.parent / self.base_name
        self.runs_dir = self.project_dir / "runs"

        self.current_run_dir: Path | None = None
        self.nodes_dir: Path | None = None
        self.lineage_csv: Path | None = None

    @property
    def max_node_id(self) -> int:
        return self._max_id

    def initialize(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_run_dir = self.runs_dir / timestamp
        self.nodes_dir = self.current_run_dir / "nodes"
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.lineage_csv = self.current_run_dir / "lineage.csv"
        log.debug(f"Storage initialized at: {self.current_run_dir}")

    def load_resume_nodes(self) -> list[tuple[int, str]]:
        if not self.resume or not self.runs_dir.exists():
            return []

        past_runs = sorted(
            [
                d
                for d in self.runs_dir.iterdir()
                if d.is_dir() and d != self.current_run_dir
            ],
            key=lambda d: d.name,
        )

        if not past_runs:
            log.info("No previous runs found to resume.")
            return []

        latest_run = past_runs[-1]
        target_nodes_dir = latest_run / "nodes"

        if not target_nodes_dir.exists():
            log.warning(f"Latest run {latest_run.name} has no 'nodes' directory.")
            return []

        log.info(f"Loading nodes to resume from latest run: {latest_run.name}")

        ext = re.escape(self.file_extension)
        file_pattern = re.compile(rf"^([0-9.]+)_(\d+){ext}$")
        parsed_files = []

        glob_pattern = f"*{self.file_extension}"
        for file_path in target_nodes_dir.glob(glob_pattern):
            match = file_pattern.match(file_path.name)
            node_id = int(match.group(2)) if match else self._max_id + 1

            self._max_id = max(self._max_id, node_id)
            parsed_files.append((node_id, file_path))

        resumed_data = []
        for node_id, file_path in parsed_files:
            try:
                content = file_path.read_text(encoding="utf-8").strip()
                if content:
                    resumed_data.append((node_id, content))
            except Exception as e:
                log.error(f"Failed to read resume node {file_path.name}: {e}")

        return sorted(resumed_data, key=lambda x: x[0])

    def save_node(self, node: SearchNode[VectorStatePayload]) -> None:
        if self.nodes_dir is None or self.lineage_csv is None:
            return

        self._max_id = max(self._max_id, node.id)

        base_fn = f"{node.score:.6f}_{node.id}"

        if node.state.payload.content:
            content_path = self.nodes_dir / f"{base_fn}{self.file_extension}"
            content_path.write_text(node.state.payload.content, encoding="utf-8")

        if self.save_raster and node.state.payload.raster_data_url:
            _, b64 = node.state.payload.raster_data_url.split(",", 1)
            png_path = self.nodes_dir / f"{base_fn}.png"
            png_path.write_bytes(base64.b64decode(b64))

        if self.save_heatmap and node.state.payload.heatmap_data_url:
            _, b64 = node.state.payload.heatmap_data_url.split(",", 1)
            heatmap_path = self.nodes_dir / f"{base_fn}.heatmap.png"
            heatmap_path.write_bytes(base64.b64decode(b64))

        content_md5 = (
            hashlib.md5(node.state.payload.content.encode()).hexdigest()
            if node.state.payload.content
            else ""
        )
        exists = self.lineage_csv.is_file()
        with self.lineage_csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(
                    [
                        "id",
                        "parent",
                        "secondary_parent",
                        "epoch",
                        "score",
                        "complexity",
                        "summary",
                        "content_md5",
                    ]
                )
            writer.writerow(
                [
                    node.id,
                    node.parent_id,
                    node.secondary_parent_id or "",
                    node.epoch,
                    f"{node.score:.6f}",
                    f"{node.complexity:.0f}",
                    node.state.payload.origin or "",
                    content_md5,
                ]
            )
