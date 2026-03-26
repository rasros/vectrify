import csv
import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path

from svgizer.search import SearchNode
from svgizer.svg.adapter import SvgStatePayload

log = logging.getLogger(__name__)


class FileStorageAdapter:
    def __init__(
        self,
        output_svg_path: str,
        resume: bool = False,
        img_dims: tuple[int, int] = (512, 512),
        openai_image_long_side: int = 512,
    ):
        self.output_svg_path = Path(output_svg_path)
        self.resume = resume
        self.img_dims = img_dims
        self.openai_image_long_side = openai_image_long_side
        self._max_id = 0

        self.base_name = self.output_svg_path.stem
        self.project_dir = self.output_svg_path.parent / self.base_name
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
        log.info(f"Storage initialized at: {self.current_run_dir}")

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

        file_pattern = re.compile(r"^([0-9.]+)_(\d+)\.svg$")
        parsed_files = []

        for file_path in target_nodes_dir.glob("*.svg"):
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

    def save_node(self, node: SearchNode[SvgStatePayload]) -> None:
        if self.nodes_dir is None or self.lineage_csv is None:
            return

        self._max_id = max(self._max_id, node.id)

        base_fn = f"{node.score:.6f}_{node.id}"

        if node.state.payload.svg:
            svg_path = self.nodes_dir / f"{base_fn}.svg"
            svg_path.write_text(node.state.payload.svg, encoding="utf-8")

        svg_md5 = (
            hashlib.md5(node.state.payload.svg.encode()).hexdigest()
            if node.state.payload.svg
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
                        "svg_md5",
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
                    node.state.payload.change_summary or "",
                    svg_md5,
                ]
            )
