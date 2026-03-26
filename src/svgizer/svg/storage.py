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
    RESUME_TOP_K = 8

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
        """
        Only if resume is true, check the latest past run directory
        and ingest the top K SVGs found in its 'nodes' folder.
        """
        if not self.resume or not self.runs_dir.exists():
            return []

        # Get all past runs, sorted by timestamp name
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

        log.info(
            f"Resuming top {self.RESUME_TOP_K} nodes from latest run: {latest_run.name}"
        )

        file_pattern = re.compile(r"score([0-9.]+)_node(\d+)")
        parsed_files = []

        for file_path in target_nodes_dir.glob("*.svg"):
            match = file_pattern.search(file_path.name)
            if match:
                score = float(match.group(1))
                node_id = int(match.group(2))
            else:
                # Fallback if filename format is unexpected
                score = float("inf")
                id_match = re.search(r"node(\d+)", file_path.name)
                node_id = int(id_match.group(1)) if id_match else (self._max_id + 1)

            self._max_id = max(self._max_id, node_id)
            parsed_files.append((score, node_id, file_path))

        # Sort by score ascending (lower is better) and take top K
        parsed_files.sort(key=lambda x: x[0])
        top_k_files = parsed_files[: self.RESUME_TOP_K]

        resumed_data = []
        for score, node_id, file_path in top_k_files:
            try:
                content = file_path.read_text(encoding="utf-8").strip()
                if content:
                    resumed_data.append((node_id, content))
                    log.info(
                        f"Found node for resume: "
                        f"{file_path.name} (ID: {node_id}, Score: {score})"
                    )
            except Exception as e:
                log.error(f"Failed to read resume node {file_path.name}: {e}")

        return sorted(resumed_data, key=lambda x: x[0])

    def save_node(self, node: SearchNode[SvgStatePayload]) -> None:
        """
        Writes the node's SVG and PNG to the current run's nodes directory.
        This is called during re-scoring (resume) and during active search.
        """
        if self.nodes_dir is None or self.lineage_csv is None:
            return

        self._max_id = max(self._max_id, node.id)

        # Standard filename: score_nodeID_parentID
        base_fn = (
            f"score{node.score:012.6f}_node{node.id:05d}_parent{node.parent_id:05d}"
        )

        # 1. Save SVG
        if node.state.payload.svg:
            svg_path = self.nodes_dir / f"{base_fn}.svg"
            svg_path.write_text(node.state.payload.svg, encoding="utf-8")

        # 2. Update Lineage CSV
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
                    f"{node.score:.6f}",
                    f"{node.complexity:.0f}",
                    node.state.payload.change_summary or "",
                    svg_md5,
                ]
            )
