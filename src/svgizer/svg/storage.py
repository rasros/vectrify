import csv
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
        base_temp: float = 1.0,
    ):
        self.output_svg_path = Path(output_svg_path)
        self.resume = resume
        self.img_dims = img_dims
        self.openai_image_long_side = openai_image_long_side
        self.base_temp = base_temp
        self._max_id = 0

        self.base_name = self.output_svg_path.stem
        self.ext = self.output_svg_path.suffix
        self.project_dir = self.output_svg_path.parent / self.base_name
        self.runs_dir = self.project_dir / "runs"
        self.legacy_nodes_dir = self.output_svg_path.parent / f"{self.base_name}_nodes"

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
        log.info(f"Storage initialized. Current run: {self.current_run_dir.name}")

    def load_resume_nodes(self) -> list[tuple[int, str]]:
        if not self.resume:
            return []

        run_candidates = []
        if self.runs_dir.exists():
            run_candidates = sorted(
                [
                    d
                    for d in self.runs_dir.iterdir()
                    if d.is_dir() and d != self.current_run_dir
                ]
            )

        resumed_data: list[tuple[int, str]] = []
        pattern = re.compile(r"node(\d+)")

        target_folders = []
        if run_candidates:
            target_folders.append(run_candidates[-1] / "nodes")
        if self.legacy_nodes_dir.exists():
            target_folders.append(self.legacy_nodes_dir)

        for folder in target_folders:
            if not folder.exists():
                continue

            log.info(f"Scanning for resume nodes in: {folder}")
            found_in_folder = []
            for file_path in folder.glob("*.svg"):
                match = pattern.search(file_path.name)
                if match:
                    try:
                        node_id = int(match.group(1))
                        with file_path.open(encoding="utf-8") as f:
                            found_in_folder.append((node_id, f.read()))
                        self._max_id = max(self._max_id, node_id)
                    except Exception as e:
                        log.error(f"Failed to read {file_path.name}: {e}")

            if found_in_folder:
                resumed_data = found_in_folder
                break

        return sorted(resumed_data, key=lambda x: x[0])

    def save_node(self, node: SearchNode[SvgStatePayload]) -> None:
        if self.nodes_dir is None or self.lineage_csv is None:
            raise RuntimeError("Storage initialized call missing.")

        self._max_id = max(self._max_id, node.id)
        fn = (
            f"score{node.score:012.6f}_node{node.id:05d}_parent{node.parent_id:05d}.svg"
        )
        path = self.nodes_dir / fn

        if node.state.payload.svg:
            with path.open("w", encoding="utf-8") as f:
                f.write(node.state.payload.svg)

        exists = self.lineage_csv.is_file()
        with self.lineage_csv.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(
                    ["id", "parent", "secondary_parent", "score", "temp", "summary"]
                )
            w.writerow(
                [
                    node.id,
                    node.parent_id,
                    node.secondary_parent_id or "",
                    f"{node.score:.6f}",
                    f"{node.state.model_temperature:.3f}",
                    node.state.payload.change_summary or "",
                ]
            )

    def save_final_svg(self, svg_content: str) -> None:
        with self.output_svg_path.open("w", encoding="utf-8") as f:
            f.write(svg_content)

        self.project_dir.mkdir(parents=True, exist_ok=True)
        final_backup = self.project_dir / f"best_{self.output_svg_path.name}"
        with final_backup.open("w", encoding="utf-8") as f:
            f.write(svg_content)

        log.info(f"Updated best SVG at: {self.output_svg_path}")

    def load_seed_svg(self, seed_path: str) -> str:
        with Path(seed_path).open(encoding="utf-8") as f:
            return f.read()
