import csv
import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path

from svgizer.score.complexity import svg_complexity
from svgizer.search import SearchNode
from svgizer.search.nsga import crowding_distance, non_dominated_sort
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

    def load_resume_nodes(self, max_nodes: int = 20) -> list[tuple[int, str]]:
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

        log.info(
            f"Resuming top {max_nodes} nodes from latest run: {latest_run.name} via Pareto front"
        )

        file_pattern = re.compile(r"^([0-9.]+)_(\d+)\.svg$")
        parsed_files = []

        for file_path in target_nodes_dir.glob("*.svg"):
            match = file_pattern.match(file_path.name)
            if match:
                score = float(match.group(1))
                node_id = int(match.group(2))
            else:
                score = float("inf")
                node_id = self._max_id + 1

            self._max_id = max(self._max_id, node_id)
            parsed_files.append((score, node_id, file_path))

        valid_nodes = []
        for score, node_id, file_path in parsed_files:
            try:
                content = file_path.read_text(encoding="utf-8").strip()
                if content:
                    comp = svg_complexity(content)
                    node = SearchNode(
                        score=score,
                        id=node_id,
                        parent_id=0,
                        state=None,  # type: ignore
                        complexity=comp,
                    )
                    valid_nodes.append((node, content))
            except Exception as e:
                log.error(f"Failed to read resume node {file_path.name}: {e}")

        if not valid_nodes:
            return []

        # Extract objectives for NSGA sorting
        nodes_only = [n for n, _ in valid_nodes]
        max_score = (
            max((n.score for n in nodes_only if n.score < float("inf")), default=1.0)
            or 1.0
        )
        max_comp = max((n.complexity for n in nodes_only), default=1.0) or 1.0

        objectives = {
            n.id: (n.score / max_score, n.complexity / max_comp) for n in nodes_only
        }

        fronts = non_dominated_sort(nodes_only, objectives)

        resumed_data = []
        node_to_content = {n.id: c for n, c in valid_nodes}

        # Pick nodes front by front until we hit max_nodes
        for front in fronts:
            if len(resumed_data) >= max_nodes:
                break

            # Sort within the front using crowding distance to maximize diversity
            distances = crowding_distance(front, objectives)
            front_sorted = sorted(front, key=lambda n: -distances[n.id])

            for node in front_sorted:
                if len(resumed_data) >= max_nodes:
                    break
                resumed_data.append((node.id, node_to_content[node.id]))
                log.info(
                    f"Found node for resume: "
                    f"ID {node.id} (Score: {node.score:.6f}, Comp: {node.complexity})"
                )

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
