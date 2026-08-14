import base64
import csv
import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path

from vectrify.formats.models import VectorStatePayload
from vectrify.llm.base import split_data_url
from vectrify.score.metrics import METRIC_NAMES
from vectrify.search import SearchNode

log = logging.getLogger(__name__)

# lineage.csv schema. The header and every row are written through this one
# list, so a row can never fall out of alignment with the header. The metric
# columns come from the registry, so registering a metric adds its column here
# without an edit.
LINEAGE_COLUMNS = [
    "id",
    "parent",
    "secondary_parent",
    "epoch",
    "score",
    *METRIC_NAMES,
    "summary",
    "content_md5",
    "evicted",
]


class FileStorageAdapter:
    def __init__(
        self,
        output_path: str,
        file_extension: str = ".svg",
        resume: bool = False,
        resume_top: int | None = None,
        save_raster: bool = False,
        save_heatmap: bool = False,
        write_lineage: bool = True,
    ):
        self.output_path = Path(output_path)
        self.file_extension = file_extension
        self.resume = resume
        self.resume_top = resume_top
        self.save_raster = save_raster
        self.save_heatmap = save_heatmap
        self.write_lineage = write_lineage
        self._max_id = 0

        self.base_name = self.output_path.stem
        # An extensionless output path has stem == filename, which would make
        # project_dir the output path itself: initialize() creates it as a
        # directory and save_best can then never write the file. Suffix the
        # project dir in that case so the two can never collide.
        project_name = self.base_name
        if not self.output_path.suffix:
            project_name = f"{self.base_name}_runs"
        self.project_dir = self.output_path.parent / project_name
        self.runs_dir = self.project_dir / "runs"

        self.current_run_dir: Path | None = None
        self.nodes_dir: Path | None = None
        self.lineage_csv: Path | None = None

    @property
    def max_node_id(self) -> int:
        return self._max_id

    def initialize(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.current_run_dir = self._claim_run_dir()
        self.nodes_dir = self.current_run_dir / "nodes"
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.lineage_csv = self.current_run_dir / "lineage.csv"
        log.debug(f"Storage initialized at: {self.current_run_dir}")

    def _claim_run_dir(self) -> Path:
        """Create and return a run directory no other run is using.

        The timestamp is second-resolution, so two runs started in the same
        second would otherwise share a directory and interleave appends into one
        stats.csv and lineage.csv. Suffixing on collision keeps the name
        readable, unlike adding sub-second precision.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for attempt in range(1, 100):
            name = timestamp if attempt == 1 else f"{timestamp}-{attempt}"
            candidate = self.runs_dir / name
            try:
                candidate.mkdir(parents=False, exist_ok=False)
                return candidate
            except FileExistsError:
                continue
        raise RuntimeError(f"Could not claim a run directory under {self.runs_dir}")

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
        # `inf` is its own alternative because save_node writes f"{score:.6f}",
        # which renders INVALID_SCORE as a bare "inf".
        file_pattern = re.compile(rf"^(inf|[0-9.]+)_(\d+){ext}$")
        parsed_files: list[tuple[int, Path, float]] = []

        glob_pattern = f"*{self.file_extension}"
        for file_path in target_nodes_dir.glob(glob_pattern):
            match = file_pattern.match(file_path.name)
            node_id = int(match.group(2)) if match else self._max_id + 1
            # Score comes from the match, not from re-splitting the stem: a
            # user-dropped file with a non-numeric prefix would otherwise raise
            # an unhandled ValueError while sorting for --resume-top.
            score = float(match.group(1)) if match else float("inf")

            self._max_id = max(self._max_id, node_id)
            parsed_files.append((node_id, file_path, score))

        if self.resume_top is not None:
            parsed_files.sort(key=lambda item: item[2])
            parsed_files = parsed_files[: self.resume_top]

        resumed_data = []
        for node_id, file_path, _score in parsed_files:
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

        # --no-write-lineage suppresses the per-node files and lineage.csv, but
        # the raster/heatmap sidecars stay under their own flags.
        if not self.write_lineage:
            self._save_sidecars(node)
            return

        base_fn = f"{node.score:.6f}_{node.id}"

        if node.state.payload.content:
            content_path = self.nodes_dir / f"{base_fn}{self.file_extension}"
            content_path.write_text(node.state.payload.content, encoding="utf-8")

        self._save_sidecars(node)

        content_md5 = (
            hashlib.md5(node.state.payload.content.encode()).hexdigest()
            if node.state.payload.content
            else ""
        )
        self._append_lineage_row(
            {
                "id": node.id,
                "parent": node.parent_id,
                "secondary_parent": node.secondary_parent_id or "",
                "epoch": node.epoch,
                "score": f"{node.score:.6f}",
                # `.6g` rather than `.0f`: the metrics are byte and
                # character counts, but region distances live in [0, 1] and an
                # integer format would write every one of them as "0".
                **{name: f"{node.metrics.get(name, 0.0):.6g}" for name in METRIC_NAMES},
                "summary": node.state.payload.origin or "",
                "content_md5": content_md5,
            }
        )

    def _save_sidecars(self, node: SearchNode[VectorStatePayload]) -> None:
        """Write the optional .png / .heatmap.png next to a node."""
        assert self.nodes_dir is not None
        base_fn = f"{node.score:.6f}_{node.id}"

        if self.save_raster and node.state.payload.raster_data_url:
            _, b64 = split_data_url(node.state.payload.raster_data_url)
            (self.nodes_dir / f"{base_fn}.png").write_bytes(base64.b64decode(b64))

        if self.save_heatmap and node.state.payload.heatmap_data_url:
            _, b64 = split_data_url(node.state.payload.heatmap_data_url)
            (self.nodes_dir / f"{base_fn}.heatmap.png").write_bytes(
                base64.b64decode(b64)
            )

    def _append_lineage_row(self, row: dict[str, object]) -> None:
        """Append one lineage row, writing the header first if the file is new.

        Rows are passed as a mapping so a caller cannot silently misalign its
        values against the header -- unset columns are written empty.
        """
        assert self.lineage_csv is not None
        exists = self.lineage_csv.is_file()
        with self.lineage_csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LINEAGE_COLUMNS, restval="")
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def save_best(self, node: SearchNode[VectorStatePayload]) -> None:
        """Write the winning candidate's content to the top-level output path."""
        content = node.state.payload.content
        if not content:
            log.warning("No valid candidate found; %s not written.", self.output_path)
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(content, encoding="utf-8")
        log.info(
            "Best candidate (score %.6f) written to %s", node.score, self.output_path
        )

    def record_eviction(self, node_id: int, tasks_completed: int) -> None:
        if self.lineage_csv is None or not self.lineage_csv.exists():
            return
        try:
            self._append_lineage_row({"id": node_id, "evicted": tasks_completed})
        except Exception as e:
            log.error(f"Failed to record eviction for node {node_id}: {e}")
