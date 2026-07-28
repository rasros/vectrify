import csv
import hashlib

import pytest

from tests.helpers import make_png
from vectrify.formats.models import VectorStatePayload
from vectrify.image_utils import png_bytes_to_data_url
from vectrify.search import ChainState, SearchNode
from vectrify.vector.storage import LINEAGE_COLUMNS, FileStorageAdapter


def _make_png(color: str = "red", size: int = 16) -> bytes:
    return make_png(color, size)


@pytest.fixture
def dummy_state() -> ChainState:
    payload = VectorStatePayload(
        content="<svg><circle r='10'/></svg>",
        raster_data_url=None,
        raster_preview_data_url=None,
        origin="Fixed circle",
    )
    return ChainState(
        score=0.123456,
        payload=payload,
    )


@pytest.fixture
def dummy_node(dummy_state) -> SearchNode:
    return SearchNode(score=0.123456, id=42, parent_id=10, state=dummy_state)


def test_adapter_path_resolution(tmp_path):
    output_path = str(tmp_path / "somedir" / "result.svg")
    adapter = FileStorageAdapter(output_path=output_path)

    assert adapter.base_name == "result"
    assert str(adapter.project_dir) == str(tmp_path / "somedir" / "result")
    assert str(adapter.runs_dir) == str(tmp_path / "somedir" / "result" / "runs")


def test_initialize_creates_directories(tmp_path):
    output_path = str(tmp_path / "nested" / "dir" / "out.svg")
    adapter = FileStorageAdapter(output_path)

    adapter.initialize()
    assert adapter.nodes_dir is not None
    assert adapter.current_run_dir is not None
    assert adapter.nodes_dir.is_dir()
    assert adapter.lineage_csv is not None
    assert adapter.lineage_csv.parent == adapter.current_run_dir


def test_save_best_writes_output_path(tmp_path, dummy_node):
    output_path = tmp_path / "sub" / "out.svg"
    adapter = FileStorageAdapter(str(output_path))
    adapter.initialize()

    adapter.save_best(dummy_node)

    assert output_path.is_file()
    assert output_path.read_text(encoding="utf-8") == dummy_node.state.payload.content


def test_save_best_skips_empty_content(tmp_path):
    output_path = tmp_path / "out.svg"
    adapter = FileStorageAdapter(str(output_path))
    empty_state = ChainState(
        score=0.0,
        payload=VectorStatePayload(None, None, None, None),
    )
    node = SearchNode(score=0.0, id=0, parent_id=0, state=empty_state)

    adapter.save_best(node)

    assert not output_path.exists()


def test_save_node_and_lineage(tmp_path, dummy_node):
    adapter = FileStorageAdapter(str(tmp_path / "out.svg"))
    adapter.initialize()

    adapter.save_node(dummy_node)

    assert adapter.nodes_dir is not None
    svg_path = adapter.nodes_dir / "0.123456_42.svg"
    assert svg_path.is_file()
    assert adapter.max_node_id == 42

    assert adapter.lineage_csv is not None
    assert adapter.lineage_csv.is_file()
    with adapter.lineage_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert list(rows[0]) == LINEAGE_COLUMNS
    row = rows[0]
    assert row["id"] == "42"
    assert row["epoch"] == "0"
    assert row["summary"] == "Fixed circle"
    assert row["visual_complexity"] == "0"
    assert row["structural_complexity"] == "0"
    expected_md5 = hashlib.md5(b"<svg><circle r='10'/></svg>").hexdigest()
    assert row["content_md5"] == expected_md5
    assert row["evicted"] == ""  # empty until evicted


def test_lineage_columns_match_the_written_header(tmp_path, dummy_node):
    """Header and rows come from one column list, so they cannot misalign."""
    adapter = FileStorageAdapter(str(tmp_path / "out.svg"))
    adapter.initialize()
    adapter.save_node(dummy_node)

    assert adapter.lineage_csv is not None
    with adapter.lineage_csv.open(encoding="utf-8", newline="") as f:
        header = next(iter(csv.reader(f)))
    assert header == LINEAGE_COLUMNS


def test_eviction_row_lands_in_the_evicted_column(tmp_path, dummy_node):
    """An eviction row is sparse; its value must not shift into a neighbouring
    column when the schema gains a field."""
    adapter = FileStorageAdapter(str(tmp_path / "out.svg"))
    adapter.initialize()
    adapter.save_node(dummy_node)
    adapter.record_eviction(dummy_node.id, tasks_completed=77)

    assert adapter.lineage_csv is not None
    with adapter.lineage_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    eviction = rows[-1]
    assert eviction["id"] == "42"
    assert eviction["evicted"] == "77"
    # Every other field is blank -- in particular the value did not land in
    # content_md5, which is where a hand-built positional row would have put it.
    assert eviction["content_md5"] == ""
    assert eviction["score"] == ""
    assert eviction["structural_complexity"] == ""


def test_load_resume_nodes(tmp_path):
    adapter = FileStorageAdapter(
        str(tmp_path / "resume_test.svg"), resume=True, file_extension=".svg"
    )
    adapter.initialize()

    valid_svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'

    prev_run_nodes = adapter.runs_dir / "2020-01-01_00-00-00" / "nodes"
    prev_run_nodes.mkdir(parents=True)
    with (prev_run_nodes / "0.555000_15.svg").open("w") as f:
        f.write(valid_svg)

    nodes = adapter.load_resume_nodes()

    assert len(nodes) == 1
    node_id, _content_text = nodes[0]
    assert node_id == 15
    assert adapter.max_node_id == 15


def test_load_resume_nodes_when_resume_false(tmp_path):
    adapter = FileStorageAdapter(str(tmp_path / "resume_test.svg"), resume=False)
    nodes = adapter.load_resume_nodes()
    assert nodes == []
    assert adapter.max_node_id == 0


def _make_node_with_raster_and_heatmap(
    raster_data_url: str | None = None,
    heatmap_data_url: str | None = None,
) -> SearchNode:
    payload = VectorStatePayload(
        content="<svg/>",
        raster_data_url=raster_data_url,
        raster_preview_data_url=None,
        origin="test",
        heatmap_data_url=heatmap_data_url,
    )
    return SearchNode(
        score=0.5,
        id=1,
        parent_id=0,
        state=ChainState(score=0.5, payload=payload),
    )


def test_save_raster_writes_png(tmp_path):
    adapter = FileStorageAdapter(str(tmp_path / "out.svg"), save_raster=True)
    adapter.initialize()
    assert adapter.nodes_dir is not None
    node = _make_node_with_raster_and_heatmap(
        raster_data_url=png_bytes_to_data_url(_make_png())
    )
    adapter.save_node(node)
    assert (adapter.nodes_dir / "0.500000_1.png").is_file()


def test_save_raster_false_does_not_write_png(tmp_path):
    adapter = FileStorageAdapter(str(tmp_path / "out.svg"), save_raster=False)
    adapter.initialize()
    assert adapter.nodes_dir is not None
    node = _make_node_with_raster_and_heatmap(
        raster_data_url=png_bytes_to_data_url(_make_png())
    )
    adapter.save_node(node)
    assert not (adapter.nodes_dir / "0.500000_1.png").is_file()


def test_save_heatmap_writes_heatmap_png(tmp_path):
    adapter = FileStorageAdapter(str(tmp_path / "out.svg"), save_heatmap=True)
    adapter.initialize()
    assert adapter.nodes_dir is not None
    node = _make_node_with_raster_and_heatmap(
        heatmap_data_url=png_bytes_to_data_url(_make_png("blue"))
    )
    adapter.save_node(node)
    assert (adapter.nodes_dir / "0.500000_1.heatmap.png").is_file()


def test_save_heatmap_false_does_not_write_heatmap_png(tmp_path):
    adapter = FileStorageAdapter(str(tmp_path / "out.svg"), save_heatmap=False)
    adapter.initialize()
    assert adapter.nodes_dir is not None
    node = _make_node_with_raster_and_heatmap(
        heatmap_data_url=png_bytes_to_data_url(_make_png("blue"))
    )
    adapter.save_node(node)
    assert not (adapter.nodes_dir / "0.500000_1.heatmap.png").is_file()


def test_save_node_content_none_does_not_write_content_file(tmp_path):
    adapter = FileStorageAdapter(str(tmp_path / "out.svg"))
    adapter.initialize()
    assert adapter.nodes_dir is not None
    payload = VectorStatePayload(
        content=None,
        raster_data_url=None,
        raster_preview_data_url=None,
        origin="no content",
    )
    node = SearchNode(
        score=0.5,
        id=99,
        parent_id=0,
        state=ChainState(score=0.5, payload=payload),
    )
    adapter.save_node(node)
    assert not (adapter.nodes_dir / "0.500000_99.svg").is_file()
    assert adapter.lineage_csv is not None
    assert adapter.lineage_csv.is_file()


def test_save_heatmap_content_is_valid_png(tmp_path):
    adapter = FileStorageAdapter(str(tmp_path / "out.svg"), save_heatmap=True)
    adapter.initialize()
    assert adapter.nodes_dir is not None
    original_png = _make_png("green")
    node = _make_node_with_raster_and_heatmap(
        heatmap_data_url=png_bytes_to_data_url(original_png)
    )
    adapter.save_node(node)
    written = (adapter.nodes_dir / "0.500000_1.heatmap.png").read_bytes()
    assert written == original_png


def test_no_write_lineage_suppresses_node_files_and_lineage(tmp_path, dummy_node):
    """Regression: --no-write-lineage was documented as suppressing lineage.csv
    and the per-node files, but the flag never reached storage at all.
    """
    adapter = FileStorageAdapter(str(tmp_path / "out.svg"), write_lineage=False)
    adapter.initialize()
    adapter.save_node(dummy_node)

    assert adapter.lineage_csv is not None
    assert not adapter.lineage_csv.exists()
    assert adapter.nodes_dir is not None
    assert list(adapter.nodes_dir.iterdir()) == []


def test_write_lineage_true_still_writes(tmp_path, dummy_node):
    adapter = FileStorageAdapter(str(tmp_path / "out.svg"), write_lineage=True)
    adapter.initialize()
    adapter.save_node(dummy_node)

    assert adapter.lineage_csv is not None
    assert adapter.lineage_csv.exists()
    assert adapter.nodes_dir is not None
    assert [p.name for p in adapter.nodes_dir.iterdir()] == ["0.123456_42.svg"]


def test_extensionless_output_does_not_collide_with_the_project_dir(
    tmp_path, dummy_node
):
    """Regression: project_dir was parent/stem, which for an extensionless path
    *is* the output path. initialize() created it as a directory and save_best
    could then never write the file -- a completed search produced no output and
    no error, because the write failure was suppressed.
    """
    output = tmp_path / "myout"
    adapter = FileStorageAdapter(str(output))
    adapter.initialize()

    assert adapter.project_dir != output
    adapter.save_best(dummy_node)
    assert output.is_file()
    assert output.read_text(encoding="utf-8") == "<svg><circle r='10'/></svg>"


def test_runs_started_in_the_same_second_get_distinct_directories(tmp_path):
    """Regression: the run dir name is second-resolution and was created with
    exist_ok=True, so concurrent runs shared one directory and interleaved
    appends into a single stats.csv and lineage.csv.
    """
    dirs = set()
    for _ in range(5):
        adapter = FileStorageAdapter(str(tmp_path / "out.svg"))
        adapter.initialize()
        dirs.add(adapter.current_run_dir)
    assert len(dirs) == 5


def test_resume_parses_inf_scored_node_files(tmp_path):
    """save_node writes f"{score:.6f}", which renders INVALID_SCORE as a bare
    "inf", so the filename grammar has to accept it.
    """
    nodes = tmp_path / "out" / "runs" / "2026-01-01_00-00-00" / "nodes"
    nodes.mkdir(parents=True)
    (nodes / "0.200000_2.svg").write_text("<svg id='2'/>", encoding="utf-8")
    (nodes / "inf_7.svg").write_text("<svg id='7'/>", encoding="utf-8")

    adapter = FileStorageAdapter(str(tmp_path / "out.svg"), resume=True)
    resumed = adapter.load_resume_nodes()

    assert {node_id for node_id, _ in resumed} == {2, 7}
    assert adapter.max_node_id == 7


def test_resume_top_tolerates_a_non_numeric_filename(tmp_path):
    """Regression: the --resume-top sort re-parsed the score out of the stem, so
    any file whose prefix was not a float raised an unhandled ValueError.
    """
    nodes = tmp_path / "out" / "runs" / "2026-01-01_00-00-00" / "nodes"
    nodes.mkdir(parents=True)
    (nodes / "0.100000_1.svg").write_text("<svg id='1'/>", encoding="utf-8")
    (nodes / "0.900000_2.svg").write_text("<svg id='2'/>", encoding="utf-8")
    (nodes / "handwritten.svg").write_text("<svg id='9'/>", encoding="utf-8")

    adapter = FileStorageAdapter(str(tmp_path / "out.svg"), resume=True, resume_top=2)
    resumed = adapter.load_resume_nodes()  # must not raise

    # The best-scoring real node is kept; the unparseable one sorts last (inf).
    assert 1 in {node_id for node_id, _ in resumed}
    assert len(resumed) == 2
