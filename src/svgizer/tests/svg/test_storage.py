import csv

import pytest

from svgizer.search import ChainState, SearchNode
from svgizer.svg.adapter import SvgStatePayload
from svgizer.svg.storage import FileStorageAdapter


@pytest.fixture
def dummy_state() -> ChainState:
    payload = SvgStatePayload(
        svg="<svg><circle r='10'/></svg>",
        raster_data_url=None,
        raster_preview_data_url=None,
        change_summary="Fixed circle",
        invalid_msg=None,
    )
    return ChainState(
        score=0.123456,
        model_temperature=0.6,
        stale_hits=0,
        payload=payload,
    )


@pytest.fixture
def dummy_node(dummy_state) -> SearchNode:
    return SearchNode(score=0.123456, id=42, parent_id=10, state=dummy_state)


def test_adapter_path_resolution(tmp_path):
    output_path = str(tmp_path / "somedir" / "result.svg")
    adapter = FileStorageAdapter(output_svg_path=output_path)

    assert adapter.base_name == "result"
    assert adapter.ext == ".svg"
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


def test_save_node_and_lineage(tmp_path, dummy_node):
    adapter = FileStorageAdapter(str(tmp_path / "out.svg"))
    adapter.initialize()

    adapter.save_node(dummy_node)

    expected_filename = "score00000.123456_node00042_parent00010.svg"

    assert adapter.nodes_dir is not None
    svg_path = adapter.nodes_dir / expected_filename
    assert svg_path.is_file()
    assert adapter.max_node_id == 42

    assert adapter.lineage_csv is not None
    assert adapter.lineage_csv.is_file()
    with adapter.lineage_csv.open(encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert reader[0] == [
            "id",
            "parent",
            "secondary_parent",
            "score",
            "temp",
            "summary",
        ]
        assert reader[1][0] == "42"
        assert reader[1][5] == "Fixed circle"


def test_load_resume_nodes(tmp_path):
    adapter = FileStorageAdapter(str(tmp_path / "resume_test.svg"), resume=True)
    adapter.initialize()

    valid_svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'
    new_format_fn = "score00000.555000_node00015_parent00010.svg"

    prev_run_nodes = adapter.runs_dir / "2020-01-01_00-00-00" / "nodes"
    prev_run_nodes.mkdir(parents=True)
    with (prev_run_nodes / new_format_fn).open("w") as f:
        f.write(valid_svg)

    nodes = adapter.load_resume_nodes()

    assert len(nodes) == 1
    node_id, _svg_text = nodes[0]
    assert node_id == 15
    assert adapter.max_node_id == 15


def test_load_resume_nodes_when_resume_false(tmp_path):
    adapter = FileStorageAdapter(str(tmp_path / "resume_test.svg"), resume=False)
    nodes = adapter.load_resume_nodes()
    assert nodes == []
    assert adapter.max_node_id == 0
