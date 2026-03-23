import csv
import logging
import os

import pytest

from svgizer.search import ChainState, SearchNode
from svgizer.svg_adapter import SvgStatePayload
from svgizer.storage import FileStorageAdapter


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
    adapter = FileStorageAdapter(output_svg_path=output_path, write_lineage=True)

    assert adapter.base_name == "result"
    assert adapter.ext == ".svg"
    assert adapter.out_dir == str(tmp_path / "somedir")
    assert adapter.nodes_dir == str(tmp_path / "somedir" / "result_nodes")
    assert adapter.lineage_csv_path == str(tmp_path / "somedir" / "result_lineage.csv")
    assert adapter.write_lineage_enabled is True


def test_initialize_creates_directories(tmp_path):
    output_path = str(tmp_path / "nested" / "dir" / "out.svg")
    adapter = FileStorageAdapter(output_path)

    assert not os.path.exists(adapter.nodes_dir)
    adapter.initialize()
    assert os.path.isdir(adapter.nodes_dir)


def test_save_node(tmp_path, dummy_node):
    adapter = FileStorageAdapter(str(tmp_path / "out.svg"))
    adapter.initialize()

    saved_path = adapter.save_node(dummy_node)

    expected_filename = "score00000.123456_node00042_parent00010.svg"
    assert os.path.basename(saved_path) == expected_filename
    assert os.path.isfile(saved_path)

    with open(saved_path, encoding="utf-8") as f:
        assert f.read() == dummy_node.state.payload.svg


def test_write_lineage(tmp_path):
    adapter = FileStorageAdapter(str(tmp_path / "out.svg"), write_lineage=True)
    adapter.initialize()

    node_info = {
        1: (0, 0.9, "/path/to/1.svg", "First try"),
        2: (1, 0.5, "/path/to/2.svg", None),
    }

    adapter.write_lineage(node_info)

    assert os.path.isfile(adapter.lineage_csv_path)

    with open(adapter.lineage_csv_path, encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 3
        assert reader[0] == ["node_id", "parent_id", "score", "path", "change_summary"]
        assert reader[1] == ["1", "0", "0.900000", "/path/to/1.svg", "First try"]
        assert reader[2] == ["2", "1", "0.500000", "/path/to/2.svg", ""]


def test_write_lineage_disabled(tmp_path):
    adapter = FileStorageAdapter(str(tmp_path / "out.svg"), write_lineage=False)
    adapter.write_lineage({1: (0, 0.9, "path", "sum")})
    assert not os.path.exists(adapter.lineage_csv_path)


def test_save_final_svg_and_load_seed(tmp_path):
    adapter = FileStorageAdapter(str(tmp_path / "final.svg"))
    test_svg = "<svg>final</svg>"

    adapter.save_final_svg(test_svg)
    assert os.path.isfile(adapter.output_svg_path)

    loaded_svg = adapter.load_seed_svg(adapter.output_svg_path)
    assert loaded_svg == test_svg


def test_load_resume_nodes(tmp_path):
    adapter = FileStorageAdapter(str(tmp_path / "resume_test.svg"), resume=True)
    adapter.initialize()

    valid_svg_new = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10" fill="red"/></svg>'
    valid_svg_old = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10" fill="blue"/></svg>'

    new_format_path = os.path.join(
        adapter.nodes_dir, "score00000.555000_node00015_parent00010.svg"
    )
    with open(new_format_path, "w") as f:
        f.write(valid_svg_new)

    old_format_path = os.path.join(
        adapter.out_dir, "resume_test_node00008_score0.999.svg"
    )
    with open(old_format_path, "w") as f:
        f.write(valid_svg_old)

    with open(os.path.join(adapter.nodes_dir, "ignore_me.svg"), "w") as f:
        f.write("junk")

    log = logging.getLogger("test")
    nodes, best_seen, max_id = adapter.load_resume_nodes(
        log=log,
        base_model_temperature=0.6,
        original_w=10,
        original_h=10,
        openai_image_long_side=10,
    )

    assert len(nodes) == 2
    assert max_id == 15

    assert nodes[0].id == 8
    assert nodes[0].score == 0.999
    assert nodes[0].parent_id == 0
    assert nodes[0].state.payload.svg == valid_svg_old
    assert nodes[0].state.payload.raster_preview_data_url.startswith("data:image/png;base64,")

    assert nodes[1].id == 15
    assert nodes[1].score == 0.555
    assert nodes[1].parent_id == 10
    assert nodes[1].state.payload.svg == valid_svg_new
    assert nodes[1].state.payload.raster_preview_data_url.startswith("data:image/png;base64,")

    assert best_seen.id == 15


def test_load_resume_nodes_when_resume_false(tmp_path):
    adapter = FileStorageAdapter(str(tmp_path / "resume_test.svg"), resume=False)
    nodes, best, max_id = adapter.load_resume_nodes(
        logging.getLogger(), 0.6, 100, 100, 512
    )
    assert nodes == []
    assert best is None
    assert max_id == 0