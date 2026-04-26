import csv
import io

import pytest
from PIL import Image

from vectrify.formats.models import VectorStatePayload
from vectrify.image_utils import png_bytes_to_data_url
from vectrify.search import ChainState, SearchNode
from vectrify.vector.storage import FileStorageAdapter


def _make_png(color: str = "red", size: int = 16) -> bytes:
    img = Image.new("RGB", (size, size), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def dummy_state() -> ChainState:
    payload = VectorStatePayload(
        content="<svg><circle r='10'/></svg>",
        raster_data_url=None,
        raster_preview_data_url=None,
        origin="Fixed circle",
        invalid_msg=None,
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
    with adapter.lineage_csv.open(encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert reader[0] == [
            "id",
            "parent",
            "secondary_parent",
            "epoch",
            "score",
            "complexity",
            "summary",
            "content_md5",
            "evicted",
        ]
        assert reader[1][0] == "42"
        assert reader[1][3] == "0"  # epoch
        assert reader[1][6] == "Fixed circle"
        import hashlib

        expected_md5 = hashlib.md5(b"<svg><circle r='10'/></svg>").hexdigest()
        assert reader[1][7] == expected_md5
        assert reader[1][8] == ""  # evicted (empty until evicted)


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
        invalid_msg=None,
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
        invalid_msg=None,
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
