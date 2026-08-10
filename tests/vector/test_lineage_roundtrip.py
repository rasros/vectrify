"""End-to-end schema contract for lineage.csv.

The writers are driven by a single LINEAGE_COLUMNS list, so a row can never
misalign with the header. Nothing pinned the other half: both analysis scripts
address columns by string literal and fall back to 0.0, so renaming a column on
either side would silently degrade every reader while all existing tests passed.
These tests write with the real adapter and read with the real script loaders.
"""

import sys
from pathlib import Path

import pytest

from vectrify.formats.models import VectorStatePayload
from vectrify.search import ChainState, SearchNode
from vectrify.vector.storage import LINEAGE_COLUMNS, FileStorageAdapter

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# The analysis scripts are not part of the installed package, so skip rather
# than fail if this runs from somewhere the repo layout does not hold.
pytest.importorskip("matplotlib", reason="scripts/plot_run.py needs matplotlib")
if not SCRIPTS.is_dir():
    pytest.skip(f"scripts/ not found at {SCRIPTS}", allow_module_level=True)


def _node(node_id: int, score: float, visual: float, structural: float) -> SearchNode:
    return SearchNode(
        score=score,
        id=node_id,
        parent_id=0,
        epoch=1,
        metrics={"visual_complexity": visual, "structural_complexity": structural},
        state=ChainState(
            score=score,
            payload=VectorStatePayload(
                content=f"<svg id='{node_id}'/>",
                raster_data_url=None,
                raster_preview_data_url=None,
                origin="mutation",
                heatmap_data_url=None,
            ),
        ),
    )


@pytest.fixture
def written_run(tmp_path):
    adapter = FileStorageAdapter(str(tmp_path / "out.svg"))
    adapter.initialize()
    adapter.save_node(_node(1, 0.25, 1000.0, 300.0))
    adapter.save_node(_node(2, 0.50, 2000.0, 800.0))
    adapter.record_eviction(2, tasks_completed=42)
    assert adapter.current_run_dir is not None
    return adapter.current_run_dir


def test_plot_run_reads_back_what_storage_wrote(written_run):
    from plot_run import load_lineage

    rows = {r["id"]: r for r in load_lineage(written_run)}

    assert rows[1]["score"] == pytest.approx(0.25)
    assert rows[1]["visual_complexity"] == pytest.approx(1000.0)
    assert rows[1]["structural_complexity"] == pytest.approx(300.0)
    assert rows[1]["epoch"] == 1
    assert rows[2]["visual_complexity"] == pytest.approx(2000.0)
    assert rows[2]["structural_complexity"] == pytest.approx(800.0)


def test_plot_run_reader_covers_every_column_it_needs(written_run):
    from plot_run import load_lineage

    rows = load_lineage(written_run)
    assert rows

    produced = set(rows[0]) - {"parent"}  # 'parent' is renamed from the column
    for key in produced:
        assert key in LINEAGE_COLUMNS, f"reader invents a column: {key}"


def test_eviction_round_trips_into_the_final_pool(written_run):
    from plot_run import load_final_pool_ids

    # Node 2 was evicted, so only node 1 remains in the final pool.
    assert load_final_pool_ids(written_run) == {1}


def test_clean_runs_reads_back_what_storage_wrote(written_run):
    from clean_runs import collect_node_files, load_complexities_from_lineage

    nodes = collect_node_files(written_run / "nodes")
    load_complexities_from_lineage(written_run / "lineage.csv", nodes)
    by_id = {n["id"]: n for n in nodes}

    assert by_id[1]["visual_complexity"] == pytest.approx(1000.0)
    assert by_id[1]["structural_complexity"] == pytest.approx(300.0)
    assert by_id[2]["visual_complexity"] == pytest.approx(2000.0)
    assert by_id[2]["structural_complexity"] == pytest.approx(800.0)


def test_legacy_single_column_runs_still_read(tmp_path):
    from plot_run import load_lineage

    run = tmp_path / "runs" / "2026-01-01_00-00-00"
    run.mkdir(parents=True)
    (run / "lineage.csv").write_text(
        "id,parent,secondary_parent,epoch,score,complexity,summary,content_md5,evicted\n"
        "1,0,,0,0.400000,1500,seed,abc,\n",
        encoding="utf-8",
    )

    row = load_lineage(run)[0]
    assert row["visual_complexity"] == pytest.approx(1500.0)
    assert row["structural_complexity"] == pytest.approx(0.0)


def test_legacy_runs_are_labelled_in_plots(tmp_path):
    from plot_run import _label, uses_legacy_complexity

    legacy = tmp_path / "runs" / "2026-01-01_00-00-00"
    legacy.mkdir(parents=True)
    (legacy / "lineage.csv").write_text(
        "id,parent,secondary_parent,epoch,score,complexity,summary,content_md5,evicted\n"
        "1,0,,0,0.400000,1500,seed,abc,\n",
        encoding="utf-8",
    )

    assert uses_legacy_complexity(legacy) is True
    assert "legacy complexity" in _label(legacy)


def test_current_runs_are_not_labelled_legacy(written_run):
    from plot_run import _label, uses_legacy_complexity

    assert uses_legacy_complexity(written_run) is False
    assert _label(written_run) == written_run.name
