from pathlib import Path

from vectrify.formats.graphviz.plugin import GraphvizPlugin
from vectrify.formats.svg.plugin import SvgPlugin
from vectrify.formats.typst.plugin import TypstPlugin
from vectrify.run_dirs import OUTPUT_EXTENSIONS, project_runs_dir, run_dirs_in


def test_output_extensions_match_plugins():
    plugin_exts = {
        SvgPlugin.file_extension,
        GraphvizPlugin.file_extension,
        TypstPlugin.file_extension,
    }
    assert plugin_exts == OUTPUT_EXTENSIONS


def test_project_runs_dir_from_output_file(tmp_path):
    for ext in OUTPUT_EXTENSIONS:
        out = tmp_path / f"result{ext}"
        assert project_runs_dir(out) == tmp_path / "result" / "runs"


def test_project_runs_dir_from_runs_dir(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    assert project_runs_dir(runs) == runs


def test_project_runs_dir_from_project_dir(tmp_path):
    (tmp_path / "runs").mkdir()
    assert project_runs_dir(tmp_path) == tmp_path / "runs"


def test_project_runs_dir_unresolvable_returns_none(tmp_path):
    assert project_runs_dir(tmp_path / "nothing") is None


def test_run_dirs_in_sorted_oldest_first(tmp_path):
    for name in ("2024-02-01_00-00-00", "2024-01-01_00-00-00"):
        (tmp_path / name).mkdir()
    (tmp_path / "stray-file.txt").write_text("x")
    dirs = run_dirs_in(tmp_path)
    assert [d.name for d in dirs] == [
        "2024-01-01_00-00-00",
        "2024-02-01_00-00-00",
    ]


def test_run_dirs_in_ignores_files(tmp_path):
    assert run_dirs_in(tmp_path) == []
    assert isinstance(project_runs_dir(Path("x.unknown")), type(None))
