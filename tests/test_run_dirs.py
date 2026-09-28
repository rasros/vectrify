from pathlib import Path

from vectrify.formats.svg.plugin import SvgPlugin
from vectrify.run_dirs import OUTPUT_EXTENSIONS, project_runs_dir, run_dirs_in


def test_output_extensions_match_svg():
    assert {SvgPlugin.file_extension} == OUTPUT_EXTENSIONS


def test_removed_output_formats_are_not_treated_as_svg_projects(tmp_path):
    assert project_runs_dir(tmp_path / "drawing.dot") is None
    assert project_runs_dir(tmp_path / "drawing.typ") is None


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
