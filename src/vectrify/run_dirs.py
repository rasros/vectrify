"""Resolve vectrify output paths to their run directories.

Shared by scripts/plot_run.py and scripts/clean_runs.py.
"""

from pathlib import Path

# Output extensions of the format plugins (SvgPlugin, GraphvizPlugin,
# TypstPlugin). Kept as a plain set so the scripts don't have to import the
# plugin stack; a test asserts it stays in sync with plugin.file_extension.
OUTPUT_EXTENSIONS = {".svg", ".dot", ".typ"}


def project_runs_dir(path: Path) -> Path | None:
    """Map an output file, project dir, or runs dir to its runs directory.

    Returns None if *path* is none of those (callers then treat it as a
    single run dir or recurse).
    """
    if path.suffix.lower() in OUTPUT_EXTENSIONS:
        return path.parent / path.stem / "runs"
    if path.name == "runs" and path.is_dir():
        return path
    if (path / "runs").is_dir():
        return path / "runs"
    return None


def run_dirs_in(runs_dir: Path) -> list[Path]:
    """Timestamped run directories inside a runs dir, oldest first."""
    return sorted((d for d in runs_dir.iterdir() if d.is_dir()), key=lambda d: d.name)
