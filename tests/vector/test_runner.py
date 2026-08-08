import inspect

import pytest
from PIL import Image

from tests.helpers import TEST_MODEL
from vectrify import cli
from vectrify.formats.svg.plugin import SvgPlugin
from vectrify.score import ScorerType
from vectrify.search import StrategyType
from vectrify.vector.runner import run_vector_search
from vectrify.vector.storage import FileStorageAdapter


def test_runner_defaults_match_cli_defaults():
    defaults = {
        p.name: p.default
        for p in inspect.signature(run_vector_search).parameters.values()
    }
    assert defaults["pool_size"] == cli.DEFAULT_POOL_SIZE
    assert defaults["epoch_diversity"] == cli.DEFAULT_EPOCH_DIVERSITY
    assert defaults["vision_model"] == cli.DEFAULT_VISION_MODEL
    # llm_rate has no static default; it is derived from --workers at call time.
    assert defaults["llm_rate"] is None


def _make_storage(tmp_path):
    plugin = SvgPlugin()
    storage = FileStorageAdapter(
        output_path=str(tmp_path / "out.svg"),
        file_extension=plugin.file_extension,
        resume=False,
    )
    return plugin, storage


def _run(image_path, plugin, storage):
    run_vector_search(
        image_path=image_path,
        storage=storage,
        workers=1,
        resolution=32,
        max_wall_seconds=1.0,
        log_level="ERROR",
        scorer_type=ScorerType.SIMPLE,
        strategy_type=StrategyType.BEAM,
        goal=None,
        reasoning="none",
        llm_provider="openai",
        llm_model=TEST_MODEL,
        format_plugin=plugin,
        write_lineage=False,
        max_epochs=None,
    )


def test_missing_image_raises_before_creating_output_dirs(tmp_path):
    plugin, storage = _make_storage(tmp_path)
    with pytest.raises(FileNotFoundError):
        _run(str(tmp_path / "does-not-exist.png"), plugin, storage)
    assert not storage.project_dir.exists()


def test_corrupt_image_raises_value_error_before_creating_output_dirs(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_text("this is not an image", encoding="utf-8")
    plugin, storage = _make_storage(tmp_path)
    with pytest.raises(ValueError, match="could not be read as an image"):
        _run(str(bad), plugin, storage)
    assert not storage.project_dir.exists()


@pytest.mark.llm
def test_run_svg_search_end_to_end(tmp_path):
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (32, 32), color="blue")
    img.save(img_path)

    out_svg_path = tmp_path / "output.svg"
    plugin = SvgPlugin()
    storage = FileStorageAdapter(
        output_path=str(out_svg_path),
        file_extension=plugin.file_extension,
        resume=False,
    )

    run_vector_search(
        image_path=str(img_path),
        storage=storage,
        workers=1,
        resolution=32,
        max_wall_seconds=10.0,
        log_level="DEBUG",
        scorer_type=ScorerType.SIMPLE,
        strategy_type=StrategyType.BEAM,
        goal="Generate a simple blue rectangle.",
        reasoning="none",
        llm_provider="openai",
        llm_model=TEST_MODEL,
        format_plugin=plugin,
        write_lineage=False,
        max_epochs=None,
    )

    assert storage.nodes_dir is not None
    assert storage.nodes_dir.is_dir()

    svg_files = list(storage.nodes_dir.glob("*.svg"))
    assert len(svg_files) > 0

    with svg_files[-1].open(encoding="utf-8") as f:
        content = f.read().lower()
        assert "<svg" in content
