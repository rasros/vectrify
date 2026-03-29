import pytest
from PIL import Image

from svgizer.formats.svg.plugin import SvgPlugin
from svgizer.score import ScorerType
from svgizer.search import StrategyType
from svgizer.vector.runner import run_vector_search
from svgizer.vector.storage import FileStorageAdapter


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
        image_long_side=64,
    )

    run_vector_search(
        image_path=str(img_path),
        storage=storage,
        workers=1,
        image_long_side=32,
        max_wall_seconds=10.0,
        log_level="DEBUG",
        scorer_type=ScorerType.SIMPLE,
        strategy_type=StrategyType.GREEDY,
        goal="Generate a simple blue rectangle.",
        reasoning="none",
        llm_provider="openai",
        llm_model="gpt-5.4-nano",
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
