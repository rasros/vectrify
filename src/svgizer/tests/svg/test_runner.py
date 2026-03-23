import pytest
from PIL import Image

from svgizer.diff import ScorerType
from svgizer.search import StrategyType
from svgizer.svg.runner import run_svg_search
from svgizer.svg.storage import FileStorageAdapter


@pytest.mark.llm
def test_run_svg_search_end_to_end(tmp_path):
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (32, 32), color="blue")
    img.save(img_path)

    out_svg_path = tmp_path / "output.svg"
    storage = FileStorageAdapter(
        output_svg_path=str(out_svg_path),
        resume=False,
        openai_image_long_side=64,
        base_temp=0.0,
    )

    run_svg_search(
        image_path=str(img_path),
        storage=storage,
        seed_svg_path=None,
        max_accepts=1,
        workers=1,
        base_model_temperature=0.0,
        image_long_side=64,
        max_wall_seconds=None,
        log_level="DEBUG",
        scorer_type=ScorerType.SIMPLE,
        strategy_type=StrategyType.GREEDY,
        goal="Generate a simple blue rectangle.",
        reasoning="none",
        llm_provider="openai",
        llm_model="gpt-5.4-nano",
        write_lineage=False,
    )

    assert out_svg_path.is_file(), "Final SVG file was not saved to storage."

    with out_svg_path.open(encoding="utf-8") as f:
        content = f.read().lower()
        assert "<svg" in content, "Output does not contain valid SVG syntax."
