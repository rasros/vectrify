import io

from PIL import Image

from svgizer.formats.models import VectorResultPayload, VectorStatePayload
from svgizer.search import GreedyHillClimbingStrategy
from svgizer.search.models import Result
from svgizer.vector.adapter import VectorStrategyAdapter


def _make_png(color: str = "red", size: int = 16) -> bytes:
    img = Image.new("RGB", (size, size), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_adapter(write_lineage: bool = False) -> VectorStrategyAdapter:
    return VectorStrategyAdapter(
        base_strategy=GreedyHillClimbingStrategy(),
        image_long_side=64,
        write_lineage=write_lineage,
    )


def _make_result(
    content: str = "<svg/>",
    raster_png: bytes | None = None,
    preview_data_url: str | None = None,
) -> Result:
    return Result(
        task_id=1,
        parent_id=0,
        worker_slot=0,
        valid=True,
        score=0.5,
        payload=VectorResultPayload(
            content=content,
            raster_png=raster_png,
            change_summary="test",
            raster_preview_data_url=preview_data_url,
        ),
        content=content,
    )


def create_payload(content_text: str | None) -> VectorStatePayload:
    return VectorStatePayload(
        content=content_text,
        raster_data_url=None,
        raster_preview_data_url=None,
        change_summary=None,
        invalid_msg=None,
    )


def test_payload_creation():
    payload = create_payload("<svg></svg>")
    assert payload.content == "<svg></svg>"


def test_create_new_state_uses_precomputed_preview():
    adapter = _make_adapter()
    precomputed = "data:image/png;base64,PRECOMPUTED"
    result = _make_result(preview_data_url=precomputed)
    state = adapter.create_new_state(result)
    assert state.payload.raster_preview_data_url == precomputed


def test_create_new_state_falls_back_to_computing_preview():
    adapter = _make_adapter()
    png = _make_png()
    result = _make_result(raster_png=png, preview_data_url=None)
    state = adapter.create_new_state(result)
    assert state.payload.raster_preview_data_url is not None
    assert state.payload.raster_preview_data_url.startswith("data:image/png;base64,")


def test_create_new_state_no_png_no_preview():
    adapter = _make_adapter()
    result = _make_result(raster_png=None, preview_data_url=None)
    state = adapter.create_new_state(result)
    assert state.payload.raster_preview_data_url is None


def test_create_new_state_precomputed_takes_priority_over_raster_png():
    adapter = _make_adapter()
    precomputed = "data:image/png;base64,WINNER"
    result = _make_result(raster_png=_make_png(), preview_data_url=precomputed)
    state = adapter.create_new_state(result)
    assert state.payload.raster_preview_data_url == precomputed


def test_create_new_state_write_lineage_sets_raster_data_url():
    adapter = _make_adapter(write_lineage=True)
    result = _make_result(raster_png=_make_png())
    state = adapter.create_new_state(result)
    assert state.payload.raster_data_url is not None
    assert state.payload.raster_data_url.startswith("data:image/png;base64,")


def test_create_new_state_no_lineage_raster_data_url_is_none():
    adapter = _make_adapter(write_lineage=False)
    result = _make_result(raster_png=_make_png())
    state = adapter.create_new_state(result)
    assert state.payload.raster_data_url is None
