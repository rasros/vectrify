import io

from PIL import Image

from vectrify.formats.models import VectorResultPayload, VectorStatePayload
from vectrify.search.models import Result
from vectrify.vector.state import VectorStateBuilder


def _make_png(color: str = "red", size: int = 16) -> bytes:
    img = Image.new("RGB", (size, size), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_builder(write_lineage: bool = False) -> VectorStateBuilder:
    return VectorStateBuilder(
        resolution_llm=64,
        write_lineage=write_lineage,
    )


def _make_result(
    content: str = "<svg/>",
    raster_png: bytes | None = None,
    preview_data_url: str | None = None,
    heatmap_png: bytes | None = None,
) -> Result:
    return Result(
        task_id=1,
        parent_id=0,
        valid=True,
        measured=True,
        payload=VectorResultPayload(
            content=content,
            raster_png=raster_png,
            origin="test",
            raster_preview_data_url=preview_data_url,
            heatmap_png=heatmap_png,
        ),
    )


def create_payload(content_text: str | None) -> VectorStatePayload:
    return VectorStatePayload(
        content=content_text,
        raster_data_url=None,
        raster_preview_data_url=None,
        origin=None,
    )


def test_uses_precomputed_preview():
    build = _make_builder()
    precomputed = "data:image/png;base64,PRECOMPUTED"
    result = _make_result(preview_data_url=precomputed)
    state = build(result)
    assert state.payload.raster_preview_data_url == precomputed


def test_falls_back_to_computing_preview():
    build = _make_builder()
    png = _make_png()
    result = _make_result(raster_png=png, preview_data_url=None)
    state = build(result)
    assert state.payload.raster_preview_data_url is not None
    assert state.payload.raster_preview_data_url.startswith("data:image/png;base64,")


def test_no_png_no_preview():
    build = _make_builder()
    result = _make_result(raster_png=None, preview_data_url=None)
    state = build(result)
    assert state.payload.raster_preview_data_url is None


def test_precomputed_takes_priority_over_raster_png():
    build = _make_builder()
    precomputed = "data:image/png;base64,WINNER"
    result = _make_result(raster_png=_make_png(), preview_data_url=precomputed)
    state = build(result)
    assert state.payload.raster_preview_data_url == precomputed


def test_write_lineage_sets_raster_data_url():
    build = _make_builder(write_lineage=True)
    result = _make_result(raster_png=_make_png())
    state = build(result)
    assert state.payload.raster_data_url is not None
    assert state.payload.raster_data_url.startswith("data:image/png;base64,")


def test_no_lineage_raster_data_url_is_none():
    build = _make_builder(write_lineage=False)
    result = _make_result(raster_png=_make_png())
    state = build(result)
    assert state.payload.raster_data_url is None


def test_heatmap_data_url_set_when_png_present():
    build = _make_builder()
    result = _make_result(heatmap_png=_make_png("blue"))
    state = build(result)
    assert state.payload.heatmap_data_url is not None
    assert state.payload.heatmap_data_url.startswith("data:image/png;base64,")


def test_heatmap_data_url_none_when_no_png():
    build = _make_builder()
    result = _make_result(heatmap_png=None)
    state = build(result)
    assert state.payload.heatmap_data_url is None


def test_heatmap_independent_of_save_raster():
    build = VectorStateBuilder(
        resolution_llm=64,
        write_lineage=False,
        save_raster=False,
    )
    result = _make_result(heatmap_png=_make_png("green"))
    state = build(result)
    assert state.payload.heatmap_data_url is not None
