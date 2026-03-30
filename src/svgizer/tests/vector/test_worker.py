import base64
import io

from PIL import Image

from svgizer.image_utils import png_bytes_to_data_url, resize_long_side
from svgizer.vector.worker import _use_llm


def _make_png(color: str = "red", size: int = 32) -> bytes:
    img = Image.new("RGB", (size, size), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_use_llm_no_svg_uses_llm_when_rate_nonzero():
    for _ in range(20):
        assert _use_llm(has_content=False, llm_rate=1.0, llm_pressure=1.0) is True


def test_use_llm_rate_zero_never_calls():
    for _ in range(20):
        assert _use_llm(has_content=False, llm_rate=0.0, llm_pressure=0.0) is False
        assert _use_llm(has_content=True, llm_rate=0.0, llm_pressure=1.0) is False


def test_use_llm_rate_one_always_calls():
    for _ in range(20):
        assert _use_llm(has_content=True, llm_rate=1.0, llm_pressure=1.0) is True


def test_use_llm_intermediate_rate_is_probabilistic():
    results = [
        _use_llm(has_content=True, llm_rate=0.5, llm_pressure=1.0) for _ in range(200)
    ]
    assert any(results), "Expected some True values at rate=0.5"
    assert not all(results), "Expected some False values at rate=0.5"


def _compute_preview(png: bytes, long_side: int) -> str:
    full_img = Image.open(io.BytesIO(png)).convert("RGB")
    preview_img = resize_long_side(full_img, long_side)
    buf = io.BytesIO()
    preview_img.save(buf, format="PNG")
    return png_bytes_to_data_url(buf.getvalue())


def test_worker_preview_downscales_image():
    png = _make_png(size=256)
    preview = _compute_preview(png, long_side=64)

    _, b64 = preview.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert max(img.size) <= 64


def test_worker_preview_preserves_small_image():
    png = _make_png(size=32)
    preview = _compute_preview(png, long_side=128)

    _, b64 = preview.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert img.size == (32, 32)
