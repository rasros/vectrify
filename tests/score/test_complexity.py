import io
import random

from PIL import Image

from vectrify.score.complexity import detail, detail_distance


def _flat(size: int = 96) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), "white").save(buf, format="PNG")
    return buf.getvalue()


def _noisy(size: int = 96) -> bytes:
    """Detail deflate cannot talk down. A regular pattern is genuinely low
    complexity -- a striped image compresses nearly as small as a blank one --
    so the busy case has to be incompressible to test what the measure claims.
    """
    rng = random.Random(11)
    img = Image.new("RGB", (size, size))
    img.putdata(
        [
            (rng.randrange(256), rng.randrange(256), rng.randrange(256))
            for _ in range(size * size)
        ]
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_detail_charges_for_busyness():
    assert detail(_noisy()) > detail(_flat()) * 10


def test_detail_distance_is_zero_at_the_reference():
    png = _noisy()
    assert detail_distance(detail(png), png) == 0.0


def test_detail_distance_charges_an_empty_drawing_too():
    """A pure minimum would make emptiness the best attainable value on this
    axis, and with an even objective count an empty candidate splits evenly
    against a good one and so cannot be dominated by it -- the regulariser
    would shelter the degenerate it exists to prevent.
    """
    reference = detail(_noisy())

    assert detail_distance(reference, _flat()) > 0.5
    assert detail_distance(reference, _noisy()) < detail_distance(reference, _flat())


def test_detail_distance_survives_a_blank_reference():
    assert detail_distance(0.0, _noisy()) == 0.0
