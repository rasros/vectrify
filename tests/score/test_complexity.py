import io
import random

from PIL import Image

from vectrify.score.complexity import detail, detail_excess


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


def test_no_charge_when_the_candidate_is_no_busier_than_the_target():
    png = _noisy()
    assert detail_excess(detail(png), png) == 0.0
    assert detail_excess(detail(png), _flat()) == 0.0


def test_charges_for_detail_the_target_does_not_have():
    reference = detail(_flat())
    assert detail_excess(reference, _noisy()) > 1.0


def test_a_noisy_target_does_not_make_a_clean_candidate_look_wrong():
    """The reason this is one-sided. Noise is the most incompressible thing an
    image can carry, and a generated or photographed target carries some: a
    symmetric distance would read a clean vector render as almost entirely
    wrong on this axis and leave adding speckle as the only way to improve it.
    """
    noisy_reference = detail(_noisy())
    clean = _flat()

    assert detail_excess(noisy_reference, clean) == 0.0
    # A symmetric reading would have charged nearly the whole reference.
    symmetric = abs(detail(clean) - noisy_reference) / noisy_reference
    assert symmetric > 0.9


def test_an_empty_candidate_wins_this_axis_and_loses_the_rest():
    """It scores 0 here, which needs no guard: it wins this one axis while
    losing colour, edge and shape, and one win against three losses is
    dominated whatever the arity."""
    from vectrify.search.nsga import _dominates

    empty = (1.0, 1.0, 1.0, 0.0)
    good = (0.2, 0.2, 0.2, 0.3)
    assert _dominates(good, empty)
    assert not _dominates(empty, good)


def test_survives_a_blank_reference():
    assert detail_excess(0.0, _noisy()) == 0.0


def test_detail_grows_with_pixel_count_at_equal_busyness():
    """Why the reference has to be measured at the size candidates render at.
    Compressed size counts bytes, so the same picture reads larger when there
    is more of it -- reading the reference at scoring resolution and candidates
    at render resolution charges every candidate for the gap between the two.
    """
    from vectrify.image_utils import resize_long_side

    rng = random.Random(5)
    big = Image.new("RGB", (256, 256))
    big.putdata(
        [
            (rng.randrange(256), rng.randrange(256), rng.randrange(256))
            for _ in range(256 * 256)
        ]
    )
    small = resize_long_side(big, 96)

    def as_png(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    assert detail(as_png(big)) > detail(as_png(small)) * 2
    # Which is exactly the false charge: the same image against itself at the
    # wrong scale reads as substantially busier than the reference.
    assert detail_excess(detail(as_png(small)), as_png(big)) > 1.0
