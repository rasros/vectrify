import io

from PIL import Image, ImageDraw

from vectrify.score.edges import edge_map, edge_score


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _square(fill: str, background: str = "white", box=(8, 8, 24, 24)) -> Image.Image:
    image = Image.new("RGB", (32, 32), background)
    ImageDraw.Draw(image).rectangle(box, fill=fill)
    return image


def test_identical_images_have_no_edge_distance():
    image = _square("black")
    assert edge_score(image, _png(image)) == 0.0


def test_edge_distance_ignores_a_shape_being_the_wrong_colour():
    """The point of the metric: pixel distance is dominated by area, so a
    recoloured shape swamps a missing one. Structure has to be judged on its
    own."""
    reference = _square("black")
    recoloured = edge_score(reference, _png(_square("red")))
    missing = edge_score(reference, _png(Image.new("RGB", (32, 32), "white")))
    assert recoloured < missing


def test_a_misplaced_shape_costs_more_than_a_correctly_placed_one():
    """Only within about one edge width. Past that the two edge maps stop
    overlapping at all and the distance saturates, so this says whether the
    structure lines up, not how far away it is."""
    reference = _square("black")
    aligned = edge_score(reference, _png(_square("black")))
    shifted = edge_score(reference, _png(_square("black", box=(10, 10, 26, 26))))
    assert aligned < shifted


def test_edge_map_is_flat_on_a_flat_image():
    assert edge_map(Image.new("RGB", (16, 16), "grey")).max() == 0.0


def test_edge_score_resizes_a_mismatched_candidate():
    """Resampling softens the boundaries, so the same drawing at twice the size
    is not free -- but it must still beat a candidate drawn wrong."""
    reference = _square("black")
    rescaled = edge_score(reference, _png(_square("black").resize((64, 64))))
    misplaced = edge_score(reference, _png(_square("black", box=(10, 10, 26, 26))))
    assert rescaled < misplaced


def test_edge_score_of_an_inverted_image_is_small():
    """Inversion keeps every boundary in place, so structure is unchanged even
    though every pixel differs."""
    reference = _square("black")
    inverted = _square("white", background="black")
    assert edge_score(reference, _png(inverted)) < 0.05
