import io
import xml.etree.ElementTree as ET
from collections import Counter

from PIL import Image, ImageDraw

from vectrify.formats.svg.operations import apply_mutation
from vectrify.formats.svg.ownership import drawable_elements
from vectrify.formats.svg.plugin import SvgPlugin
from vectrify.formats.svg.targets import element_targets

NS = "http://www.w3.org/2000/svg"

# A right-hand square that matches the target and a left-hand one that does not.
DRAWING = (
    f'<svg xmlns="{NS}" viewBox="0 0 64 64">'
    '<rect x="0" y="0" width="64" height="64" fill="#ffffff"/>'
    '<rect x="4" y="24" width="16" height="16" fill="#ff0000"/>'
    '<rect x="44" y="24" width="16" height="16" fill="#000000"/>'
    "</svg>"
)


def _target_png() -> bytes:
    image = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(image).rectangle((44, 24, 60, 40), fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_targets_blame_the_element_that_does_not_match():
    targets = element_targets(DRAWING, _target_png(), size=64)

    # index 1 is the red square the target does not have; index 2 already matches.
    assert targets[1] > targets[2]


def test_targets_are_empty_for_unparseable_content():
    assert element_targets("<svg", _target_png(), size=64) == {}


def test_mutation_favours_the_element_answering_for_the_error():
    """Uniform choice spends a run in proportion to how many elements there are
    rather than to where the drawing is wrong."""
    targets = element_targets(DRAWING, _target_png(), size=64)

    def touched(weights) -> Counter:
        counts: Counter = Counter()
        for _ in range(300):
            mutated, _ = apply_mutation(DRAWING, "Mutation: color tweak", weights)
            before = drawable_elements(ET.fromstring(DRAWING))
            after = drawable_elements(ET.fromstring(mutated))
            for index, ((_, a), (_, b)) in enumerate(zip(before, after, strict=True)):
                if a.attrib != b.attrib:
                    counts[index] += 1
        return counts

    assert touched(targets)[1] > touched(None)[1]


def test_every_element_stays_reachable_under_targeting():
    """Attributed error says where the error is, not where a fix is available,
    so nothing may become unreachable."""
    targets = element_targets(DRAWING, _target_png(), size=64)
    targets = {**targets, 2: 0.0}

    seen = set()
    for _ in range(400):
        mutated, _ = apply_mutation(DRAWING, "Mutation: color tweak", targets)
        before = drawable_elements(ET.fromstring(DRAWING))
        after = drawable_elements(ET.fromstring(mutated))
        for index, ((_, a), (_, b)) in enumerate(zip(before, after, strict=True)):
            if a.attrib != b.attrib:
                seen.add(index)
    assert 2 in seen


def test_mutation_without_targets_is_unchanged():
    mutated, label = apply_mutation(DRAWING, "Mutation: color tweak")
    assert SvgPlugin().validate(mutated)[0]
    assert label == "Mutation: color tweak"


def test_an_element_too_small_to_own_a_pixel_is_still_reachable():
    """Error is attributed over the pixels an element owns at MASK_SIZE. Under
    about a pixel it owns none, scoring exactly zero -- and a zero weight means
    mutation can never pick it again. On one real drawing that locked out 28 of
    63 elements, including a nostril that stayed 14px from where it belonged for
    all 11,834 nodes of a run."""
    from vectrify.formats.svg.targets import element_targets

    content = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" '
        'width="100" height="100">'
        '<rect width="100" height="100" fill="#ffffff"/>'
        '<circle cx="50" cy="50" r="30" fill="#000000"/>'
        '<circle cx="10" cy="10" r="0.05" fill="#ff0000"/>'
        "</svg>"
    )
    targets = element_targets(content, _target_png())
    assert targets, "attribution produced nothing"
    assert min(targets.values()) > 0.0
    # And attribution still decides the bulk of it rather than going uniform.
    assert max(targets.values()) > min(targets.values()) * 5


def test_the_weights_still_sum_to_one():
    from vectrify.formats.svg.targets import element_targets

    content = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" '
        'width="100" height="100">'
        '<circle cx="30" cy="30" r="20" fill="#000000"/>'
        '<circle cx="70" cy="70" r="10" fill="#ff0000"/>'
        "</svg>"
    )
    targets = element_targets(content, _target_png())
    assert abs(sum(targets.values()) - 1.0) < 1e-9
