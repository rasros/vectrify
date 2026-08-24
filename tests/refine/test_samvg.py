import sys
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
from PIL import Image

import vectrify.refine.samvg as samvg
from vectrify.refine.samvg import (
    MaskLayer,
    TextLayer,
    _components,
    _fit_cubic,
    _is_crop_edge_mask,
    _text_svg_attributes,
    automatic_masks,
    coverage_prompt_points,
    detect_text,
    filter_by_impact,
    generate_svg,
    mask_path,
    mask_stroke,
    recolour_visible_layers,
    residual_prompt_points,
)


def test_detect_text_retains_high_confidence_editable_words(monkeypatch):
    class Reader:
        def __init__(self, languages, *, gpu, verbose):
            assert languages == ["en"]
            assert gpu is True
            assert verbose is False

        def readtext(self, source, **kwargs):
            assert source.shape == (16, 32, 3)
            assert kwargs == {"detail": 1, "paragraph": False}
            return [
                ([[2, 3], [20, 3], [20, 11], [2, 11]], "Cats & dogs", 0.94),
                ([[2, 12], [4, 12], [4, 14], [2, 14]], "I", 0.99),
                ([[2, 3], [20, 3], [20, 11], [2, 11]], "blur", 0.2),
            ]

    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=Reader))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )

    layers = detect_text(Image.new("RGB", (32, 16), "white"))

    assert layers == [TextLayer("Cats & dogs", 2.0, 3.0, 18.0, 8.0, (255, 255, 255))]
    assert _text_svg_attributes(layers[0])["font-family"] == "sans-serif"


def test_generate_svg_writes_detected_words_as_editable_text(monkeypatch):
    monkeypatch.setattr(samvg, "retrieve_layers", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        samvg,
        "detect_text",
        lambda _image: [TextLayer("Cats & dogs", 2, 3, 18, 8, (20, 30, 40))],
    )

    root = ET.fromstring(generate_svg(Image.new("RGB", (32, 16))))
    text = root.find("{http://www.w3.org/2000/svg}text")

    assert text is not None
    assert text.text == "Cats & dogs"
    assert text.get("font-size") == "8.00"


def test_automatic_masks_uses_source_sized_first_layer_crops(monkeypatch):
    calls = []

    class Generator:
        device = "cuda:0"

        def __call__(self, source, **_kwargs):
            calls.append(source.size)
            return {"masks": [Image.new("1", source.size, 1)]}

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(pipeline=lambda *_args, **_kwargs: Generator()),
    )

    masks = automatic_masks(Image.new("RGB", (12, 8)))

    assert calls[0] == (12, 8)
    assert sorted(calls[1:]) == [(7, 5)] * 4
    assert len(masks) == 5
    assert all(mask.shape == (8, 12) for mask in masks)


def test_filter_by_impact_keeps_useful_nested_masks_in_layer_order():
    image = Image.new("RGB", (12, 12), "white")
    pixels = np.asarray(image).copy()
    pixels[2:10, 2:10] = (220, 30, 30)
    pixels[5:7, 5:7] = (20, 40, 230)
    image = Image.fromarray(pixels)
    outer = np.zeros((12, 12), dtype=bool)
    outer[2:10, 2:10] = True
    centre = np.zeros((12, 12), dtype=bool)
    centre[5:7, 5:7] = True

    layers = filter_by_impact(image, [centre, outer], min_pixels=1, min_impact=0.00001)

    assert [int(layer.mask.sum()) for layer in layers] == [64, 4]
    assert layers[1].colour == (20, 40, 230)
    assert all(layer.impact > 0 for layer in layers)


def test_recolour_uses_only_each_layers_visible_pixels():
    image = Image.new("RGB", (8, 8), (220, 30, 30))
    pixels = np.asarray(image).copy()
    pixels[2:6, 2:6] = (20, 40, 230)
    image = Image.fromarray(pixels)
    outer = np.ones((8, 8), dtype=bool)
    inner = np.zeros((8, 8), dtype=bool)
    inner[2:6, 2:6] = True

    recoloured = recolour_visible_layers(
        image,
        [MaskLayer(outer, (0, 0, 0), 1.0), MaskLayer(inner, (0, 0, 0), 1.0)],
    )

    assert recoloured[0].colour == (220, 30, 30)
    assert recoloured[1].colour == (20, 40, 230)


def test_recolour_preserves_texture_overlap_metadata():
    image = Image.new("RGB", (4, 4), "white")
    layer = MaskLayer(np.ones((4, 4), dtype=bool), (0, 0, 0), 1.0, 1)

    recoloured = recolour_visible_layers(image, [layer])

    assert recoloured[0].overlap_pixels == 1


def test_components_are_separate_and_do_not_fill_meaningful_holes():
    mask = np.zeros((12, 12), dtype=bool)
    mask[1:6, 1:6] = True
    mask[2:5, 2:5] = False
    mask[8:11, 8:11] = True

    components = _components(mask, min_pixels=4)

    assert [int(component.sum()) for component in components] == [16, 9]
    assert not components[0][3, 3]


def test_components_fill_only_tiny_enclosed_holes():
    mask = np.ones((8, 8), dtype=bool)
    mask[3:5, 3:5] = False

    components = _components(mask, min_pixels=4)

    assert len(components) == 1
    assert components[0].all()


def test_crop_edge_masks_are_rejected_unless_they_reach_the_image_edge():
    cropped = np.ones((20, 30), dtype=bool)
    at_image_edge = np.zeros((20, 30), dtype=bool)
    at_image_edge[5:15, :10] = True

    assert _is_crop_edge_mask(cropped, (30, 20, 60, 40), (100, 80))
    assert not _is_crop_edge_mask(at_image_edge, (0, 0, 60, 40), (100, 80))


def test_mask_path_keeps_a_hole_as_a_second_even_odd_subpath():
    mask = np.ones((8, 8), dtype=bool)
    mask[2:6, 2:6] = False

    path = mask_path(mask)

    assert path is not None
    assert path.count("M ") == 2
    assert path.count(" Z") == 2


def test_generate_svg_creates_editable_layered_paths_from_supplied_masks():
    image = Image.new("RGB", (10, 8), "white")
    pixels = np.asarray(image).copy()
    pixels[1:7, 2:8] = (20, 130, 220)
    image = Image.fromarray(pixels)
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:7, 2:8] = True

    svg = generate_svg(image, [mask], min_pixels=1, min_impact=0.00001)
    root = ET.fromstring(svg)
    paths = list(root.findall("{http://www.w3.org/2000/svg}path"))

    assert root.get("viewBox") == "0 0 10 8"
    assert len(paths) == 1
    assert paths[0].get("fill") == "#1482dc"


def test_thin_single_contour_mask_is_emitted_as_a_round_stroke():
    image = Image.new("RGB", (12, 32), "white")
    pixels = np.asarray(image).copy()
    pixels[4:28, 5:8] = (20, 130, 220)
    image = Image.fromarray(pixels)
    mask = np.zeros((32, 12), dtype=bool)
    mask[4:28, 5:8] = True

    stroke = mask_stroke(mask)
    svg = generate_svg(image, [mask], min_pixels=1, min_impact=0.00001)
    path = ET.fromstring(svg).find("{http://www.w3.org/2000/svg}path")

    assert stroke is not None
    assert path is not None
    assert path.get("fill") == "none"
    assert path.get("stroke") == "#1482dc"
    assert path.get("stroke-linecap") == "round"
    assert " Z" not in path.get("d", "")


def test_coverage_prompt_points_selects_the_centre_of_a_large_empty_region():
    occupied = np.zeros((32, 32), dtype=bool)
    occupied[:, :12] = True
    points = coverage_prompt_points(
        [MaskLayer(occupied, (10, 20, 30), 1.0)],
        (32, 32),
        radius_fraction=0.15,
    )

    assert points
    assert all(x >= 16 for x, _y in points)


def test_residual_points_use_summed_rgb_difference_at_the_paper_threshold():
    target = Image.new("RGB", (32, 32), (255, 255, 255))
    rendered = Image.new("RGB", (32, 32), (0, 0, 0))

    points = residual_prompt_points(
        target, rendered, radius_fraction=0.15, threshold=0.784
    )

    assert points


def test_cubic_fit_reparameterises_nonuniform_curve_samples():
    start = np.array((0.0, 0.0))
    expected_a = np.array((8.0, 20.0))
    expected_b = np.array((22.0, -16.0))
    end = np.array((30.0, 4.0))
    parameters = np.linspace(0.0, 1.0, 25) ** 2
    inverse = 1 - parameters
    points = (
        inverse[:, None] ** 3 * start
        + 3 * inverse[:, None] ** 2 * parameters[:, None] * expected_a
        + 3 * inverse[:, None] * parameters[:, None] ** 2 * expected_b
        + parameters[:, None] ** 3 * end
    )

    uniform = _fit_cubic(points, reparameterize=False)
    refined = _fit_cubic(points)

    assert np.linalg.norm(np.hstack(refined) - np.hstack((expected_a, expected_b))) < (
        np.linalg.norm(np.hstack(uniform) - np.hstack((expected_a, expected_b)))
    )
