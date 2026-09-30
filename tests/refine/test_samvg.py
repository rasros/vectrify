import sys
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
from PIL import Image

import vectrify.refine.samvg as samvg
from vectrify.refine.samvg import (
    MaskLayer,
    TextLayer,
    _binary_dilation,
    _components,
    _distance_transform_edt,
    _fit_cubic,
    _is_crop_edge_mask,
    _label,
    _text_svg_attributes,
    arrange_layers,
    automatic_masks,
    backdrop_colour,
    coverage_prompt_points,
    detect_text,
    filter_by_impact,
    generate_svg,
    mask_path,
    recolour_visible_layers,
    residual_prompt_points,
    thinner_than,
)
from vectrify.refine.samvg_runtime import device_name, pipeline_options


def test_sam_runtime_uses_cpu_pipeline_without_cuda():
    torch = SimpleNamespace(
        float16="fp16", cuda=SimpleNamespace(is_available=lambda: False)
    )

    assert pipeline_options(torch, "sam") == {"model": "sam", "device": -1}
    assert device_name(torch) == "cpu"


def test_sam_runtime_uses_half_precision_cuda_pipeline():
    torch = SimpleNamespace(
        float16="fp16", cuda=SimpleNamespace(is_available=lambda: True)
    )

    assert pipeline_options(torch, "sam") == {
        "model": "sam",
        "device": 0,
        "dtype": "fp16",
    }
    assert device_name(torch) == "cuda"


def test_sam_runtime_loads_transformers_on_cpu(monkeypatch):
    calls = []
    generator = SimpleNamespace(device="cpu")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float16="fp16", cuda=SimpleNamespace(is_available=lambda: False)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            pipeline=lambda name, **kwargs: calls.append((name, kwargs)) or generator
        ),
    )

    assert samvg._sam_runtime(model="example/sam").generator is generator
    assert calls == [("mask-generation", {"model": "example/sam", "device": -1})]


def test_detect_text_retains_high_confidence_editable_words(monkeypatch):
    class Inputs(dict):
        input_ids = SimpleNamespace(shape=(1, 4))

        def to(self, device):
            assert device == "cuda"
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"][0]["image"].size == (32, 16)
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "prompt"

        def __call__(self, **kwargs):
            assert kwargs["text"] == ["prompt"]
            assert kwargs["images"][0].size == (32, 16)
            return Inputs()

        def batch_decode(self, generated, **kwargs):
            assert generated.shape == (1, 1)
            assert kwargs == {
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": False,
            }
            return [
                '[{"text":"Cats & dogs","box":[2,3,20,11],"confidence":0.94},'
                '{"text":"I","box":[2,12,4,14],"confidence":0.99},'
                '{"text":"blur","box":[2,3,20,11],"confidence":0.2}]'
            ]

    class Model:
        def to(self, device):
            assert device == "cuda"
            return self

        def generate(self, **kwargs):
            assert kwargs == {"max_new_tokens": 768, "do_sample": False}
            return np.zeros((1, 5), dtype=int)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=lambda _model: Processor()),
            Qwen2_5_VLForConditionalGeneration=SimpleNamespace(
                from_pretrained=lambda _model, **_kwargs: Model()
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            bfloat16="bf16",
            float32="float32",
            cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None),
            inference_mode=nullcontext,
        ),
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


def test_generate_svg_keeps_only_pixel_improving_text(monkeypatch):
    target = Image.new("RGB", (32, 16), "black")
    monkeypatch.setattr(samvg, "retrieve_layers", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        samvg,
        "detect_text",
        lambda _image: [
            TextLayer("keep", 2, 3, 18, 8, (20, 30, 40)),
            TextLayer("discard", 2, 3, 18, 8, (20, 30, 40)),
        ],
    )
    monkeypatch.setattr(
        samvg,
        "_render_svg",
        lambda svg, _image, _rasterize: (
            Image.new("RGB", (32, 16), "white")
            if "discard" in svg
            else target
            if "keep" in svg
            else Image.new("RGB", (32, 16), "white")
        ),
    )

    root = ET.fromstring(generate_svg(target, rasterize=lambda *_args: b""))
    labels = [
        element.text for element in root.findall("{http://www.w3.org/2000/svg}text")
    ]

    assert labels == ["keep"]


def test_pixel_gate_allows_a_small_font_or_placement_mismatch(monkeypatch):
    target = Image.new("RGB", (32, 16), "black")
    monkeypatch.setattr(
        samvg,
        "_render_svg",
        lambda svg, _image, _rasterize: (
            Image.new("RGB", (32, 16), (2, 2, 2)) if "near" in svg else target
        ),
    )

    result = samvg._accept_text_layers(
        '<svg xmlns="http://www.w3.org/2000/svg" />',
        target,
        [TextLayer("near", 2, 3, 18, 8, (20, 30, 40))],
        lambda *_args: b"",
    )

    assert "near" in result


def test_automatic_masks_uses_source_sized_first_layer_crops(monkeypatch):
    calls = []

    class Generator:
        device = "cuda:0"

        def __call__(self, source, **kwargs):
            calls.append((source.size, kwargs))
            return {"masks": [Image.new("1", source.size, 1)]}

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(pipeline=lambda *_args, **_kwargs: Generator()),
    )

    masks = automatic_masks(Image.new("RGB", (12, 8)))

    assert calls[0][0] == (12, 8)
    assert sorted(size for size, _kwargs in calls[1:]) == [(7, 5)] * 4
    assert all(
        kwargs["points_per_batch"] == samvg.SAMVG_POINTS_PER_BATCH
        and kwargs["points_per_crop"] == 32
        and kwargs["crops_n_layers"] == 0
        for _size, kwargs in calls
    )
    assert len(masks) == 5
    assert all(mask.shape == (8, 12) for mask in masks)


def test_automatic_mask_finalization_suppresses_across_crop_sources():
    import torch

    masks = torch.tensor(
        [
            [[True, True], [True, True]],
            [[False, False], [False, True]],
        ]
    )
    # These candidates represent the same image-space crop box.  The second
    # candidate has a different raster but a higher SAM IoU, so AMG's one
    # image-global NMS must retain it instead of allowing each source to keep
    # its own duplicate.
    retained = samvg._finalize_automatic_masks(
        masks,
        torch.tensor([0.8, 0.9]),
        torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]]),
    )

    assert len(retained) == 1
    assert retained[0][1, 1]
    assert not retained[0][0, 0]


def test_retrieve_layers_reuses_one_runtime_for_automatic_and_coverage_prompts(
    monkeypatch,
):
    runtime = samvg._SamRuntime(generator=object())
    seen = {}
    monkeypatch.setattr(
        samvg,
        "automatic_masks",
        lambda _image, **kwargs: seen.setdefault("automatic", kwargs) or [],
    )
    monkeypatch.setattr(samvg, "filter_by_impact", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(samvg, "coverage_prompt_points", lambda *_args: [(2, 3)])
    monkeypatch.setattr(
        samvg,
        "prompted_masks",
        lambda _image, _points, **kwargs: seen.setdefault("prompted", kwargs) or [],
    )

    samvg.retrieve_layers(Image.new("RGB", (8, 8)), _runtime=runtime)

    assert seen["automatic"]["_runtime"] is runtime
    assert seen["prompted"]["_runtime"] is runtime


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


def test_filter_by_impact_scores_a_disconnected_mask_before_emitting_components():
    pixels = np.zeros((12, 12, 3), dtype=np.uint8)
    pixels[2:5, 2:5] = (220, 20, 20)
    pixels[7:10, 7:10] = (20, 20, 220)
    image = Image.fromarray(pixels)
    mask = np.zeros((12, 12), dtype=bool)
    mask[2:5, 2:5] = True
    mask[7:10, 7:10] = True

    layers = filter_by_impact(image, [mask], min_pixels=1, min_impact=0)

    assert [int(layer.mask.sum()) for layer in layers] == [9, 9]
    assert {layer.colour for layer in layers} == {(120, 20, 120)}
    assert layers[0].impact == layers[1].impact


def test_filter_by_impact_residual_canvas_does_not_charge_covered_pixels_as_blank():
    image = Image.new("RGB", (8, 8), (128, 128, 128))
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    fitted = np.full((8, 8, 3), 128, dtype=np.uint8)

    layers = filter_by_impact(
        image,
        [mask],
        initial_canvas=fitted,
        initial_coverage=np.ones((8, 8), dtype=bool),
        min_pixels=1,
        min_impact=1e-6,
    )

    assert layers == []


def test_incremental_impact_scoring_matches_full_canvas_recomputation():
    pixels = np.full((16, 16, 3), 255, dtype=np.uint8)
    pixels[2:12, 2:12] = (180, 60, 30)
    pixels[5:14, 5:14] = (30, 140, 220)
    image = Image.fromarray(pixels)
    first = np.zeros((16, 16), dtype=bool)
    first[2:12, 2:12] = True
    second = np.zeros((16, 16), dtype=bool)
    second[5:14, 5:14] = True
    third = np.zeros((16, 16), dtype=bool)
    third[7:9, 7:9] = True

    target = np.asarray(image, dtype=np.uint8)
    canvas = np.zeros_like(target)
    coverage = np.zeros(target.shape[:2], dtype=bool)
    error = samvg._impact_error_map(target, canvas, coverage).mean()
    expected = []
    for mask in sorted(
        [first, second, third], key=lambda item: int(item.sum()), reverse=True
    ):
        colour = tuple(int(value) for value in np.rint(target[mask].mean(axis=0)))
        next_canvas = canvas.copy()
        next_coverage = coverage | mask
        next_canvas[mask] = colour
        next_error = samvg._impact_error_map(target, next_canvas, next_coverage).mean()
        impact = error - next_error
        if impact >= 1e-5:
            expected.append((mask, colour, impact))
            canvas, coverage, error = next_canvas, next_coverage, next_error

    actual = filter_by_impact(
        image, [first, second, third], min_pixels=1, min_impact=1e-5
    )

    assert len(actual) == len(expected)
    for layer, (mask, colour, impact) in zip(actual, expected, strict=True):
        assert np.array_equal(layer.mask, mask)
        assert layer.colour == colour
        assert np.isclose(layer.impact, impact)


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


def test_bounded_component_hole_checks_match_full_canvas_semantics():
    def full_canvas(mask: np.ndarray, min_pixels: int) -> list[np.ndarray]:
        result = []
        for runs in samvg._run_components(mask):
            if sum(end - start for _y, start, end in runs) < min_pixels:
                continue
            component = np.zeros(mask.shape, dtype=bool)
            for y, start, end in runs:
                component[y, start:end] = True
            for hole in samvg._run_components(~component):
                area = sum(end - start for _y, start, end in hole)
                touches_border = any(
                    y in {0, mask.shape[0] - 1} or start == 0 or end == mask.shape[1]
                    for y, start, end in hole
                )
                if area <= min_pixels and not touches_border:
                    for y, start, end in hole:
                        component[y, start:end] = True
            result.append(component)
        return result

    mask = np.zeros((20, 24), dtype=bool)
    mask[1:12, 1:12] = True
    mask[4:6, 4:6] = False
    mask[7:10, 7:10] = False
    mask[3:5, 18:20] = True
    mask[14:17, 2:5] = True

    bounded = _components(mask, min_pixels=4)
    original = full_canvas(mask, min_pixels=4)

    assert len(bounded) == len(original)
    assert all(
        np.array_equal(left, right)
        for left, right in zip(bounded, original, strict=True)
    )


def test_internal_morphology_matches_scipy_default_connectivity():
    mask = np.array(
        [[False, True, True], [True, True, True], [True, True, True]], dtype=bool
    )

    labels, count = _label(np.array([[True, False], [False, True]], dtype=bool))
    distance = _distance_transform_edt(mask)
    dilated = _binary_dilation(np.array([[False, True, False]], dtype=bool), 1)

    assert count == 2
    assert labels.tolist() == [[1, 0], [0, 2]]
    assert np.allclose(distance, [[0, 1, 2], [1, 2**0.5, 5**0.5], [2, 5**0.5, 8**0.5]])
    assert dilated.tolist() == [[True, True, True]]


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


def test_smoothing_takes_the_steps_out_of_an_enlarged_raster_edge():
    import re

    # A diagonal edge made at a third of the size and enlarged, as a mask SAM
    # made at a lower resolution than the image is: steps three pixels wide.
    small = np.tril(np.ones((40, 40), dtype=bool), -1)
    small[:, 30:] = False
    mask = (
        np.asarray(
            Image.fromarray(small.astype(np.uint8) * 255).resize(
                (120, 120), Image.Resampling.NEAREST
            )
        )
        > 0
    )

    def spread(smooth):
        path = mask_path(mask, segments=64, smooth=smooth)
        assert path is not None
        values = np.array([float(v) for v in re.findall(r"-?\d+\.?\d*", path)])
        points = values.reshape(-1, 2)
        near = points[
            (points[:, 0] > 10)
            & (points[:, 0] < 80)
            & (points[:, 1] > 10)
            & (np.abs(points[:, 1] - points[:, 0]) < 8)
        ]
        return float(np.std(near[:, 1] - near[:, 0]))

    assert spread(3.0) < 0.5 * spread(0.0)


def test_mask_path_supports_the_variable_segment_tracing_variation():
    mask = np.zeros((48, 48), dtype=bool)
    mask[8:40, 8:40] = True
    mask[16:32, 16:32] = False

    path = mask_path(mask, curvature_threshold=0.8, maximum_segments=6)

    assert path is not None
    assert path.count("M ") == 2
    assert 6 <= path.count("C ") <= 12


def test_variable_corners_retains_nearby_local_extrema(monkeypatch):
    monkeypatch.setattr(
        samvg,
        "_curvature_scores",
        lambda _loop: np.array((1.0, -0.9, 1.0, -0.8, 1.0, -0.7, 1.0, -0.6)),
    )

    corners = samvg._variable_corners([(0.0, 0.0)] * 8, threshold=0, maximum=16)

    assert corners == [1, 3, 5, 7]


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


def test_generate_svg_refits_each_visible_fill_colour_after_mask_selection():
    pixels = np.full((12, 12, 3), (220, 30, 30), dtype=np.uint8)
    pixels[4:8, 4:8] = (20, 40, 230)
    image = Image.fromarray(pixels)
    outer = np.ones((12, 12), dtype=bool)
    inner = np.zeros((12, 12), dtype=bool)
    inner[4:8, 4:8] = True

    root = ET.fromstring(
        generate_svg(
            image,
            [outer, inner],
            min_pixels=1,
            min_impact=0,
            ocr=False,
        )
    )
    paths = list(root.findall("{http://www.w3.org/2000/svg}path"))

    assert [path.get("fill") for path in paths] == ["#dc1e1e", "#1428e6"]


def test_regions_thinner_than_the_minimum_width_are_left_out():
    image = Image.new("RGB", (32, 32), "white")
    pixels = np.asarray(image).copy()
    pixels[4:28, 5:7] = (20, 130, 220)
    pixels[4:28, 14:26] = (200, 30, 30)
    image = Image.fromarray(pixels)
    line = np.zeros((32, 32), dtype=bool)
    line[4:28, 5:7] = True
    block = np.zeros((32, 32), dtype=bool)
    block[4:28, 14:26] = True

    def fills(**options):
        svg = generate_svg(image, [line, block], min_pixels=1, min_impact=0, **options)
        return sorted(
            p.get("fill", "")
            for p in ET.fromstring(svg).iter("{http://www.w3.org/2000/svg}path")
        )

    assert fills(min_width=0) == ["#1482dc", "#c81e1e"]
    assert fills(min_width=3) == ["#c81e1e"]


def test_thinner_than_measures_the_widest_part():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:18, 4:6] = True
    assert thinner_than(mask, 3)
    mask[8:13, 8:13] = True
    assert not thinner_than(mask, 5)
    assert not thinner_than(mask, 0)


def test_coverage_prompt_points_selects_the_centre_of_a_large_empty_region():
    occupied = np.zeros((32, 32), dtype=bool)
    occupied[:, :12] = True
    points = coverage_prompt_points(
        [MaskLayer(occupied, (10, 20, 30), 1.0)],
        (32, 32),
        radius_fraction=0.15,
        max_points=10,
    )

    assert points
    assert points == [
        (27, 20),
        (27, 10),
        (25, 25),
        (25, 15),
        (25, 5),
        (20, 27),
        (20, 20),
        (20, 15),
        (20, 10),
        (20, 4),
    ]


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


def test_fixed_cubic_tracing_does_not_duplicate_each_curve_endpoint(monkeypatch):
    loop = [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (2.0, 2.0),
        (1.0, 2.0),
        (0.0, 2.0),
        (0.0, 1.0),
    ]
    samples = []
    monkeypatch.setattr(samvg, "_corners", lambda *_args: [0, 2, 4, 6])

    def fit(points):
        samples.append(points.copy())
        return points[0], points[-1]

    monkeypatch.setattr(samvg, "_fit_cubic", fit)

    samvg._cubic_loop(loop, segments=4)

    assert [len(points) for points in samples] == [3, 3, 3, 3]
    assert all(not np.array_equal(points[-1], points[-2]) for points in samples)


def test_sam_input_cap_restores_binary_masks_to_the_original_canvas():
    image = Image.new("RGB", (100, 50), "white")

    capped, scale = samvg._sam_image(image, 32)
    restored = samvg._restore_mask(
        np.ones((capped.height, capped.width), dtype=bool), image.size
    )

    assert capped.size == (32, 16)
    assert scale == 0.32
    assert restored.shape == (50, 100)
    assert restored.all()


def test_generate_svg_forwards_the_optional_sam_size_cap(monkeypatch):
    seen = {}

    def retrieve(_image, **kwargs):
        seen["max_side"] = kwargs["max_side"]
        return []

    monkeypatch.setattr(
        samvg,
        "retrieve_layers",
        retrieve,
    )

    generate_svg(Image.new("RGB", (80, 40)), max_side=32, ocr=False)

    assert seen == {"max_side": 32}


def test_generate_svg_defaults_to_sam_native_input_size(monkeypatch):
    seen = {}

    def retrieve(_image, **kwargs):
        seen["max_side"] = kwargs["max_side"]
        return []

    monkeypatch.setattr(samvg, "retrieve_layers", retrieve)

    generate_svg(Image.new("RGB", (80, 40)), ocr=False)

    assert seen == {"max_side": samvg.SAMVG_MAX_SIDE}


def _layer(mask, colour=(0, 0, 0)):
    return MaskLayer(mask, colour, 1.0)


def test_hidden_layers_are_dropped_and_flattening_removes_overlap():
    below = np.zeros((10, 10), dtype=bool)
    below[2:6, 2:6] = True
    hidden = np.zeros((10, 10), dtype=bool)
    hidden[3:5, 3:5] = True
    top = np.zeros((10, 10), dtype=bool)
    top[3:7, 3:7] = True
    beside = np.zeros((10, 10), dtype=bool)
    beside[0:10, 7:10] = True
    layers = [_layer(below), _layer(hidden), _layer(beside), _layer(top)]

    kept = arrange_layers(layers, drop_hidden=True)
    assert [id(layer.mask) for layer in kept] == [id(below), id(beside), id(top)]

    flat = arrange_layers(layers, flatten=True)
    assert np.all(np.sum([layer.mask for layer in flat], axis=0) <= 1)
    assert np.any(flat[0].mask)
    assert not np.any(flat[0].mask & top)


def test_flattening_gives_a_sliver_to_its_neighbour_without_a_gap():
    below = np.zeros((20, 20), dtype=bool)
    below[0:20, 0:20] = True
    top = np.zeros((20, 20), dtype=bool)
    # Covers all of the layer below but one row, which would be a sliver.
    top[1:20, 0:20] = True
    flat = arrange_layers(
        [_layer(below), _layer(top)], flatten=True, min_width=3, min_pixels=4
    )

    assert len(flat) == 1
    assert flat[0].mask.all()


def test_the_backdrop_takes_the_colour_of_what_no_layer_claims():
    pixels = np.full((8, 8, 3), 200, dtype=np.uint8)
    pixels[:, 4] = (30, 40, 50)
    image = Image.fromarray(pixels)
    left = np.zeros((8, 8), dtype=bool)
    left[:, :4] = True
    right = np.zeros((8, 8), dtype=bool)
    right[:, 5:] = True
    assert backdrop_colour(image, [_layer(left), _layer(right)]) == (30, 40, 50)

    svg = generate_svg(image, [left, right], min_pixels=1, min_impact=0, backdrop=True)
    first = next(iter(ET.fromstring(svg)))
    assert first.tag.endswith("rect")
    assert first.get("fill") == "#1e2832"
