import io
import sys
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
from PIL import Image

import vectrify.refine.paths as paths
import vectrify.refine.samvg as samvg
from vectrify.formats.svg.plugin import SvgPlugin
from vectrify.refine.paths import fit_svg_primitives_locally
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
    automatic_masks,
    coverage_prompt_points,
    detect_text,
    filter_by_impact,
    generate_svg,
    mask_path,
    mask_stroke,
    mask_strokes,
    recolour_visible_layers,
    residual_prompt_points,
)


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


def test_accepted_fit_uses_bounded_fill_coordinate_descent(monkeypatch):
    seen = {}

    def bounded(svg, image, *, rasterize, steps):
        seen["image"] = image.size
        seen["steps"] = steps
        assert rasterize is not None
        return svg

    monkeypatch.setattr(paths, "fit_filled_svg_bounded", bounded)
    image = Image.new("RGB", (16, 16), "white")
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" />'
    plugin = SvgPlugin()

    fitted, rendered = samvg._accepted_fit(
        svg, image, rasterize=plugin.rasterize, steps=7
    )

    assert fitted == svg
    assert rendered.size == image.size
    assert seen == {"image": (16, 16), "steps": 7}


def test_vectorize_svg_runs_a_second_residual_recovery_phase(monkeypatch):
    image = Image.new("RGB", (16, 16), "white")
    base = np.zeros((16, 16), dtype=bool)
    base[2:10, 2:10] = True
    added = np.zeros((16, 16), dtype=bool)
    added[10:14, 10:14] = True
    initial_layer = MaskLayer(base, (10, 20, 30), 1.0)
    added_layer = MaskLayer(added, (40, 50, 60), 1.0)
    calls = []
    monkeypatch.setattr(samvg, "_sam_runtime", lambda: object())
    monkeypatch.setattr(
        samvg, "retrieve_layers", lambda *_args, **_kwargs: [initial_layer]
    )
    monkeypatch.setattr(samvg, "residual_prompt_points", lambda *_args: [(12, 12)])
    monkeypatch.setattr(samvg, "prompted_masks", lambda *_args, **_kwargs: [added])
    monkeypatch.setattr(
        samvg,
        "filter_by_impact",
        lambda _image, _masks, **kwargs: [*kwargs["existing"], added_layer],
    )

    def accepted(svg, _image, *, rasterize, steps):
        assert rasterize is not None
        calls.append(
            (
                sum(
                    element.tag.split("}")[-1] == "path"
                    for element in ET.fromstring(svg).iter()
                ),
                steps,
            )
        )
        return svg, image

    monkeypatch.setattr(samvg, "_accepted_fit", accepted)
    result = samvg.vectorize_svg(image, rasterize=SvgPlugin().rasterize, steps=3)

    assert calls == [(1, 3), (2, 3)]
    root = ET.fromstring(result)
    paths = list(root.findall("{http://www.w3.org/2000/svg}path"))
    assert len(paths) == 2
    assert all(path.get("stroke") is None for path in paths)


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
    error = samvg._impact_error(target, canvas, coverage)
    expected = []
    for mask in sorted(
        [first, second, third], key=lambda item: int(item.sum()), reverse=True
    ):
        colour = tuple(int(value) for value in np.rint(target[mask].mean(axis=0)))
        next_canvas = canvas.copy()
        next_coverage = coverage | mask
        next_canvas[mask] = colour
        next_error = samvg._impact_error(target, next_canvas, next_coverage)
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
    svg = generate_svg(
        image, [mask], min_pixels=1, min_impact=0.00001, hybrid_strokes=True
    )
    path = ET.fromstring(svg).find("{http://www.w3.org/2000/svg}path")

    assert stroke is not None
    assert path is not None
    assert path.get("fill") == "none"
    assert path.get("stroke") == "#1482dc"
    assert path.get("stroke-linecap") == "round"
    assert " Z" not in path.get("d", "")


def test_curved_thin_mask_uses_a_multisegment_skeleton_stroke():
    mask = np.zeros((48, 48), dtype=bool)
    for x in range(6, 42):
        y = round(24 + 10 * np.sin((x - 6) / 35 * np.pi))
        mask[y - 1 : y + 2, x] = True

    stroke = mask_stroke(mask, segments=8)

    assert stroke is not None
    assert stroke[0].count("C ") >= 2


def test_thin_branch_mask_emits_independent_width_aware_strokes():
    mask = np.zeros((48, 48), dtype=bool)
    mask[6:42, 22:25] = True
    mask[6:9, 10:37] = True

    strokes = mask_strokes(mask, segments=8)

    assert len(strokes) >= 3
    assert all(data.startswith("M ") and width >= 1 for data, width in strokes)


def test_branched_stroke_seed_roundtrips_through_unified_local_fitter():
    mask = np.zeros((48, 48), dtype=bool)
    mask[6:42, 22:25] = True
    mask[6:9, 10:37] = True
    image = Image.new("RGB", (48, 48), "white")
    svg = generate_svg(image, [mask], min_pixels=1, min_impact=0, ocr=False)
    plugin = SvgPlugin()
    reference = plugin.rasterize(svg, 48, 48)

    fitted = fit_svg_primitives_locally(
        svg, reference, rasterize=plugin.rasterize, steps=1
    )

    assert fitted != svg
    assert fitted.count("stroke-width=") >= 3
    Image.open(io.BytesIO(plugin.rasterize(fitted, 48, 48))).verify()


def test_coverage_prompt_points_selects_the_centre_of_a_large_empty_region():
    occupied = np.zeros((32, 32), dtype=bool)
    occupied[:, :12] = True
    points = coverage_prompt_points(
        [MaskLayer(occupied, (10, 20, 30), 1.0)],
        (32, 32),
        radius_fraction=0.15,
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
