"""Fit lifetimes preserve packed geometry, replay derivatives and editor hooks."""

import os
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vectrify.refine import paths, soft_coverage
from vectrify.refine.filled_fit_optimize import FilledFitOptimizer
from vectrify.refine.filled_fit_raster import FilledFitRaster
from vectrify.refine.filled_fit_scene import (
    FilledFitHooks,
    FilledFitOptions,
    prepare_scene,
)

SOURCE = (
    '<svg width="96" height="64">'
    '<path id="back" d="M8.3 7.2 L40.3 7.2 L40.3 38.2 L8.3 38.2 Z" '
    'fill="#4080c0" fill-opacity=".65"/>'
    '<path id="hole" d="M24.2 12.3 L60.2 12.3 L60.2 44.3 L24.2 44.3 Z '
    'M32.2 20.3 L44.2 20.3 L44.2 30.3 L32.2 30.3 Z" '
    'fill="#c06040" fill-rule="evenodd" fill-opacity=".8"/>'
    '<path id="front" d="M38.1 17.4 L72.1 23.4 L56.1 52.4 Z" '
    'fill="#60c080" fill-opacity=".4"/>'
    "</svg>"
)


def _fit_state(*, device="cpu", learn_alpha=False, hooks=None, source=SOURCE):
    pytest.importorskip("torch")
    options = FilledFitOptions(
        xing_weight=0.02,
        subpixels=2,
        curve_samples=None,
        sparse_replay=False,
        max_point_displacement=None,
    )
    hooks = FilledFitHooks() if hooks is None else hooks
    target = Image.fromarray(np.arange(64 * 96 * 3, dtype=np.uint8).reshape(64, 96, 3))
    scene = prepare_scene(
        source,
        target,
        options=options,
        hooks=hooks,
        optimisation_long_side=None,
        monolithic=True,
        backdrop=Image.new("RGB", target.size, "#203040"),
        learn_alpha=learn_alpha,
        fit_context=None,
        device=device,
    )
    raster = FilledFitRaster(scene, options)
    optimizer = FilledFitOptimizer(
        scene,
        raster,
        options,
        hooks,
        point_learning_rate=0.1,
        color_learning_rate=0.1,
    )
    return scene, raster, optimizer


def test_filled_helpers_import_without_loading_optional_torch():
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class NoTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch" or fullname.startswith("torch."):
                    raise AssertionError("filled fit imported optional Torch eagerly")
                return None

        sys.meta_path.insert(0, NoTorch())
        from vectrify.refine import paths
        from vectrify.refine import filled_fit_scene
        from vectrify.refine import filled_fit_raster
        from vectrify.refine import filled_fit_optimize
        assert len(paths.parse_filled_cubics("M0 0 L4 0 L4 4 Z")[0]) == 3
        assert "torch" not in sys.modules
        """
    )
    environment = dict(os.environ, PYTHONPATH=str(Path(paths.__file__).parents[2]))
    subprocess.run(
        [sys.executable, "-c", code], env=environment, check=True, capture_output=True
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("learn_alpha", [False, True])
def test_extracted_backward_paths_preserve_painter_order_gradients(device, learn_alpha):
    torch = pytest.importorskip("torch")
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required")
        from vectrify.refine.cuda_renderer import available

        if not available():
            pytest.skip("optional CUDA renderer extension is not installed")
    gradients = {}
    for strategy in ("monolithic", "dense", "sparse"):
        scene, _raster, optimizer = _fit_state(device=device, learn_alpha=learn_alpha)
        controls = torch.cat([control for path in scene.controls for control in path])
        if strategy == "monolithic":
            optimizer.backward_monolithic(0, controls)
        elif strategy == "dense":
            optimizer.backward_dense(controls)
        else:
            optimizer.backward_sparse(controls)
        gradients[strategy] = (
            scene.control_storage.grad.detach().clone(),
            scene.color_storage.grad.detach().clone(),
            None if scene.alpha_values is None else scene.alpha_values.grad.clone(),
        )
    for strategy in ("dense", "sparse"):
        for actual, expected in zip(
            gradients[strategy][:2], gradients["monolithic"][:2], strict=True
        ):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    if learn_alpha:
        # Keep the established replay opacity derivative, including its
        # separate coverage contribution; this extraction changes no policy.
        torch.testing.assert_close(
            gradients["dense"][2], gradients["sparse"][2], atol=1e-6, rtol=1e-5
        )


def test_packed_contour_views_keep_original_lengths_and_storage():
    torch = pytest.importorskip("torch")
    scene, _raster, _optimizer = _fit_state()
    assert scene.contour_sizes == [4, 4, 4, 3]
    assert scene.path_storage_spans == [(0, 1), (1, 3), (3, 4)]
    with torch.no_grad():
        scene.controls[2][0][0, 1, 0] += 0.75
    assert scene.controls[2][0][0, 1, 0] == scene.control_storage[3, 0, 1, 0]
    scene.close_contours(FilledFitOptions(0.02, 2, None, False, None), FilledFitHooks())
    exported = ET.fromstring(scene.serialize(None))
    assert [element.get("id") for element in exported] == ["back", "hole", "front"]
    assert [
        [len(contour) for contour in paths.parse_filled_cubics(element.get("d", ""))]
        for element in exported
    ] == [[4], [4, 4], [3]]
    assert exported[1].get("fill-rule") == "evenodd"


def test_raster_caches_refresh_only_after_consuming_the_guard_band():
    torch = pytest.importorskip("torch")
    scene, raster, _optimizer = _fit_state()
    groups = raster.initial_simple_groups
    with torch.no_grad():
        scene.control_storage[..., 0] += 1
    raster.refresh_simple_tiles()
    assert raster.initial_simple_groups is groups
    with torch.no_grad():
        scene.control_storage[..., 0] += 1
    raster.refresh_simple_tiles()
    assert raster.initial_simple_groups is not groups
    assert raster.initial_simple_groups == raster.cropped_simple_groups()
    torch.testing.assert_close(raster.simple_tile_reference, scene.control_storage)


def test_raster_retains_paths_helper_monkeypatches_and_tile_gradients(monkeypatch):
    torch = pytest.importorskip("torch")
    scene, raster, _optimizer = _fit_state()
    # Exercise the batched fallback boundary on CPU without requiring a GPU.
    scene.soft = False
    calls = []

    def coverage(controls, box, **kwargs):
        calls.append((box, kwargs))
        return controls.mean(dim=(1, 2, 3))[:, None, None].expand(-1, box[3], box[2])

    monkeypatch.setattr(paths, "_fill_coverages", coverage)
    (_shape, rule, width, height), items = next(
        iter(raster.initial_simple_groups.items())
    )
    alphas = raster.rasterise_simple(rule, width, height, items)
    sum(alpha.sum() for _index, alpha in alphas).backward()
    assert calls[0][0] == (0, 0, width, height)
    assert calls[0][1]["dynamic_fuse"] is False
    assert scene.control_storage.grad.abs().sum() > 0
    assert all(alpha.shape == (64, 96) for _index, alpha in alphas)
    assert torch.isfinite(scene.control_storage.grad).all()


def test_portable_renderer_lookup_survives_separate_fit_calls(monkeypatch):
    scene, raster, _optimizer = _fit_state()
    calls = []
    original = soft_coverage.soft_coverage

    def coverage(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(soft_coverage, "soft_coverage", coverage)
    raster.soft_multi(1, scene.controls[1])
    assert len(calls) == 1


@pytest.mark.parametrize("transform", [False, True])
def test_projection_and_observation_order_survives_optimizer_extraction(transform):
    torch = pytest.importorskip("torch")
    events = []
    observed = []

    def observe(step, controls, _colours):
        events.append(("observe", step))
        observed.append([[contour.detach().clone() for contour in controls[0]]])
        return step < 1

    def project(controls):
        events.append(("project", None))
        # The first point is independent when a differentiable transform is
        # supplied; projecting it must precede closing the duplicate endpoint.
        controls[0][0][0, 0, 0] = 9.25

    def control_transform(controls):
        events.append(("transform", None))
        return [[control * 1 for control in path] for path in controls]

    def penalty(controls):
        events.append(("penalty", None))
        return 0.01 * torch.cat(controls[0]).square().mean()

    source = '<svg><path d="M8 8 L24 8 L24 24 L8 24 Z" fill="#4080c0"/></svg>'
    fitted = paths.fit_filled_svg(
        source,
        Image.new("RGB", (32, 32)),
        steps=5,
        device="cpu",
        monolithic=True,
        observe=observe,
        project_controls=project,
        control_transform=control_transform if transform else None,
        control_loss=penalty,
    )
    expected = [("observe", 0), ("penalty", None)]
    if transform:
        expected.append(("transform", None))
    expected.extend([("project", None), ("observe", 1), ("observe", 1)])
    assert events == expected
    assert len(observed) == 3
    final = paths.parse_filled_cubics(ET.fromstring(fitted)[0].get("d", ""))[0]
    assert final[0][0][0] == 9.2
    if transform:
        assert len(final) == 4
        assert final[-1][3] == final[0][0]


def test_control_transform_maps_gradients_back_to_packed_independent_parameters():
    torch = pytest.importorskip("torch")
    plain, _raster, direct = _fit_state()
    transformed, _raster, mapped = _fit_state(
        hooks=FilledFitHooks(
            project_controls=lambda _controls: None,
            control_transform=lambda controls: [
                [control * 2 for control in path] for path in controls
            ],
        )
    )
    for scene, optimizer in ((plain, direct), (transformed, mapped)):
        optimizer.backward_monolithic(
            0, torch.cat([control for path in scene.controls for control in path])
        )
    torch.testing.assert_close(
        transformed.control_storage.grad, 2 * plain.control_storage.grad
    )
    torch.testing.assert_close(transformed.color_storage.grad, plain.color_storage.grad)
    # The triangle's padded fourth cubic is storage only; it must not become
    # an independent coordinate in the transform's Jacobian.
    assert torch.count_nonzero(transformed.control_storage.grad[3, 3]) == 0


def test_rgba_context_passes_affine_response_and_target_through_loss_hook():
    torch = pytest.importorskip("torch")
    source = '<svg><path d="M2 2 L14 2 L14 10 L2 10 Z" fill="#804020"/></svg>'
    target = Image.new("RGBA", (16, 12), (60, 90, 120, 128))
    context = tuple(
        np.full((12, 16, 4), value, dtype=np.float32) for value in (0.2, 0.1, 0.4)
    )
    seen = []

    def coverage(_controls, alphas):
        seen.append("coverage")
        return alphas * 0 + 0.5

    def comparison(rendered, goal, step):
        seen.append(("loss", step))
        # A single layer paints half its RGB over the affine context base.
        colour = torch.tensor([128, 64, 32], dtype=torch.float32) / 255
        expected = torch.full((12, 16, 4), 0.2)
        expected[..., :3] += 0.05 + 0.2 * colour
        expected[..., 3] += 0.25
        torch.testing.assert_close(rendered, expected)
        alpha = 128 / 255
        white_rgb = np.asarray(target.convert("RGB"))[0, 0] * alpha + 255 * (1 - alpha)
        # on_white uses the image library's integer white-backed RGB rounding.
        torch.testing.assert_close(
            goal[0, 0, :3], torch.tensor(np.round(white_rgb) / 255, dtype=torch.float32)
        )
        assert float(goal[0, 0, 3]) == pytest.approx(alpha)
        return rendered, goal

    paths.fit_filled_svg(
        source,
        target,
        device="cpu",
        monolithic=True,
        fit_context=context,
        steps=1,
        point_learning_rate=0,
        color_learning_rate=0,
        coverage_transform=coverage,
        loss_transform=comparison,
    )
    assert seen == ["coverage", ("loss", 0)]
