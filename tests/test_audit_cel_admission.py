"""Admission diagnosis inventories exact current policy failures and coordinates."""

import numpy as np
import pytest
from PIL import Image

from scripts.audit_cel_admission import analyze, crossing_locations, save_crops
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.score import render, svg_metrics


def svg(body, size=64):
    return f'<svg width="{size}" height="{size}">{body}</svg>'


@pytest.mark.parametrize("opacity", [1 / 255, 0.5, 1])
def test_native_component_inventory_includes_every_lost_faint_support(opacity):
    truth = np.zeros((64, 64, 4), np.float32)
    truth[10:30, 10:30] = (0.25, 0.5, 0.75, opacity)
    truth[42:44, 42:46] = (0, 0, 0, opacity)
    actual = truth.copy()
    actual[10:30, 10:30, 3] = 0
    actual[42:44, 42:46, 3] = 0
    policy = Policy(truth)
    baseline = policy.evaluate(svg(""), pixels=truth)
    policy.establish(baseline)
    report, findings, _ = analyze(policy, svg(""), pixels=actual)
    lost = [f for f in findings if f.reason == "translucent-component-lost"]
    assert [f.box for f in lost] == [(10, 10, 20, 20), (42, 42, 4, 2)]
    assert sum(f.details["pixels"] for f in lost) == 408
    assert report["lost_component_alpha_mass"] == pytest.approx(408 * opacity)
    assert report["validation_rejections"] == list(
        policy.evaluate(svg(""), pixels=actual).rejections
    )
    assert policy.baseline is baseline
    assert report["residuals"]["translucent-interior-gap"]["pixels"] == 256


def test_pixel_inventory_uses_native_thresholds_and_fixed_baseline_allowance():
    truth = np.zeros((64, 64, 4), np.float32)
    truth[8:56, 8:56] = 1
    actual = truth.copy()
    actual[20:24, 20:24, 3] = 0.6  # Opacity loss, but not an opaque gap.
    actual[2:5, 2:5] = (0, 0, 0, 1 / 255)  # Spill is visible below 0.05 alpha.
    report, findings, _ = analyze(Policy(truth), svg(""), pixels=actual)
    assert report["residuals"]["opaque-interior-gap"]["pixels"] == 0
    assert report["residuals"]["silhouette-spill"]["pixels"] == 9
    assert report["residuals"]["translucent-interior-gap"]["pixels"] == 16
    assert {f.reason for f in findings} == {
        "silhouette-spill",
        "translucent-interior-gap",
    }


def test_hole_inventory_uses_actual_source_ceiling_and_all_supports():
    truth = np.zeros((64, 64, 4), np.float32)
    truth[8:56, 8:56] = (0.5, 0, 0, 0.5)
    truth[20:24, 20:24, 3] = 0.02
    truth[40:44, 40:44, 3] = 0
    policy = Policy(truth)
    actual = truth.copy()
    actual[20:24, 20:24, 3] = 0.5
    actual[40:44, 40:44, 3] = 0.5
    _, findings, _ = analyze(policy, svg(""), pixels=actual)
    holes = [f for f in findings if f.reason == "protected-hole-lost"]
    assert len(holes) == 2
    assert holes[0].details["ceiling"] > holes[1].details["ceiling"]


def test_crossing_inventory_includes_use_instances_and_group_frames():
    drawing = svg(
        '<defs><path id="asset" d="M0 0L8 8L0 8L8 0Z"/></defs>'
        '<g transform="translate(10 20)"><use id="copy" href="#asset"/></g>'
        '<path id="flipped" transform="translate(50 40) scale(-2 2)" '
        'd="M0 0L8 8L0 8L8 0Z"/>'
    )
    locations = crossing_locations(drawing)
    assert (
        len(locations)
        == svg_metrics(drawing, include_crossings=True)["self_crossings"]
        == 2
    )
    assert {row["path"]: row["xy"] for row in locations} == {
        "copy": [14, 24],
        "flipped": [42, 48],
    }
    policy = Policy(render(drawing, (64, 64)))
    report, findings, _ = analyze(policy, drawing)
    assert report["validation_rejections"] == ["new-self-crossing"]
    assert {f.box for f in findings} == {(14, 24, 1, 1), (42, 48, 1, 1)}
    assert [row["sampled_smaller_loop_area"] for row in locations] == [16, 64]


def test_crossing_instances_include_xy_offsets_and_asset_transforms():
    drawing = svg(
        '<defs><path id="asset" transform="translate(2 1)" '
        'd="M0 0L8 8L0 8L8 0Z"/></defs>'
        '<g transform="translate(10 20)"><use href="#asset" x="3" y="5"/></g>'
    )
    assert crossing_locations(drawing)[0]["xy"] == [19, 30]


@pytest.mark.parametrize(
    ("aspect", "xy"),
    [
        ("xMidYMid meet", [40, 45]),
        ("none", [40, 40]),
        ("xMaxYMax meet", [40, 70]),
        ("xMidYMid slice", [30, 40]),
    ],
)
def test_crossing_native_coordinates_follow_viewbox_and_aspect_ratio(aspect, xy):
    drawing = (
        '<svg width="100" height="100" viewBox="10 20 100 50" '
        f'preserveAspectRatio="{aspect}">'
        '<path d="M30 20L70 60L30 60L70 20Z"/></svg>'
    )
    truth = render(drawing, (100, 100))
    report, findings, _ = analyze(Policy(truth), drawing)
    assert report["crossings"][0]["root_xy"] == [50, 40]
    assert report["crossings"][0]["xy"] == xy
    assert [f.box for f in findings] == [(*xy, 1, 1)]


def test_crops_keep_native_coordinates_and_report_unsaved_findings(tmp_path):
    truth = np.zeros((64, 64, 4), np.float32)
    truth[2:4, 2:6] = (0, 0, 0, 1 / 255)
    truth[60:62, 60:64] = (0, 0, 0, 0.5)
    report, findings, actual = analyze(Policy(truth), svg(""))
    save_crops(tmp_path, truth, actual, report, findings, limit=1)
    assert report["saved_crops"] == 1
    assert report["unsaved_findings"] == 1
    saved = [f for f in report["findings"] if "crop" in f]
    assert saved[0]["box"] == (60, 60, 4, 2)
    with Image.open(tmp_path / saved[0]["crop"]) as image:
        assert image.size == (400, 104)
        # Native candidate/source panels remain unmarked; only panel five is annotated.
        assert image.getpixel((64, 24 + 64)) == (128, 128, 128)
        assert image.getpixel((80 + 64, 24 + 64)) == (255, 255, 255)
        assert image.getpixel((320 + 64, 24 + 64)) == (255, 0, 255)
    with pytest.raises(ValueError, match="nonnegative"):
        save_crops(tmp_path, truth, actual, report, findings, limit=-1)
