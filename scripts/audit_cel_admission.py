"""Locate current hard admission failures without changing generator policy.

    PYTHONPATH=src:. python scripts/audit_cel_admission.py --out .bench/admission
    PYTHONPATH=src:. python scripts/audit_cel_admission.py \
        --candidate legacy=.bench/planned-baseline/sword/cel/drawing.svg

Human geometry is diagnostic input only. The native policy and conservative
baseline are built from the source raster before any candidate is evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import find_objects, label

from scripts.bench_cel_planned import DATA, MANIFEST, load_case, source_hash
from vectrify.document import import_svg
from vectrify.document.transforms import object_matrix
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.export import export
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.model import Options, Work
from vectrify.refine.cel_plan.opacity import VISIBLE
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.score import composite, foreground_mask, render
from vectrify.refine.crossings import contour_line, crossing_pairs


@dataclass
class Finding:
    reason: str
    box: tuple[int, int, int, int]
    support: np.ndarray
    details: dict


def crossing_locations(svg, *, size=None):
    """Locate the hard crossing count in root space, or requested native pixels."""
    document = import_svg(svg)
    rows: list[dict] = []
    vx, vy, vw, vh = document.artboard()
    sx = sy = 1.0
    tx = ty = 0.0
    if size is not None:
        width, height = size
        sx, sy = width / vw, height / vh
        aspect = (document.root.get("preserveAspectRatio") or "xMidYMid meet").split()
        if aspect[0] == "defer":
            aspect = aspect[1:]
        if aspect[0] != "none":
            sx = sy = max(sx, sy) if "slice" in aspect else min(sx, sy)
            horizontal = {"xMin": 0, "xMid": 0.5, "xMax": 1}[aspect[0][:4]]
            vertical = {"YMin": 0, "YMid": 0.5, "YMax": 1}[aspect[0][4:]]
            tx, ty = (width - vw * sx) * horizontal, (height - vh * sy) * vertical
        tx, ty = tx - vx * sx, ty - vy * sy
    for element in document.elements():
        if element.tag not in {"path", "use"} or any(
            a.tag in {"defs", "clipPath", "mask"} for a in document.ancestry(element.id)
        ):
            continue
        asset = element
        while asset.tag == "use":
            asset = document.element((asset.get("href") or "")[1:])
        if asset.tag != "path":
            continue
        a, b, c, d, e, f = object_matrix(document, element.id)
        for contour in document.geometry_for(element.id).subpaths:
            controls = []
            start = np.asarray(contour.nodes[0].endpoint)
            for node in contour.nodes[1:]:
                points = np.asarray(node.values).reshape(-1, 2)
                controls.append(np.vstack([start, points]))
                start = points[-1]
            line, _ = contour_line(controls)
            if contour.closed and len(line) and np.any(line[-1] != line[0]):
                line = np.vstack([line, line[:1]])
            for i, j in crossing_pairs(line, contour.closed):
                p, q, r, s = line[i], line[i + 1], line[j], line[j + 1]
                delta, other = q - p, s - r
                denominator = delta[0] * other[1] - delta[1] * other[0]
                t = ((r - p)[0] * other[1] - (r - p)[1] * other[0]) / denominator
                x, y = p + t * delta
                root_x, root_y = a * x + c * y + e, b * x + d * y + f
                loop = np.vstack([p + t * delta, line[i + 1 : j + 1], p + t * delta])
                # Both loops use the same sampled geometry as the crossing proof.
                loops = [loop]
                if contour.closed:
                    loops.append(
                        np.vstack(
                            [p + t * delta, line[j + 1 :], line[: i + 1], p + t * delta]
                        )
                    )
                areas = [
                    abs(
                        float(
                            np.sum(
                                points[:-1, 0] * points[1:, 1]
                                - points[1:, 0] * points[:-1, 1]
                            )
                        )
                        / 2
                    )
                    * abs(a * d - b * c)
                    * sx
                    * sy
                    for points in loops
                ]
                rows.append(
                    {
                        "path": element.id,
                        "contour": contour.id,
                        "closed": contour.closed,
                        "xy": [float(root_x * sx + tx), float(root_y * sy + ty)],
                        "root_xy": [float(root_x), float(root_y)],
                        "sampled_loop_areas": areas,
                        "sampled_smaller_loop_area": min(areas),
                    }
                )
    return rows


def analyze(policy, svg, *, pixels=None):
    """Inventory every failed pixel/support rule, not just the first rejection."""
    actual = (
        render(svg, (policy.truth.shape[1], policy.truth.shape[0]))
        if pixels is None
        else pixels
    )
    evaluated = policy.evaluate(svg, pixels=actual)
    allowance = max(4, round(policy.area * 0.0005))
    alpha = policy.truth[..., 3]
    difference = actual[..., 3] - alpha
    findings = []
    residuals = {}
    checks = (
        (
            "interior_missing_pixels",
            "opaque-interior-gap",
            policy.inside & (actual[..., 3] < 0.5),
        ),
        (
            "outside_spill_pixels",
            "silhouette-spill",
            policy.outside & (actual[..., 3] > VISIBLE),
        ),
        (
            "opacity_missing_pixels",
            "translucent-interior-gap",
            policy.opacity_inside & (difference < -policy.opacity_tolerance),
        ),
        (
            "opacity_excess_pixels",
            "translucent-opacity-excess",
            policy.opacity_inside & (difference > policy.opacity_tolerance),
        ),
    )
    for term, reason, mask in checks:
        count = int(mask.sum())
        if count != evaluated.terms[term]:
            raise ValueError(f"Audit mask disagrees with policy term: {term}")
        ceiling = (policy.baseline.terms[term] if policy.baseline else 0) + allowance
        residuals[reason] = {"pixels": count, "ceiling": ceiling}
        if count <= ceiling:
            continue
        components, n = label(mask)
        for index, box in enumerate(find_objects(components, max_label=n), 1):
            if box is None:
                continue
            support = components[box] == index
            y, x = box
            findings.append(
                Finding(
                    reason,
                    (x.start, y.start, x.stop - x.start, y.stop - y.start),
                    support,
                    {"pixels": int(support.sum())},
                )
            )
    for index, (feature, ceiling) in enumerate(
        zip(policy.holes, policy.hole_ceilings, strict=True)
    ):
        x, y, w, h = feature.box
        predicted = actual[y : y + h, x : x + w, 3][feature.support]
        if float(predicted.mean()) > ceiling:
            findings.append(
                Finding(
                    "protected-hole-lost",
                    feature.box,
                    feature.support,
                    {
                        "index": index,
                        "mean_actual_alpha": float(predicted.mean()),
                        "ceiling": ceiling,
                    },
                )
            )
    lost_mass = 0.0
    for index, (feature, retained) in enumerate(
        zip(policy.opacity_components, policy.component_retention, strict=True)
    ):
        x, y, w, h = feature.box
        original = alpha[y : y + h, x : x + w][feature.support]
        predicted = actual[y : y + h, x : x + w, 3][feature.support]
        mass = float(original.sum())
        kept = float(np.minimum(predicted, original).sum())
        details = {
            "index": index,
            "pixels": len(original),
            "source_mass": mass,
            "source_peak": float(original.max()),
            "actual_mass": float(predicted.sum()),
            "retained_fraction": kept / mass,
            "required_fraction": retained,
        }
        if kept < mass * retained:
            lost_mass += mass - kept
            findings.append(
                Finding(
                    "translucent-component-lost", feature.box, feature.support, details
                )
            )
        if float(predicted.sum()) > mass * 1.25 + len(original) / 255:
            findings.append(
                Finding(
                    "translucent-component-opacity-excess",
                    feature.box,
                    feature.support,
                    details,
                )
            )
    locations = crossing_locations(svg, size=(alpha.shape[1], alpha.shape[0]))
    if len(locations) != evaluated.structure["self_crossings"]:
        raise ValueError("Audit crossing inventory disagrees with policy")
    crossing_limit = (
        policy.baseline.structure["self_crossings"] if policy.baseline else 0
    )
    if len(locations) > crossing_limit:
        height, width = alpha.shape
        for row in locations:
            x, y = row["xy"]
            # A coordinate outside the canvas remains reported, with a border crop.
            xx = min(width - 1, max(0, math.floor(x)))
            yy = min(height - 1, max(0, math.floor(y)))
            findings.append(
                Finding("new-self-crossing", (xx, yy, 1, 1), np.ones((1, 1), bool), row)
            )
    found = {f.reason for f in findings}
    # Policy reports only the first component failure (loss or strengthening).
    # Preserve both in this exhaustive inventory; all other reasons must agree.
    component_reasons = {
        "translucent-component-lost",
        "translucent-component-opacity-excess",
    }
    expected = set(evaluated.rejections)
    if found - component_reasons != expected - component_reasons or bool(
        found & component_reasons
    ) != bool(expected & component_reasons):
        raise ValueError("Audit reasons disagree with policy")
    report: dict = {
        "svg_sha256": hashlib.sha256(svg.encode()).hexdigest(),
        **evaluated.metrics(),
        "residuals": residuals,
        "crossings": locations,
        "lost_component_alpha_mass": lost_mass,
        "findings": [],
    }
    for finding in findings:
        x, y, w, h = finding.box
        box = np.s_[y : y + h, x : x + w]
        support = finding.support
        error = max(
            float(
                np.max(
                    np.abs(
                        composite(actual[box], bg) - composite(policy.truth[box], bg)
                    )[support]
                )
            )
            for bg in (0, 1)
        )
        report["findings"].append(
            {
                "reason": finding.reason,
                "box": finding.box,
                "max_backdrop_channel_difference_bytes": error * 255,
                **finding.details,
            }
        )
    return report, findings, actual


def save_crops(output, truth, actual, report, findings, *, limit=32):
    """Bound saved crops; all unsaved findings remain in the machine inventory."""
    if limit < 0:
        raise ValueError("Crop limit must be nonnegative")
    output.mkdir(parents=True, exist_ok=True)
    chosen = sorted(
        range(len(findings)),
        key=lambda i: (
            -report["findings"][i]["max_backdrop_channel_difference_bytes"],
            i,
        ),
    )[:limit]
    for index in chosen:
        finding = findings[index]
        x, y, w, h = finding.box
        left, top = max(0, x - 16), max(0, y - 16)
        right, bottom = min(truth.shape[1], x + w + 16), min(truth.shape[0], y + h + 16)
        box = np.s_[top:bottom, left:right]
        size = (right - left, bottom - top)
        # Large components are shown at native resolution; small defects are enlarged.
        scale = max(1, min(4, 640 // max(size)))
        panel = (size[0] * scale, size[1] * scale)
        sheet = Image.new("RGB", (5 * panel[0], panel[1] + 24), "white")
        draw = ImageDraw.Draw(sheet)
        for j, (image, background, title) in enumerate(
            (
                (truth, 1, "Source white"),
                (actual, 1, "Actual white"),
                (truth, 0, "Source black"),
                (actual, 0, "Actual black"),
                (truth, 1, "Failed support"),
            )
        ):
            pixels = np.rint(composite(image[box], background) * 255).astype(np.uint8)
            crop = Image.fromarray(pixels).resize(panel, Image.Resampling.NEAREST)
            if j == 4:
                marker = ImageDraw.Draw(crop)
                yy, xx = np.nonzero(finding.support)
                for cx, cy in zip(xx, yy, strict=True):
                    px, py = (x + int(cx) - left) * scale, (y + int(cy) - top) * scale
                    marker.rectangle(
                        (px, py, px + scale - 1, py + scale - 1), outline="#ff00ff"
                    )
            sheet.paste(crop, (j * panel[0], 24))
            draw.text((j * panel[0] + 2, 4), title, fill="black")
        name = f"{index:04d}-{finding.reason}.png"
        sheet.save(output / name)
        report["findings"][index]["crop"] = name
    report["saved_crops"] = len(chosen)
    report["unsaved_findings"] = len(findings) - len(chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="sword")
    parser.add_argument("--out", type=Path, default=Path(".bench/cel-admission"))
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=SVG")
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--crops", type=int, default=32)
    args = parser.parse_args()
    if args.crops < 0 or not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("Crops must be nonnegative and seconds finite and positive")
    named = {}
    for entry in args.candidate:
        name, separator, path = entry.partition("=")
        if (
            not separator
            or not name
            or Path(name).name != name
            or name in {".", "..", "human"}
            or name in named
        ):
            parser.error("Candidates require distinct safe names: NAME=SVG")
        named[name] = Path(path).read_text()
    cases = {c["name"]: c for c in json.loads(MANIFEST.read_text())["cases"]}
    if args.case not in cases:
        parser.error(f"Unknown case: {args.case}")
    case = cases[args.case]
    start_hash = source_hash()
    human, reference = load_case(DATA / case["file"])
    truth = np.asarray(reference, dtype=np.float32) / 255
    mask_hash = hashlib.sha256(foreground_mask(truth).tobytes()).hexdigest()
    if mask_hash != case["mask_sha256"]:
        raise ValueError("Fixture mask differs from frozen benchmark")
    options, work = Options(refine=False), Work.start(args.seconds)
    evidence = collect(reference, None, options, work)
    graph = build(evidence, work=work)
    policy = Policy.from_evidence(evidence, graph)
    fallback, _ = export(evidence, evidence.labels, options, work, conservative=True)
    baseline = policy.evaluate(fallback)
    if not baseline.valid:
        raise ValueError(
            f"Source-only conservative baseline is invalid: {baseline.rejections}"
        )
    policy.establish(baseline)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "baseline.svg").write_text(fallback)
    rows = {}
    for name, svg in {"human": human, **named}.items():
        report, findings, actual = analyze(policy, svg)
        save_crops(args.out / name, truth, actual, report, findings, limit=args.crops)
        rows[name] = report
        print(
            f"{name}: {len(findings)} findings; {report['validation_rejections']}",
            flush=True,
        )
    if source_hash() != start_hash:
        raise ValueError("Algorithm source changed during audit")
    report = {
        "source_sha256": start_hash,
        "mask_sha256": mask_hash,
        "reference_rgba_sha256": hashlib.sha256(truth.tobytes()).hexdigest(),
        "purpose": "diagnostic only; human geometry never supplied to generation",
        "options": {"complexity": 50, "quality": "balanced", "refine": False},
        "policy": policy.metadata(),
        "baseline": baseline.metrics(),
        "baseline_svg_sha256": hashlib.sha256(fallback.encode()).hexdigest(),
        "rows": rows,
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
