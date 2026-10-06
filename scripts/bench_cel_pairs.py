"""Compare CEL candidates with independent clean targets on the paired corpus.

    PYTHONPATH=src:. python scripts/bench_cel_pairs.py --out .bench/cel-pairs

Defaults to tuning artwork. --heldout is an explicit release-evaluation action.
The planned method is called at its planner boundary to observe all evaluated
SVGs; other methods use the normal operation/apply benchmark. Clean pixels,
geometry and evaluation features are never passed to any generator. This does
not replace operation-placement, human-redraw or independent review gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy
from PIL import Image
from PIL import __version__ as pillow_version

from scripts.bench_cel_planned import generate, save_render, source_hash
from scripts.bench_lines import score as line_score
from scripts.cel_pairs import DATA, cases, pair
from vectrify.operations.methods.cel_planned import SETTINGS
from vectrify.operations.settings import read_settings
from vectrify.refine.cel_plan.frontier import Observation
from vectrify.refine.cel_plan.model import Options
from vectrify.refine.cel_plan.pipeline import vectorize
from vectrify.refine.cel_plan.score import (
    foreground_mask,
    measurements,
    render,
    svg_metrics,
)

BENCH_VERSION = 2
MAX_POOL_BYTES = 64 * 1024 * 1024
MAX_POOL_ENTRIES = 256
DEGRADATIONS = ("clean", "blur", "resized", "jpeg", "noise", "alpha")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_hash(image: Image.Image) -> str:
    header = f"{image.mode}:{image.width}:{image.height}:".encode()
    return digest(header + image.tobytes())


def finite_json(value):
    """JSON null means unavailable (e.g. line F1 for an image with no ink)."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(finite_json(value), indent=2, allow_nan=False) + "\n")


def feature_boxes(case: dict, size: tuple[int, int]) -> dict:
    """Manifest rectangles are fractions of the clean canvas, for scoring only."""
    boxes = {}
    for name, values in case.get("features", {}).items():
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            raise ValueError("Evaluation features require four finite coordinates")
        x, y, width, height = values
        if min(x, y) < 0 or min(width, height) <= 0 or x + width > 1 or y + height > 1:
            raise ValueError("Evaluation features must lie within the clean canvas")
        left, top = math.floor(x * size[0]), math.floor(y * size[1])
        right = min(size[0], math.ceil((x + width) * size[0]))
        bottom = min(size[1], math.ceil((y + height) * size[1]))
        boxes[name] = (left, top, right - left, bottom - top)
    return boxes


class Pool:
    """A bounded diagnostic sink, separate from the production frontier."""

    def __init__(
        self, max_bytes: int = MAX_POOL_BYTES, max_entries: int = MAX_POOL_ENTRIES
    ):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.bytes = 0
        self.observations: list[Observation] = []
        self.omitted: list[dict] = []

    def observe(self, observation: Observation) -> None:
        size = len(observation.svg.encode())
        if (
            self.bytes + size > self.max_bytes
            or len(self.observations) >= self.max_entries
        ):
            self.omitted.append(
                {
                    "key": observation.key,
                    "label": observation.label,
                    "reason": "benchmark-pool-memory-limit",
                }
            )
            return
        self.bytes += size
        self.observations.append(observation)

    @property
    def complete(self) -> bool:
        return not self.omitted


def oracle(rows: list[dict], selected: dict, node_budget: int = 0) -> dict:
    """Best measured clean-target error in the observed, hard-valid pool.

    The cost-constrained oracle keeps the selected representation ceiling;
    the unconstrained oracle can reveal useful but more expensive proposals.
    Neither is a production score or a claim of human preference.
    """
    valid = [row for row in rows if row["hard_valid"] and "clean" in row]
    if node_budget and any(row["nodes"] <= node_budget for row in valid):
        valid = [row for row in valid if row["nodes"] <= node_budget]
    if not valid:
        return {"available": False}

    def best(choices):
        found = min(
            choices,
            key=lambda row: (
                row["clean"]["mse"],
                row["representation_cost"],
                row["key"],
            ),
        )
        return {
            "key": found["key"],
            "label": found["label"],
            "clean_mse": found["clean"]["mse"],
            "representation_cost": found["representation_cost"],
            "selected_mse_gap": selected["clean"]["mse"] - found["clean"]["mse"],
        }

    constrained = [
        row
        for row in valid
        if row["representation_cost"] <= selected["representation_cost"]
    ]
    return {
        "available": True,
        "metric": "clean foreground RGB MSE; diagnostic only",
        "best_in_pool": best(valid),
        "best_at_selected_cost": best(constrained) if constrained else None,
    }


def candidate_rows(
    pool: Pool,
    truth: np.ndarray,
    mask: np.ndarray,
    features: dict,
    output: Path,
) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for observation in pool.observations:
        evaluation = observation.evaluation
        row = {
            "key": observation.key,
            "label": observation.label,
            "decision": observation.decision,
            "hard_valid": bool(evaluation is not None and evaluation.valid),
            "exact_evaluated": evaluation is not None,
            "details": observation.details,
            "svg": f"candidates/{observation.key}.svg",
        }
        (output / f"{observation.key}.svg").write_text(observation.svg)
        if evaluation is not None:
            row.update(evaluation.metrics())
            # Hard-rejected, renderable candidates remain diagnostic examples;
            # oracle selection excludes them regardless of their pixel error.
            pixels = render(observation.svg, (truth.shape[1], truth.shape[0]))
            row["clean"] = measurements(pixels, truth, mask, features=features)
        write_json(output / f"{observation.key}.json", row)
        rows.append(row)
    return rows


def compare(
    case: dict,
    degradation: str,
    name: str,
    settings: dict,
    output: Path,
    *,
    long_side: int = 1000,
    seconds: float | None = None,
    composition_opacity: float = 1,
) -> dict:
    clean_svg, clean_image, reference = pair(
        case, degradation, long_side, composition_opacity
    )
    truth = np.asarray(clean_image, dtype=np.float32) / 255
    degraded = np.asarray(reference, dtype=np.float32) / 255
    mask = foreground_mask(truth)
    features = feature_boxes(case, clean_image.size)
    pool = Pool()
    output.mkdir(parents=True, exist_ok=True)
    (output / "clean.svg").write_text(clean_svg)
    clean_image.save(output / "clean.png")
    reference.save(output / "reference.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(output / "mask.png")
    row = {
        "case": case["name"],
        "family": case["family"],
        "split": case["set"],
        "degradation": degradation,
        "composition_opacity": composition_opacity,
        "target_variant": "original" if composition_opacity == 1 else "uniform-opacity",
        "ranker_training_eligible": case["set"] == "tuning"
        and degradation in {"clean", "blur", "jpeg", "noise"},
        "method": name,
        "settings": settings,
        "dimensions": list(clean_image.size),
        "long_side": long_side,
        "time_limit": seconds,
        "clean_svg_sha256": digest(clean_svg.encode()),
        "clean_pixels_sha256": image_hash(clean_image),
        "input_pixels_sha256": image_hash(reference),
        "mask_sha256": digest(mask.tobytes()),
        "evaluation_features": features,
        "execution": "planner" if name == "cel-planned" else "operation",
        "status": "ready",
    }
    began = time.monotonic()
    try:
        if name == "cel-planned":
            options = Options(**read_settings(settings, SETTINGS, "planned cel"))
            result = vectorize(
                reference, options=options, seconds=seconds, observe=pool.observe
            )
            svg, details = result.svg, result.metrics
            elapsed = time.monotonic() - began
        else:
            svg, details, elapsed = generate(reference, name, settings, seconds)
    except (ValueError, RuntimeError) as exc:
        elapsed = time.monotonic() - began
        row.update(status="failed", error=str(exc), seconds_generation=elapsed)
        # A rerun that fails must not leave a previous success looking current.
        for filename in ("drawing.svg", "drawing.png"):
            (output / filename).unlink(missing_ok=True)
        for feature in features:
            (output / f"feature-{feature}.png").unlink(missing_ok=True)
    else:
        pixels = render(svg, clean_image.size)
        row.update(
            seconds_generation=elapsed,
            selected_key=digest(svg.encode()),
            **svg_metrics(svg, include_crossings=True),
            clean=measurements(pixels, truth, mask, features=features),
            degraded=measurements(pixels, degraded, mask),
            # Degradation affects the input only. Ground-truth line geometry
            # remains clean even when the generator sees blur/JPEG/noise.
            lines=line_score(clean_svg, svg, *clean_image.size, "clean"),
            details=details,
        )
        (output / "drawing.svg").write_text(svg)
        save_render(output / "drawing.png", pixels)
        for feature, (x, y, width, height) in features.items():
            sheet = Image.new("RGBA", (3 * width, height), "white")
            drawing = Image.fromarray((pixels * 255).round().astype(np.uint8))
            for index, image in enumerate((clean_image, reference, drawing)):
                crop = image.crop((x, y, x + width, y + height))
                sheet.alpha_composite(crop, (index * width, 0))
            sheet.convert("RGB").save(output / f"feature-{feature}.png")
    evaluated = time.monotonic()
    candidates = candidate_rows(pool, truth, mask, features, output / "candidates")
    row.update(
        candidates=candidates,
        candidate_pool_complete=pool.complete,
        candidate_pool_bytes=pool.bytes,
        candidate_pool_omitted=pool.omitted,
        candidate_oracle=oracle(candidates, row, settings.get("node_budget", 0))
        if pool.complete and row["status"] == "ready" and name == "cel-planned"
        else {"available": False},
        seconds_candidate_evaluation=time.monotonic() - evaluated,
        seconds_evaluation=time.monotonic() - began - elapsed,
    )
    write_json(output / "metrics.json", row)
    return row


def environment() -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pillow": pillow_version,
        "threads": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "planned_device": "cpu",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("cel", "cel-planned", "colour-regions"),
        default=["cel", "cel-planned"],
    )
    parser.add_argument("--cases", nargs="+")
    parser.add_argument(
        "--degradations",
        nargs="+",
        choices=DEGRADATIONS,
        default=["clean", "blur", "jpeg", "noise"],
    )
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--long-side", type=int, default=1000)
    parser.add_argument("--seconds", type=float)
    parser.add_argument(
        "--composition-opacity",
        type=float,
        default=1,
        help="Apply opacity to the target before clean/input renders (0 < value <= 1)",
    )
    parser.add_argument(
        "--method-settings",
        default="{}",
        help="JSON mapping from method names to setting objects",
    )
    parser.add_argument("--out", type=Path, default=Path(".bench/cel-pairs"))
    args = parser.parse_args()
    if args.long_side < 32 or (
        args.seconds is not None
        and (args.seconds <= 0 or not math.isfinite(args.seconds))
    ):
        parser.error("Long side must be at least 32 and time limit must be positive")
    if (
        not math.isfinite(args.composition_opacity)
        or not 0 < args.composition_opacity <= 1
    ):
        parser.error("Composition opacity must be finite and between zero and one")
    selected = [
        case
        for case in cases(args.heldout)
        if args.cases is None or case["name"] in args.cases
    ]
    if not selected or (
        args.cases and set(args.cases) - {case["name"] for case in selected}
    ):
        parser.error("Requested cases must belong to the selected corpus split")
    try:
        settings = json.loads(args.method_settings)
    except json.JSONDecodeError as exc:
        parser.error(str(exc))
    if not isinstance(settings, dict) or any(
        not isinstance(value, dict) for value in settings.values()
    ):
        parser.error("Method settings must map method names to setting objects")
    if set(settings) - set(args.methods):
        parser.error("Settings supplied for a method that is not being evaluated")
    manifest_path = DATA / "planned_pairs.json"
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    report = {
        "benchmark_version": BENCH_VERSION,
        "revision": revision,
        "source_sha256": source_hash(),
        "manifest_sha256": digest(manifest_path.read_bytes()),
        "provenance": json.loads(manifest_path.read_text())["provenance"],
        "environment": environment(),
        "split": "heldout" if args.heldout else "tuning",
        "heldout_evaluation": args.heldout,
        "rows": [],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for case in selected:
        variant_name = (
            case["name"]
            if args.composition_opacity == 1
            else f"{case['name']}-opacity-{args.composition_opacity:g}"
        )
        for kind in args.degradations:
            for name in args.methods:
                row = compare(
                    case,
                    kind,
                    name,
                    settings.get(name, {}),
                    args.out / variant_name / kind / name,
                    long_side=args.long_side,
                    seconds=args.seconds,
                    composition_opacity=args.composition_opacity,
                )
                report["rows"].append(row)
                # Persist completed work after every case, including failures.
                write_json(args.out / "summary.json", report)
                print(
                    json.dumps(
                        finite_json(
                            {
                                "case": row["case"],
                                "degradation": kind,
                                "method": name,
                                "status": row["status"],
                                "seconds": row.get("seconds_generation"),
                                "nodes": row.get("nodes"),
                                "clean_mse": row.get("clean", {}).get("mse"),
                                "line_f": row.get("lines", {}).get("line_f"),
                                "oracle": row["candidate_oracle"],
                            }
                        ),
                        allow_nan=False,
                    ),
                    flush=True,
                )
    if any(row["status"] != "ready" for row in report["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
