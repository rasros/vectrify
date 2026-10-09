"""Compare cel generators with human redraws on frozen native-resolution masks.

    uv run python scripts/bench_cel_planned.py --methods cel --out .bench/human
    uv run python scripts/bench_cel_planned.py --methods cel-planned --check

The reference remains the input; the human geometry is used only by the
benchmark scorer. Algorithm code never receives the human project.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from vectrify.document import (
    Editor,
    Selection,
    export_svg,
    import_svg,
    load_project,
)
from vectrify.operations import Budget, Job, OperationRequest, Permissions, method
from vectrify.project_file import decode_source
from vectrify.refine.cel_plan.score import (
    foreground_mask,
    measurements,
    render,
    svg_metrics,
)

DATA = Path(__file__).parent / "bench_data"
MANIFEST = DATA / "human.json"


def source_hash() -> str:
    """Identify dirty algorithm sources as well as committed benchmark runs."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    files = sorted((root / "src/vectrify").rglob("*.py"))
    files.extend(sorted((root / "scripts").glob("*cel*.py")))
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_case(path: Path):
    project = json.loads(decode_source(path.read_bytes()))
    document, _ = load_project(json.dumps(project["document"]))
    data = base64.b64decode(project["reference"]["data_url"].split(",", 1)[1])
    with Image.open(io.BytesIO(data)) as image:
        reference = image.convert("RGBA")
    return export_svg(document), reference


def generate(reference: Image.Image, name: str, settings: dict, seconds=None):
    width, height = reference.size
    editor = Editor(
        import_svg(f'<svg width="{width}" height="{height}"/>'),
        selection=Selection(whole_document=True),
    )
    request = OperationRequest(
        "generate",
        name,
        editor.snapshot,
        editor,
        Permissions(geometry=True, structure=True, paint=True),
        reference=reference,
        settings=settings,
        budget=Budget(seconds=seconds),
    )
    job = Job(method("generate", name), request)
    began = time.monotonic()
    job.run()
    state = job.state()
    if state["status"] != "ready":
        raise RuntimeError(f"{name}: {state.get('error', state['status'])}")
    job.apply()
    return (
        export_svg(editor.snapshot.document),
        state["result"]["metrics"],
        time.monotonic() - began,
    )


def save_render(path: Path, rgba: np.ndarray):
    Image.fromarray(np.clip(rgba * 255, 0, 255).round().astype(np.uint8)).save(path)


def compare(case: dict, name: str, settings: dict, output: Path, seconds=None):
    human_svg, reference = load_case(DATA / case["file"])
    truth = np.asarray(reference, dtype=np.float32) / 255
    human = render(human_svg, reference.size)
    mask = foreground_mask(truth)
    mask_hash = hashlib.sha256(mask.tobytes()).hexdigest()
    if mask_hash != case["mask_sha256"]:
        raise ValueError("Human fixture changed: review and version the scoring mask")
    svg, details, elapsed = generate(reference, name, settings, seconds)
    actual = render(svg, reference.size)
    row = {
        "case": case["name"],
        "method": name,
        "settings": settings,
        "seconds": elapsed,
        **svg_metrics(svg, include_crossings=True),
        "reference": measurements(actual, truth, mask, features=case["features"]),
        "human": measurements(actual, human, mask, features=case["features"]),
        "human_structure": svg_metrics(human_svg),
        "mask_sha256": mask_hash,
        "details": details,
        "targets": case["targets"],
    }
    row["passes"] = (
        row["nodes"] <= case["targets"]["nodes"]
        and row["contours"] <= case["targets"]["contours"]
        and row["human"]["mse"] <= case["targets"]["human_mse"]
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "drawing.svg").write_text(svg)
    (output / "human.svg").write_text(human_svg)
    reference.save(output / "reference.png")
    save_render(output / "drawing.png", actual)
    save_render(output / "human.png", human)
    Image.fromarray(mask.astype(np.uint8) * 255).save(output / "mask.png")
    for feature, (x, y, width, height) in case["features"].items():
        crops = [reference.crop((x, y, x + width, y + height))]
        for image in (human, actual):
            pixels = image[y : y + height, x : x + width]
            crops.append(Image.fromarray((pixels * 255).round().astype(np.uint8)))
        sheet = Image.new("RGBA", (3 * width, height), "white")
        for i, crop in enumerate(crops):
            sheet.alpha_composite(crop, (i * width, 0))
        sheet.convert("RGB").save(output / f"{feature.replace(' ', '-')}.png")
    (output / "metrics.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["cel"])
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--settings", default="{}", help="JSON method settings")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--out", type=Path, default=Path(".bench/cel-planned"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    cases = [
        case
        for case in manifest["cases"]
        if args.cases is None or case["name"] in args.cases
    ]
    if not cases:
        parser.error("No matching human cases")
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    rows = []
    for case in cases:
        for name in args.methods:
            row = compare(
                case,
                name,
                json.loads(args.settings),
                args.out / case["name"] / name,
                args.seconds,
            )
            row["revision"] = revision
            row["source_sha256"] = source_hash()
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    if args.check and not all(row["passes"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
