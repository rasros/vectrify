"""Paired test: does the difference map make an LLM edit better?

The map is a third image in every refinement prompt, so it costs roughly 12% of
the input tokens. Whether it earns that has never been measured.

Full A/B runs cannot answer it: the same config has produced final scores of
0.0118, 0.0106 and 0.0092, so search stochasticity swamps a 12% effect and it
would take many runs per arm to see through. Instead each parent is its own
control -- the identical prompt is sent twice, once with the map and once
without -- which removes the dominant variance ("which parent did we start
from") and makes a signed-rank test on the pairs the right read.

Parents are stratified across the score range because the map plausibly helps
only at one end: when a candidate is far off there is error everywhere and the
map says little, and when it is nearly right there may be nothing bright left.

Usage: eval_diff_map.py RUN_DIR IMAGE --pairs N [--model M] [--out results.json]
"""

import argparse
import io
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

from PIL import Image

from vectrify.formats.svg.plugin import SvgPlugin
from vectrify.image_utils import pixel_diff_png, png_bytes_to_data_url
from vectrify.llm import LLMConfig
from vectrify.llm.models import DEFAULT_MODELS
from vectrify.llm.openai import OpenAIProvider
from vectrify.score.regions import worst_region_score
from vectrify.score.vision import VisionScorer


def load_parents(run_dir: Path, count: int, seed: int = 0) -> list[tuple[float, str]]:
    """Stratified sample across the score range, worst to best."""
    scored: list[tuple[float, str]] = []
    pattern = re.compile(r"^(inf|[0-9.]+)_(\d+)\.svg$")
    for f in (run_dir / "nodes").glob("*.svg"):
        m = pattern.match(f.name)
        if not m or m.group(1) == "inf":
            continue
        scored.append((float(m.group(1)), f.read_text()))
    scored.sort(key=lambda t: t[0])
    if not scored:
        raise SystemExit(f"no scored nodes in {run_dir}")

    rng = random.Random(seed)
    strata, per = 3, max(1, count // 3)
    out: list[tuple[float, str]] = []
    size = len(scored) // strata
    for i in range(strata):
        chunk = (
            scored[i * size : (i + 1) * size] if i < strata - 1 else scored[i * size :]
        )
        out.extend(rng.sample(chunk, min(per, len(chunk))))
    return out[:count]


def wilcoxon_signed_rank(deltas: list[float]) -> tuple[float, float]:
    """Return (W, two-sided p) via normal approximation with tie correction.

    Hand-rolled because scipy is not a dependency and pulling it in for one
    test is not worth it. Zeros are dropped, which is the standard treatment.
    """
    nonzero = [d for d in deltas if d != 0.0]
    n = len(nonzero)
    if n < 6:
        return 0.0, 1.0
    order = sorted(range(n), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nonzero[order[j + 1]]) == abs(nonzero[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nonzero, strict=True) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nonzero, strict=True) if d < 0)
    w = min(w_plus, w_minus)
    mean = n * (n + 1) / 4
    sd = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    if sd == 0:
        return w, 1.0
    z = (w - mean + 0.5) / sd
    # Two-sided normal tail
    p = 2 * 0.5 * (1 - _erf(abs(z) / (2**0.5)))
    return w, min(1.0, max(0.0, p))


def _erf(x: float) -> float:
    import math

    return math.erf(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("image", type=Path)
    ap.add_argument("--pairs", type=int, default=60)
    ap.add_argument("--model", default=DEFAULT_MODELS["openai"])
    ap.add_argument("--resolution-llm", type=int, default=512)
    ap.add_argument("--out", type=Path, default=Path("diff_map_eval.json"))
    args = ap.parse_args()

    plugin = SvgPlugin()
    original = Image.open(args.image).convert("RGB")
    w, h = original.size
    buf = io.BytesIO()
    original.save(buf, format="PNG")
    original_png = buf.getvalue()
    target_url = png_bytes_to_data_url(_downscale(original_png, args.resolution_llm))

    scorer = VisionScorer()
    scorer._load_dependencies()
    ref = scorer.prepare_reference(original)

    provider = OpenAIProvider()
    config = LLMConfig(model=args.model, reasoning="medium")

    parents = load_parents(args.run_dir, args.pairs)
    print(f"parents: {len(parents)} (stratified across the score range)")

    rows = []
    for idx, (parent_score, parent_svg) in enumerate(parents, 1):
        parent_png = plugin.rasterize(parent_svg, out_w=w, out_h=h)
        preview_url = png_bytes_to_data_url(_downscale(parent_png, args.resolution_llm))
        diff_url = png_bytes_to_data_url(
            pixel_diff_png(original, parent_png, args.resolution_llm)
        )

        row: dict = {"parent_score": parent_score, "index": idx}
        for arm, dmap in (("with_map", diff_url), ("without_map", None)):
            try:
                blocks = plugin.build_generate_prompt(
                    target_url,
                    idx,
                    content_prev=parent_svg,
                    raster_preview_url=preview_url,
                    goal=None,
                    canvas=(w, h),
                )
                if dmap is not None:
                    # Spliced in here rather than by the prompt builder: the
                    # map lost its measurement and was removed from the
                    # product, so reproducing the comparison means the harness
                    # adding the third image itself.
                    blocks = [
                        *blocks,
                        {"type": "input_text", "text": "Difference Map:"},
                        {"type": "input_image", "image_url": dmap},
                    ]
                raw = provider.generate(blocks, config)
                usage = dict(provider.last_usage)
                child = plugin.apply_edit(parent_svg, raw)
                valid, _ = plugin.validate(child)
                if not valid:
                    row[arm] = {"valid": False, "usage": usage}
                    continue
                child_png = plugin.rasterize(child, out_w=w, out_h=h)
                score = scorer.score(ref, child_png)
                grid = scorer.region_distance_grid(ref, child_png)
                row[arm] = {
                    "valid": True,
                    "score": score,
                    "delta": parent_score - score,
                    "worst_region": worst_region_score(grid)
                    if grid is not None
                    else None,
                    "usage": usage,
                    "changed": child.strip() != parent_svg.strip(),
                }
            except Exception as exc:
                row[arm] = {"valid": False, "error": str(exc)[:200]}
            time.sleep(0.2)
        rows.append(row)
        a, b = row.get("with_map", {}), row.get("without_map", {})
        print(
            f"  {idx:3d}/{len(parents)} parent={parent_score:.6f} "
            f"with={a.get('delta', float('nan')):+.6f} "
            f"without={b.get('delta', float('nan')):+.6f}",
            flush=True,
        )

    args.out.write_text(json.dumps(rows, indent=2))
    report(rows)
    return 0


def _downscale(png: bytes, long_side: int) -> bytes:
    from vectrify.image_utils import downscale_png_bytes

    return downscale_png_bytes(png, long_side)


def report(rows: list[dict]) -> None:
    both = [
        r
        for r in rows
        if r.get("with_map", {}).get("valid") and r.get("without_map", {}).get("valid")
    ]
    print(f"\npairs where both arms produced valid SVG: {len(both)}/{len(rows)}")

    for arm in ("with_map", "without_map"):
        valid = sum(1 for r in rows if r.get(arm, {}).get("valid"))
        changed = sum(1 for r in rows if r.get(arm, {}).get("changed"))
        toks = [
            r[arm]["usage"].get("prompt_tokens", 0)
            for r in rows
            if r.get(arm, {}).get("usage")
        ]
        print(
            f"  {arm:12s} valid {valid}/{len(rows)}  changed {changed}  "
            f"median input tokens {statistics.median(toks) if toks else 0:.0f}"
        )

    if len(both) < 6:
        print("\ntoo few valid pairs for a signed-rank test")
        return

    with_d = [r["with_map"]["delta"] for r in both]
    without_d = [r["without_map"]["delta"] for r in both]
    paired = [a - b for a, b in zip(with_d, without_d, strict=True)]

    print(f"\nmedian improvement with map:    {statistics.median(with_d):+.6f}")
    print(f"median improvement without map: {statistics.median(without_d):+.6f}")
    print(f"median paired difference:       {statistics.median(paired):+.6f}")
    print(f"map better in {sum(1 for d in paired if d > 0)}/{len(paired)} pairs")

    w, p = wilcoxon_signed_rank(paired)
    print(f"Wilcoxon signed-rank: W={w:.1f}, p={p:.4f}")

    mw = statistics.median(with_d)
    mo = statistics.median(without_d)
    if mo != 0:
        print(
            f"relative change in median improvement: {100 * (mw - mo) / abs(mo):+.1f}%"
        )
    print("\nDecision rule (pre-registered): keep the map only if it improves")
    print("median delta by >=10% with p < 0.05.")


if __name__ == "__main__":
    sys.exit(main())
