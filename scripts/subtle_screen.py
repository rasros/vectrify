#!/usr/bin/env python3
"""A screen of subtle path damage, with known severity order.

The distortion screen next to this one scores gross damage -- a deleted
element, a recoloured region, a blurred raster -- and that is what chose the
panel's members and, for a while, its 5x5 tiling. It cannot see the difference
between a beak fitted well and one fitted badly, so a panel could sit at chance
on stroke-level accuracy and still score 91% there.

That is not hypothetical: measured here, the panel of the day ordered subtle
path damage 50.7% of the time, which is a coin flip, while two of its three
members reported a beak displaced by 8px as BETTER than the undisplaced one.

Each family below damages one real drawing by a growing amount, so the true
order is known by construction and a scorer either reproduces it or does not.
Run both screens before changing the panel: a setup that wins here and loses
there has traded gross accuracy for fine and needs saying out loud.
"""

import io
import random
import re
from pathlib import Path

from PIL import Image

from vectrify.formats.svg.plugin import SvgPlugin

P = SvgPlugin()
_NUM = re.compile(r"-?(?:\d+\.\d+|\.\d+|\d+)")


def _group(svg, gid):
    return re.search(rf'<g[^>]*id="{gid}"[^>]*>(.*?)</g>', svg, re.S)


def _map_points(d, fn):
    """Apply *fn* to every coordinate pair in absolute path data."""
    nums = [float(x) for x in _NUM.findall(d)]
    pairs = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
    flat = [v for p in fn(pairs) for v in p]
    out, i = [], 0
    for token in re.split(r"(-?(?:\d+\.\d+|\.\d+|\d+))", d):
        if _NUM.fullmatch(token or "") and i < len(flat):
            out.append(f"{flat[i]:.2f}")
            i += 1
        else:
            out.append(token or "")
    return "".join(out)


def _edit_paths(svg, gid, fn):
    m = _group(svg, gid)
    out = svg
    for d in re.findall(r'd="([^"]+)"', m.group(1)):
        out = out.replace(f'd="{d}"', f'd="{_map_points(d, fn)}"', 1)
    return out


def _wobble(pairs, amp, seed):
    r = random.Random(seed)
    return [(x + r.uniform(-amp, amp), y + r.uniform(-amp, amp)) for x, y in pairs]


def families(svg):
    """name -> [(severity, svg)], severity 0 being intact."""
    out = {}
    out["beak shifted"] = [
        (
            s,
            _edit_paths(
                svg, "duck_beak_and_head", lambda k, s=s: [(x + s, y) for x, y in k]
            ),
        )
        for s in (0, 1, 2, 3, 5, 8)
    ]
    cx, cy = 183, 250
    out["beak scaled"] = [
        (
            s,
            _edit_paths(
                svg,
                "duck_beak_and_head",
                lambda k, f=1 + s / 100: [
                    ((x - cx) * f + cx, (y - cy) * f + cy) for x, y in k
                ],
            ),
        )
        for s in (0, 1, 2, 4, 7, 12)
    ]
    out["beak wobbled"] = [
        (s, _edit_paths(svg, "duck_beak_and_head", lambda k, s=s: _wobble(k, s, 7)))
        for s in (0, 0.5, 1, 2, 3, 5)
    ]
    out["body wobbled"] = [
        (s, _edit_paths(svg, "body_outline", lambda k, s=s: _wobble(k, s, 11)))
        for s in (0, 0.5, 1, 2, 3, 5)
    ]
    m = _group(svg, "duck_beak_and_head")
    base = float(re.search(r'stroke-width="([\d.]+)"', m.group(0)).group(1))
    out["beak thickened"] = [
        (
            s,
            svg.replace(
                m.group(0),
                m.group(0).replace(
                    f'stroke-width="{base}"', f'stroke-width="{base + s}"'
                ),
            ),
        )
        for s in (0, 0.2, 0.5, 1.0, 1.6, 2.5)
    ]
    return out


def render(svg, size=700):
    return Image.open(io.BytesIO(P.rasterize(svg, size, size))).convert("RGB")


def main() -> None:
    import itertools

    from vectrify.score.ensemble import EnsembleScorer

    target = Image.open("connect-the-dots-little-duck.png").convert("RGB")
    scorer = EnsembleScorer()
    reference = scorer.prepare_reference(target)
    families_ = families(Path("duck-v17.svg").read_text(encoding="utf-8"))

    right = total = 0
    print(f"{'family':<20}{'ordered correctly':>18}")
    for name, levels in families_.items():
        severities = [s for s, _ in levels]
        values = [
            scorer.score(reference, P.rasterize(svg, 700, 700)) for _, svg in levels
        ]
        hits = pairs = 0
        for i, j in itertools.combinations(range(len(severities)), 2):
            if severities[i] == severities[j]:
                continue
            pairs += 1
            worse = j if severities[j] > severities[i] else i
            better = i if worse == j else j
            hits += values[worse] > values[better]
        right += hits
        total += pairs
        print(f"{name:<20}{hits:>8}/{pairs:<4}{hits / pairs:>7.0%}")
    print(f"\n{'OVERALL':<20}{right:>8}/{total:<4}{right / total:>7.1%}")


if __name__ == "__main__":
    main()
