"""Regenerate the benchmark corpus.

Targets are drawn here as raster at 4x and downsampled, so they carry
anti-aliased edges and gradients that no shipped SVG encodes. The seeds each
case starts from live in bench/seeds.py.

Run from the repo root: uv run python bench/generate.py
"""

import itertools
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

SIDE = 384
SS = 4
S = SIDE * SS

BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FAMILY = "DejaVu Sans"

HEAD = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIDE} {SIDE}">'


def _font(path: str, size: float) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, round(size * SS))
    except OSError as exc:
        raise SystemExit(f"font not found: {path} ({exc})") from exc


def _canvas(color: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (S, S), color)
    return img, ImageDraw.Draw(img)


def _finish(img: Image.Image) -> Image.Image:
    return img.resize((SIDE, SIDE), Image.LANCZOS)


def _centred(draw, text, font, cx, cy, fill) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (cx * SS - (right - left) / 2, cy * SS - (bottom - top) / 2 - top),
        text,
        font=font,
        fill=fill,
    )


def _box(cx: float, cy: float, rx: float, ry: float) -> list[float]:
    return [(cx - rx) * SS, (cy - ry) * SS, (cx + rx) * SS, (cy + ry) * SS]


def _text(x, y, size, fill, body, weight="normal") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="{FAMILY}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="middle" fill="{fill}">{body}</text>'
    )


MASCOT_PARTS = [
    (124, 88, 28, 28),
    (260, 88, 28, 28),
    (192, 277, 42, 45),
    (192, 168, 88, 84),
]
MASCOT_EYES = [(155, 163, 19, 23), (229, 163, 19, 23)]
MASCOT_PUPILS = [(156, 165, 9, 11), (230, 165, 9, 11)]
MASCOT_GLINTS = [(154, 160, 3, 3), (228, 160, 3, 3)]
MASCOT_CHEEKS = [(132, 206, 14, 10), (252, 206, 14, 10)]


def mascot_target() -> Image.Image:
    body, line = "#3fb6a8", "#12403c"
    img, d = _canvas("#eef6fb")
    for part in MASCOT_PARTS:
        d.ellipse(_box(*part), fill=body, outline=line, width=5 * SS)
    for eye in MASCOT_EYES:
        d.ellipse(_box(*eye), fill="#ffffff", outline=line, width=5 * SS)
    for pupil in MASCOT_PUPILS:
        d.ellipse(_box(*pupil), fill=line)
    for glint in MASCOT_GLINTS:
        d.ellipse(_box(*glint), fill="#ffffff")
    for cheek in MASCOT_CHEEKS:
        d.ellipse(_box(*cheek), fill="#f58fa4")
    d.arc(_box(192, 202, 30, 24), 20, 160, fill=line, width=5 * SS)
    d.polygon(
        [(192 * SS, 194 * SS), (181 * SS, 208 * SS), (203 * SS, 208 * SS)], fill=line
    )
    return _finish(img)


def mascot_seed() -> str:
    body, line = "#59c2b0", "#26514c"
    out = [
        HEAD,
        f'<rect width="{SIDE}" height="{SIDE}" fill="#e6f1f7"/>',
        '<g id="parts">',
    ]
    for cx, cy, rx, ry in MASCOT_PARTS:
        out.append(
            f'<ellipse cx="{cx - 4}" cy="{cy + 5}" rx="{round(rx * 1.1)}" '
            f'ry="{round(ry * 0.92)}" fill="{body}" stroke="{line}" stroke-width="3"/>'
        )
    out.append('</g><g id="eyes">')
    for cx, cy, rx, ry in MASCOT_EYES:
        out.append(
            f'<ellipse cx="{cx + 3}" cy="{cy - 4}" rx="{round(rx * 0.85)}" '
            f'ry="{round(ry * 1.12)}" fill="#f2f2f2" stroke="{line}" stroke-width="2"/>'
        )
    for cx, cy, rx, ry in MASCOT_PUPILS:
        out.append(
            f'<ellipse cx="{cx + 2}" cy="{cy - 3}" rx="{round(rx * 1.3)}" '
            f'ry="{round(ry * 0.8)}" fill="#1d3c4a"/>'
        )
    for cx, cy, rx, _ry in MASCOT_GLINTS:
        out.append(f'<circle cx="{cx - 2}" cy="{cy + 2}" r="{rx + 2}" fill="#e8e8e8"/>')
    out.append('</g><g id="face">')
    for cx, cy, rx, ry in MASCOT_CHEEKS:
        out.append(
            f'<ellipse cx="{cx + 6}" cy="{cy - 6}" rx="{round(rx * 0.7)}" '
            f'ry="{round(ry * 1.4)}" fill="#e2a5ad"/>'
        )
    out.append(
        '<path d="M166 198 Q192 224 218 198" fill="none" stroke="#26514c" '
        'stroke-width="6"/>'
        '<path d="M192 190 L176 212 L208 212 Z" fill="#2c5a54"/></g></svg>'
    )
    return "".join(out)


def wordmark_target() -> Image.Image:
    img, d = _canvas("#ffffff")
    d.rounded_rectangle(_box(141, 119, 45, 45), radius=22 * SS, fill="#f07c2b")
    d.rounded_rectangle(_box(191, 163, 45, 45), radius=22 * SS, fill="#1f4e9c")
    d.ellipse(_box(191, 159, 21, 21), fill="#ffffff")
    _centred(d, "NOVA", _font(BOLD, 46), 192, 262, "#12233f")
    _centred(d, "SYSTEMS", _font(REGULAR, 17), 192, 302, "#5c6b80")
    return _finish(img)


def wordmark_seed() -> str:
    return (
        f'{HEAD}<rect width="{SIDE}" height="{SIDE}" fill="#fbfbfb"/>'
        '<g id="mark">'
        '<rect x="100" y="80" width="84" height="84" rx="14" fill="#d9862f"/>'
        '<rect x="150" y="126" width="84" height="84" rx="14" fill="#2f5aa0"/>'
        '<circle cx="194" cy="156" r="17" fill="#f0f0f0"/>'
        '</g><g id="text">'
        + _text(196, 274, 38, "#243b57", "NOVA", weight="bold")
        + _text(188, 308, 13, "#7b8698", "SYSTEMS")
        + "</g></svg>"
    )


STAR_OUTER, STAR_INNER = 74, 32


def _star(cx: float, cy: float, outer: float, inner: float):
    points = []
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5
        radius = outer if i % 2 == 0 else inner
        points.append((cx + radius * math.cos(angle), cy - radius * math.sin(angle)))
    return points


def emblem_target() -> Image.Image:
    img, d = _canvas("#fdfaf3")
    d.ellipse(_box(192, 192, 140, 140), fill="#173a5e")
    d.ellipse(_box(192, 192, 124, 124), outline="#d9b45b", width=4 * SS)
    d.ellipse(_box(192, 192, 100, 100), fill="#1e4c78")
    d.polygon(
        [(x * SS, y * SS) for x, y in _star(192, 192, STAR_OUTER, STAR_INNER)],
        fill="#d9b45b",
    )
    _centred(d, "EST 1994", _font(BOLD, 20), 192, 296, "#fdfaf3")
    return _finish(img)


def emblem_seed() -> str:
    star = _star(190, 186, STAR_OUTER * 0.78, STAR_INNER * 1.35)
    d = "M" + " L".join(f"{round(x, 1)} {round(y, 1)}" for x, y in star) + " Z"
    return (
        f'{HEAD}<rect width="{SIDE}" height="{SIDE}" fill="#f7f2e6"/>'
        '<g id="rings">'
        '<circle cx="192" cy="192" r="132" fill="#1d4468"/>'
        '<circle cx="192" cy="192" r="118" fill="none" stroke="#c2a86e" '
        'stroke-width="7"/>'
        '<circle cx="192" cy="192" r="94" fill="#265a86"/>'
        "</g>"
        f'<g id="star"><path d="{d}" fill="#c9a961"/></g>'
        '<g id="text">'
        + _text(192, 302, 15, "#f2ece0", "EST 1994", weight="bold")
        + "</g></svg>"
    )


SKY_TOP, SKY_BOTTOM = (255, 170, 90), (94, 62, 140)


GLOW_INNER, GLOW_OUTER = (255, 233, 168), (255, 170, 90)


def sunset_target() -> Image.Image:
    """The effects case: linear gradient sky, radial sun glow, translucent haze."""
    img = Image.new("RGB", (S, S), "#ffffff")
    d = ImageDraw.Draw(img)
    for y in range(S):
        t = y / S
        band = tuple(
            round(a + (b - a) * t) for a, b in zip(SKY_TOP, SKY_BOTTOM, strict=True)
        )
        d.line([(0, y), (S, y)], fill=band)

    glow = Image.new("RGB", (S, S), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    steps = 40
    for i in range(steps, 0, -1):
        t = i / steps
        radius = 52 + 58 * t
        color = tuple(
            round(a + (b - a) * t) for a, b in zip(GLOW_INNER, GLOW_OUTER, strict=True)
        )
        gd.ellipse(_box(192, 148, radius, radius), fill=color)
    mask = Image.new("L", (S, S), 0)
    md = ImageDraw.Draw(mask)
    for i in range(steps, 0, -1):
        t = i / steps
        md.ellipse(_box(192, 148, 52 + 58 * t, 52 + 58 * t), fill=round(220 * (1 - t)))
    md.ellipse(_box(192, 148, 52, 52), fill=255)
    img = Image.composite(glow, img, mask)

    img = img.filter(ImageFilter.GaussianBlur(0.8 * SS))
    d = ImageDraw.Draw(img)
    d.polygon(
        [(0, 300 * SS), (110 * SS, 214 * SS), (232 * SS, 300 * SS)], fill="#3c2f5a"
    )
    d.polygon(
        [(150 * SS, 300 * SS), (272 * SS, 196 * SS), (384 * SS, 300 * SS)],
        fill="#2a2142",
    )
    haze = Image.new("RGB", (S, S), "#c98fb4")
    hmask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(hmask).rectangle([0, 250 * SS, S, 300 * SS], fill=64)
    img = Image.composite(haze, img, hmask)
    ImageDraw.Draw(img).rectangle([0, 300 * SS, S, S], fill="#1b1730")
    return _finish(img)


def sunset_seed() -> str:
    """Same effects, wrong values: gradient stops and opacity are both mutable."""
    return (
        f"{HEAD}<defs>"
        '<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#e08a55"/>'
        '<stop offset="1" stop-color="#6b4f9e"/>'
        "</linearGradient>"
        '<radialGradient id="glow" cx="0.5" cy="0.5" r="0.5">'
        '<stop offset="0.4" stop-color="#f2dfa4"/>'
        '<stop offset="1" stop-color="#e69a5e" stop-opacity="0"/>'
        "</radialGradient>"
        "</defs>"
        f'<rect width="{SIDE}" height="{SIDE}" fill="url(#sky)"/>'
        '<g id="sun">'
        '<circle cx="188" cy="142" r="112" fill="url(#glow)"/>'
        '<circle cx="188" cy="142" r="58" fill="#f2dfa4"/>'
        "</g>"
        '<g id="hills">'
        '<path d="M0 300 L104 222 L226 300 Z" fill="#4a3d68"/>'
        '<path d="M156 300 L266 204 L384 300 Z" fill="#332a4c"/>'
        "</g>"
        '<g id="haze">'
        '<rect y="248" width="384" height="52" fill="#b98aa8" opacity="0.45"/>'
        "</g>"
        '<g id="ground"><rect y="300" width="384" height="84" fill="#241f38"/></g>'
        "</svg>"
    )


DUCK = [
    (70, 150),
    (100, 138),
    (128, 108),
    (152, 130),
    (160, 168),
    (200, 186),
    (250, 196),
    (306, 176),
    (330, 196),
    (296, 214),
    (270, 244),
    (210, 262),
    (168, 244),
    (150, 200),
    (128, 176),
    (96, 158),
]
DOT_COUNT = 22


def _even_points(vertices, count):
    closed = [*vertices, vertices[0]]
    spans, total = [], 0.0
    for a, b in itertools.pairwise(closed):
        length = math.dist(a, b)
        spans.append((total, length, a, b))
        total += length

    out = []
    for i in range(count):
        target = total * i / count
        for start, length, a, b in spans:
            if target <= start + length or (start, length) == spans[-1][:2]:
                t = (target - start) / length if length else 0.0
                out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
                break
    return out


def _dot_layout():
    points = _even_points(DUCK, DOT_COUNT)
    cx = sum(x for x, _ in points) / len(points)
    cy = sum(y for _, y in points) / len(points)
    out = []
    for i, (x, y) in enumerate(points):
        dx, dy = x - cx, y - cy
        norm = math.hypot(dx, dy) or 1.0
        out.append((x, y, x + 13 * dx / norm, y + 13 * dy / norm, str(i + 1)))
    return out


def connect_dots_target() -> Image.Image:
    """A connect-the-dots puzzle: numbered dots, the outline never drawn."""
    img, d = _canvas("#ffffff")
    font = _font(REGULAR, 11)
    for x, y, lx, ly, label in _dot_layout():
        d.ellipse(_box(x, y, 2.6, 2.6), fill="#111111")
        _centred(d, label, font, lx, ly, "#111111")
    d.ellipse(_box(153, 121, 3, 3), fill="#111111")
    return _finish(img)


def connect_dots_seed() -> str:
    out = [
        HEAD,
        f'<rect width="{SIDE}" height="{SIDE}" fill="#ffffff"/>',
        '<g id="dots">',
    ]
    for x, y, lx, ly, label in _dot_layout():
        out.append(
            f'<circle cx="{round(x + 3, 1)}" cy="{round(y - 2, 1)}" r="4" '
            'fill="#3a3a3a"/>'
        )
        out.append(_text(round(lx - 4, 1), round(ly + 6, 1), 16, "#3a3a3a", label))
    out.append('</g><g id="eye"><circle cx="156" cy="118" r="4" fill="#3a3a3a"/></g>')
    out.append("</svg>")
    return "".join(out)


LEAF = [
    (192, 56),
    (286, 96),
    (306, 196),
    (196, 320),
    (196, 320),
    (88, 200),
    (104, 98),
    (192, 56),
]
VEIN = [(192, 300), (200, 220), (196, 140), (192, 70)]
SPRIG = [(192, 232), (150, 214), (128, 186), (120, 150)]


def _cubic(points, steps: int = 60):
    """Sample the cubic segments of a path given as start + 3n control points."""
    out = []
    for i in range(0, len(points) - 3, 3):
        p0, p1, p2, p3 = points[i : i + 4]
        for step in range(steps + 1):
            t = step / steps
            u = 1 - t
            out.append(
                (
                    u**3 * p0[0]
                    + 3 * u * u * t * p1[0]
                    + 3 * u * t * t * p2[0]
                    + t**3 * p3[0],
                    u**3 * p0[1]
                    + 3 * u * u * t * p1[1]
                    + 3 * u * t * t * p2[1]
                    + t**3 * p3[1],
                )
            )
    return out


def _path_d(points, shift=(0.0, 0.0), scale=1.0) -> str:
    """Emit 'M x y C ...' from the same control points the target samples."""
    cx = sum(x for x, _ in points) / len(points)
    cy = sum(y for _, y in points) / len(points)

    def moved(p):
        return (
            round(cx + (p[0] - cx) * scale + shift[0], 1),
            round(cy + (p[1] - cy) * scale + shift[1], 1),
        )

    head = moved(points[0])
    parts = [f"M{head[0]} {head[1]}"]
    for i in range(1, len(points) - 2, 3):
        a, b, c = (moved(p) for p in points[i : i + 3])
        parts.append(f"C{a[0]} {a[1]} {b[0]} {b[1]} {c[0]} {c[1]}")
    return " ".join(parts)


def leaf_target() -> Image.Image:
    """Bezier geometry only: every shape here is a path in the seed."""
    img, d = _canvas("#f6f7f1")
    d.polygon([(x * SS, y * SS) for x, y in _cubic(LEAF)], fill="#4f8a3d")
    d.line(
        [(x * SS, y * SS) for x, y in _cubic(VEIN)],
        fill="#2f5a24",
        width=5 * SS,
        joint="curve",
    )
    d.line(
        [(x * SS, y * SS) for x, y in _cubic(SPRIG)],
        fill="#2f5a24",
        width=3 * SS,
        joint="curve",
    )
    return _finish(img)


def leaf_seed() -> str:
    return (
        f'{HEAD}<rect width="{SIDE}" height="{SIDE}" fill="#f2f4ee"/>'
        '<g id="blade">'
        f'<path d="{_path_d(LEAF, shift=(6, -5), scale=0.88)}" fill="#5f9b52"/>'
        '</g><g id="veins">'
        f'<path d="{_path_d(VEIN, shift=(-5, 4), scale=1.15)}" fill="none" '
        'stroke="#3d6b33" stroke-width="8"/>'
        f'<path d="{_path_d(SPRIG, shift=(8, 6), scale=0.8)}" fill="none" '
        'stroke="#3d6b33" stroke-width="6"/>'
        "</g></svg>"
    )


CASES = {
    "leaf": (leaf_target, leaf_seed),
    "mascot": (mascot_target, mascot_seed),
    "wordmark": (wordmark_target, wordmark_seed),
    "emblem": (emblem_target, emblem_seed),
    "sunset": (sunset_target, sunset_seed),
    "connect-dots": (connect_dots_target, connect_dots_seed),
}


def main() -> None:
    # Running this as a script puts bench/ on sys.path rather than the repo
    # root, so `from bench.seeds` needs the root added either way.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from bench.seeds import SEEDS
    from vectrify.formats.svg.plugin import SvgPlugin

    plugin = SvgPlugin()
    root = Path(__file__).parent / "cases"
    for name, (target, _legacy_seed) in CASES.items():
        case_dir = root / name
        seeds_dir = case_dir / "seeds"
        seeds_dir.mkdir(parents=True, exist_ok=True)
        for stale in (*case_dir.glob("*.svg"), *seeds_dir.glob("*.svg")):
            stale.unlink()

        variants = SEEDS[name]
        for index, svg in enumerate(variants, start=1):
            ok, err = plugin.validate(svg)
            if not ok:
                raise SystemExit(f"{name}/seeds/{index}.svg is invalid: {err}")
            (seeds_dir / f"{index}.svg").write_text(svg, encoding="utf-8")

        target().save(case_dir / "target.png")
        print(f"{name}: wrote target.png and {len(variants)} seeds")

    try:
        from bench.non_svg import CASES as NON_SVG_CASES
        from bench.non_svg import generate as generate_non_svg

        generate_non_svg(root)
        for name, (_format, _target, variants) in NON_SVG_CASES.items():
            print(f"{name}: wrote target.png and {len(variants)} seeds")
    except (ImportError, OSError) as exc:
        print(f"non-SVG corpus skipped (renderer unavailable: {exc})")


if __name__ == "__main__":
    main()
