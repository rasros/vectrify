"""Seed pools for the benchmark corpus."""

import math

SIDE = 384
HEAD = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIDE} {SIDE}">'
FAMILY = "DejaVu Sans"


def _n(value: float) -> str:
    """Trim a coordinate to one decimal without a trailing ``.0``."""
    return f"{round(value, 1):g}"


def _text(x, y, size, fill, body, weight="normal") -> str:
    return (
        f'<text x="{_n(x)}" y="{_n(y)}" font-family="{FAMILY}" '
        f'font-size="{_n(size)}" font-weight="{weight}" text-anchor="middle" '
        f'fill="{fill}">{body}</text>'
    )


def _bg(color: str) -> str:
    return f'<rect x="0" y="0" width="{SIDE}" height="{SIDE}" fill="{color}"/>'


def _linear(gid: str, top: str, bottom: str, vertical: bool = True) -> str:
    x2, y2 = ("0", "1") if vertical else ("1", "0")
    return (
        f'<linearGradient id="{gid}" x1="0" y1="0" x2="{x2}" y2="{y2}">'
        f'<stop offset="0" stop-color="{top}"/>'
        f'<stop offset="1" stop-color="{bottom}"/>'
        "</linearGradient>"
    )


def _radial(gid: str, inner: str, outer: str, mid: str = "0.5") -> str:
    return (
        f'<radialGradient id="{gid}" cx="0.5" cy="0.5" r="0.5">'
        f'<stop offset="{mid}" stop-color="{inner}"/>'
        f'<stop offset="1" stop-color="{outer}"/>'
        "</radialGradient>"
    )


def _circle_path(cx: float, cy: float, r: float) -> str:
    """A circle written as four cubics, for seeds that draw discs as paths."""
    k = r * 0.5523
    return (
        f"M{_n(cx - r)} {_n(cy)} "
        f"C{_n(cx - r)} {_n(cy - k)} {_n(cx - k)} {_n(cy - r)} {_n(cx)} {_n(cy - r)} "
        f"C{_n(cx + k)} {_n(cy - r)} {_n(cx + r)} {_n(cy - k)} {_n(cx + r)} {_n(cy)} "
        f"C{_n(cx + r)} {_n(cy + k)} {_n(cx + k)} {_n(cy + r)} {_n(cx)} {_n(cy + r)} "
        f"C{_n(cx - k)} {_n(cy + r)} {_n(cx - r)} {_n(cy + k)} "
        f"{_n(cx - r)} {_n(cy)} Z"
    )


def _round_rect_path(x, y, w, h, r, quad: bool = False) -> str:
    """A rounded rectangle as a path, with cubic or quadratic corners."""
    if quad:

        def corner(px, py, ex, ey):
            return f"Q{_n(px)} {_n(py)} {_n(ex)} {_n(ey)}"

        return (
            f"M{_n(x + r)} {_n(y)} L{_n(x + w - r)} {_n(y)} "
            + corner(x + w, y, x + w, y + r)
            + f" L{_n(x + w)} {_n(y + h - r)} "
            + corner(x + w, y + h, x + w - r, y + h)
            + f" L{_n(x + r)} {_n(y + h)} "
            + corner(x, y + h, x, y + h - r)
            + f" L{_n(x)} {_n(y + r)} "
            + corner(x, y, x + r, y)
            + " Z"
        )
    k = r * 0.4477
    return (
        f"M{_n(x + r)} {_n(y)} L{_n(x + w - r)} {_n(y)} "
        f"C{_n(x + w - k)} {_n(y)} {_n(x + w)} {_n(y + k)} {_n(x + w)} {_n(y + r)} "
        f"L{_n(x + w)} {_n(y + h - r)} "
        f"C{_n(x + w)} {_n(y + h - k)} {_n(x + w - k)} {_n(y + h)} "
        f"{_n(x + w - r)} {_n(y + h)} "
        f"L{_n(x + r)} {_n(y + h)} "
        f"C{_n(x + k)} {_n(y + h)} {_n(x)} {_n(y + h - k)} {_n(x)} {_n(y + h - r)} "
        f"L{_n(x)} {_n(y + r)} "
        f"C{_n(x)} {_n(y + k)} {_n(x + k)} {_n(y)} {_n(x + r)} {_n(y)} Z"
    )

_MASCOT_1 = (
    f"{HEAD}{_bg('#e9f3fa')}"
    '<g id="parts">'
    '<ellipse cx="129" cy="93" rx="31" ry="25" fill="#4cc0b0" stroke="#1a4b46" '
    'stroke-width="3"/>'
    '<ellipse cx="255" cy="93" rx="31" ry="25" fill="#4cc0b0" stroke="#1a4b46" '
    'stroke-width="3"/>'
    '<ellipse cx="187" cy="283" rx="47" ry="40" fill="#4cc0b0" stroke="#1a4b46" '
    'stroke-width="3"/>'
    '<ellipse cx="187" cy="174" rx="97" ry="76" fill="#4cc0b0" stroke="#1a4b46" '
    'stroke-width="3"/>'
    '</g><g id="eyes">'
    '<ellipse cx="160" cy="157" rx="16" ry="26" fill="#f1f1f1" stroke="#1a4b46" '
    'stroke-width="2"/>'
    '<ellipse cx="234" cy="157" rx="16" ry="26" fill="#f1f1f1" stroke="#1a4b46" '
    'stroke-width="2"/>'
    '<ellipse cx="158" cy="161" rx="12" ry="9" fill="#1b3f4c"/>'
    '<ellipse cx="232" cy="161" rx="12" ry="9" fill="#1b3f4c"/>'
    '<circle cx="150" cy="164" r="5" fill="#e6e6e6"/>'
    '<circle cx="224" cy="164" r="5" fill="#e6e6e6"/>'
    '</g><g id="face">'
    '<ellipse cx="138" cy="199" rx="10" ry="14" fill="#e9959f"/>'
    '<ellipse cx="258" cy="199" rx="10" ry="14" fill="#e9959f"/>'
    '<line x1="164" y1="216" x2="222" y2="216" stroke="#1a4b46" '
    'stroke-width="7"/>'
    '<path d="M188 188 L174 204 L204 204 Z" fill="#1a4b46"/>'
    "</g></svg>"
)

_MASCOT_2 = (
    f"{HEAD}{_bg('#f2f7fc')}"
    '<g id="body">'
    '<circle cx="197" cy="270" r="50" fill="#34a396"/>'
    '<circle cx="197" cy="270" r="50" fill="none" stroke="#0f3835" '
    'stroke-width="7"/>'
    '</g><g id="head">'
    '<circle cx="198" cy="162" r="80" fill="#34a396"/>'
    '<circle cx="198" cy="162" r="80" fill="none" stroke="#0f3835" '
    'stroke-width="7"/>'
    '</g><g id="ears">'
    '<circle cx="130" cy="82" r="33" fill="#34a396"/>'
    '<circle cx="130" cy="82" r="33" fill="none" stroke="#0f3835" '
    'stroke-width="7"/>'
    '<circle cx="266" cy="82" r="33" fill="#34a396"/>'
    '<circle cx="266" cy="82" r="33" fill="none" stroke="#0f3835" '
    'stroke-width="7"/>'
    '</g><g id="eyes">'
    '<circle cx="161" cy="158" r="22" fill="#ffffff"/>'
    '<circle cx="161" cy="158" r="22" fill="none" stroke="#0f3835" '
    'stroke-width="4"/>'
    '<circle cx="235" cy="158" r="22" fill="#ffffff"/>'
    '<circle cx="235" cy="158" r="22" fill="none" stroke="#0f3835" '
    'stroke-width="4"/>'
    '<circle cx="162" cy="160" r="11" fill="#0f3835"/>'
    '<circle cx="236" cy="160" r="11" fill="#0f3835"/>'
    '<circle cx="158" cy="154" r="5" fill="#ffffff"/>'
    '<circle cx="232" cy="154" r="5" fill="#ffffff"/>'
    '</g><g id="face">'
    '<ellipse cx="137" cy="212" rx="17" ry="12" fill="#e884a0"/>'
    '<ellipse cx="247" cy="212" rx="17" ry="12" fill="#e884a0"/>'
    '<path d="M168 212 Q198 236 226 212" fill="none" stroke="#0f3835" '
    'stroke-width="4"/>'
    '<circle cx="197" cy="206" r="9" fill="#0f3835"/>'
    "</g></svg>"
)

_MASCOT_3 = (
    f"{HEAD}<defs>{_linear('hide', '#5fc7b2', '#2c7f74')}</defs>"
    f"{_bg('#eaf2f9')}"
    f'<path d="{_circle_path(129, 94, 25)}" fill="url(#hide)" stroke="#174742" '
    'stroke-width="8"/>'
    f'<path d="{_circle_path(255, 94, 25)}" fill="url(#hide)" stroke="#174742" '
    'stroke-width="8"/>'
    '<ellipse cx="197" cy="271" rx="37" ry="51" fill="url(#hide)" '
    'stroke="#174742" stroke-width="8"/>'
    f'<path d="{_circle_path(187, 174, 82)}" fill="url(#hide)" stroke="#174742" '
    'stroke-width="8"/>'
    '<ellipse cx="151" cy="168" rx="22" ry="19" fill="#ececec" stroke="#174742" '
    'stroke-width="8"/>'
    '<ellipse cx="225" cy="168" rx="22" ry="19" fill="#ececec" stroke="#174742" '
    'stroke-width="8"/>'
    f'<path d="{_circle_path(152, 170, 8)}" fill="#174742"/>'
    f'<path d="{_circle_path(226, 170, 8)}" fill="#174742"/>'
    '<ellipse cx="149" cy="164" rx="5" ry="4" fill="#f4f4f4"/>'
    '<ellipse cx="223" cy="164" rx="5" ry="4" fill="#f4f4f4"/>'
    '<rect x="112" y="203" width="30" height="14" fill="#f0a8b6"/>'
    '<rect x="240" y="203" width="30" height="14" fill="#f0a8b6"/>'
    '<path d="M166 208 C176 228 208 228 218 208" fill="none" stroke="#174742" '
    'stroke-width="3"/>'
    '<path d="M190 190 L178 206 L202 206 Z" fill="#174742"/>'
    "</svg>"
)

_MASCOT_4 = (
    f"{HEAD}<defs>{_radial('fur', '#3fb0a2', '#1d6f68', mid='0.35')}</defs>"
    f"{_bg('#f0f7fa')}"
    '<g id="core">'
    '<line x1="196" y1="260" x2="196" y2="292" stroke="#153c38" '
    'stroke-width="98" stroke-linecap="round"/>'
    '<line x1="196" y1="260" x2="196" y2="292" stroke="#2f9f95" '
    'stroke-width="82" stroke-linecap="round"/>'
    '<ellipse cx="196" cy="164" rx="80" ry="92" fill="url(#fur)" '
    'stroke="#153c38" stroke-width="9"/>'
    '</g><g id="left">'
    '<ellipse cx="130" cy="82" rx="25" ry="32" fill="#2f9f95" stroke="#153c38" '
    'stroke-width="9"/>'
    '<ellipse cx="159" cy="168" rx="22" ry="20" fill="#f6f6f6" stroke="#153c38" '
    'stroke-width="9"/>'
    '<ellipse cx="160" cy="170" rx="8" ry="13" fill="#153c38"/>'
    '<rect x="152" y="162" width="8" height="6" fill="#f6f6f6"/>'
    '</g><g id="right">'
    '<ellipse cx="254" cy="82" rx="25" ry="32" fill="#2f9f95" stroke="#153c38" '
    'stroke-width="9"/>'
    '<ellipse cx="233" cy="168" rx="22" ry="20" fill="#f6f6f6" stroke="#153c38" '
    'stroke-width="9"/>'
    '<ellipse cx="234" cy="170" rx="8" ry="13" fill="#153c38"/>'
    '<rect x="226" y="162" width="8" height="6" fill="#f6f6f6"/>'
    '</g><g id="muzzle">'
    '<path d="M168 210 Q196 232 224 210" fill="none" stroke="#153c38" '
    'stroke-width="9"/>'
    '<path d="M196 198 L186 212 L208 212 Z" fill="#153c38"/>'
    '</g><g id="cheeks">'
    '<ellipse cx="128" cy="202" rx="16" ry="12" fill="#e79aae" '
    'fill-opacity="0.7"/>'
    '<ellipse cx="256" cy="202" rx="16" ry="12" fill="#e79aae" '
    'fill-opacity="0.7"/>'
    "</g></svg>"
)

_MASCOT_5 = (
    f"{HEAD}{_bg('#eaf4fb')}"
    '<g id="flats">'
    '<ellipse cx="196" cy="176" rx="84" ry="88" fill="#4fbcac"/>'
    '<rect x="104" y="76" width="48" height="46" fill="#4fbcac"/>'
    '<rect x="234" y="76" width="48" height="46" fill="#4fbcac"/>'
    '<ellipse cx="188" cy="286" rx="38" ry="52" fill="#4fbcac"/>'
    '<circle cx="152" cy="159" r="21" fill="#fafafa"/>'
    '<circle cx="226" cy="159" r="21" fill="#fafafa"/>'
    '<ellipse cx="153" cy="161" rx="8" ry="10" fill="#154440"/>'
    '<ellipse cx="227" cy="161" rx="8" ry="10" fill="#154440"/>'
    '<circle cx="149" cy="155" r="4" fill="#fafafa"/>'
    '<circle cx="223" cy="155" r="4" fill="#fafafa"/>'
    '<rect x="119" y="199" width="28" height="18" rx="9" ry="9" fill="#eb93a6"/>'
    '<rect x="241" y="199" width="28" height="18" rx="9" ry="9" fill="#eb93a6"/>'
    '</g><g id="linework">'
    '<ellipse cx="196" cy="176" rx="84" ry="88" fill="none" stroke="#154440" '
    'stroke-width="2"/>'
    '<rect x="104" y="76" width="48" height="46" fill="none" stroke="#154440" '
    'stroke-width="2"/>'
    '<rect x="234" y="76" width="48" height="46" fill="none" stroke="#154440" '
    'stroke-width="2"/>'
    '<ellipse cx="188" cy="286" rx="38" ry="52" fill="none" stroke="#154440" '
    'stroke-width="2"/>'
    '<circle cx="152" cy="159" r="21" fill="none" stroke="#154440" '
    'stroke-width="2"/>'
    '<circle cx="226" cy="159" r="21" fill="none" stroke="#154440" '
    'stroke-width="2"/>'
    '<path d="M166 208 C176 226 212 226 220 208" fill="none" stroke="#154440" '
    'stroke-width="2"/>'
    '<path d="M194 196 L182 210 L208 210 Z" fill="#154440"/>'
    "</g></svg>"
)

_WORDMARK_1 = (
    f"{HEAD}{_bg('#fbfbfb')}"
    '<g id="mark">'
    '<rect x="102" y="80" width="84" height="84" rx="14" fill="#dd8434"/>'
    '<rect x="152" y="124" width="84" height="84" rx="14" fill="#2b5498"/>'
    '<rect x="178" y="146" width="34" height="34" fill="#f0f0f0"/>'
    '</g><g id="type">'
    + _text(196, 272, 38, "#243b57", "NOVA", weight="bold")
    + _text(188, 308, 13, "#7b8698", "SYSTEMS")
    + "</g></svg>"
)

_WORDMARK_2 = (
    f"{HEAD}<defs>{_linear('warm', '#f2a95c', '#c96a1e', vertical=False)}</defs>"
    f"{_bg('#f6f4f0')}"
    '<g id="squares">'
    '<ellipse cx="139" cy="116" rx="50" ry="48" fill="url(#warm)"/>'
    '<rect x="140" y="112" width="98" height="96" rx="30" ry="30" fill="#173f80"/>'
    '<circle cx="186" cy="164" r="24" fill="#fcfcfc"/>'
    '</g><g id="type">'
    + _text(198, 268, 52, "#0c1c34", "NOVA", weight="bold")
    + _text(198, 306, 22, "#48566b", "SYSTEMS")
    + "</g></svg>"
)

_WORDMARK_3 = (
    f"{HEAD}{_bg('#ffffff')}"
    f'<path d="{_round_rect_path(152, 124, 80, 80, 18)}" fill="#2f5eab"/>'
    '<circle cx="188" cy="154" r="24" fill="#f4f4f4"/>'
    f'<path d="{_round_rect_path(102, 80, 80, 80, 18)}" fill="#e98637"/>'
    '<g id="type">'
    + _text(160, 268, 44, "#1b2f4e", "NO", weight="bold")
    + _text(228, 268, 44, "#1b2f4e", "VA", weight="bold")
    + _text(192, 310, 14, "#6a7789", "SYSTEMS")
    + "</g></svg>"
)

_WORDMARK_4 = (
    f"{HEAD}{_bg('#f4f4f6')}"
    '<line x1="142" y1="102" x2="142" y2="138" stroke="#e07b2c" '
    'stroke-width="92" stroke-linecap="round"/>'
    '<line x1="192" y1="146" x2="192" y2="182" stroke="#26559b" '
    'stroke-width="92" stroke-linecap="round"/>'
    f'<path d="{_circle_path(194, 162, 18)}" fill="#ffffff"/>'
    + _text(194, 268, 42, "#1a2b46", "NOVA", weight="bold")
    + _text(192, 306, 14, "#6b7a8d", "SYSTEMS")
    + "</svg>"
)

_WORDMARK_5 = (
    f"{HEAD}{_bg('#fdfaf6')}"
    '<g id="mark">'
    f'<path d="{_round_rect_path(100, 78, 88, 88, 28, quad=True)}" fill="#e77c26"/>'
    f'<path d="{_round_rect_path(150, 122, 88, 88, 28, quad=True)}" fill="#173f8c"/>'
    '<circle cx="186" cy="164" r="18" fill="#f8f8f8" fill-opacity="0.8"/>'
    '</g><g id="nova">'
    + _text(190, 258, 52, "#101f3a", "NOVA", weight="bold")
    + '</g><g id="tagline">'
    + _text(150, 304, 15, "#55637a", "S")
    + _text(164, 304, 15, "#55637a", "Y")
    + _text(178, 304, 15, "#55637a", "S")
    + _text(192, 304, 15, "#55637a", "T")
    + _text(204, 304, 15, "#55637a", "E")
    + _text(218, 304, 15, "#55637a", "M")
    + _text(234, 304, 15, "#55637a", "S")
    + "</g></svg>"
)

def _star_points(cx, cy, outer, inner, rot=0.0):
    out = []
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5 + rot
        radius = outer if i % 2 == 0 else inner
        out.append((cx + radius * math.cos(angle), cy - radius * math.sin(angle)))
    return out


def _star_path(cx, cy, outer, inner, rot=0.0) -> str:
    pts = _star_points(cx, cy, outer, inner, rot)
    body = " L".join(f"{_n(x)} {_n(y)}" for x, y in pts)
    return f"M{body} Z"


def _star_quad_path(cx, cy, outer, inner, rot=0.0) -> str:
    """The same star with quadratic sides, so the arms bow and stay bowed."""
    pts = _star_points(cx, cy, outer, inner, rot)
    parts = [f"M{_n(pts[0][0])} {_n(pts[0][1])}"]
    for i in range(1, 11):
        prev = pts[(i - 1) % 10]
        nxt = pts[i % 10]
        mx = (prev[0] + nxt[0]) / 2 + (cx - (prev[0] + nxt[0]) / 2) * 0.22
        my = (prev[1] + nxt[1]) / 2 + (cy - (prev[1] + nxt[1]) / 2) * 0.22
        parts.append(f"Q{_n(mx)} {_n(my)} {_n(nxt[0])} {_n(nxt[1])}")
    return " ".join(parts) + " Z"


def _star_pieces(cx, cy, outer, inner, fill, rot=0.0) -> str:
    """The star as a pentagon core plus five arm triangles, stacked."""
    pts = _star_points(cx, cy, outer, inner, rot)
    core = " L".join(f"{_n(pts[i][0])} {_n(pts[i][1])}" for i in (1, 3, 5, 7, 9))
    out = [f'<path d="M{core} Z" fill="{fill}"/>']
    for i in (0, 2, 4, 6, 8):
        tip = pts[i]
        left = pts[(i - 1) % 10]
        right = pts[(i + 1) % 10]
        out.append(
            f'<path d="M{_n(left[0])} {_n(left[1])} L{_n(tip[0])} {_n(tip[1])} '
            f'L{_n(right[0])} {_n(right[1])} Z" fill="{fill}"/>'
        )
    return "".join(out)


def _beaded_ring(cx, cy, r, count, color, width) -> str:
    """A ring chopped into short chords -- never a smooth circle."""
    out = []
    for i in range(count):
        a = 2 * math.pi * i / count
        b = 2 * math.pi * (i + 0.62) / count
        out.append(
            f'<line x1="{_n(cx + r * math.cos(a))}" '
            f'y1="{_n(cy + r * math.sin(a))}" '
            f'x2="{_n(cx + r * math.cos(b))}" '
            f'y2="{_n(cy + r * math.sin(b))}" '
            f'stroke="{color}" stroke-width="{width}"/>'
        )
    return "".join(out)


_EMBLEM_1 = (
    f"{HEAD}{_bg('#f9f5ea')}"
    '<g id="rings">'
    '<circle cx="192" cy="192" r="132" fill="#1d4468"/>'
    '<circle cx="192" cy="192" r="118" fill="none" stroke="#c2a86e" '
    'stroke-width="7"/>'
    '<circle cx="192" cy="192" r="94" fill="#265a86"/>'
    "</g>"
    f'<g id="star"><path d="{_star_quad_path(190, 186, 66, 36)}" '
    'fill="#c9a961"/></g>'
    '<g id="type">' + _text(192, 302, 15, "#f2ece0", "EST 1994", weight="bold") + "</g>"
    "</svg>"
)

_EMBLEM_2 = (
    f"{HEAD}{_bg('#fffdf6')}"
    '<g id="disc">'
    '<circle cx="186" cy="198" r="146" fill="#0f2c4c"/>'
    '<circle cx="186" cy="198" r="130" fill="#e6c98a"/>'
    '<circle cx="186" cy="198" r="122" fill="#0f2c4c"/>'
    '<circle cx="186" cy="198" r="106" fill="#2a6091"/>'
    "</g>"
    f'<g id="star"><path d="{_star_path(186, 198, 80, 36)}" fill="#e6c98a"/></g>'
    '<g id="type">'
    + _text(160, 290, 26, "#fbf6ea", "EST", weight="bold")
    + _text(222, 290, 26, "#fbf6ea", "1994", weight="bold")
    + "</g></svg>"
)

_EMBLEM_3 = (
    f"{HEAD}<defs>{_radial('plate', '#2c6597', '#132f4d', mid='0.3')}</defs>"
    f"{_bg('#f0e9d6')}"
    '<ellipse cx="198" cy="186" rx="126" ry="146" fill="#2a628f"/>'
    '<ellipse cx="198" cy="186" rx="112" ry="132" fill="none" stroke="#a98b3e" '
    'stroke-width="12"/>'
    '<ellipse cx="198" cy="186" rx="90" ry="106" fill="url(#plate)"/>'
    + _text(198, 288, 28, "#f4efe2", "EST 1994", weight="bold")
    + _star_pieces(198, 186, 84, 38, "#bf9a4e")
    + "</svg>"
)

_EMBLEM_4 = (
    f"{HEAD}<defs>{_linear('well', '#3873ab', '#10375c')}</defs>"
    f"{_bg('#fbf8ee')}"
    '<g id="plate">'
    '<circle cx="188" cy="198" r="73" fill="none" stroke="#1a3f66" '
    'stroke-width="146"/>'
    '<circle cx="188" cy="198" r="116" fill="none" stroke="#eac977" '
    'stroke-width="2"/>'
    f'<path d="{_circle_path(188, 198, 96)}" fill="url(#well)"/>'
    "</g>"
    f'<g id="star"><path d="{_star_path(188, 198, 70, 36, rot=0.42)}" '
    'fill="#eac977"/></g>'
    + _text(188, 302, 13, "#efe7d6", "EST 1994", weight="bold")
    + "</svg>"
)

_EMBLEM_5 = (
    f"{HEAD}{_bg('#f9f6ec')}"
    '<ellipse cx="196" cy="190" rx="148" ry="132" fill="#264a6d"/>'
    + _beaded_ring(196, 190, 126, 16, "#d0b878", 9)
    + '<ellipse cx="196" cy="190" rx="106" ry="94" fill="#12406e"/>'
    f'<path d="{_star_path(196, 190, 82, 28)}" fill="#c8a955"/>'
    + _text(200, 294, 24, "#faf4e6", "EST 1994", weight="bold")
    + "</svg>"
)

_SKY_BANDS = [
    "#f8a95c",
    "#e89a66",
    "#d78a70",
    "#c47b78",
    "#ae6a7e",
    "#95587e",
    "#78477a",
    "#563a74",
]

_SUNSET_1 = (
    f"{HEAD}<defs>"
    + _linear("sky", "#f09a5c", "#5c4290")
    + '<radialGradient id="glow" cx="0.5" cy="0.5" r="0.5">'
    '<stop offset="0.4" stop-color="#f7e4a8"/>'
    '<stop offset="1" stop-color="#f0a066" stop-opacity="0"/>'
    "</radialGradient>"
    "</defs>"
    f'<rect x="0" y="0" width="{SIDE}" height="{SIDE}" fill="url(#sky)"/>'
    '<g id="sun">'
    '<circle cx="188" cy="142" r="118" fill="url(#glow)"/>'
    '<circle cx="188" cy="142" r="44" fill="#f7e4a8"/>'
    "</g>"
    '<g id="hills">'
    '<line x1="4" y1="296" x2="108" y2="220" stroke="#443563" '
    'stroke-width="44" stroke-linecap="round"/>'
    '<line x1="108" y1="220" x2="228" y2="296" stroke="#443563" '
    'stroke-width="44" stroke-linecap="round"/>'
    '<path d="M156 300 L268 202 L384 300 Z" fill="#2c2444"/>'
    "</g>"
    '<rect x="0" y="248" width="384" height="52" fill="#c186ad" opacity="0.45"/>'
    '<rect x="0" y="300" width="384" height="84" fill="#201c34"/>'
    "</svg>"
)

_SUNSET_2 = (
    f"{HEAD}"
    + "".join(
        f'<rect x="0" y="{i * 48}" width="384" height="48" fill="{color}"/>'
        for i, color in enumerate(_SKY_BANDS)
    )
    + '<rect x="0" y="252" width="384" height="26" fill="#d09ac0" '
    'fill-opacity="0.2"/>'
    '<rect x="0" y="278" width="384" height="26" fill="#d09ac0" '
    'fill-opacity="0.2"/>'
    '<g id="halo">'
    '<circle cx="196" cy="154" r="118" fill="#f0b070" fill-opacity="0.3"/>'
    '<circle cx="196" cy="154" r="96" fill="#f4c184" fill-opacity="0.35"/>'
    '<circle cx="196" cy="154" r="74" fill="#f8d79a" fill-opacity="0.45"/>'
    '<circle cx="196" cy="154" r="58" fill="#fbe6b4"/>'
    "</g>"
    '<g id="hills">'
    '<path d="M0 294 L116 210 L238 294 Z" fill="#443160"/>'
    '<path d="M146 294 L278 192 L384 294 Z" fill="#251d3a"/>'
    "</g>"
    '<rect x="0" y="304" width="384" height="80" fill="#151128"/>'
    "</svg>"
)

_SUNSET_3 = (
    f"{HEAD}<defs>"
    + _linear("sky", "#ffbf78", "#764a94")
    + '<radialGradient id="glow" cx="0.5" cy="0.5" r="0.5">'
    '<stop offset="0.3" stop-color="#fff0c4"/>'
    '<stop offset="1" stop-color="#ffb070" stop-opacity="0"/>'
    "</radialGradient>"
    "</defs>"
    f'<rect x="0" y="0" width="{SIDE}" height="{SIDE}" fill="url(#sky)"/>'
    '<rect x="70" y="26" width="248" height="248" fill="url(#glow)"/>'
    '<ellipse cx="184" cy="140" rx="62" ry="46" fill="#ffeeb6"/>'
    '<rect x="0" y="242" width="384" height="64" fill="#c47fae" opacity="0.6"/>'
    '<g id="hills">'
    '<path d="M0 304 L104 226 L104 304 Z" fill="#443563"/>'
    '<path d="M104 226 L222 304 L104 304 Z" fill="#443563"/>'
    '<path d="M158 304 L262 208 L262 304 Z" fill="#2f2648"/>'
    '<path d="M262 208 L384 304 L262 304 Z" fill="#2f2648"/>'
    "</g>"
    '<path d="M0 304 L384 304 L384 384 L0 384 Z" fill="#2c2646"/>'
    "</svg>"
)

_SUNSET_4 = (
    f"{HEAD}<defs>"
    + _linear("skytop", "#ffa860", "#d08a86")
    + _linear("skylow", "#b07a92", "#61468f")
    + '<radialGradient id="glow" cx="0.5" cy="0.5" r="0.5">'
    '<stop offset="0.45" stop-color="#ffe6a4"/>'
    '<stop offset="1" stop-color="#ffa860" stop-opacity="0"/>'
    "</radialGradient>"
    "</defs>"
    '<rect x="0" y="0" width="384" height="160" fill="url(#skytop)"/>'
    '<rect x="0" y="160" width="384" height="224" fill="url(#skylow)"/>'
    '<circle cx="196" cy="142" r="104" fill="url(#glow)"/>'
    '<circle cx="196" cy="142" r="58" fill="#ffefb0"/>'
    '<path d="M0 296 L104 210 L104 296 L236 296 L104 210 Z" fill="#372a54"/>'
    '<path d="M152 296 L276 190 L384 296 Z" fill="#241c3c"/>'
    '<rect x="0" y="254" width="384" height="42" fill="#cf92b6" opacity="0.35"/>'
    '<rect x="0" y="296" width="384" height="88" fill="#1e1a32"/>'
    "</svg>"
)

_SUNSET_5 = (
    f"{HEAD}<defs>"
    + _linear("sky", "#f89a52", "#553c88")
    + _linear("soil", "#2a2450", "#0c0a1c")
    + '<radialGradient id="glow" cx="0.5" cy="0.5" r="0.5">'
    '<stop offset="0.42" stop-color="#ffe8b0"/>'
    '<stop offset="1" stop-color="#f6a862" stop-opacity="0"/>'
    "</radialGradient>"
    "</defs>"
    f'<rect x="0" y="0" width="{SIDE}" height="{SIDE}" fill="url(#sky)"/>'
    '<path d="M0 306 L118 226 L240 306 Z" fill="#4c3c6c"/>'
    '<path d="M154 306 L266 204 L384 306 Z" fill="#332a4c"/>'
    '<rect x="0" y="256" width="384" height="50" fill="#b884ac" '
    'fill-opacity="0.4"/>'
    '<rect x="0" y="306" width="384" height="78" fill="url(#soil)"/>'
    '<circle cx="188" cy="156" r="126" fill="url(#glow)"/>'
    f'<path d="{_circle_path(188, 156, 46)}" fill="#ffeeb6"/>'
    "</svg>"
)

# (dot x, dot y, label x, label y, label)
_DOTS = [
    (70.0, 150.0, 57.6, 146.1, "1"),
    (97.9, 138.8, 86.3, 132.9, "2"),
    (118.9, 117.7, 109.4, 108.8, "3"),
    (140.4, 119.3, 132.3, 109.1, "4"),
    (154.9, 143.9, 146.3, 134.2, "5"),
    (165.0, 170.2, 153.9, 163.3, "6"),
    (192.4, 182.6, 186.8, 170.8, "7"),
    (221.2, 190.2, 234.2, 190.7, "8"),
    (250.7, 195.8, 263.6, 197.3, "9"),
    (279.0, 185.7, 291.9, 185.1, "10"),
    (307.0, 176.8, 319.9, 175.4, "11"),
    (329.9, 196.1, 342.9, 196.7, "12"),
    (303.4, 210.1, 316.1, 212.6, "13"),
    (281.8, 230.4, 293.5, 236.0, "14"),
    (258.5, 247.5, 268.0, 256.3, "15"),
    (229.7, 256.1, 235.6, 267.7, "16"),
    (201.3, 258.3, 202.3, 271.2, "17"),
    (173.7, 246.4, 169.0, 258.6, "18"),
    (159.0, 221.9, 149.2, 230.5, "19"),
    (145.7, 195.3, 132.8, 196.8, "20"),
    (124.6, 174.1, 111.9, 171.4, "21"),
    (98.5, 159.4, 86.0, 155.6, "22"),
]
_EYE = (153.0, 121.0)


def _dots_1() -> str:
    """Round marks, but every two-digit number is set as two <text>s whose
    spacing never closes up into one numeral."""
    marks = "".join(
        f'<circle cx="{_n(x + 4)}" cy="{_n(y - 3)}" r="4" fill="#2e2e2e"/>'
        for x, y, _lx, _ly, _t in _DOTS
    )
    labels = []
    for _x, _y, lx, ly, t in _DOTS:
        if len(t) == 1:
            labels.append(_text(lx - 4, ly + 8, 14, "#2e2e2e", t))
        else:
            labels.append(_text(lx - 8, ly + 8, 14, "#2e2e2e", t[0]))
            labels.append(_text(lx + 1, ly + 8, 14, "#2e2e2e", t[1]))
    return (
        f"{HEAD}{_bg('#ffffff')}"
        f'<g id="dots">{marks}</g>'
        f'<g id="labels">{"".join(labels)}</g>'
        '<g id="eye"><circle cx="157" cy="118" r="4" fill="#2e2e2e"/></g>'
        "</svg>"
    )


def _dots_2() -> str:
    """One group per dot, each holding its mark and its number. Stuck: the
    marks are <rect>s with no rx and stay square."""
    body = "".join(
        f'<g id="d{t}">'
        f'<rect x="{_n(x - 5)}" y="{_n(y + 2)}" width="7" height="6" '
        'fill="#222222"/>' + _text(lx - 5, ly + 4, 12, "#222222", t) + "</g>"
        for x, y, lx, ly, t in _DOTS
    )
    return (
        f"{HEAD}{_bg('#fafafa')}"
        f"{body}"
        f'<rect x="{_n(_EYE[0] - 5)}" y="{_n(_EYE[1] + 2)}" width="7" '
        'height="6" fill="#222222"/>'
        "</svg>"
    )


def _dots_3() -> str:
    """Numbers first, then rounded squares under a radial gradient the target
    paints flat -- round marks and whole numbers, loose on size."""
    labels = "".join(
        _text(lx + 6, ly + 10, 9, "#4a4a4a", t) for _x, _y, lx, ly, t in _DOTS
    )
    marks = "".join(
        f'<rect x="{_n(x + 3)}" y="{_n(y + 2)}" width="8" height="8" rx="4" '
        'ry="4" fill="url(#ink)"/>'
        for x, y, _lx, _ly, _t in _DOTS
    )
    return (
        f"{HEAD}<defs>{_radial('ink', '#6e6e6e', '#111111', mid='0.2')}</defs>"
        f"{_bg('#fdfdfd')}"
        f'<g id="labels">{labels}</g>'
        f'<g id="marks">{marks}</g>'
        '<rect x="156" y="123" width="8" height="8" rx="4" ry="4" '
        'fill="url(#ink)"/>'
        "</svg>"
    )


def _dots_4() -> str:
    """Dots as small cubic-path discs, numbers whole but far too large and
    drawn in reverse document order."""
    marks = "".join(
        f'<path d="{_circle_path(x - 7, y - 4, 3.6)}" fill="#1f1f1f"/>'
        for x, y, _lx, _ly, _t in _DOTS
    )
    labels = "".join(
        _text(lx + 3, ly + 3, 18, "#1f1f1f", t) for _x, _y, lx, ly, t in reversed(_DOTS)
    )
    return (
        f"{HEAD}{_bg('#ffffff')}"
        f'<path d="{_circle_path(_EYE[0] - 7, _EYE[1] - 4, 3.6)}" fill="#1f1f1f"/>'
        f"{marks}{labels}"
        "</svg>"
    )


def _dots_5() -> str:
    """Two halves of the puzzle, each its own group. Stuck: the marks are
    stroked rings with a gradient stroke and never fill in solid."""

    def half(rows, gid):
        body = "".join(
            f'<circle cx="{_n(x + 3)}" cy="{_n(y + 7)}" r="3" fill="none" '
            'stroke="url(#ring)" stroke-width="2"/>'
            + _text(lx + 5, ly + 2, 12, "#333333", t)
            for x, y, lx, ly, t in rows
        )
        return f'<g id="{gid}">{body}</g>'

    return (
        f"{HEAD}<defs>{_linear('ring', '#5c5c5c', '#0e0e0e')}</defs>"
        f"{_bg('#f7f7f5')}"
        + half(_DOTS[:11], "first")
        + half(_DOTS[11:], "second")
        + '<g id="eye"><circle cx="156" cy="128" r="3" fill="none" '
        'stroke="url(#ring)" stroke-width="2"/></g>'
        "</svg>"
    )

_LEAF_1 = (
    f"{HEAD}{_bg('#f0f2ea')}"
    '<g id="blade">'
    '<path d="M198 62 C284 104 296 198 202 312 C202 312 96 202 112 106 Z" '
    'fill="#619b4d"/>'
    '</g><g id="veins">'
    '<path d="M198 306 C206 226 202 146 198 76" fill="none" stroke="#3f6b34" '
    'stroke-width="8"/>'
    '<path d="M198 238 C158 220 136 192 128 156" fill="none" stroke="#3f6b34" '
    'stroke-width="6"/>'
    "</g></svg>"
)

_LEAF_2 = (
    f"{HEAD}<defs>{_linear('blade', '#4f9040', '#2c5c24', vertical=False)}</defs>"
    f"{_bg('#eef1e6')}"
    '<g id="blade">'
    '<path d="M190 52 C288 94 306 192 194 322 Z" fill="url(#blade)"/>'
    '<path d="M194 322 C88 200 94 100 108 96 L190 52 Z" fill="url(#blade)"/>'
    '</g><g id="veins">'
    '<line x1="190" y1="304" x2="196" y2="66" stroke="#28502a" '
    'stroke-width="3"/>'
    '<path d="M190 228 Q146 202 122 148" fill="none" stroke="#28502a" '
    'stroke-width="9"/>'
    "</g></svg>"
)

_LEAF_3 = (
    f"{HEAD}{_bg('#f4f6ee')}"
    '<g id="blade">'
    '<ellipse cx="196" cy="258" rx="72" ry="60" fill="#589040"/>'
    '<ellipse cx="196" cy="196" rx="96" ry="70" fill="#589040"/>'
    '<ellipse cx="190" cy="132" rx="64" ry="58" fill="#589040"/>'
    '</g><g id="veins">'
    '<path d="M190 294 C200 226 194 148 190 74" fill="none" stroke="#33602a" '
    'stroke-width="2"/>'
    '<path d="M190 226 C150 210 130 182 122 148" fill="none" stroke="#33602a" '
    'stroke-width="2"/>'
    '</g><g id="tips">'
    '<path d="M140 260 L200 334 L252 256 Z" fill="#589040"/>'
    '<path d="M128 112 L192 46 L258 116 Z" fill="#589040"/>'
    "</g></svg>"
)

_LEAF_4 = (
    f"{HEAD}<defs>{_radial('lens', '#5fae42', '#255a1e', mid='0.25')}</defs>"
    f"{_bg('#f6f7f1')}"
    '<g id="blade">'
    '<path d="M192 52 C282 118 292 226 196 322 C100 226 110 118 192 52 Z" '
    'fill="url(#lens)"/>'
    '</g><g id="veins">'
    '<line x1="194" y1="302" x2="198" y2="226" stroke="#2c5824" '
    'stroke-width="10"/>'
    '<line x1="198" y1="226" x2="196" y2="146" stroke="#2c5824" '
    'stroke-width="10"/>'
    '<line x1="196" y1="146" x2="192" y2="66" stroke="#2c5824" '
    'stroke-width="10"/>'
    '<line x1="194" y1="236" x2="150" y2="204" stroke="#2c5824" '
    'stroke-width="4"/>'
    '<line x1="150" y1="204" x2="124" y2="154" stroke="#2c5824" '
    'stroke-width="4"/>'
    "</g></svg>"
)

_LEAF_5 = (
    f"{HEAD}{_bg('#f8f9f0')}"
    '<path d="M190 60 Q306 152 198 314 Q88 190 116 104 L190 60 Z" '
    'fill="#568a46" fill-opacity="0.85"/>'
    '<path d="M188 238 Q154 212 130 166" fill="none" stroke="#4a7a3c" '
    'stroke-width="2"/>'
    '<path d="M188 296 Q196 220 192 138 L190 72" fill="none" stroke="#4a7a3c" '
    'stroke-width="6"/>'
    "</svg>"
)


SEEDS: dict[str, list[str]] = {
    "mascot": [_MASCOT_1, _MASCOT_2, _MASCOT_3, _MASCOT_4, _MASCOT_5],
    "wordmark": [
        _WORDMARK_1,
        _WORDMARK_2,
        _WORDMARK_3,
        _WORDMARK_4,
        _WORDMARK_5,
    ],
    "emblem": [_EMBLEM_1, _EMBLEM_2, _EMBLEM_3, _EMBLEM_4, _EMBLEM_5],
    "sunset": [_SUNSET_1, _SUNSET_2, _SUNSET_3, _SUNSET_4, _SUNSET_5],
    "connect-dots": [_dots_1(), _dots_2(), _dots_3(), _dots_4(), _dots_5()],
    "leaf": [_LEAF_1, _LEAF_2, _LEAF_3, _LEAF_4, _LEAF_5],
}
