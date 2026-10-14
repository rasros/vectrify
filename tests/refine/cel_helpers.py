"""Shared builders for the cel-plan tests."""

from dataclasses import replace

import numpy as np
import pathops
from PIL import Image

from vectrify.document import (
    Editor,
    Geometry,
    PathNode,
    Selection,
    Subpath,
    export_svg,
    import_svg,
)
from vectrify.document.join import curve_path, path_geometry
from vectrify.document.redraw import root_matrix
from vectrify.refine import cel
from vectrify.refine.cel_plan.attached_outlines import AttachedOutlines
from vectrify.refine.cel_plan.core_cells import CoreCells
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.export import export
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.frontier import Frontier
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.line_fidelity import SourceLineGuard, SourceProfile
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Evidence, Options, Work
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.proposals import bounds
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import Proposal, State, identity
from vectrify.refine.tracing import _loops


def one_atom():
    evidence = stripes(hole=True)
    return replace(evidence, labels=np.where(evidence.empty, 0, 1).astype(np.int32))


def capacity_source():
    """54 mixed parents need 66 binary cuts, but only 120 direct children."""
    shape = (24, 162)
    target = np.full((*shape, 3), 180, dtype=np.float32)
    rgba = np.concatenate((target / 255, np.full((*shape, 1), 0.5)), axis=-1)
    zero = np.zeros(shape, np.float32)
    evidence = replace(
        stripes(),
        rgba=rgba,
        target=target,
        smooth=target,
        coarse=target,
        empty=zero.astype(bool),
        drawn=zero.astype(bool),
        line=zero.astype(bool),
        foreground=np.ones(shape, bool),
        darkness=zero,
        texture=zero,
        labels=np.broadcast_to(np.arange(162)[None, :] // 3, shape)
        .copy()
        .astype(np.int32),
        source_size=(162, 24),
        opacity=rgba[..., 3],
    )
    y, x = np.indices(shape)
    classes = np.where(x < 36, y // 8, np.where(y < 12, 0, 2)).astype(np.int32)
    return evidence, classes


def outline_setup():
    document, partition, evidence, guard = fixture()
    svg = export_svg(document)
    frontier = Frontier(Policy(evidence.rgba))
    assert frontier.add(
        svg, "Material parent", {"planning_surfaces": partition.metadata()}
    )
    frontier.freeze_normalizer()
    entry = frontier.baseline
    assert entry is not None
    state = State(
        document,
        svg,
        LocalPolicy(frontier.policy).start(svg, entry.evaluation),
        identity(svg, partition),
        entry.details,
        partition=partition,
    )
    options = Options(quality="high", refine=False)
    operator = AttachedOutlines(evidence, options, guard=lambda _work: guard)
    return state, frontier, evidence, options, guard, operator


OWNER = "M10 10 C25 10 40 25 50 30 H90 V70 H50 V34 C40 29 25 14 10 14 Z"


BASE = "M10 14 C25 14 40 29 50 34 H90 V80 H10 Z"


LINE = "M10 12 C25 12 40 27 50 32"


def fixture(*, opacity=1, transform="translate(0 0)", reverse=False):
    head = (
        '<svg width="160" height="160"><g id="component" '
        f'opacity="{opacity}" transform="{transform}">'
    )
    background = '<path id="bg" d="M0 0H160V160H0Z" fill="#b49150"/>'
    base = f'<path id="base" d="{BASE}" fill="#b49150"/>'
    bar = (
        '<path id="bar" d="M10 12L50 32" fill="none" stroke="#060301" '
        'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    document = import_svg(
        head
        + background
        + base
        + f'<path id="ink" d="{OWNER}" fill="#271b10"/>'
        + bar
        + "</g></svg>"
    )
    raw = render(
        head
        + background
        + base
        + '<path d="M50 34H90V70H50Z" fill="#271b10"/>'
        + f'<path d="{LINE}" fill="none" stroke="#060301" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        + bar
        + "</g></svg>",
        (160, 160),
    )
    evidence = collect(
        Image.fromarray(np.rint(raw * 255).astype(np.uint8)),
        None,
        Options(),
        Work.start(10),
    )
    t = np.linspace(0.2, 0.8, 25)[:, None]
    points = (
        (1 - t) ** 3 * [10, 12]
        + 3 * (1 - t) ** 2 * t * [25, 12]
        + 3 * (1 - t) * t**2 * [40, 27]
        + t**3 * [50, 32]
    )
    a, b, c, d, e, f = root_matrix(document, "ink")
    points = points @ np.array(((a, b), (c, d))) + (e, f)
    profiles = [
        SourceProfile.at(points, 2),
        SourceProfile.at(
            np.linspace((10, 12), (50, 32), 20) @ np.array(((a, b), (c, d))) + (e, f),
            1.5,
        ),
    ]
    if reverse:
        profiles.reverse()
    guard = SourceLineGuard(evidence.rgba, profiles)
    part = Partition(
        (Surface("bg", (1, 2), "underlay"), Surface("base", (1,)), Surface("ink", (2,)))
    )
    return document, part, evidence, guard


def material(alpha=128, hole=False):
    evidence = step(alpha, hole=hole)
    target, rgba = evidence.target.copy(), evidence.rgba.copy()
    _y, x = np.indices(evidence.labels.shape)
    target[(x >= 70) & ~evidence.empty] = (60, 100, 200)
    target[(x >= 35) & (x < 70) & ~evidence.empty] = (180, 100, 60)
    rgba[~evidence.empty, :3] = target[~evidence.empty] / 255
    return replace(
        evidence,
        target=target,
        smooth=target,
        coarse=target,
        rgba=rgba,
        opacity=rgba[..., 3],
    )


def prepared(evidence, *, layers=False):
    options = Options(refine=False)
    svg, details = export(
        evidence,
        evidence.labels,
        options,
        Work.start(10),
        layers=layers,
        conservative=evidence.opacity is None,
    )
    policy = Policy(evidence.rgba)
    frontier = Frontier(policy)
    assert frontier.add(svg, "Partitioned", details)
    frontier.freeze_normalizer()
    entry = frontier.entries[0]
    state = State(
        import_svg(svg),
        svg,
        LocalPolicy(policy).start(svg, entry.evaluation),
        entry.key,
        details,
        partition=Partition.from_metadata(details["planning_surfaces"]),
    )
    return frontier, state, options


def fragmented(gap=False):
    evidence = material(128, hole=False)
    y, x = np.indices(evidence.labels.shape)
    ink = (y >= 18) & (y < 24) & (x >= 15) & (x < 81)
    if gap:
        ink &= (x < 46) | (x >= 50)
    rgb, rgba, labels = (
        evidence.target.copy(),
        evidence.rgba.copy(),
        evidence.labels.copy(),
    )
    rgb[ink], rgba[ink, :3] = 20, 20 / 255
    # Each source pixel has independent native ownership. One complete fitted
    # chain can replace those observations together, preserving real gaps.
    labels[ink] = np.arange(9, 9 + ink.sum())
    return replace(
        evidence,
        target=rgb,
        smooth=rgb,
        coarse=rgb,
        rgba=rgba,
        labels=labels,
        drawn=ink,
        line=ink,
    ), ink


def line():
    return np.column_stack((np.arange(10.5, 90), np.full(80, 40.5)))


def exterior_profile(alpha=1):
    target = np.full((80, 100, 3), 180.0)
    target[:40] = 255
    target[40:44, 8:92] = 15
    opacity = np.zeros((80, 100), float)
    opacity[40:] = alpha
    opacity[40] *= 0.2
    points = line().copy()
    points[:, 1] = 42.5
    return points, target, opacity


def ring(*, alpha=128, hole=False, gap=False, irregular=False):
    evidence = source(alpha=alpha)
    y, x = np.mgrid[:64, :96]
    dx, dy = (x + 0.5 - 48) / 24, (y + 0.5 - 32) / 20
    angle = np.arctan2(dy, dx)
    radius = np.hypot(dx, dy)
    if irregular:
        radius /= 1 + 0.14 * np.cos(7 * angle)
    inside = radius < 0.78
    ink = (radius < 1) & ~inside
    if gap:
        ink &= np.abs(angle) > 0.3
    labels = inside.astype(np.int32)
    labels[ink] = (
        2 + np.floor((angle[ink] + np.pi) * 12 / (2 * np.pi)).astype(np.int32) % 12
    )
    palette = np.array(
        [(220, 180, 100), (60, 150, 90), *[(12 + i, 8 + i, 5 + i) for i in range(12)]],
        dtype=np.float32,
    )
    target = palette[labels]
    empty = inside if hole else np.zeros(inside.shape, bool)
    opacity = (~empty).astype(np.float32) * (alpha / 255)
    rgba = np.concatenate((target / 255, opacity[..., None]), axis=-1)
    return replace(
        evidence,
        labels=labels,
        target=target,
        smooth=target,
        coarse=target,
        rgba=rgba,
        opacity=opacity,
        empty=empty,
        foreground=~empty,
        line=ink,
        drawn=ink,
        darkness=ink.astype(float) * 180,
        texture=np.zeros(ink.shape),
    )


def rims(factory, state):
    return [
        p
        for p in factory(state, Work.start(10))
        if p.details["ink_replacement"].get("boundary_model") == "source-rim"
    ]


def drawing(alpha=128, *, gap=False):
    line = "M20 28 L76 60 M20 60 L76 28"
    if gap:
        line = "M20 28 L42 28 M54 28 L76 28"
    svg = (
        '<svg width="96" height="96"><g opacity="'
        + str(alpha / 255)
        + '"><path d="M8 8 H88 V88 H8Z" fill="#ad8665"/>'
        '<path d="M48 8 H88 V88 H48Z" fill="#9d8065"/>'
        f'<path d="{line}" fill="none" stroke="#202020" '
        'stroke-width="3" stroke-linecap="round"/></g></svg>'
    )
    rgba = render(svg, (96, 96))
    image = Image.fromarray((rgba * 255).round().astype(np.uint8))
    options = Options(refine=False)
    evidence = collect(image, None, options, Work.start(10))
    ink = (~evidence.empty) & (cel.lightness(evidence.target) < 65)
    return evidence, ink


def source(*, alpha=128, tapered=False, gap=False):
    evidence = stripes(alpha=alpha)
    y, x = np.mgrid[:64, :96]
    labels = (y >= 32).astype(np.int32)
    width = 2 + (x - 8) / 8 if tapered else 4
    ink = (np.abs(y - 31.5) < width / 2) & (x >= 8) & (x < 88)
    if gap:
        ink &= (x < 38) | (x >= 58)
    labels[ink] = 2 + (x[ink] - 8) // 10
    palette = np.array(
        [
            (220, 180, 100),
            (160, 100, 70),
            *[(12 + i, 8 + i, 5 + i) for i in range(8)],
        ],
        dtype=np.float32,
    )
    target = palette[labels]
    opacity = np.full((64, 96), alpha / 255, dtype=np.float32)
    rgba = np.concatenate((target / 255, opacity[..., None]), axis=-1)
    empty = np.zeros(labels.shape, dtype=bool)
    return replace(
        evidence,
        rgba=rgba,
        target=target,
        smooth=target,
        coarse=target,
        labels=labels,
        empty=empty,
        foreground=~empty,
        line=ink,
        drawn=ink,
        darkness=ink.astype(float) * 180,
        texture=empty.astype(float),
        opacity=opacity,
    )


def evidence():
    y, x = np.mgrid[:120, :120]
    labels = (x >= 60).astype(np.int32)
    labels[(x - 60) ** 2 + (y - 60) ** 2 <= 12**2] = 2
    palette = np.array(
        ((200, 150, 100), (140, 100, 65), (60, 150, 110)), dtype=np.float32
    )
    target = palette[labels]
    rgba = np.concatenate((target / 255, np.ones((120, 120, 1))), axis=-1).astype(
        np.float32
    )
    clear = np.zeros(labels.shape, dtype=bool)
    zeros = np.zeros(labels.shape, dtype=float)
    return Evidence(
        rgba,
        target,
        target,
        target,
        clear,
        ~clear,
        clear,
        clear,
        zeros,
        zeros,
        labels,
        (120, 120),
        (0, 0),
        (1, 1),
        None,
        False,
    )


def fidelity_source(*, alpha=1, gap=False, shifted=0, blank=False):
    path = "M10 32H46 M50 32H86" if gap else f"M10 {32 + shifted}H86"
    return render(
        '<svg width="96" height="64">'
        f'<g opacity="{alpha}"><path d="M0 0H96V64H0Z" fill="#c4b79c"/>'
        + (
            ""
            if blank
            else f'<path d="{path}" fill="none" stroke="#202020" '
            'stroke-width="3" stroke-linecap="butt"/>'
        )
        + "</g></svg>",
        (96, 64),
    )


def overlay_source(*, alpha=128, hole=False, irregular=False, gradient=False):
    opaque = evidence()
    y, x = np.mgrid[:120, :120]
    own = opaque.labels == 2
    if irregular:
        own = (np.abs(x - 60) + np.abs(y - 60) <= 14) & ~((x < 60) & (y > 63))
    if hole:
        own[57:64, 57:64] = False
    labels = (x >= 60).astype(np.int32)
    labels[own] = 2 + np.clip((x[own] - 48) // 6, 0, 3)
    target = np.array(
        [(200, 150, 100), (140, 100, 65), *[(60, 150, 110)] * 4], dtype=np.float32
    )[labels]
    if gradient:
        target[own, 0] = 40 + 2 * (x[own] - 48)
    opacity = np.full(labels.shape, alpha / 255, dtype=np.float32)
    rgba = np.concatenate((target / 255, opacity[..., None]), axis=-1)
    return replace(
        opaque,
        labels=labels,
        target=target,
        smooth=target,
        coarse=target,
        rgba=rgba,
        opacity=opacity,
    )


def stripes(*, alpha=255, gradient=False, hole=False):
    height, width = 64, 96
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[8:56, 8:88] = (176, 100, 60, alpha)
    if gradient:
        pixels[8:56, 8:88, 0] = np.arange(80, dtype=np.uint8)[None, :] + 136
    if hole:
        pixels[24:40, 40:56] = 0
    shown = pixels[..., 3] > 0
    labels = np.where(shown, 1 + (np.arange(width)[None, :] - 8) // 10, 0).astype(
        np.int32
    )
    rgba = pixels.astype(np.float32) / 255
    target = np.where(shown[..., None], pixels[..., :3], 255).astype(np.float32)
    zero = np.zeros((height, width), dtype=np.float32)
    return Evidence(
        rgba,
        target,
        target.copy(),
        target.copy(),
        ~shown,
        shown,
        zero.astype(bool),
        zero.astype(bool),
        zero.copy(),
        zero.copy(),
        labels,
        (width, height),
        (0, 0),
        (1, 1),
        None,
        False,
        rgba[..., 3].copy() if alpha != 255 else None,
    )


def step(alpha=255, hole=False, diagonal=False, gradient=False):
    evidence = stripes(alpha=alpha, gradient=True, hole=hole)
    y, x = np.indices(evidence.labels.shape)
    side = (x + 0.5 + y + 0.5) / np.sqrt(2) < 56 if diagonal else x < 43
    target = evidence.target.copy()
    target[(~evidence.empty) & side] = (32, 100, 60)
    target[(~evidence.empty) & ~side] = (180, 100, 60)
    if gradient:
        target[(~evidence.empty) & side, 0] += 0.8 * x[(~evidence.empty) & side]
        target[(~evidence.empty) & ~side, 0] += 0.6 * y[(~evidence.empty) & ~side]
    rgba = evidence.rgba.copy()
    rgba[~evidence.empty, :3] = target[~evidence.empty] / 255
    return replace(evidence, target=target, smooth=target, coarse=target, rgba=rgba)


def marked_step(alpha=255):
    evidence = step(alpha)
    labels, target, rgba = (
        evidence.labels.copy(),
        evidence.target.copy(),
        evidence.rgba.copy(),
    )
    labels[24:40, 32:56] = 9
    target[24:40, 32:56] = (250, 220, 30)
    rgba[24:40, 32:56, :3] = np.array((250, 220, 30)) / 255
    return replace(
        evidence, labels=labels, target=target, smooth=target, coarse=target, rgba=rgba
    )


def indented_family(state):
    points = [(8.0, 56.0), (8.0, 10.0)]
    points.extend((float(x), 10.0 if x % 2 == 0 else 11.0) for x in range(9, 89))
    points.append((88.0, 56.0))
    wave = Geometry(
        "indentations",
        (
            Subpath(
                "indentations-loop",
                tuple(
                    PathNode(f"w{i}", "M" if i == 0 else "L", p)
                    for i, p in enumerate(points)
                ),
                True,
            ),
        ),
    )
    ids = tuple(
        s.id
        for s in state.partition.surfaces
        if s.role == "surface"
        and s.id not in state.details.get("paint_constraints", ())
    )
    editor = Editor(state.document, selection=Selection(whole_document=True))
    for oid in ids:
        clipped = pathops.op(
            curve_path(state.document.geometry_for(oid)),
            curve_path(wave),
            pathops.PathOp.INTERSECTION,
        )
        with editor.transaction("Fragment union with incidental dents") as tx:
            tx.replace_geometry(oid, path_geometry(clipped))
    changed = replace(state, document=editor.snapshot.document)
    return replace(changed, svg=export_svg(changed.document)), ids


def factory(evidence, options, **kwargs):
    return CoreCells(
        Families(evidence, build(evidence), options),
        options,
        joint=True,
        grouping="ward",
        ink_support="connected",
        ink_fit="source-intervals",
        **kwargs,
    )


INITIAL = (
    '<svg width="160" height="128"><defs>'
    '<linearGradient id="a"><stop offset="0" stop-color="#ac5030"/>'
    '<stop offset="1" stop-color="#b45030"/></linearGradient>'
    '<linearGradient id="b"><stop offset="0" stop-color="#40907c"/>'
    '<stop offset="1" stop-color="#409084"/></linearGradient></defs>'
    '<path id="left" fill="url(#a)" d="M8 8H72V120H8Z"/>'
    '<path id="right" fill="url(#b)" d="M88 8H152V120H88Z"/></svg>'
)


TARGET = INITIAL.replace('fill="url(#a)"', 'fill="#b05030"').replace(
    'fill="url(#b)"', 'fill="#409080"'
)


def setup(initial=INITIAL, target=TARGET, *, size=(160, 128)):
    options = Options(refine=False)
    truth = render(target, size)
    evidence = collect(
        Image.fromarray(np.rint(truth * 255).astype(np.uint8)),
        None,
        options,
        Work.start(10),
    )
    frontier = Frontier(Policy(truth))
    assert frontier.add(initial, "Initial")
    frontier.freeze_normalizer()
    return frontier, evidence, options


def color(document, oid, value):
    editor = Editor(document, selection=Selection(whole_document=True))
    with editor.transaction("Test proposal") as transaction:
        transaction.set_fill(oid, value)
    return editor.snapshot.document


def proposal(state, oid, value, *, operator="paint", parent=None):
    document = color(state.document, oid, value)
    return Proposal(
        operator,
        (oid,),
        (value,),
        parent or state.key,
        document,
        bounds(state.document, document, (oid,)),
    )


def circle(alpha=1, *, opaque=False, gap=False):
    svg = '<svg width="96" height="96">'
    if opaque:
        svg += '<path d="M0 0H96V96H0Z" fill="white"/>'
    svg += (
        f'<g opacity="{alpha}"><circle cx="48" cy="48" r="30" '
        'fill="#c4b79c" stroke="#202020" stroke-width="3"/></g></svg>'
    )
    rgba = render(svg, (96, 96))
    if gap:
        rgba[14:23, 45:51, :3] = np.array((196, 183, 156)) / 255
    evidence = collect(
        Image.fromarray(np.rint(rgba * 255).astype(np.uint8)),
        None,
        Options(refine=False),
        Work.start(10),
    )
    ink = ~evidence.empty & (evidence.target.max(-1) < 65)
    return replace(evidence, drawn=ink, line=ink)


def perimeter(evidence):
    mask = ~evidence.empty & evidence.foreground
    if evidence.opacity is not None:
        mask &= evidence.opacity >= evidence.opacity.max() * 0.5
    loop = _loops(mask)[0]
    return np.array([*loop, loop[0]], float)


def observed_gap():
    evidence, _mask = drawing(gap=True)
    points = np.column_stack((np.arange(20.25, 76.26, 4), np.full(15, 28.25)))
    return evidence, [SourceProfile.at(points, 3)]


def bands_fixture(*, alpha=1, diagonal=False, larger=False):
    side = 240 if larger else 96
    frame = (
        'transform="translate(70.12345 60.23456)"'
        if larger
        else 'transform="matrix(.8 .6 -.6 .8 20 -4)"'
        if diagonal
        else ""
    )
    source = (
        f'<svg width="{side}" height="{side}"><g opacity="{alpha}">'
        f'<path id="bg" d="M0 0H{side}V{side}H0Z" fill="#ad8665"/>'
        f'<path id="line" {frame} d="M20.25 28.25H76.25" fill="none" '
        'stroke="#121008" stroke-width="3" stroke-linecap="butt"/></g></svg>'
    )
    rgba = render(source, (side, side))
    evidence = collect(
        Image.fromarray(np.rint(rgba * 255).astype(np.uint8)),
        None,
        Options(),
        Work.start(10),
    )
    document = import_svg(
        source.replace('id="line"', 'id="ink"')
        .replace(
            'd="M20.25 28.25H76.25" fill="none"',
            'd="M20.25 25.75H76.25V30.75H20.25Z M70 70H72V72H70Z" fill="#121008"',
        )
        .replace('stroke="#121008"', 'stroke="none"')
    )
    anchors = np.column_stack((np.linspace(20.25, 76.25, 20), np.full(20, 28.25)))
    if diagonal or larger:
        a, b, c, d, e, f = root_matrix(document, "ink")
        anchors = anchors @ np.array([[a, b], [c, d]]) + (e, f)
    profile = SourceProfile.at(anchors, 3)
    guard = SourceLineGuard(evidence.rgba, (profile,))
    geometry = document.geometry_for("ink")
    main = replace(geometry, subpaths=geometry.subpaths[:1])
    marks = replace(geometry, subpaths=geometry.subpaths[1:])
    return evidence, guard, document, main, marks


def junction_drawing(alpha=255, *, gap=False):
    path = "M20 48H44 M50 48H76 M44 20V48 M50 48V76"
    path += " M44 48H46 M48 48H50" if gap else " M44 48H50"
    svg = (
        '<svg width="96" height="96"><g opacity="'
        + str(alpha / 255)
        + '"><path d="M8 8H88V88H8Z" fill="#ad8665"/>'
        f'<path d="{path}" stroke="#202020" stroke-width="2" '
        'fill="none" stroke-linecap="butt"/></g></svg>'
    )
    rgba = render(svg, (96, 96))
    evidence = collect(
        Image.fromarray((rgba * 255).round().astype(np.uint8)),
        None,
        Options(),
        Work.start(10),
    )
    mask = (~evidence.empty) & (cel.lightness(evidence.target) < 65)
    return evidence, mask


def ramp(alpha=255, hole=False):
    evidence = stripes(alpha=alpha, gradient=True, hole=hole)
    target = evidence.target.copy()
    shown = ~evidence.empty
    values = np.broadcast_to(16 + 3 * (np.arange(96) - 8), shown.shape)
    target[shown, 0] = values[shown]
    rgba = evidence.rgba.copy()
    rgba[..., :3][shown] = target[shown] / 255
    return replace(evidence, target=target, smooth=target, coarse=target, rgba=rgba)
