"""Joint fill fitting respects the real SVG context and editor constraints."""

import xml.etree.ElementTree as ET
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from cairosvg.colors import color

from vectrify.document import export_svg, import_svg
from vectrify.document.join import path_style
from vectrify.document.transforms import root_matrix
from vectrify.refine.crossings import crossings
from vectrify.refine.joint import _context, _groups, polish
from vectrify.refine.paths import _composite_in_context
from vectrify.refine.selected import FitOptions
from vectrify.refine.simplify import curved
from vectrify.svg_render import render_image

SVG = """<svg width="64" height="64">
<defs><clipPath id="clip"><path d="M8 8 H55 V55 H8 Z"/></clipPath></defs>
BACKGROUND
<g opacity="0.7" clip-path="url(#clip)" transform="translate(1 2)">
<path id="a" fill="maroon" fill-opacity="0.6" transform="translate(2 1)"
 d="M8 8 L35 8 L35 38 L8 38 Z"/>
<path id="b" fill="#38b" opacity="0.8" fill-rule="evenodd"
 d="M25 20 L58 20 L58 50 L25 50 Z M32 28 L40 28 L40 35 L32 35 Z"/>
</g><rect id="front" x="30" y="25" width="30" height="10" fill="blue"/></svg>"""


@pytest.mark.parametrize("transparent", [False, True])
@pytest.mark.parametrize("stroke", [False, True])
def test_joint_compositing_matches_cairo_with_clipping_group_opacity_and_front_paint(
    transparent,
    stroke,
):
    torch = pytest.importorskip("torch")
    svg = SVG.replace(
        "BACKGROUND",
        "" if transparent else '<rect width="64" height="64" fill="#eef"/>',
    )
    if stroke:
        svg = svg.replace(
            'fill="#38b"',
            'fill="none" stroke="#38b" stroke-width="0.7" '
            'stroke-linejoin="round" stroke-linecap="round"',
        )
    document = import_svg(svg)
    target = render_image(svg, (0, 0, 64, 64), (64, 64), alpha=transparent)
    context = _context(document, ("a", "b"), target, FitOptions())
    left, top, right, bottom = context.crop
    box = (left, top, right - left, bottom - top)
    alphas, colours = [], []
    # Independent Cairo coverages include each path's own opacity/transform.
    # Common ancestry effects are supplied only by the production context.
    for oid in ("a", "b"):
        element = document.element(oid)
        geometry = export_svg(document)

        root = ET.fromstring(geometry)
        path = next(e for e in root.iter() if e.get("id") == oid)
        isolated = ET.Element("svg", width="64", height="64")
        attrs = dict(path.attrib)
        attrs["transform"] = (
            "matrix(" + " ".join(str(v) for v in root_matrix(document, oid)) + ")"
        )
        isolated.append(ET.Element("path", attrs))
        image = render_image(
            ET.tostring(isolated, encoding="unicode"),
            box,
            context.size,
            alpha=True,
        )
        alphas.append(np.asarray(image)[..., 3] / 255)
        style = path_style(document, element)
        colours.append(
            color(style["stroke"] if style["fill"] == "none" else style["fill"])[:3]
        )
    alphas = torch.tensor(np.array(alphas), dtype=torch.float32, requires_grad=True)
    reconstructed = _composite_in_context(
        alphas,
        torch.tensor(colours, dtype=torch.float32),
        tuple(
            torch.tensor(a) for a in (context.base, context.delta, context.transmission)
        ),
    )
    expected = context.array(render_image(svg, box, context.size, alpha=transparent))
    assert np.max(abs(reconstructed.detach().numpy() - expected)) < 0.025
    reconstructed.sum().backward()
    assert torch.isfinite(alphas.grad).all()
    assert alphas.grad[0].abs().sum() > 0
    assert alphas.grad[1].abs().sum() > 0


PAIR = """<svg width="64" height="64"><g id="g">
<path id="a" fill="#a23" d="M10 10 L30 10 L30 50 L10 50 Z"/>
<path id="b" fill="#38b" d="M34 10 L54 10 L54 50 L34 50 Z"/>
</g></svg>"""

SHARED = """<svg width="64" height="64"><g>
<path id="a" fill="maroon" fill-opacity="0"
 d="M8 8 L48 8 L48 24 L28 24 L8 24 Z"/>
<path id="b" fill="navy" d="M8 24 L28 24 L48 24 L48 48 L8 48 Z"/>
</g></svg>"""


@pytest.mark.parametrize("reverse_links", [False, True])
def test_joint_shared_edge_uses_the_visible_neighbours_gradient(reverse_links):
    """An invisible selected copy must not overwrite its visible neighbour."""
    pytest.importorskip("torch")
    from vectrify.refine.shared import follow, frozen_points, links

    document = import_svg(SHARED)
    for oid in ("a", "b"):
        document = document.replace_geometry(curved(document.geometry_for(oid)))
    shared = links(document, ("a", "b"), ("a", "b"))
    assert len(shared) == 2
    if reverse_links:
        shared.reverse()
    middle = next(
        n
        for n in document.geometry_for("b").subpaths[0].nodes
        if n.endpoint == (28, 24)
    )
    held = frozenset(
        node.id
        for oid in ("a", "b")
        for subpath in document.geometry_for(oid).subpaths
        for node in subpath.nodes
        if node.endpoint != (28, 24)
    ) | frozen_points(document, shared)
    geometry = document.geometry_for("b")
    target_document = document.replace_geometry(
        geometry.replace_node(replace(middle, values=(*middle.values[:-2], 28, 26)))
    )
    target = render_image(
        export_svg(target_document), (0, 0, 64, 64), (64, 64), alpha=True
    )
    result = polish(
        document,
        ("a", "b"),
        target,
        FitOptions(snap=False, handles=False, steps=20, displacement=2, resolution=64),
        held=held,
        shared=tuple(shared),
    )
    assert result.geometry_for("b").node(middle.id).endpoint[1] > 24.5
    # Either directed following order is now redundant: the fitted edge is
    # already shared, with the original IDs and no changes to held points.
    assert follow(result, shared)[0] == result
    for oid in ("a", "b"):
        original = document.geometry_for(oid)
        fitted = result.geometry_for(oid)
        for subpath in original.subpaths:
            for node in subpath.nodes:
                updated = fitted.node(node.id)
                if node.id in held:
                    # Opposite line-to-cubic conversions can differ by one
                    # double rounding unit when following the shared edge.
                    np.testing.assert_allclose(
                        updated.values, node.values, atol=1e-12, rtol=0
                    )
                assert (
                    np.linalg.norm(
                        np.asarray(updated.endpoint) - np.asarray(node.endpoint)
                    )
                    <= 2.00001
                )
    pixels = np.asarray(
        render_image(export_svg(result), (0, 0, 64, 64), (64, 64), alpha=True),
        dtype=float,
    )
    before = np.asarray(
        render_image(export_svg(document), (0, 0, 64, 64), (64, 64), alpha=True),
        dtype=float,
    )
    expected = np.asarray(target, dtype=float)
    assert np.mean((pixels - expected) ** 2) < np.mean((before - expected) ** 2) * 0.8


@pytest.mark.parametrize("pinned_path", ["a", "b"])
def test_joint_shared_point_obeys_a_pin_on_either_copy(pinned_path):
    pytest.importorskip("torch")
    from vectrify.refine.shared import frozen_points, links

    document = import_svg(SHARED)
    for oid in ("a", "b"):
        document = document.replace_geometry(curved(document.geometry_for(oid)))
    geometry = document.geometry_for(pinned_path)
    middle = next(n for n in geometry.subpaths[0].nodes if n.endpoint == (28, 24))
    document = document.replace_geometry(
        geometry.replace_node(replace(middle, pinned=True))
    )
    shared = links(document, ("a", "b"), ("a", "b"))
    held = frozenset(
        node.id
        for oid in ("a", "b")
        for subpath in document.geometry_for(oid).subpaths
        for node in subpath.nodes
        if node.endpoint != (28, 24)
    ) | frozen_points(document, shared)
    target = render_image(
        SHARED.replace("L28 24", "L28 26"), (0, 0, 64, 64), (64, 64), alpha=True
    )
    result = polish(
        document,
        ("a", "b"),
        target,
        FitOptions(snap=False, handles=False, steps=10, displacement=2, resolution=64),
        held=held,
        shared=tuple(shared),
    )
    assert result == document


@pytest.mark.parametrize("opacity", [1.0, 0.6])
@pytest.mark.parametrize("reference_alpha", [255, 253])
def test_joint_fit_closes_a_thin_transparent_seam_between_white_fills(
    opacity, reference_alpha
):
    """Opacity, rather than RGB over white, supplies the fitting direction."""
    pytest.importorskip("torch")
    svg = """<svg width="64" height="64"><g>
    <path id="a" fill="white" d="M8 8 L31.8 8 L31.8 56 L8 56 Z"/>
    <path id="b" fill="white" d="M32.2 8 L56 8 L56 56 L32.2 56 Z"/>
    </g></svg>"""
    svg = svg.replace('fill="white"', f'fill="white" fill-opacity="{opacity}"')
    document = import_svg(svg)
    target = render_image(
        svg.replace("31.8", "33").replace("32.2", "31"),
        (0, 0, 64, 64),
        (64, 64),
        alpha=True,
    )
    if reference_alpha != 255:
        pixels = np.array(target)
        pixels[..., 3] = np.rint(pixels[..., 3].astype(float) * reference_alpha / 255)
        from PIL import Image

        target = Image.fromarray(pixels)
    result = polish(
        document,
        ("a", "b"),
        target,
        FitOptions(snap=False, steps=20, displacement=2, resolution=64),
    )
    before = np.asarray(
        render_image(export_svg(document), (0, 0, 64, 64), (64, 64), alpha=True)
    )
    after = np.asarray(
        render_image(export_svg(result), (0, 0, 64, 64), (64, 64), alpha=True)
    )
    seam = (slice(12, 52), slice(31, 33))
    expected = np.asarray(target)[seam][..., 3].astype(float)
    before_error = ((expected - before[seam][..., 3].astype(float)) ** 2).sum()
    after_error = ((expected - after[seam][..., 3].astype(float)) ** 2).sum()
    assert after_error < before_error * 0.2
    for oid in ("a", "b"):
        assert result.element(oid).attributes == document.element(oid).attributes


def test_joint_fit_moves_a_join_when_only_its_outgoing_curve_is_visible():
    """The hidden incoming endpoint must not discard the visible side's signal."""
    pytest.importorskip("torch")
    svg = """<svg width="64" height="64"><g>
    <path id="a" fill="maroon" d="M8 23 L24 23 L24 48 L8 48 Z"/>
    <path id="b" fill="navy" d="M0 0 L64 0 L64 26 L0 26 Z"/>
    </g></svg>"""
    document = import_svg(svg)
    corner = document.geometry_for("a").subpaths[0].nodes[1]
    held = frozenset(
        node.id
        for oid in ("a", "b")
        for subpath in document.geometry_for(oid).subpaths
        for node in subpath.nodes
        if node.id != corner.id
    )
    target = render_image(
        svg.replace("L24 23", "L26 23"), (0, 0, 64, 64), (64, 64), alpha=True
    )
    result = polish(
        document,
        ("a", "b"),
        target,
        FitOptions(snap=False, handles=False, steps=20, displacement=2, resolution=64),
        held=held,
    )
    moved = result.geometry_for("a").node(corner.id).endpoint
    assert moved[0] > corner.endpoint[0] + 0.5
    before = np.asarray(
        render_image(export_svg(document), (0, 0, 64, 64), (64, 64), alpha=True),
        dtype=float,
    )
    after = np.asarray(
        render_image(export_svg(result), (0, 0, 64, 64), (64, 64), alpha=True),
        dtype=float,
    )
    expected = np.asarray(target, dtype=float)
    assert np.mean((after - expected) ** 2) < np.mean((before - expected) ** 2) * 0.8
    for oid in ("a", "b"):
        original = document.geometry_for(oid)
        fitted = result.geometry_for(oid)
        assert {n.id for s in original.subpaths for n in s.nodes} == {
            n.id for s in fitted.subpaths for n in s.nodes
        }
        for subpath in original.subpaths:
            for node in subpath.nodes:
                if node.id in held:
                    assert fitted.node(node.id).endpoint == node.endpoint


def test_joint_fit_closes_a_gap_without_changing_paint_structure_or_pinned_points():
    pytest.importorskip("torch")
    document = import_svg(PAIR)
    first = document.geometry_for("a")
    held = first.subpaths[0].nodes[-1].id
    second = document.geometry_for("b")
    pin = second.subpaths[0].nodes[1]
    document = document.replace_geometry(
        replace(
            second,
            subpaths=(
                replace(
                    second.subpaths[0],
                    nodes=tuple(
                        replace(n, pinned=True) if n.id == pin.id else n
                        for n in second.subpaths[0].nodes
                    ),
                ),
            ),
        )
    )
    target_svg = (
        PAIR.replace("L30 10 L30 50", "L32 10 L32 50")
        .replace("M34 10", "M32 10")
        .replace("L34 50", "L32 50")
    )
    target = render_image(target_svg, (0, 0, 64, 64), (64, 64), alpha=True)

    def score(candidate):
        pixels = np.asarray(
            render_image(export_svg(candidate), (0, 0, 64, 64), (64, 64), alpha=True),
            dtype=float,
        )
        # Independent exact alpha error detects the original transparent seam.
        return np.mean((pixels[..., 3] - np.asarray(target)[..., 3]) ** 2)

    result = polish(
        document,
        ("b", "a"),
        target,
        FitOptions(steps=20, displacement=2, resolution=64),
        held=frozenset({held}),
        score=score,
    )
    assert score(result) < score(document) * 0.7
    assert result.root == document.root
    for oid in ("a", "b"):
        assert crossings(result.geometry_for(oid)) <= crossings(
            document.geometry_for(oid)
        )
        curved_nodes = {
            n.id: n
            for sub in curved(document.geometry_for(oid)).subpaths
            for n in sub.nodes
        }
        original_nodes = {
            n.id: n for s in document.geometry_for(oid).subpaths for n in s.nodes
        }
        result_nodes = {
            n.id: n for s in result.geometry_for(oid).subpaths for n in s.nodes
        }
        assert result_nodes.keys() == original_nodes.keys()
        for nid, node in result_nodes.items():
            original = original_nodes[nid]
            distance = np.linalg.norm(np.array(node.values[-2:]) - original.values[-2:])
            assert distance <= 2.0001
            if nid == held:
                assert node.values == curved_nodes[nid].values
            if original.pinned:
                assert node.values[-2:] == original.values[-2:]


def test_only_compatible_contiguous_selected_siblings_form_a_joint_run():
    document = import_svg(
        PAIR.replace(
            "</g>",
            '<rect id="front" width="2" height="2"/>'
            '<path id="c" fill="#333" d="M1 1 L5 1 L5 5 Z"/></g>',
        )
    )
    assert _groups(document, ("c", "b", "a"), FitOptions()) == [("a", "b")]
    assert _groups(document, ("a", "c"), FitOptions()) == []
    stroke = document.replace_element(
        replace(
            document.element("b"),
            attributes=(*document.element("b").attributes, ("stroke", "black")),
        )
    )
    assert _groups(stroke, ("a", "b"), FitOptions()) == []
    singular = document.replace_element(
        replace(
            document.element("b"),
            attributes=(*document.element("b").attributes, ("transform", "scale(0)")),
        )
    )
    assert _groups(singular, ("a", "b"), FitOptions()) == []
    stopped = Event()
    stopped.set()
    assert polish(document, ("a", "b"), None, FitOptions(), stop=stopped) == document


def test_an_empty_joint_contour_keeps_the_individually_verified_document():
    pytest.importorskip("torch")
    document = import_svg(PAIR.replace("M10 10 L30 10 L30 50 L10 50 Z", "M10 10"))
    target = render_image(PAIR, (0, 0, 64, 64), (64, 64), alpha=True)
    assert polish(document, ("a", "b"), target, FitOptions()) == document


def test_joint_fit_moves_a_fill_and_open_round_outline_together():
    pytest.importorskip("torch")
    svg = (
        '<svg width="64" height="64"><g id="g">'
        '<path id="fill" fill="#a23" d="M10 10 L31.5 10 L31.5 50 L10 50 Z"/>'
        '<path id="outline" fill="none" stroke="black" stroke-width="0.7" '
        'stroke-linecap="round" stroke-linejoin="round" d="M32.6 10 L32.6 50"/>'
        "</g></svg>"
    )
    target_svg = svg.replace("31.5", "32").replace("32.6", "32")
    target = render_image(target_svg, (0, 0, 64, 64), (64, 64), alpha=True)
    document = import_svg(svg)
    result = polish(
        document,
        ("fill", "outline"),
        target,
        FitOptions(steps=30, resolution=64, displacement=2),
    )
    assert result.root == document.root
    for oid in ("fill", "outline"):
        assert result.geometry_for(oid) != document.geometry_for(oid)
        assert {n.id for s in result.geometry_for(oid).subpaths for n in s.nodes} == {
            n.id for s in document.geometry_for(oid).subpaths for n in s.nodes
        }
    assert not result.geometry_for("outline").subpaths[0].closed
    context = _context(document, ("fill", "outline"), target, FitOptions())
    left, top, right, bottom = context.crop
    box = left, top, right - left, bottom - top

    def error(d):
        pixels = context.array(
            render_image(export_svg(d), box, context.size, alpha=True)
        )
        expected = context.array(context.target)
        squared = (pixels - expected) ** 2
        return np.mean((squared[..., :3].sum(-1) + 3 * squared[..., 3]) / 6)

    assert error(result) < error(document) * 0.65
