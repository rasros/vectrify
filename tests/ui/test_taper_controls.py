"""Tapers retain the centerline while selection and exported paint use its outline."""

import io
import xml.etree.ElementTree as ET

import cairosvg
import pytest
from PIL import Image
from shapely.geometry import Point

from tests.ui.helpers import send
from vectrify.document import DocumentError, HitIndex, import_svg
from vectrify.document.project import load_project, save_project
from vectrify.document.svg import export_svg
from vectrify.document.taper import END, OWNER, START
from vectrify.ui.agent import Agent
from vectrify.ui.session import Session

SVG = (
    '<svg width="100" height="100"><g stroke="red" stroke-width="10" '
    'stroke-opacity="0.5"><path id="p" fill="none" d="M10 50 L90 50"/></g></svg>'
)


def render(svg):
    data = cairosvg.svg2png(bytestring=svg.encode())
    assert isinstance(data, bytes)
    return Image.open(io.BytesIO(data)).convert("RGBA")


def pixel(image, x, y):
    value = image.getpixel((x, y))
    assert isinstance(value, tuple)
    return value


def tapered(start=20, end=20):
    session = Session(import_svg(SVG))
    send(session, "select", objects=["p"])
    send(session, "paint", changes={START: str(start), END: str(end)})
    return session


def test_tapers_preserve_nodes_undo_project_and_svg_roundtrip():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["p"])
    before = session.editor.snapshot.document
    send(session, "paint", changes={START: "20", END: "30"})
    document = session.editor.snapshot.document
    assert document.geometry_for("p") == before.geometry_for("p")
    assert HitIndex(document).painted_area("p") == pytest.approx(550)
    reopened, _ = load_project(save_project(document))
    assert reopened == document
    svg = export_svg(document)
    root = ET.fromstring(svg)
    [outline] = [node for node in root.iter() if node.get(OWNER)]
    assert outline.get("fill") == "red"
    assert outline.get("fill-opacity") == "0.5"
    reopened = import_svg(svg)
    assert len(reopened.elements()) == len(document.elements())
    assert (
        reopened.geometry_for("p").path_data() == document.geometry_for("p").path_data()
    )
    assert reopened.element("p").get("stroke-opacity") is None
    assert export_svg(reopened) == svg
    send(session, "undo")
    assert session.editor.snapshot.document == before
    send(session, "redo")
    assert session.editor.snapshot.document == document


def test_hits_contours_and_raster_match_tapered_width():
    document = tapered().editor.snapshot.document
    hit = HitIndex(document)
    area = hit.area("p")
    assert area is not None
    assert area.covers(Point(12, 50))
    assert not area.covers(Point(12, 53))
    assert not hit.contours_in("p", Point(12, 53).buffer(0.1))
    assert hit.contours_in("p", Point(50, 53).buffer(0.1))
    # Render using standard SVG support: no custom taper renderer is required.
    image = render(export_svg(document))
    assert pixel(image, 12, 53)[3] == 0
    assert 120 <= pixel(image, 50, 50)[3] <= 130
    assert pixel(image, 88, 53)[3] == 0


@pytest.mark.parametrize("cap", ["butt", "round", "square"])
def test_one_taper_retains_the_other_cap_and_zero_disables(cap):
    session = tapered(20, 0)
    send(session, "paint", changes={"stroke-linecap": cap})
    document = session.editor.snapshot.document
    area = HitIndex(document).area("p")
    assert area is not None
    assert area.bounds[0] == pytest.approx(10)
    assert area.bounds[2] == pytest.approx(90 if cap == "butt" else 95)
    send(session, "paint", changes={START: "0"})
    assert OWNER not in export_svg(session.editor.snapshot.document)


def test_overlapping_tapers_fit_short_contours_and_very_large_lengths():
    for length in (100, 1e308):
        document = tapered(length, length).editor.snapshot.document
        area = HitIndex(document).area("p")
        assert area is not None
        assert area.area == pytest.approx(400)
        assert area.covers(Point(50, 54.9))
        assert area.bounds == pytest.approx((10, 45, 90, 55))
        import_svg(export_svg(document))


@pytest.mark.parametrize("value", ["-1", "NaN", "inf", "bad", ""])
def test_invalid_taper_is_atomic(value):
    session = tapered()
    before = session.editor.snapshot
    with pytest.raises(DocumentError):
        send(session, "paint", changes={END: value})
    assert session.editor.snapshot == before


@pytest.mark.parametrize("lock", ["paint", START])
def test_paint_and_property_locks_refuse_tapers(lock):
    session = tapered()
    send(session, "locks", object="p", locks=[lock])
    before = session.editor.snapshot
    with pytest.raises(DocumentError, match="locked"):
        send(session, "paint", changes={START: "30"})
    assert session.editor.snapshot == before


def test_curves_mixed_contours_arrows_and_stroke_to_path_agree():
    session = Session(
        import_svg(
            '<svg width="100" height="100"><path id="p" fill="none" stroke="blue" '
            'stroke-width="4" d="M10 50 C30 10 70 90 90 50 '
            'M20 80 L40 80 L40 90 Z"/></svg>'
        )
    )
    send(session, "select", objects=["p"])
    send(
        session,
        "paint",
        changes={
            START: "12",
            END: "20",
            "stroke-linejoin": "round",
            "data-vectrify-arrow-end": "arrow",
        },
    )
    before = session.editor.snapshot.document
    footprint = HitIndex(before, tolerance=0.02).area("p")
    assert footprint is not None
    send(session, "line_to_fill")
    after = session.editor.snapshot.document
    outlined = HitIndex(after, tolerance=0.02).area("p")
    assert outlined is not None
    assert footprint.symmetric_difference(outlined).area < 0.5
    assert after.element("p").get(START) is None
    assert after.element("p").get(END) is None
    assert OWNER not in export_svg(after)


def test_tapered_lines_transform_clip_and_restore_own_opacity():
    document = import_svg(
        '<svg width="100" height="100"><defs><clipPath id="clip">'
        '<rect width="60" height="60"/></clipPath></defs><g transform="translate(5 5)">'
        '<line id="p" x1="10" y1="50" x2="90" y2="50" stroke="red" '
        'stroke-width="10" stroke-opacity="0.7" clip-path="url(#clip)" '
        'data-vectrify-taper-start="20"/></g></svg>'
    )
    before = HitIndex(document).area("p")
    reopened = import_svg(export_svg(document))
    after = HitIndex(reopened).area("p")
    assert before is not None
    assert after is not None
    assert before.equals(after)
    assert reopened.element("p").tag == "line"
    assert reopened.element("p").get("stroke-opacity") == "0.7"


def test_agent_taper_properties_and_node_edits_update_outline():
    session = Session(import_svg(SVG))
    agent = Agent(session)
    agent.call(
        "properties",
        {"seen": [session.epoch, 0], "ids": ["p"], "start_taper": 12, "end_taper": 18},
    )
    document = session.editor.snapshot.document
    last = document.geometry_for("p").subpaths[0].nodes[-1]
    agent.call(
        "set_points",
        {"seen": [session.epoch, 1], "changes": {"p": {last.id: [90, 80]}}},
    )
    root = ET.fromstring(session.state()["svg"])
    [outline] = [node for node in root.iter() if node.get(OWNER)]
    assert "90.0 80.0" in (outline.get("d") or "")


@pytest.mark.parametrize("source_paint", [True, False])
def test_use_instances_render_tapers_with_source_or_inherited_paint(source_paint):
    attrs = 'stroke="red" stroke-opacity="0.5"' if source_paint else ""
    document = import_svg(
        '<svg width="100" height="100"><defs>'
        f'<path id="p" {attrs} stroke-width="10" fill="none" '
        'transform="translate(0 5)" d="M10 30 L80 30" '
        'data-vectrify-taper-start="20" data-vectrify-taper-end="20"/>'
        '</defs><use id="u" href="#p" x="5" y="10" '
        'stroke="blue" stroke-opacity="0.5"/></svg>'
    )
    svg = export_svg(document)
    reopened = import_svg(svg)
    assert reopened.element("u").get("href") == "#p"
    assert len(reopened.elements()) == len(document.elements())
    assert export_svg(reopened) == svg
    image = render(svg)
    center = pixel(image, 50, 45)
    assert center[:3] == ((255, 0, 0) if source_paint else (0, 0, 255))
    assert 120 <= center[3] <= 130
    assert pixel(image, 17, 48)[3] == 0


def test_zero_length_taper_exports_an_empty_footprint():
    document = import_svg(
        '<svg width="100" height="100"><path id="p" d="M50 50 L50 50" '
        'stroke="red" stroke-width="10" fill="none" stroke-linecap="round" '
        'data-vectrify-taper-start="20"/></svg>'
    )
    assert HitIndex(document).area("p") is None
    assert import_svg(export_svg(document)).element("p").get(START) == "20"


def test_invalid_exported_taper_metadata_is_reported_as_unsupported_svg():
    from vectrify.document.svg import UnsupportedSvgError

    svg = export_svg(tapered().editor.snapshot.document)
    with pytest.raises(UnsupportedSvgError):
        import_svg(
            svg.replace(
                'data-vectrify-taper-start="20"', 'data-vectrify-taper-start="bad"'
            )
        )
