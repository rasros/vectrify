"""Complete vector restoration floors cannot waive native alpha or old ink removal."""

from dataclasses import replace
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from vectrify.document import (
    Editor,
    Selection,
    export_svg,
    import_svg,
    load_project,
    save_project,
)
from vectrify.document.holes import reversed_subpath
from vectrify.document.join import curve_path
from vectrify.document.paint import GradientStop, LinearGradient
from vectrify.document.svg import parse_path
from vectrify.refine.cel_plan.local import _native_raster
from vectrify.refine.cel_plan.model import StageInterruptedError, Work
from vectrify.refine.cel_plan.source_faces import SourceFace
from vectrify.refine.cel_plan.source_family import SourceFamily
from vectrify.refine.cel_plan.source_spans import SourceSpan
from vectrify.refine.cel_plan.span_restoration import construct


def fixture(
    *, opacity=1, transform="translate(0 0)", gradient=False, field="M8 8H24V10H8Z"
):
    document = import_svg(
        '<svg width="48" height="48"><g id="component" '
        f'opacity="{opacity}" transform="{transform}">'
        '<path id="base" d="M4 4H28V28H4Z" fill="#b49150"/>'
        '<path id="owner" d="M8 8H24V24H8Z" fill="#271b10"/>'
        '<path id="face" d="M16 10H24V24H16Z" fill="#584022"/>'
        '</g><path id="external" d="M32 8H40V20H32Z" fill="#806060"/></svg>'
    )
    if gradient:
        editor = Editor(document, selection=Selection(whole_document=True))
        with editor.transaction("Original material gradient") as tx:
            tx.set_fill(
                "base",
                LinearGradient(
                    (4, 4),
                    (28, 28),
                    (GradientStop(0, "#b49150"), GradientStop(1, "#806040")),
                ),
            )
        document = editor.snapshot.document
    selected = parse_path(field)
    original = document.geometry_for("owner")
    retained = replace(
        original,
        subpaths=(
            *original.subpaths,
            *(reversed_subpath(s) for s in selected.subpaths),
        ),
    )
    span = SourceSpan(selected, retained, np.linspace((10, 9), (22, 9), 20))
    shade = parse_path("M16 8H24V10H16Z")
    remainder = replace(
        selected, subpaths=(*selected.subpaths, reversed_subpath(shade.subpaths[0]))
    )
    return document, span, SourceFace("face", 0, shade, remainder, 1)


@pytest.mark.parametrize(
    ("opacity", "transform", "gradient"),
    [
        (1, "translate(0 0)", False),
        (0.6, "translate(.35 .2)", False),
        (0.6, "translate(3 2)", True),
    ],
)
@pytest.mark.parametrize("adjoining", [False, True])
def test_complete_floor_keeps_native_alpha_locality_materials_and_shadow(
    adjoining, opacity, transform, gradient
):
    document, span, face = fixture(
        opacity=opacity, transform=transform, gradient=gradient
    )
    result = construct(
        document,
        "owner",
        span,
        "base",
        (48, 48),
        Work.start(10),
        face=face if adjoining else None,
    )
    assert result is not None
    assert result.proof["native_alpha_exact"]
    assert result.proof["outside_removed_field_rgba_exact"]
    assert not result.proof["positive_body_proved"]
    assert not result.proof["accepted"]
    assert len(result.parts) == 4 if adjoining else len(result.parts) == 3
    for oid in ("base", "face", "external"):
        assert result.document.element(oid) == document.element(oid)
        assert result.document.geometry_for(oid) == document.geometry_for(oid)
    assert result.document.element("owner").get("fill") == "none"
    assert result.document.element("owner").get("stroke") == "none"
    loaded, _ = load_project(save_project(result.document))
    np.testing.assert_array_equal(
        _native_raster(ET.fromstring(export_svg(loaded)), (48, 48)).root,
        _native_raster(ET.fromstring(export_svg(result.document)), (48, 48)).root,
    )
    residual = next(p.id for p in result.parts if p.role == "residual")
    assert (
        curve_path(result.document.geometry_for(residual)).area
        == curve_path(span.retained).area
    )
    # No stroke means the intermediate cannot publish a physical source family.
    with pytest.raises(ValueError, match="genuine open"):
        SourceFamily.bind(
            document,
            loaded,
            "owner",
            span.field,
            result.parts,
            (48, 48),
            Work.start(10),
        )


def test_generated_floor_binds_after_a_real_positive_stroke_is_installed():
    document, span, face = fixture()
    result = construct(
        document, "owner", span, "base", (48, 48), Work.start(10), face=face
    )
    assert result is not None
    editor = Editor(result.document, selection=Selection(whole_document=True))
    with editor.transaction("Complete genuine outline") as tx:
        tx.replace_geometry("owner", parse_path("M10 9H22"))
        tx.set_attributes(
            "owner",
            {"stroke": "#271b10", "stroke-width": "1", "stroke-linecap": "round"},
        )
    candidate = editor.snapshot.document
    family = SourceFamily.bind(
        document, candidate, "owner", span.field, result.parts, (48, 48), Work.start(10)
    )
    family.validate(candidate)


def test_equivalent_reparsed_faces_keep_physical_part_identities():
    document, span, face = fixture()
    reparsed = replace(
        face,
        source_index=99,
        field=parse_path(face.field.path_data()),
        remainder=parse_path(face.remainder.path_data()),
    )
    first = construct(
        document, "owner", span, "base", (48, 48), Work.start(10), face=face
    )
    second = construct(
        document, "owner", span, "base", (48, 48), Work.start(10), face=reparsed
    )
    assert first is not None
    assert second is not None
    assert first.parts == second.parts


@pytest.mark.parametrize(
    "corruption", ["resurrect", "prefix", "outside", "open", "extent"]
)
def test_invalid_complete_field_or_residual_is_excluded(corruption):
    document, span, _face = fixture()
    if corruption == "resurrect":
        span = replace(span, retained=document.geometry_for("owner"))
    elif corruption == "prefix":
        sub = span.retained.subpaths[0]
        changed = replace(
            sub, nodes=(replace(sub.nodes[0], values=(9, 8)), *sub.nodes[1:])
        )
        span = replace(
            span,
            retained=replace(
                span.retained, subpaths=(changed, *span.retained.subpaths[1:])
            ),
        )
    elif corruption == "outside":
        span = replace(span, field=parse_path("M7 8H24V10H7Z"))
    elif corruption == "open":
        span = replace(span, field=parse_path("M8 8H24V10H8"))
    else:
        span = replace(span, field=parse_path("M8 8H240V10H8Z"))
    assert construct(document, "owner", span, "base", (48, 48), Work.start(10)) is None


@pytest.mark.parametrize(
    "change", ["fill-opacity", "effects", "singular-frame", "unshared-face"]
)
def test_unsupported_material_or_unshared_face_is_excluded(change):
    document, span, face = fixture()
    if change == "unshared-face":
        face = replace(face, field=parse_path("M16 7H24V9H16Z"))
    else:
        element = document.element("base")
        attrs = dict(element.attributes)
        attrs.update(
            {"fill-opacity": ".5"}
            if change == "fill-opacity"
            else {"filter": "url(#missing)"}
            if change == "effects"
            else {"transform": "scale(0)"}
        )
        document = document.replace_element(
            replace(element, attributes=tuple(attrs.items()))
        )
    assert (
        construct(document, "owner", span, "base", (48, 48), Work.start(10), face=face)
        is None
    )


def test_alpha_change_cannot_be_hidden_by_valid_boolean_geometry():
    document, span, _face = fixture()
    # The donor is opaque but lives elsewhere: restoring its paint cannot
    # reproduce the original antialiased silhouette at the removed boundary.
    geometry = document.geometry_for("base")
    document = replace(
        document,
        geometries=tuple(
            replace(g, subpaths=parse_path("M30 30H40V40H30Z").subpaths)
            if g.id == geometry.id
            else g
            for g in document.geometries
        ),
    )
    fractional = import_svg(
        export_svg(document).replace("translate(0 0)", "translate(.4 .4)")
    )
    selected = span.field
    original = fractional.geometry_for("owner")
    span = replace(
        span,
        retained=replace(
            original,
            subpaths=(*original.subpaths, reversed_subpath(selected.subpaths[0])),
        ),
    )
    assert (
        construct(fractional, "owner", span, "base", (48, 48), Work.start(10)) is None
    )


def test_native_bounds_and_interruption_are_not_partial_floor_results():
    document, span, face = fixture()
    assert (
        construct(
            document, "owner", span, "base", (2000, 2000), Work.start(10), face=face
        )
        is None
    )
    with pytest.raises(ValueError, match="viewport"):
        construct(document, "owner", span, "base", (0, 48), Work.start(10))
    with pytest.raises(StageInterruptedError):
        construct(document, "owner", span, "base", (48, 48), Work.start(0))
