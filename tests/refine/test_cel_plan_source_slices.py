"""A whole physical stroke can be removed from a connected filled shadow."""

from dataclasses import replace

import numpy as np
import pathops
import pytest
from PIL import Image

from vectrify.document import (
    Editor,
    Selection,
    export_svg,
    import_svg,
    load_project,
    save_project,
)
from vectrify.document.join import curve_path, path_style
from vectrify.document.svg import parse_path
from vectrify.refine.cel_plan import atoms as atom_module
from vectrify.refine.cel_plan import band_plans as band_module
from vectrify.refine.cel_plan.atoms import Atoms
from vectrify.refine.cel_plan.band_plans import (
    BandPlans,
    attached_locality,
    color_quantization,
)
from vectrify.refine.cel_plan.component_edits import ComponentEdit
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.line_fidelity import SourceLineGuard, SourceProfile
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.proposals import Operators, bounds
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import Proposal, State
from vectrify.refine.cel_plan.source_slices import source_slice


def fixture(*, alpha=1, group_alpha=1, gap=False):
    prefix = (
        '<svg width="96" height="96"><g id="scene" '
        f'opacity="{alpha}"><path id="bg" d="M0 0H96V96H0Z" fill="#ad8665"/>'
    )
    suffix = "</g></svg>"
    if group_alpha != 1:
        prefix += f'<g id="paint" opacity="{group_alpha}">'
        suffix = "</g>" + suffix
    shadow = '<path d="M76.25 25.75H90V70H76.25Z" fill="#121008"/>'
    line = "M20.25 28.25H44 M50 28.25H76.25" if gap else "M20.25 28.25H76.25"
    source = (
        prefix
        + shadow
        + f'<path d="{line}" fill="none" stroke="#121008" stroke-width="3"/>{suffix}'
    )
    document = import_svg(
        prefix + '<path id="ink" d="M20.25 25.75H90V70H76.25V30.75H20.25Z" '
        f'fill="#121008"/>{suffix}'
    )
    rgba = render(source, (96, 96))
    e = collect(
        Image.fromarray(np.rint(rgba * 255).astype(np.uint8)),
        None,
        Options(),
        Work.start(10),
    )
    points = np.column_stack((np.linspace(20.25, 76.25, 24), np.full(24, 28.25)))
    guard = SourceLineGuard(e.rgba, (SourceProfile.at(points, 3),))
    return e, guard, document


@pytest.mark.parametrize("alpha", [1, 0.75])
def test_subtracted_band_keeps_original_shadow_controls_and_has_no_old_ink_fill(alpha):
    _, guard, document = fixture(alpha=alpha)
    observed = guard.source_breaks(guard.original_profiles()[0])
    result = source_slice(document, "ink", observed, Work.start(10))
    assert result is not None
    removed, retained = result
    original = document.geometry_for("ink")
    assert retained.subpaths[: len(original.subpaths)] == original.subpaths
    difference = pathops.op(
        curve_path(original), curve_path(removed), pathops.PathOp.DIFFERENCE
    )
    assert pathops.op(curve_path(retained), difference, pathops.PathOp.XOR).area == 0
    assert (
        pathops.op(
            curve_path(retained), curve_path(removed), pathops.PathOp.INTERSECTION
        ).area
        == 0
    )


def test_real_gap_fragment_and_broad_shadow_are_not_sliced_into_fake_complete_lines():
    _, guard, document = fixture(gap=True)
    observed = guard.source_breaks(guard.original_profiles()[0])
    assert observed is not None
    assert observed.gaps.any()
    assert source_slice(document, "ink", observed, Work.start(10)) is None
    _, guard, document = fixture()
    observed = guard.source_breaks(guard.original_profiles()[0])
    assert observed is not None
    assert (
        source_slice(document, "ink", replace(observed, anchors=None), Work.start(10))
        is None
    )
    broad = import_svg(
        '<svg width="96" height="96"><path id="ink" d="M10 10H90V70H10Z"/></svg>'
    )
    assert source_slice(broad, "ink", observed, Work.start(10)) is None
    with pytest.raises(ValueError, match="tolerance"):
        source_slice(document, "ink", observed, Work.start(10), tolerance=float("nan"))
    with pytest.raises(StageInterruptedError):
        source_slice(document, "ink", observed, Work.start(0))


@pytest.mark.parametrize("reverse", [False, True])
def test_removal_port_extension_keeps_complete_field_and_physical_source_ports(reverse):
    _, guard, document = fixture()
    observed = guard.source_breaks(guard.original_profiles()[0])
    assert observed is not None
    assert observed.anchors is not None
    original_anchors = observed.anchors.copy()
    if reverse:
        observed = replace(observed, anchors=observed.anchors[::-1])
    plain = source_slice(document, "ink", observed, Work.start(10))
    extended = source_slice(
        document,
        "ink",
        observed,
        Work.start(10),
        port_extension=(0.5, 0) if reverse else (0, 0.5),
    )
    assert plain is not None
    assert extended is not None
    selected, retained = extended
    old = document.geometry_for("ink")
    # The domain can remove a little more of the attached fill; it cannot leave
    # any part of the complete former outline behind or move the source ports.
    assert (
        pathops.op(
            curve_path(plain[0]), curve_path(selected), pathops.PathOp.DIFFERENCE
        ).area
        == 0
    )
    assert curve_path(selected).bounds[2] == 76.75
    assert curve_path(selected).area > curve_path(plain[0]).area
    assert retained.subpaths[: len(old.subpaths)] == old.subpaths
    assert (
        pathops.op(
            curve_path(retained), curve_path(selected), pathops.PathOp.INTERSECTION
        ).area
        == 0
    )
    assert (
        pathops.op(
            curve_path(retained),
            pathops.op(
                curve_path(old), curve_path(selected), pathops.PathOp.DIFFERENCE
            ),
            pathops.PathOp.XOR,
        ).area
        == 0
    )
    original_observation = guard.source_breaks(guard.original_profiles()[0])
    assert original_observation is not None
    assert original_observation.anchors is not None
    assert np.array_equal(original_observation.anchors, original_anchors)


@pytest.mark.parametrize("extension", [(-0.5, 0), (0, 2.01), (0, np.nan), (0,)])
def test_removal_port_extension_is_bounded_and_does_not_override_source_gaps(extension):
    _, guard, document = fixture()
    observed = guard.source_breaks(guard.original_profiles()[0])
    with pytest.raises(ValueError, match="port extensions"):
        source_slice(
            document, "ink", observed, Work.start(10), port_extension=extension
        )
    _, guard, document = fixture(gap=True)
    observed = guard.source_breaks(guard.original_profiles()[0])
    assert (
        source_slice(document, "ink", observed, Work.start(10), port_extension=(0, 0.5))
        is None
    )


def test_color_quantization_cannot_hide_alpha_or_more_than_one_premultiplied_byte():
    before = np.array([[[80, 90, 100, 255], [32, 32, 32, 10]]], np.uint8)
    outside = np.ones((1, 2), bool)
    after = before.copy()
    after[0, 0, 0] += 1
    assert color_quantization(before, after, outside) == {
        "changed_pixels": 1,
        "max_premultiplied_byte_delta": 1.0,
        "alpha_exact": True,
    }
    after[0, 0, 0] += 1
    assert color_quantization(before, after, outside) is None
    after = before.copy()
    after[0, 1, 3] += 1
    assert color_quantization(before, after, outside) is None


@pytest.mark.parametrize(
    ("alpha", "group_alpha", "hole", "proof_mode"),
    [
        (1, 1, False, "normal"),
        (0.75, 1, False, "normal"),
        (0.5, 1, False, "normal"),
        (0.75, 0.6, False, "normal"),
        (1, 1, True, "normal"),
        (0.75, 1, True, "normal"),
        (0.5, 1, True, "normal"),
        (1, 1, "terminal", "normal"),
        (0.75, 1, "terminal", "normal"),
        (0.5, 1, "terminal", "normal"),
        (1, 1, True, "deferred"),
        (1, 1, True, "reject_final"),
        (1, 1, False, "ghost_outline"),
        (1, 1, False, "filled_stroke"),
    ],
)
def test_attached_stroke_replays_ancestor_and_roundtrips_partial_group_opacity(
    alpha, group_alpha, hole, proof_mode, monkeypatch
):
    e, guard, document = fixture(alpha=alpha, group_alpha=group_alpha)
    if hole:
        # The old filled outline supplies all coverage in this material hole.
        # Replacing it by the thinner source stroke must restore that original
        # opaque coverage with material, rather than retain filled old ink.
        editor = Editor(document, selection=Selection(whole_document=True))
        with editor.transaction("Material hole beneath old outline") as tx:
            right = 79 if hole == "terminal" else 74
            tx.replace_geometry(
                "bg", parse_path(f"M0 0H96V96H0Z M22 26V30H{right}V26H22Z")
            )
        document = editor.snapshot.document
    labels = np.zeros(e.labels.shape, np.int32)
    labels[e.target.max(axis=-1) < e.target[0, 0].max() - 25] = 1
    e = replace(e, labels=labels)
    graph = build(e)
    old = Partition((Surface("bg", (0,), covered=(1,)), Surface("ink", (1,))))
    policy = Policy.from_evidence(e, graph)
    svg = export_svg(document)
    state = State(
        document,
        svg,
        LocalPolicy(policy).start(svg, policy.evaluate(svg)),
        "ancestor",
        {},
        partition=old,
    )
    classes = (np.indices(labels.shape)[1] >= 48).astype(np.int32)
    atoms, groups = Atoms.original(graph).partition(
        graph, (1,), classes, 2, Work.start(10), compact=True, retain_parent=True
    )
    part = old.split(
        ("ink",), (Surface("ink", tuple(sorted(groups[0] + groups[1]))),), atoms
    )
    edit = Proposal(
        "joint-core-cells",
        ("ink",),
        (),
        state.key,
        document,
        bounds(document, document, ("ink",)),
        partition=part,
        component=ComponentEdit.bind(
            document, old, document.ancestry("ink")[-2].id, Work.start(10)
        ),
    )
    monkeypatch.setattr(atom_module, "MAX_CUTS", 1)
    proof_calls = []
    if proof_mode in {"deferred", "reject_final"}:

        def pending(before, after, outside):
            proof_calls.append(None)
            # Neither incomplete construction is a publication proof. A missing
            # complete-parent proof must still exclude the whole candidate.
            if len(proof_calls) <= 2 or proof_mode == "reject_final":
                return None
            return color_quantization(before, after, outside)

        monkeypatch.setattr(band_module, "color_quantization", pending)
    if proof_mode in {"ghost_outline", "filled_stroke"}:
        from vectrify.refine.cel_plan.band_fit import BandFit
        from vectrify.refine.cel_plan.ink_replace import identified

        original_fit = BandFit.fit

        def resurrect(fitter, before, assembled, oid, seed, work, **kwargs):
            result = original_fit(fitter, before, assembled, oid, seed, work, **kwargs)
            assert result is not None
            fitted, witness = result
            editor = Editor(fitted, selection=Selection(whole_document=True))
            with editor.transaction("Incorrectly restore old filled outline") as tx:
                if proof_mode == "ghost_outline":
                    tx.replace_geometry(
                        "ink-band-marks",
                        identified(document.geometry_for("ink"), "ink-band-marks"),
                    )
                else:
                    tx.set_fill(
                        oid, path_style(document, document.element("ink"))["fill"]
                    )

            ghost = editor.snapshot.document
            # The restored filled outline has the same paint as the new stroke.
            # Native locality alone can accept its unchanged alpha, even
            # though the promised field removal has been undone under the ink.
            assert np.array_equal(
                render(export_svg(ghost), (96, 96))[..., 3],
                render(svg, (96, 96))[..., 3],
            )
            if proof_mode == "filled_stroke":
                # A straight open path has zero fill area, so an incorrect fill
                # attribute is invisible until somebody edits the centerline.
                assert np.array_equal(
                    render(export_svg(ghost), (96, 96)),
                    render(export_svg(fitted), (96, 96)),
                )
            selected = source_slice(
                document, "ink", guard.source_breaks(guard.original_profiles()[0]), work
            )
            assert selected is not None
            assert (
                attached_locality(
                    document, ghost, "ink", selected[0], (), (96, 96), work
                )
                is not None
            )
            proof_calls.append(None)
            return ghost, witness

        monkeypatch.setattr(BandFit, "fit", resurrect)
    planner = BandPlans(e, graph, graph, guard=lambda _work: guard)
    alternatives = list(
        planner.proposals(state, edit, Work.start(20), source_fit=True, attached=True)
    )
    if proof_mode in {"deferred", "reject_final"}:
        assert len(proof_calls) >= 3
    if proof_mode in {"reject_final", "ghost_outline", "filled_stroke"}:
        assert alternatives == []
        assert proof_calls
        return
    assert len(alternatives) == 1, planner.diagnostics
    candidate = alternatives[0]
    assert candidate.partition.follows(old)
    assert len(candidate.partition.atoms.cuts) == 1
    Operators(e, graph, Options()).validate_partition(
        candidate.partition, Work.start(10)
    )
    candidate.component.validate(
        document,
        candidate.document,
        old,
        candidate.partition,
        candidate.ids,
        candidate.bounds,
        Work.start(10),
    )
    assert (
        path_style(candidate.document, candidate.document.element("ink"))["fill"]
        == "none"
    )
    assert len(candidate.document.geometry_for("ink").subpaths[0].nodes) == 2
    record = candidate.details["planned_band_stroke"]
    assert record["attached"]
    assert record["source_line_comparison"]["rejections"] == []
    assert record["source_fit"]["retained_shadow"] == {
        "id": record["marks"],
        "geometry_exact": True,
        "paint_exact": True,
    }
    assert record["source_fit"]["fit"]["native_body_absence"]
    assert record["source_fit"]["fit"]["native_alpha_exact"]
    if proof_mode == "deferred":
        assert record["cut_quantization"] is None
        assert record["source_fit"]["continuation"]["color_quantization"] is None
    else:
        assert record["cut_quantization"]["alpha_exact"]
    proof = record["source_fit"]["native_locality"]
    assert proof["scope"] == "complete-final-candidate-vs-material-parent"
    assert proof["native_alpha_exact"]
    assert proof["native_footprint_exact"]
    if hole:
        continuation = record["source_fit"]["continuation"]
        assert continuation["material"] == "bg"
        assert continuation["patch_nodes"] > 0
        assert not continuation["native_alpha_exact"]
        assert continuation["native_alpha_scope"] == "continuation-vs-complement"
        if hole == "terminal":
            assert continuation["scope"] == "opaque-native-removed-field-cells"
        assert candidate.document.geometry_for("bg") != document.geometry_for("bg")
    removed = parse_path(record["removed"])
    mutations: tuple[tuple[str, dict[str, str | None]], ...] = (
        ("bg", {"fill": "#ffffff"}),
        ("scene", {"opacity": "0.25"}),
        (record["marks"], {"transform": "translate(8 0)"}),
    )
    for target, attrs in mutations:
        editor = Editor(candidate.document, selection=Selection(whole_document=True))
        with editor.transaction("Unrelated change must not pass locality") as tx:
            tx.set_attributes(target, attrs)
        assert (
            attached_locality(
                document,
                editor.snapshot.document,
                "ink",
                removed,
                (),
                (96, 96),
                Work.start(10),
            )
            is None
        )
    actual = render(export_svg(candidate.document), (96, 96))
    np.testing.assert_array_equal(actual[..., 3], render(svg, (96, 96))[..., 3])
    assert policy.evaluate(export_svg(candidate.document)).valid
    residual = candidate.document.geometry_for(record["marks"])
    original = document.geometry_for("ink")
    difference = pathops.op(
        curve_path(original), curve_path(removed), pathops.PathOp.DIFFERENCE
    )
    assert pathops.op(curve_path(residual), difference, pathops.PathOp.XOR).area == 0
    assert (
        pathops.op(
            curve_path(residual), curve_path(removed), pathops.PathOp.INTERSECTION
        ).area
        == 0
    )
    # Published residuals receive fresh editing ids; commands and controls
    # must still retain the original shadow geometry exactly.
    assert (
        replace(
            residual, subpaths=residual.subpaths[: len(original.subpaths)]
        ).path_data()
        == original.path_data()
    )
    actual = render(export_svg(candidate.document), (96, 96))
    restored, _ = load_project(save_project(candidate.document))
    assert np.array_equal(actual, render(export_svg(restored), (96, 96)))
