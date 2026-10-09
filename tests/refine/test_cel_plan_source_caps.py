"""Cap recovery keeps physical source ports and gaps, paint and other controls."""

from dataclasses import replace

import numpy as np
import pytest

from tests.refine.test_cel_plan_source_absence import observed_gap
from vectrify.document import export_svg, import_svg
from vectrify.document.join import transformed_geometry
from vectrify.document.svg import parse_path
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan import source_caps
from vectrify.refine.cel_plan.line_fidelity import SourceLineGuard
from vectrify.refine.cel_plan.model import StageInterruptedError, Work
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.source_absence import SourceAbsence
from vectrify.refine.cel_plan.source_caps import SourceCaps


def fixture():
    evidence, profiles = observed_gap()
    guard = SourceLineGuard(evidence.rgba, profiles)
    document = import_svg(
        '<svg width="96" height="96"><path id="bg" d="M0 0H96V96H0Z" '
        'fill="#ccbb99"/><path id="ink" d="M20.25 28.25H40.25 '
        'M55.75 28.25H76.25" fill="none" stroke="#121212" '
        'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        '<path id="shadow" d="M10 60H80V75H10Z" fill="#987654"/></svg>'
    )
    return evidence, guard, document


def test_recovery_uses_own_gap_and_qualified_samples_without_changing_physical_ports():
    evidence, guard, document = fixture()
    result = SourceCaps(guard).extend(
        document, ("ink",), (0, 0, 96, 96), Work.start(10), margin=1
    )
    assert result is not None
    proposed, witnesses = result
    assert len(witnesses) == 2
    old = document.geometry_for("ink").subpaths
    new = proposed.geometry_for("ink").subpaths
    assert new[0].nodes[0] == old[0].nodes[0]
    assert new[-1].nodes[-1] == old[-1].nodes[-1]
    assert new[0].nodes[-1].endpoint[0] > old[0].nodes[-1].endpoint[0]
    assert new[1].nodes[0].endpoint[0] < old[1].nodes[0].endpoint[0]
    assert proposed.element("ink") == document.element("ink")
    for oid in ("bg", "shadow"):
        assert proposed.element(oid) == document.element(oid)
        assert proposed.geometry_for(oid) == document.geometry_for(oid)
    for witness in witnesses:
        profile = guard.original_profiles()[witness["profile"]]
        observed = guard.source_breaks(profile)
        assert observed is not None
        assert observed.qualified[witness["to_sample"]]
        assert not observed.gaps[
            min(witness["from_sample"], witness["to_sample"]) : max(
                witness["from_sample"], witness["to_sample"]
            )
            + 1
        ].any()
        np.testing.assert_array_equal(
            witness["to"], observed.points[witness["to_sample"]]
        )
    comparison = guard.compare(
        render(export_svg(document), evidence.source_size),
        render(export_svg(proposed), evidence.source_size),
    )
    assert comparison["rejections"] == []
    assert comparison["new_gap_completed"] == 0
    absence = SourceAbsence(evidence, guard.original_profiles(), Work.start(10))
    assert absence.permits(proposed.geometry_for("ink"), 3, "round", Work.start(10))


def test_surviving_cubic_controls_and_sibling_subpaths_remain_exact():
    _, guard, document = fixture()
    from vectrify.document import Editor, Selection

    editor = Editor(document, selection=Selection(whole_document=True))
    with editor.transaction("Control fixture") as tx:
        tx.replace_geometry(
            "ink", parse_path("M20.25 28.25C25 28.25 32 28.25 40.25 28.25")
        )
    document = editor.snapshot.document
    result = SourceCaps(guard).extend(
        document, ("ink",), (0, 0, 96, 96), Work.start(10)
    )
    assert result is not None
    proposed, _ = result
    before = document.geometry_for("ink").subpaths[0].nodes
    after = proposed.geometry_for("ink").subpaths[0].nodes
    assert before[0] == after[0]
    assert before[-1].values[:4] == after[-1].values[:4]
    assert before[-1].id == after[-1].id


def test_unknown_truncation_and_a_chain_crossing_the_gap_cannot_supply_a_cap():
    _, guard, document = fixture()
    from vectrify.document import Editor, Selection

    for data in ("M21.1 28.25H40.25", "M40.25 28.25H55.75"):
        editor = Editor(document, selection=Selection(whole_document=True))
        with editor.transaction("Unproved fixture") as tx:
            tx.replace_geometry("ink", parse_path(data))
        assert (
            SourceCaps(guard).extend(
                editor.snapshot.document, ("ink",), (0, 0, 96, 96), Work.start(10)
            )
            is None
        )


def test_duplicate_source_identities_are_ambiguous():
    evidence, guard, document = fixture()
    original = guard.original_profiles()[0]
    duplicated = SourceLineGuard(evidence.rgba, (original, replace(original)))
    assert (
        SourceCaps(duplicated).extend(
            document, ("ink",), (0, 0, 96, 96), Work.start(10)
        )
        is None
    )


def test_scope_pins_and_inherited_effects_exclude_optional_cap_changes():
    _, guard, document = fixture()
    from vectrify.document import Editor, Selection

    assert (
        SourceCaps(guard).extend(document, ("ink",), (0, 50, 96, 96), Work.start(10))
        is None
    )
    editor = Editor(document, selection=Selection(whole_document=True))
    old = document.geometry_for("ink")
    pinned = replace(
        old,
        subpaths=tuple(
            replace(sub, nodes=tuple(replace(node, pinned=True) for node in sub.nodes))
            for sub in old.subpaths
        ),
    )
    with editor.transaction("Pinned fixture") as tx:
        tx.replace_geometry("ink", pinned)
    assert (
        SourceCaps(guard).extend(
            editor.snapshot.document, ("ink",), (0, 0, 96, 96), Work.start(10)
        )
        is None
    )
    effect = import_svg(
        '<svg width="96" height="96"><defs><clipPath id="clip">'
        '<path d="M0 0H96V96H0Z"/></clipPath></defs><g clip-path="url(#clip)">'
        '<path id="ink" d="M20.25 28.25H40.25" fill="none" stroke="black" '
        'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></g></svg>'
    )
    assert (
        SourceCaps(guard).extend(effect, ("ink",), (0, 0, 96, 96), Work.start(10))
        is None
    )


def test_cap_matching_uses_the_actual_object_frame():
    _, guard, document = fixture()
    from vectrify.document import Editor, Selection

    matrix = (0.9, 0.03, 0.08, 0.85, 0.25, 1.5)
    local = transformed_geometry(document.geometry_for("ink"), inverse_matrix(matrix))
    editor = Editor(document, selection=Selection(whole_document=True))
    with editor.transaction("Framed fixture") as tx:
        tx.replace_geometry("ink", local)
        tx.set_attributes("ink", {"transform": "matrix(0.9 0.03 0.08 0.85 0.25 1.5)"})
    result = SourceCaps(guard).extend(
        editor.snapshot.document, ("ink",), (0, 0, 96, 96), Work.start(10)
    )
    assert result is not None
    proposed, witnesses = result
    native = transformed_geometry(proposed.geometry_for("ink"), matrix)
    for sub, witness in zip(native.subpaths, witnesses, strict=True):
        point = sub.nodes[-1 if witness["end"] == "end" else 0].endpoint
        np.testing.assert_allclose(point, witness["to"], atol=1e-12)
    assert proposed.element("ink") == editor.snapshot.document.element("ink")


@pytest.mark.parametrize("bound", ["MAX_PATHS", "MAX_NODES", "MAX_MOVEMENT"])
def test_bounds_and_cancellation_never_publish_a_partial_document(monkeypatch, bound):
    _, guard, document = fixture()
    monkeypatch.setattr(source_caps, bound, 0)
    assert (
        SourceCaps(guard).extend(document, ("ink",), (0, 0, 96, 96), Work.start(10))
        is None
    )
    with pytest.raises(StageInterruptedError):
        SourceCaps(guard).extend(document, ("ink",), (0, 0, 96, 96), Work.start(0))
    assert document.geometry_for("ink").path_data() == (
        "M20.25 28.25 L40.25 28.25 M55.75 28.25 L76.25 28.25"
    )


def test_source_profile_mutation_cannot_change_the_copied_cap_observations():
    _, guard, document = fixture()
    original = guard.original_profiles()[0]
    expected = SourceCaps(guard).extend(
        document, ("ink",), (0, 0, 96, 96), Work.start(10)
    )
    original.points.flags.writeable = True
    original.points[:] = 0
    actual = SourceCaps(guard).extend(
        document, ("ink",), (0, 0, 96, 96), Work.start(10)
    )
    assert expected == actual


def test_a_qualified_cap_proposal_still_needs_complete_native_gap_validation():
    evidence, guard, document = fixture()
    result = SourceCaps(guard).extend(
        document, ("ink",), (0, 0, 96, 96), Work.start(10), margin=0
    )
    assert result is not None
    proposed, _ = result
    comparison = guard.compare(
        render(export_svg(document), evidence.source_size),
        render(export_svg(proposed), evidence.source_size),
    )
    assert comparison["new_gap_completed"] > 0
    assert any(
        r["reason"] == "source-line-gap-completed" for r in comparison["rejections"]
    )
