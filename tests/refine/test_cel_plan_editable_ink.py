"""Explicit stroke quality cannot borrow filled paint or invent source queries."""

from dataclasses import replace

import numpy as np
import pytest

from tests.refine.test_cel_plan_attached_outlines import setup as attached_setup
from tests.refine.test_cel_plan_attached_spans import fixture as curved_fixture
from vectrify.document import (
    Editor,
    Selection,
    export_svg,
    import_svg,
    load_project,
    save_project,
)
from vectrify.document.svg import parse_path
from vectrify.refine.cel_plan import editable_ink
from vectrify.refine.cel_plan.attached_outlines import AttachedOutlines
from vectrify.refine.cel_plan.editable_ink import EditableInk
from vectrify.refine.cel_plan.frontier import Frontier
from vectrify.refine.cel_plan.line_fidelity import SourceLineGuard, SourceProfile
from vectrify.refine.cel_plan.local import Box, LocalPolicy
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.score import render, representation
from vectrify.refine.cel_plan.search import Proposal, identity, search


def line(path="M16.5 32.5H80.5", *, opacity=1, extra=""):
    return import_svg(
        '<svg width="96" height="96"><g id="body" '
        f'opacity="{opacity}"><path id="bg" d="M0 0H96V96H0Z" fill="#c2a46d"/>'
        f'<path id="ink" d="{path}" fill="none" stroke="#111111" stroke-width="2"/>'
        + extra
        + "</g></svg>"
    )


def bank(document):
    truth = render(export_svg(document), (96, 96))
    profile = SourceProfile.at(np.linspace((20.5, 32.5), (76.5, 32.5), 30), 2)
    return truth, profile, SourceLineGuard(truth, (profile,))


def change(document, *, path=None, **attrs):
    editor = Editor(document, selection=Selection(whole_document=True))
    with editor.transaction("Quality control") as tx:
        if path is not None:
            tx.replace_geometry("ink", parse_path(path))
        if attrs:
            tx.set_attributes("ink", attrs)
    return editor.snapshot.document


def test_pixel_identical_filled_contour_cannot_supply_editable_support():
    before = line()
    _, _, guard = bank(before)
    quality = EditableInk(guard, before, None, Work.start(10))
    filled = change(
        before, path="M16.5 31.5H80.5V33.5H16.5Z", fill="#111111", stroke="none"
    )
    np.testing.assert_array_equal(
        render(export_svg(before), (96, 96)), render(export_svg(filled), (96, 96))
    )
    assert quality.observe(before)["missing_samples"] == 0
    observed = quality.observe(filled)
    assert observed["missing_samples"] == observed["qualified_samples"]


@pytest.mark.parametrize(
    "kind", ["hidden", "same-background", "white", "transparent", "effect"]
)
def test_hidden_bright_or_unsupported_strokes_cannot_borrow_source_ink(kind):
    before = line()
    _, _, guard = bank(before)
    quality = EditableInk(guard, before, None)
    if kind == "hidden":
        cover = '<path id="cover" d="M0 0H96V96H0Z" fill="#c2a46d"/>'
        after = line(extra=cover)
    elif kind == "same-background":
        bg = before.element("bg")
        after = before.replace_element(replace(bg, attributes=(("fill", "#111111"),)))
    else:
        if kind == "effect":
            ink = before.element("ink")
            after = before.replace_element(
                replace(ink, attributes=(*ink.attributes, ("stroke-dasharray", "2 2")))
            )
            with pytest.raises(ValueError, match="Unsupported attribute"):
                quality.observe(after)
            return
        attrs = (
            {"stroke": "white"}
            if kind == "white"
            else (
                {"stroke-opacity": "0"}
                if kind == "transparent"
                else {"stroke-dasharray": "2 2"}
            )
        )
        after = change(before, **attrs)
    observed = quality.observe(after)
    assert observed["missing_samples"] == observed["qualified_samples"]


def test_new_unsealed_stroke_cannot_create_quality_credit():
    truth, _, guard = bank(line())
    before = line("M16.5 32.5H40")
    quality = EditableInk(guard, before, None)
    after = line(
        "M16.5 32.5H40",
        extra=(
            '<path id="fake" d="M40 32.5H80.5" fill="none" '
            'stroke="#111111" stroke-width="2"/>'
        ),
    )
    np.testing.assert_array_equal(render(export_svg(after), (96, 96)), truth)
    old, new = quality.observe(before), quality.observe(after)
    assert old["missing_samples"] > 0
    assert new["missing_samples"] == old["missing_samples"]
    assert new["eligible_strokes"] == ["ink"]


def test_original_raw_positive_windows_are_frozen_and_not_point_samples():
    before = line()
    _, profile, guard = bank(before)
    quality = EditableInk(guard, before, None)
    offset = change(before, path="M16.5 33.1H80.5", **{"stroke-width": "1"})
    expected = quality.observe(offset)
    assert expected["missing_samples"] == 0
    profile.points.flags.writeable = True
    profile.points[:] += 15
    assert quality.observe(offset) == expected
    outside = change(before, path="M16.5 38.5H80.5", **{"stroke-width": "1"})
    assert quality.observe(outside)["missing_samples"] == quality.qualified


def test_completing_a_real_raw_gap_is_a_hard_rejection():
    before = line("M16.5 32.5H44.5 M52.5 32.5H80.5")
    _, profile, guard = bank(before)
    observed = guard.source_breaks(profile)
    assert observed is not None
    assert observed.gaps.any()
    quality = EditableInk(guard, before, None)
    assert quality.observe(before)["new_gap_completed"] == 0
    joined = quality.observe(change(before, path="M16.5 32.5H80.5"))
    assert joined["new_gap_completed"] > 0
    assert joined["rejections"] == ["editable-ink-gap-completed"]


def test_shared_group_opacity_is_counted_once():
    _, _, guard = bank(line())
    before = line(
        opacity=0.04,
        extra=(
            '<path id="copy" d="M16.5 32.5H80.5" fill="none" '
            'stroke="black" stroke-width="2"/>'
        ),
    )
    quality = EditableInk(guard, before, None)
    observed = quality.observe(before)
    assert observed["missing_samples"] == observed["qualified_samples"]


@pytest.mark.parametrize("kind", ["hidden", "same-background", "white"])
def test_invisible_stroke_cannot_borrow_a_faint_overlapping_strokes_visibility(kind):
    truth, _, guard = bank(line())
    paint = {"hidden": "#111111", "same-background": "#c2a46d", "white": "white"}[kind]
    cover = (
        '<path id="cover" d="M0 0H96V96H0Z" fill="#c2a46d"/>'
        if kind == "hidden"
        else ""
    )
    before = import_svg(
        '<svg width="96" height="96">'
        '<path id="bg" d="M0 0H96V96H0Z" fill="#c2a46d"/>'
        f'<path id="ink" d="M16.5 64.5H80.5" fill="none" stroke="{paint}" '
        'stroke-width="2"/>'
        + cover
        + '<path id="faint" d="M16.5 32.5H80.5" fill="none" '
        'stroke="#111111" stroke-width="2" stroke-opacity="0.04"/></svg>'
    )
    after = change(before, path="M16.5 32.5H80.5")
    if kind != "white":
        np.testing.assert_array_equal(
            render(export_svg(before), (96, 96)), render(export_svg(after), (96, 96))
        )
    quality = EditableInk(guard, before, None)
    old, new = quality.observe(before), quality.observe(after)
    assert old["missing_samples"] == new["missing_samples"] == quality.qualified
    policy = Policy(truth, editable_ink=quality)
    if kind == "white":
        return
    frontier = Frontier(policy)
    assert frontier.add(export_svg(before), "Parent", document=before)
    frontier.freeze_normalizer()

    def candidate(current, _work):
        if not current.edits:
            yield Proposal(
                "hidden-stroke-control",
                ("ink",),
                (),
                current.key,
                after,
                Box(0, 0, 96, 96),
            )

    report = search(
        frontier,
        Options(quality="high"),
        Work.start(30),
        candidate,
        seed_document=before,
    )
    assert report["accepted"] == report["checkpointed"] == 0
    assert report["score_disagreements"] == 0
    assert frontier.select(100).svg == export_svg(before)


@pytest.mark.parametrize(("opacity", "supported"), [(0.96, False), (0.8, True)])
def test_partial_occlusion_is_applied_before_the_body_support_threshold(
    opacity, supported
):
    _, _, guard = bank(line())
    before = line(
        extra=f'<path id="cover" d="M0 0H96V96H0Z" fill="#c2a46d" opacity="{opacity}"/>'
    )
    quality = EditableInk(guard, before, None)
    observed = quality.observe(before)
    assert observed["missing_samples"] == (0 if supported else quality.qualified)


def test_overlapping_opaque_genuine_strokes_keep_visible_quality_credit():
    _, _, guard = bank(line())
    before = line(
        extra='<path id="copy" d="M16.5 32.5H80.5" fill="none" '
        'stroke="#111111" stroke-width="2"/>'
    )
    quality = EditableInk(guard, before, None)
    assert quality.observe(before)["missing_samples"] == 0


@pytest.mark.parametrize("paint", ["#eeeeee", "#dddddd", "rgba(0,0,0,0.04)"])
def test_faint_paint_cannot_amplify_native_rounding_into_body_support(paint):
    _, _, guard = bank(line())
    opacity = "1" if paint.startswith("rgba") else "0.04"
    before = import_svg(
        '<svg width="96" height="96"><path d="M0 0H96V96H0Z" fill="white"/>'
        f'<path id="ink" d="M16.5 32.5H80.5" fill="none" stroke="{paint}" '
        f'stroke-width="2" stroke-opacity="{opacity}"/></svg>'
    )
    quality = EditableInk(guard, before, None)
    assert quality.observe(before)["missing_samples"] == quality.qualified


def fitted():
    state, _, evidence, options, guard, operator = attached_setup()
    assert state.partition is not None
    quality = EditableInk(guard, state.document, state.partition, Work.start(10))
    proposal = next(operator(state, Work.start(30)))
    return state, evidence, options, quality, proposal


def test_complete_family_improves_quality_and_survives_native_reload():
    state, _, _, quality, proposal = fitted()
    old = quality.observe(state.document, state.partition)
    new = quality.observe(proposal.document, proposal.partition)
    assert new["qualified_samples"] == old["qualified_samples"]
    assert new["missing_samples"] == 0 < old["missing_samples"]
    assert new["rejections"] == []
    loaded, _ = load_project(save_project(proposal.document))
    assert quality.observe(loaded, proposal.partition) == new
    assert (
        quality.observe(proposal.document)["missing_samples"] >= old["missing_samples"]
    )


def test_rotated_translucent_complete_family_has_real_visible_source_support():
    state, _, options, _, _ = fitted()
    document, partition, evidence, guard = curved_fixture(
        opacity=0.6, transform="matrix(.8 .6 -.6 .8 60 20)", reverse=True
    )
    quality = EditableInk(guard, document, partition, Work.start(10))
    current = replace(
        state,
        document=document,
        svg=export_svg(document),
        key=identity(export_svg(document), partition),
        partition=partition,
    )
    operator = AttachedOutlines(evidence, options, guard=lambda _work: guard)
    proposal = next(operator(current, Work.start(30)))
    observed = quality.observe(proposal.document, proposal.partition)
    assert (
        observed["missing_samples"]
        < quality.observe(document, partition)["missing_samples"]
    )
    assert observed["rejections"] == []


def test_unrelated_additions_cannot_increase_the_frozen_qualification():
    state, _, _, quality, proposal = fitted()
    expected = quality.observe(proposal.document, proposal.partition)
    extra = import_svg(
        '<svg><path id="unrelated" d="M120 130H150" fill="none" '
        'stroke="black" stroke-width="2"/></svg>'
    )
    editor = Editor(proposal.document, selection=Selection(whole_document=True))
    with editor.transaction("Unrelated source-free stroke") as tx:
        tx.insert_object(
            "component",
            extra.element("unrelated"),
            geometries=(extra.geometry_for("unrelated"),),
        )
    observed = quality.observe(editor.snapshot.document, proposal.partition)
    assert observed["qualified_samples"] == expected["qualified_samples"]
    assert observed["missing_samples"] == expected["missing_samples"]
    assert "unrelated" not in observed["eligible_strokes"]
    assert state.partition is not None


def test_corrupt_family_and_interrupted_observation_do_not_produce_quality():
    _, _, _, quality, proposal = fitted()
    assert proposal.partition is not None
    invalid = replace(proposal.partition.families[0], original="M0 0H1V1H0Z")
    with pytest.raises(ValueError, match=r"original|family|Family"):
        quality.observe(
            proposal.document, replace(proposal.partition, families=(invalid,))
        )
    restored = Partition.from_metadata(proposal.partition.metadata())
    assert restored is not None
    assert quality.observe(proposal.document, restored) == quality.observe(
        proposal.document, proposal.partition
    )
    with pytest.raises(StageInterruptedError):
        quality.observe(proposal.document, proposal.partition, Work.start(0))


def test_cloning_old_filled_ink_under_the_new_stroke_cannot_earn_removal_credit():
    state, _, _, quality, proposal = fitted()
    owner = state.document.element("ink")
    cloned_geometry = parse_path(state.document.geometry_for("ink").path_data())
    clone = replace(owner, id="resurrected-ink", geometry_id=cloned_geometry.id)
    editor = Editor(proposal.document, selection=Selection(whole_document=True))
    with editor.transaction("Clone old filled ink") as tx:
        tx.insert_object("component", clone, geometries=(cloned_geometry,))
    after = editor.snapshot.document
    group = after.element("component")
    after = after.replace_element(
        replace(
            group,
            children=(
                *(e for e in group.children if e.id not in {"resurrected-ink", "ink"}),
                clone,
                after.element("ink"),
            ),
        )
    )
    assert proposal.partition is not None
    proposal.partition.validate(after)
    observed = quality.observe(after, proposal.partition)
    assert observed["excluded_interference"] == ["ink"]
    assert (
        observed["missing_samples"]
        >= quality.observe(state.document, state.partition)["missing_samples"]
    )


def test_policy_source_digest_context_and_ordinary_default_are_explicit():
    state, evidence, _, quality, proposal = fitted()
    policy = Policy(evidence.rgba, editable_ink=quality)
    ordinary = Policy(evidence.rgba)
    svg = export_svg(proposal.document)
    raw = ordinary.evaluate(svg)
    observed = policy.evaluate(
        svg, document=proposal.document, partition=proposal.partition
    )
    assert observed.cost == raw.cost
    assert observed.terms["raster_visual"] == raw.visual
    assert observed.terms["editable_ink_missing_fraction"] == 0
    assert "editable_ink_missing_fraction" not in raw.terms
    with pytest.raises(ValueError, match="original native source"):
        Policy(np.zeros_like(evidence.rgba), editable_ink=quality)
    with pytest.raises(ValueError, match="native document"):
        policy.evaluate(svg, partition=proposal.partition)
    with pytest.raises(ValueError, match="scored SVG"):
        policy.evaluate(svg, document=state.document)


def test_local_full_parity_and_complexity_tradeoff_use_common_search():
    state, evidence, options, quality, proposal = fitted()
    assert state.partition is not None
    policy = Policy(evidence.rgba, editable_ink=quality)
    frontier = Frontier(policy)
    assert frontier.add(
        state.svg,
        "Parent",
        {"planning_surfaces": state.partition.metadata()},
        document=state.document,
        partition=state.partition,
    )
    frontier.freeze_normalizer()
    before = frontier.baseline
    assert before is not None
    evaluator = LocalPolicy(policy)
    snapshot = evaluator.start(state.svg, before.evaluation)
    svg = export_svg(proposal.document)
    updated = evaluator.update(
        snapshot,
        svg,
        proposal.bounds,
        representation(proposal.document).metrics(),
        document=proposal.document,
        partition=proposal.partition,
        work=Work.start(10),
    )
    native = policy.evaluate(
        svg, document=proposal.document, partition=proposal.partition
    )
    for term, value in native.terms.items():
        assert updated.evaluation.terms[term] == pytest.approx(value, abs=2e-7)

    def candidate(current, _work):
        if not current.edits:
            yield replace(proposal, parent=current.key)

    report = search(
        frontier, options, Work.start(30), candidate, seed_document=state.document
    )
    assert report["accepted"] == report["checkpointed"] == 1
    assert report["score_disagreements"] == 0
    assert frontier.select(0).svg == state.svg
    assert frontier.select(100).svg == svg


@pytest.mark.parametrize(
    "kind", ["queries", "painters", "pixels", "viewport", "interrupted"]
)
def test_unsupported_budget_or_frame_cannot_publish_a_partial_score(kind, monkeypatch):
    before = line()
    _, _, guard = bank(before)
    if kind in {"queries", "painters", "pixels"}:
        name = {
            "queries": "MAX_QUERIES",
            "painters": "MAX_PAINTERS",
            "pixels": "MAX_NATIVE_PIXELS",
        }[kind]
        monkeypatch.setattr(editable_ink, name, 1)
    elif kind == "viewport":
        before = replace(
            before,
            root=replace(
                before.root,
                attributes=(
                    ("width", "96"),
                    ("height", "96"),
                    ("viewBox", "0 0 192 192"),
                ),
            ),
        )
    if kind == "interrupted":
        with pytest.raises(StageInterruptedError):
            EditableInk(guard, before, None, Work.start(0))
    else:
        with pytest.raises(ValueError, match=r"bound|viewport"):
            EditableInk(guard, before, None, Work.start(10))
