"""Interior fitting preserves the exact native complement of each path."""

from dataclasses import replace

import numpy as np
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
from vectrify.document.join import path_style, union_geometry
from vectrify.document.redraw import root_matrix
from vectrify.refine import shared
from vectrify.refine.cel_plan import constraints as chains
from vectrify.refine.cel_plan.export import export
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.frontier import Frontier
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Evidence, Options, Work
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.proposals import Operators, bounds
from vectrify.refine.cel_plan.refine import _geometry_proposal, refine
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import State, search

SVG = (
    '<svg width="64" height="64"><g transform="translate(4 2) scale(.8)">'
    '<path id="a" fill="#ab6040" d="M8 8L32 8L32 24L32 40L8 40Z"/>'
    '<path id="b" fill="#b06040" d="M32 8L56 8L56 40L32 40L32 24Z"/>'
    "</g></svg>"
)


def permission(document):
    paths = {}
    for oid in ("a", "b"):
        paths[oid] = {
            "geometry": chains.fingerprint(document.geometry_for(oid).path_data()),
            "matrix": root_matrix(document, oid),
            "free": [
                value
                for _a, _b, value in chains.segments(document, oid)
                if all(point[0] == 32 for point in value)
            ],
            "chains": [{"id": "interior", "members": [1 if oid == "a" else 2]}],
        }
    return {"version": 1, "paths": paths}


def test_native_contacts_hold_while_both_copies_of_an_interior_chain_simplify():
    document = import_svg(SVG)
    metadata = permission(document)
    links = shared.links(document, ["a"], ["b"])
    held = {oid: chains.bind(document, oid, metadata) for oid in ("a", "b")}
    proposed = _geometry_proposal(
        document,
        "a",
        Image.new("RGB", (64, 64)),
        Options(),
        Work.start(10),
        frozenset({"a", "b"}),
        move=False,
        chain_constraints=metadata,
    )
    assert proposed is not None
    assert all(shared.intact(proposed, link) for link in links)
    assert all(hold.intact(proposed, oid) for oid, hold in held.items())
    assert sum(
        len(s.nodes) for oid in ("a", "b") for s in proposed.geometry_for(oid).subpaths
    ) < sum(
        len(s.nodes) for oid in ("a", "b") for s in document.geometry_for(oid).subpaths
    )
    updated = chains.refresh(metadata, document, proposed, ("a", "b"))
    assert all(chains.bind(proposed, oid, updated) is not None for oid in ("a", "b"))
    assert all(chains.bind(proposed, oid, metadata) is None for oid in ("a", "b"))
    assert metadata == permission(document)
    assert updated["paths"]["a"]["chains"] == metadata["paths"]["a"]["chains"]
    saved, _ = load_project(save_project(proposed))
    assert chains.bind(saved, "a", updated) is not None
    np.testing.assert_array_equal(
        render(export_svg(saved), (64, 64)), render(SVG, (64, 64))
    )


def test_a_whole_path_hold_still_blocks_an_unproved_neighbor():
    document = import_svg(SVG)
    metadata = permission(document)
    metadata["paths"].pop("b")
    assert (
        _geometry_proposal(
            document,
            "a",
            Image.new("RGB", (64, 64)),
            Options(),
            Work.start(10),
            frozenset({"a", "b"}),
            move=False,
            chain_constraints=metadata,
        )
        is None
    )


def test_stale_permissions_cannot_authorize_a_changed_native_contact():
    document = import_svg(SVG)
    metadata = permission(document)
    geometry = document.geometry_for("a")
    first = geometry.subpaths[0].nodes[0]
    changed = document.replace_geometry(
        geometry.replace_node(replace(first, values=(7.5, 8)))
    )
    assert chains.bind(changed, "a", metadata) is None
    assert (
        _geometry_proposal(
            changed,
            "a",
            Image.new("RGB", (64, 64)),
            Options(),
            Work.start(10),
            frozenset({"a", "b"}),
            move=False,
            chain_constraints=metadata,
        )
        is None
    )


def test_changed_ancestor_transform_invalidates_native_chain_permissions():
    document = import_svg(SVG)
    metadata = permission(document)
    parent = document.ancestry("a")[-2]
    changed = document.replace_element(
        replace(parent, attributes=(("transform", "translate(5 2) scale(.8)"),))
    )
    assert changed.geometry_for("a") == document.geometry_for("a")
    assert chains.bind(changed, "a", metadata) is None


def evidence():
    height, width = 80, 96
    y, x = np.indices((height, width))
    visible = (x >= 8) & (x < 88) & (y >= 8) & (y < 72)
    cut = 48 + np.rint(3 * np.sin(y / 4)).astype(int)
    labels = np.where(visible, np.where(x < cut, 1, 2), 0).astype(np.int32)
    labels[24:30, 18:24] = 0
    labels[2:6, 2:4] = 3
    shown = labels > 0
    target = np.full((height, width, 3), 255, dtype=np.float32)
    target[labels == 1] = (176, 100, 60)
    target[labels == 2] = (180, 100, 60)
    target[labels == 3] = (176, 100, 60)
    alpha = np.where(shown, 128 / 255, 0).astype(np.float32)
    alpha[labels == 3] = 1 / 255
    rgba = np.dstack((target / 255, alpha))
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
        alpha,
    )


def prepared(*, structure=False):
    source = evidence()
    options = Options(tolerance=0.1, gradients=False, refine=False)
    svg, details = export(
        source, source.labels, options, Work.start(10), layers=True, structure=structure
    )
    frontier = Frontier(Policy(source.rgba))
    assert frontier.add(svg, "Native contacts", details)
    frontier.freeze_normalizer()
    entry = frontier.entries[0]
    state = State(
        import_svg(svg),
        svg,
        LocalPolicy(frontier.policy).start(svg, entry.evaluation),
        entry.key,
        details,
        partition=Partition.from_metadata(details["planning_surfaces"]),
    )
    return source, frontier, state, options


@pytest.mark.parametrize("structure", [False, True])
def test_exported_permissions_reach_native_acceptance_with_holes_and_faint_marks(
    structure,
):
    source, frontier, state, options = prepared(structure=structure)
    metadata = state.details["chain_constraints"]
    assert state.details["native_hold_reasons"]["transparent-contact"] > 0
    assert state.details["native_hold_reasons"]["thin-component"] > 0
    assert set(metadata["paths"]) == {"cel-fill-1", "cel-fill-2"}
    assert all(
        chains.bind(state.document, oid, metadata) is not None
        for oid in metadata["paths"]
    )
    operators = Operators(source, build(source), replace(options, tolerance=2))
    proposals = list(operators.geometry(state, Work.start(10)))
    assert proposals
    before = render(state.svg, source.source_size)
    for proposal in proposals:
        after = render(export_svg(proposal.document), source.source_size)
        np.testing.assert_array_equal(after[..., 3], before[..., 3])
        assert after[24:30, 18:24, 3].max() == 0
        np.testing.assert_array_equal(after[2:6, 2:4], before[2:6, 2:4])
    result = search(frontier, options, Work.start(10), operators.geometry)
    assert result["accepted"] > 0
    assert result["checkpointed"] > 0
    assert result["score_disagreements"] == 0
    selected = frontier.select(50)
    assert selected.metrics["nodes"] < state.snapshot.evaluation.structure["nodes"]
    partition = Partition.from_metadata(selected.metrics["planning_surfaces"])
    partition.validate(import_svg(selected.svg))


@pytest.mark.parametrize("structure", [False, True])
def test_cpu_refinement_uses_chain_permissions_and_refreshes_accepted_geometry(
    structure,
):
    source, frontier, state, options = prepared(structure=structure)
    # Isolate geometry in this test; separate tests exercise paint/width fitting.
    state.details["paint_constraints"] = [s.id for s in state.partition.surfaces]
    frontier.entries[0].details["paint_constraints"] = state.details[
        "paint_constraints"
    ]
    result = refine(
        frontier, source, replace(options, refine=True, tolerance=2), Work.start(10)
    )
    assert result["accepted"] > 0
    assert any(
        stages.get("simplify", {}).get("accepted", 0) for stages in result["stages"]
    )
    selected = frontier.select(50)
    assert selected.metrics["nodes"] < state.snapshot.evaluation.structure["nodes"]
    document = import_svg(selected.svg)
    assert all(
        chains.bind(document, oid, selected.metrics["chain_constraints"]) is not None
        for oid in selected.metrics["chain_constraints"]["paths"]
    )
    np.testing.assert_array_equal(
        render(selected.svg, source.source_size)[..., 3],
        render(state.svg, source.source_size)[..., 3],
    )


def test_protected_cubic_handles_are_checked_in_addition_to_endpoints():
    document = import_svg(SVG.replace("M8 8L32 8", "M8 8C16 7.9 24 7.9 32 8"))
    metadata = permission(document)
    hold = chains.bind(document, "a", metadata)
    geometry = document.geometry_for("a")
    node = geometry.subpaths[0].nodes[1]
    changed = document.replace_geometry(
        geometry.replace_node(replace(node, command="L", values=node.endpoint))
    )
    assert not hold.intact(changed, "a")
    assert "a" not in chains.refresh(metadata, document, changed, ("a",))["paths"]


def test_stopped_fitting_does_not_publish_a_partly_fitted_native_contact():
    document = import_svg(SVG)
    work = Work.start(10)
    work.stop.set()
    assert (
        _geometry_proposal(
            document,
            "a",
            Image.new("RGB", (64, 64)),
            Options(),
            work,
            frozenset({"a", "b"}),
            move=False,
            chain_constraints=permission(document),
        )
        is None
    )


def test_family_replacement_discards_permissions_but_preserves_its_whole_path_hold():
    source, _frontier, state, options = prepared()
    proposal = next(Families(source, build(source), options)(state, Work.start(10)))
    assert not set(proposal.ids).intersection(
        proposal.details["chain_constraints"]["paths"]
    )
    survivor = next(
        s.id for s in proposal.partition.surfaces if set(s.members) == {1, 2}
    )
    assert survivor in proposal.details["geometry_constraints"]
    assert state.details["chain_constraints"]["paths"]


def union_fixture():
    document = import_svg(
        '<svg width="64" height="64"><g opacity=".5" '
        'transform="translate(4 2) scale(.8)">'
        '<path id="base" fill="#b05030" d="M8 8H56V40H8Z"/>'
        '<path id="a" fill="#b05030" d="M8 8H24V40H8Z"/>'
        '<path id="b" fill="#b05030" d="M24 8H40C40 12 40 20 41 24'
        'C42 28 42 36 42 40H24Z"/>'
        '<path id="c" fill="#b05030" d="M40 8H56V40H42'
        'C42 36 42 28 41 24C40 20 40 12 40 8Z"/></g></svg>'
    )
    records = {}
    for oid in ("a", "b", "c"):
        records[oid] = {
            "geometry": chains.fingerprint(document.geometry_for(oid).path_data()),
            "matrix": root_matrix(document, oid),
            "free": [
                value
                for _a, _b, value in chains.segments(document, oid)
                if len(value) == 4 or all(point[0] == 24 for point in value)
            ],
            "chains": [{"id": f"source-{oid}", "members": [ord(oid) - ord("a") + 1]}],
        }
    geometry = union_geometry(
        [document.geometry_for(oid) for oid in ("a", "b")],
        [path_style(document, document.element(oid)) for oid in ("a", "b")],
    )
    editor = Editor(document, selection=Selection(whole_document=True))
    with editor.transaction("Owned material union") as transaction:
        transaction.delete_objects(frozenset({"a"}))
        transaction.replace_geometry("b", geometry)
    return document, editor.snapshot.document, {"version": 1, "paths": records}


def test_exact_union_keeps_surviving_permissions_for_native_shared_fitting():
    before, document, metadata = union_fixture()
    parent_metadata = repr(metadata)
    updated = chains.merged(metadata, before, document, ("a", "b"), "b", Work.start(10))
    assert repr(metadata) == parent_metadata
    assert "a" not in updated["paths"]
    assert len(updated["paths"]["b"]["free"]) == 2
    hold = chains.bind(document, "b", updated)
    assert hold is not None
    svg = export_svg(document)
    truth = render(svg, (64, 64))
    proposed = _geometry_proposal(
        document,
        "b",
        Image.fromarray(np.rint(truth[..., :3] * 255).astype(np.uint8)),
        Options(tolerance=2),
        Work.start(10),
        frozenset({"base", "b", "c"}),
        move=False,
        chain_constraints=updated,
    )
    assert proposed is not None
    assert proposed.geometry_for("c") != document.geometry_for("c")
    assert proposed.geometry_for("base") == document.geometry_for("base")
    assert hold.intact(proposed, "b")
    actual = render(export_svg(proposed), (64, 64))
    np.testing.assert_array_equal(actual[..., 3], truth[..., 3])
    frontier = Frontier(Policy(truth))
    assert frontier.add(svg, "Owned union")
    full = frontier.policy.evaluate(export_svg(proposed))
    assert full.valid
    assert full.cost < frontier.baseline.evaluation.cost
    evaluator = LocalPolicy(frontier.policy)
    local = evaluator.update(
        evaluator.start(svg, frontier.baseline.evaluation),
        export_svg(proposed),
        bounds(document, proposed, ("b", "c")),
        full.structure,
    )
    assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
    assert local.canvas.matches(actual)
    refreshed = chains.refresh(updated, document, proposed, ("b", "c"))
    saved, _ = load_project(save_project(proposed))
    assert chains.bind(saved, "b", refreshed) is not None
    assert chains.bind(saved, "c", refreshed) is not None


@pytest.mark.parametrize("failure", ["stale", "frame", "stop", "bound"])
def test_union_never_recovers_stale_unframed_or_interrupted_permissions(
    failure, monkeypatch
):
    before, after, metadata = union_fixture()
    work = Work.start(10)
    if failure == "stale":
        metadata["paths"]["b"]["geometry"] = "stale"
    elif failure == "frame":
        editor = Editor(after, selection=Selection(whole_document=True))
        with editor.transaction("Changed survivor frame") as transaction:
            transaction.set_attributes("b", {"transform": "translate(1 0)"})
        after = editor.snapshot.document
    elif failure == "stop":
        work.stop.set()
    else:
        monkeypatch.setattr(chains, "MAX_PATH_NODES", 1)
    original = repr(metadata)
    updated = chains.merged(metadata, before, after, ("a", "b"), "b", work)
    assert "a" not in updated["paths"]
    assert "b" not in updated["paths"]
    assert updated["paths"]["c"] == metadata["paths"]["c"]
    assert repr(metadata) == original


def test_boolean_subdivisions_do_not_inherit_unproved_curve_permissions():
    before, _after, metadata = union_fixture()
    geometry = before.geometry_for("b")
    nodes = list(geometry.subpaths[0].nodes)
    curves = [i for i, n in enumerate(nodes) if n.command == "C"]
    nodes[curves[0]] = replace(nodes[curves[0]], values=(42, 12, 38, 20, 40, 24))
    nodes[curves[1]] = replace(nodes[curves[1]], values=(42, 28, 38, 36, 40, 40))
    before = before.replace_geometry(
        replace(geometry, subpaths=(replace(geometry.subpaths[0], nodes=tuple(nodes)),))
    )
    metadata["paths"]["b"]["geometry"] = chains.fingerprint(
        before.geometry_for("b").path_data()
    )
    metadata["paths"]["b"]["free"] = [
        value for _a, _b, value in chains.segments(before, "b") if len(value) == 4
    ]
    combined = union_geometry(
        [before.geometry_for(oid) for oid in ("a", "b")],
        [path_style(before, before.element(oid)) for oid in ("a", "b")],
    )
    after = before.replace_geometry(replace(combined, id=geometry.id))
    assert sum(n.command == "C" for s in combined.subpaths for n in s.nodes) == 4
    updated = chains.merged(metadata, before, after, ("a", "b"), "b", Work.start(10))
    assert "b" not in updated["paths"]


def test_structured_ellipse_stays_exact_beside_a_refinable_interior_curve():
    source = evidence()
    y, x = np.indices(source.labels.shape)
    labels = source.labels.copy()
    labels[(x - 26) ** 2 + (y - 52) ** 2 <= 10**2] = 4
    source = replace(source, labels=labels)
    options = Options(tolerance=1.2, gradients=False, refine=False)
    svg, metadata = export(source, labels, options, Work.start(10), structure=True)
    document = import_svg(svg)
    assert any(m["model"] == "ellipse" for m in metadata["geometry_models"])
    assert "cel-fill-4" in metadata["geometry_constraints"]
    assert chains.bind(document, "cel-fill-4", metadata["chain_constraints"]) is None
    assert sum(len(s.nodes) for s in document.geometry_for("cel-fill-4").subpaths) == 5
    assert (
        chains.bind(document, "cel-fill-1", metadata["chain_constraints"]) is not None
    )


@pytest.mark.parametrize("limit", ["chains", "segments", "source_points"])
def test_exhausted_permission_bounds_keep_conservative_whole_path_holds(
    limit, monkeypatch
):
    monkeypatch.setattr(
        chains,
        {
            "chains": "MAX_CHAINS",
            "segments": "MAX_SEGMENTS",
            "source_points": "MAX_SOURCE_POINTS",
        }[limit],
        0,
    )
    _source, _frontier, state, _options = prepared()
    assert not state.details["chain_constraints"]["paths"]
    assert state.details["chain_constraints"]["omitted_chains"] > 0
    assert state.details["geometry_constraints"]
