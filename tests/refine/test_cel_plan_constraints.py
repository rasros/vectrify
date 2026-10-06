"""Interior fitting preserves the exact native complement of each path."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from vectrify.document import export_svg, import_svg, load_project, save_project
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
from vectrify.refine.cel_plan.proposals import Operators
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


def prepared():
    source = evidence()
    options = Options(tolerance=0.1, gradients=False, refine=False)
    svg, details = export(source, source.labels, options, Work.start(10), layers=True)
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


def test_exported_permissions_reach_native_acceptance_with_holes_and_faint_marks():
    source, frontier, state, options = prepared()
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


def test_cpu_refinement_uses_chain_permissions_and_refreshes_accepted_geometry():
    source, frontier, state, options = prepared()
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
