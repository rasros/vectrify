"""One original owner can render a stroke, exact residual and material patches."""

import json
from dataclasses import replace

import pytest

from tests.refine.test_cel_plan_search import setup
from vectrify.document import (
    Editor,
    Element,
    Selection,
    export_svg,
    import_svg,
    load_project,
    save_project,
)
from vectrify.document.holes import reversed_subpath
from vectrify.document.model import paint_server
from vectrify.document.paint import GradientStop, LinearGradient
from vectrify.document.svg import parse_path
from vectrify.refine.cel_plan.atoms import Atoms, Cut
from vectrify.refine.cel_plan.component_edits import ComponentEdit
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import Box, LocalPolicy
from vectrify.refine.cel_plan.model import StageInterruptedError, Work
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.proposals import Operators, bounds
from vectrify.refine.cel_plan.search import Proposal, State, identity, search
from vectrify.refine.cel_plan.source_family import FamilyPart, SourceFamily, revision

ORIGINAL = "M8 8H24V24H8Z"
REMOVED = "M8 8H24V10H8Z"


def fixture(*, opacity=1, transform="translate(0 0)", width=1, restoration=REMOVED):
    opening = (
        '<svg id="scene" width="40" height="40"><g id="body" '
        f'opacity="{opacity}" transform="{transform}">'
        '<path id="material" d="M4 4H28V28H4Z" fill="#bb9944"/>'
    )
    before = import_svg(
        opening + f'<path id="ink" d="{ORIGINAL}" fill="#222222"/></g></svg>'
    )
    reversed_field = reversed_subpath(parse_path(REMOVED).subpaths[0])
    hole = replace(parse_path(REMOVED), subpaths=(reversed_field,)).path_data()
    after = import_svg(
        opening + f'<path id="residual" d="{ORIGINAL} {hole}" fill="#222222"/>'
        f'<path id="restoration" d="{restoration}" fill="#bb9944"/>'
        f'<path id="ink" d="M10 9H22" fill="none" stroke="#222222" '
        f'stroke-width="{width}" stroke-linecap="round"/></g></svg>'
    )
    # Preserve the actual original donor object and its native geometry IDs.
    after = after.replace_element(before.element("material"))
    after = replace(
        after, geometries=(*after.geometries, before.geometry_for("material"))
    )
    parts = (
        FamilyPart("ink", "stroke"),
        FamilyPart("residual", "residual"),
        FamilyPart("restoration", "restoration", "material"),
    )
    return before, after, parts


def bind(before, after, parts):
    return SourceFamily.bind(
        before, after, "ink", parse_path(REMOVED), parts, (40, 40), Work.start(10)
    )


def paint(document, oid, key, value):
    element = document.element(oid)
    attributes = dict(element.attributes)
    attributes[key] = value
    return document.replace_element(
        replace(element, attributes=tuple(attributes.items()))
    )


def reseal(family, document):
    # Corruption controls rebind digests so geometry/paint checks, rather than
    # merely stale hashes, must reject the bad physical interpretation.
    return replace(
        family,
        revisions=tuple(
            sorted((oid, revision(document, oid)) for oid in family.dependencies)
        ),
    )


@pytest.mark.parametrize(
    ("opacity", "transform"), [(1, "translate(0 0)"), (0.6, "translate(3 2)")]
)
def test_physical_family_retains_original_ownership_and_survives_native_reload(
    opacity, transform
):
    before, after, parts = fixture(opacity=opacity, transform=transform)
    family = bind(before, after, parts)
    original = Partition((Surface("ink", (1,)), Surface("material", (2,))))
    planned = original.with_family(family)
    assert planned.surfaces is original.surfaces
    assert planned.owners == original.owners
    assert planned.atoms is original.atoms
    assert planned.follows(original)
    planned.validate(after)
    assert planned.metadata()["version"] == 2
    assert original.metadata()["version"] == 1
    loaded, _ = load_project(save_project(after))
    restored = Partition.from_metadata(json.loads(json.dumps(planned.metadata())))
    assert restored == planned
    assert restored is not None
    restored.validate(loaded)
    assert identity(export_svg(after), planned) != identity(export_svg(after), original)
    contract = ComponentEdit.bind(before, original, "body", Work.start(10))
    assert contract.validate(
        before,
        after,
        original,
        planned,
        tuple(family.dependencies),
        bounds(before, after, tuple(family.dependencies)),
        Work.start(10),
    )


def test_saturated_atom_ledger_is_unchanged_by_physical_path_declaration():
    before, after, parts = fixture()
    # All 64 existing entries and 128 children remain exact. This declaration
    # changes physical rendering only, never the original pixel subdivisions.
    atoms = Atoms(
        "a" * 64,
        (1, 130),
        65,
        tuple(Cut(i, ((0, 2 * i, 2 * i + 1),), (1, 1)) for i in range(64)),
    )
    original = Partition(
        (Surface("ink", (64,)), Surface("material", tuple(range(65, 193)))), atoms
    )
    planned = original.with_family(bind(before, after, parts))
    planned.validate(after)
    assert planned.atoms is atoms
    assert planned.owners == original.owners
    assert planned.atoms.metadata() == atoms.metadata()
    assert len(planned.atoms.cuts) == 64
    assert planned.atoms.namespace_count - planned.atoms.count == 128


def test_private_gradient_clone_and_equivalent_opacity_preserve_material():
    before, after, parts = fixture(opacity=0.6)
    gradient = LinearGradient(
        (4, 4), (28, 28), (GradientStop(0, "#bb9944"), GradientStop(1, "#887722"))
    )
    documents = []
    for document in (before, after):
        editor = Editor(document, selection=Selection(whole_document=True))
        with editor.transaction("Private material gradients") as tx:
            tx.set_fill("material", gradient)
            if document is after:
                tx.set_fill("restoration", gradient)
                tx.set_attributes("restoration", {"fill-opacity": "1.0"})
        documents.append(editor.snapshot.document)
    before, after = documents
    original_server = paint_server(before.element("material").get("fill"))
    after_server = paint_server(after.element("material").get("fill"))
    assert original_server is not None
    assert after_server is not None
    after = after.replace_element(before.element("material"))
    definitions = after.ancestry(after_server)[-2]
    after = after.replace_element(
        replace(
            definitions,
            children=tuple(
                before.element(original_server) if e.id == after_server else e
                for e in definitions.children
            ),
        )
    )
    family = bind(before, after, parts)
    loaded, _ = load_project(save_project(after))
    family.validate(loaded)
    patch_server = paint_server(loaded.element("restoration").get("fill"))
    assert patch_server is not None
    changed = paint(
        loaded, loaded.element(patch_server).children[0].id, "stop-color", "red"
    )
    with pytest.raises(ValueError, match="paint/frame"):
        reseal(family, changed).validate(changed)


@pytest.mark.parametrize("oid", ["ink", "residual", "restoration", "material", "body"])
def test_family_revisions_reject_changed_part_material_or_ancestor(oid):
    before, after, parts = fixture()
    family = bind(before, after, parts)
    changed = paint(after, oid, "transform", "translate(1 0)")
    with pytest.raises(ValueError, match="changed"):
        family.validate(changed)


@pytest.mark.parametrize(
    "kind", ["filled", "residual", "material", "escape", "hole", "order"]
)
def test_resealed_invalid_geometry_cannot_claim_physical_family(kind):
    before, after, parts = fixture()
    family = bind(before, after, parts)
    if kind == "filled":
        changed = paint(after, "ink", "fill", "black")
    elif kind == "residual":
        geometry = after.geometry_for("residual")
        changed = replace(
            after,
            geometries=tuple(
                replace(g, subpaths=g.subpaths[:1]) if g.id == geometry.id else g
                for g in after.geometries
            ),
        )
    elif kind == "material":
        changed = paint(after, "restoration", "fill", "black")
    elif kind in {"escape", "hole"}:
        changed = fixture(
            restoration="M7 8H24V10H7Z" if kind == "escape" else "M8 8H23V10H8Z"
        )[1]
    else:
        group = after.element("body")
        changed = after.replace_element(
            replace(group, children=tuple(reversed(group.children)))
        )
    with pytest.raises(ValueError, match="Source family"):
        reseal(family, changed).validate(changed)


def test_alpha_changes_and_original_material_edits_reject_binding():
    before, after, parts = fixture(width=30)
    with pytest.raises(ValueError, match="native alpha"):
        bind(before, after, parts)
    before, after, parts = fixture()
    changed = paint(after, "material", "fill", "blue")
    with pytest.raises(ValueError, match="original material"):
        bind(before, changed, parts)


def test_component_contract_requires_all_parts_and_material_dependencies():
    before, after, parts = fixture()
    original = Partition((Surface("ink", (1,)), Surface("material", (2,))))
    planned = original.with_family(bind(before, after, parts))
    contract = ComponentEdit.bind(before, original, "body", Work.start(10))
    with pytest.raises(ValueError, match="every new family"):
        contract.validate(
            before,
            after,
            original,
            planned,
            ("ink", "residual", "restoration"),
            Box(0, 0, 40, 40),
            Work.start(10),
        )


def test_family_is_not_lost_or_aliased_in_structural_ownership():
    before, after, parts = fixture()
    original = Partition((Surface("ink", (1,)), Surface("material", (2,))))
    planned = original.with_family(bind(before, after, parts))
    assert not original.follows(planned)
    with pytest.raises(ValueError, match="atomic family"):
        planned.replace(("ink",), (Surface("new", (1,)),))
    changed = planned.replace(("material",), (Surface("other", (2,)),))
    assert changed.families == planned.families
    assert changed.follows(planned)
    with pytest.raises(ValueError, match="distinct primary"):
        replace(planned, surfaces=(*planned.surfaces, Surface("residual", (3,))))
    with pytest.raises(ValueError, match="distinct primary"):
        planned.with_family(planned.families[0])
    metadata = planned.metadata()
    metadata["version"] = 1
    with pytest.raises(ValueError, match="version 2"):
        Partition.from_metadata(metadata)


def test_component_rejects_forged_parent_snapshot_and_reassigned_source_family():
    before, after, parts = fixture()
    original = Partition((Surface("ink", (1,)), Surface("material", (2,))))
    family = bind(before, after, parts)
    contract = ComponentEdit.bind(before, original, "body", Work.start(10))
    changed = paint(before, "ink", "fill", "#232323")
    forged = replace(
        family, attributes=tuple(sorted(changed.element("ink").attributes))
    )
    planned = original.with_family(forged)
    with pytest.raises(ValueError, match=r"retained shadow|original owner"):
        contract.validate(
            before,
            after,
            original,
            planned,
            tuple(family.dependencies),
            Box(0, 0, 40, 40),
            Work.start(10),
        )
    reassigned = Partition(
        (Surface("ink", (2,)), Surface("material", (1,))), families=(family,)
    )
    with pytest.raises(ValueError, match="reassigned physical family"):
        contract.validate(
            before,
            after,
            original,
            reassigned,
            tuple(family.dependencies),
            Box(0, 0, 40, 40),
            Work.start(10),
        )


def test_sealed_parent_snapshot_cannot_be_forged_even_when_final_geometry_matches():
    before, after, parts = fixture()
    family = bind(before, after, parts)
    altered = paint(before, "ink", "fill", "#232323")
    with pytest.raises(ValueError, match="original owner"):
        family.validate_before(altered, after)


@pytest.mark.parametrize("connected", [False, True])
def test_operator_scheduling_preserves_family_while_allowing_independent_work(
    monkeypatch,
    connected,
):
    before, after, parts = connected_fixture() if connected else fixture()
    geometry = parse_path("M30 8H36V14H30Z")
    element = Element(
        "sibling", "path", (("fill", "#906030"),), geometry_id=geometry.id
    )
    documents = []
    for document in (before, after):
        editor = Editor(document, selection=Selection(whole_document=True))
        with editor.transaction("Add independent control") as tx:
            tx.insert_object("body", element, geometries=(geometry,))
        documents.append(editor.snapshot.document)
    before, after = documents
    family = (
        connected_bind(before, after, parts)
        if connected
        else bind(before, after, parts)
    )
    original = Partition(
        (Surface("ink", (1,)), Surface("material", (2,)), Surface("sibling", (3,)))
    )
    planned = original.with_family(family)
    svg = export_svg(after)
    frontier, evidence, options = setup(initial=svg, target=svg, size=(40, 40))
    entry = frontier.baseline
    assert entry is not None
    state = State(
        after,
        svg,
        LocalPolicy(frontier.policy).start(svg, entry.evaluation),
        identity(svg, planned),
        entry.details,
        partition=planned,
    )
    operators = Operators(evidence, build(evidence), options)
    calls = []
    for name in (
        "families",
        "overlays",
        "paint",
        "geometry",
        "ink",
        "replacements",
        "ridges",
    ):

        def edits(current, _work, name=name):
            calls.append(name)
            assert family.dependencies <= set(current.details["paint_constraints"])
            assert family.dependencies <= set(current.details["geometry_constraints"])
            if name != "families":
                return
            for oid in (("bar",) if connected else ()) + ("material", "sibling"):
                changed = paint(current.document, oid, "fill", "#887733")
                yield Proposal(
                    "control",
                    (oid,),
                    (),
                    current.key,
                    changed,
                    bounds(current.document, changed, (oid,)),
                    partition=original,
                )

        monkeypatch.setattr(operators, name, edits)
    proposals = list(operators(state, Work.start(10)))
    assert len(proposals) == 1
    assert proposals[0].ids == ("sibling",)
    assert proposals[0].partition == planned
    assert set(calls) == {
        "families",
        "overlays",
        "paint",
        "geometry",
        "ink",
        "replacements",
        "ridges",
    }


def test_interrupted_binding_does_not_publish_a_partial_family():
    before, after, parts = fixture()
    work = Work.start(0)
    with pytest.raises(StageInterruptedError):
        SourceFamily.bind(
            before, after, "ink", parse_path(REMOVED), parts, (40, 40), work
        )


@pytest.mark.parametrize("field", ["parts", "removed", "frame", "revisions"])
def test_incomplete_family_metadata_is_rejected(field):
    before, after, parts = fixture()
    metadata = bind(before, after, parts).metadata()
    metadata.pop(field)
    with pytest.raises(ValueError, match="metadata"):
        SourceFamily.from_metadata(metadata)


def connected_fixture(**kwargs):
    before, after, parts = fixture(**kwargs)
    geometry = parse_path("M22 9H10")
    element = Element(
        "bar",
        "path",
        (("fill", "none"), ("stroke", "#222222"), ("stroke-width", "1")),
        geometry_id=geometry.id,
    )
    documents = []
    for document in (before, after):
        editor = Editor(document, selection=Selection(whole_document=True))
        with editor.transaction("Add original junction") as tx:
            tx.insert_object("body", element, geometries=(geometry,))
        documents.append(editor.snapshot.document)
    return documents[0], documents[1], parts


def connected_bind(before, after, parts):
    return SourceFamily.bind(
        before,
        after,
        "ink",
        parse_path(REMOVED),
        parts,
        (40, 40),
        Work.start(10),
        junction="bar",
    )


@pytest.mark.parametrize("transform", ["translate(0 0)", "matrix(.8 .6 -.6 .8 15 2)"])
def test_original_junction_ports_and_dependency_survive_native_reload(transform):
    before, after, parts = connected_fixture(opacity=0.6, transform=transform)
    family = connected_bind(before, after, parts)
    assert family.junction == "bar"
    assert "bar" in family.dependencies
    assert "bar" not in family.paths
    loaded, _ = load_project(save_project(after))
    restored = SourceFamily.from_metadata(json.loads(json.dumps(family.metadata())))
    assert restored == family
    restored.validate(loaded)
    original = Partition((Surface("ink", (1,)), Surface("material", (2,))))
    planned = original.with_family(restored)
    assert "bar" in planned.family_dependencies
    contract = ComponentEdit.bind(before, original, "body", Work.start(10))
    with pytest.raises(ValueError, match="every new family"):
        contract.validate(
            before,
            after,
            original,
            planned,
            tuple(family.dependencies - {"bar"}),
            Box(0, 0, 40, 40),
            Work.start(10),
        )
    assert contract.validate(
        before,
        after,
        original,
        planned,
        tuple(family.dependencies),
        bounds(before, after, tuple(family.dependencies)),
        Work.start(10),
    )


@pytest.mark.parametrize("kind", ["stroke", "junction", "closed", "fill", "frame"])
def test_resealed_family_cannot_lose_its_literal_connection(kind):
    before, after, parts = connected_fixture()
    family = connected_bind(before, after, parts)
    oid = "ink" if kind == "stroke" else "bar"
    if kind in {"stroke", "junction", "closed"}:
        old = after.geometry_for(oid)
        path = "M10 9H21" if kind != "closed" else "M10 9H22Z"
        changed = replace(
            after,
            geometries=tuple(
                replace(g, subpaths=parse_path(path).subpaths) if g.id == old.id else g
                for g in after.geometries
            ),
        )
    else:
        changed = paint(
            after,
            "bar",
            "fill" if kind == "fill" else "transform",
            "black" if kind == "fill" else "translate(1 0)",
        )
    with pytest.raises(ValueError, match="junction"):
        reseal(family, changed).validate(changed)


def test_junction_must_be_an_unchanged_original_and_cannot_alias_material():
    before, after, parts = connected_fixture()
    changed = paint(after, "bar", "stroke", "red")
    with pytest.raises(ValueError, match="original junction"):
        connected_bind(before, changed, parts)
    family = connected_bind(before, after, parts)
    with pytest.raises(ValueError, match="literal junction"):
        replace(family, junction="material")
    without = fixture()[0]
    with pytest.raises(ValueError, match="actual parent"):
        connected_bind(without, after, parts)
    with pytest.raises(ValueError, match="invented"):
        family.validate_before(without, after)


@pytest.mark.parametrize("field", ["junction", "ports"])
def test_partial_junction_metadata_is_rejected(field):
    before, after, parts = connected_fixture()
    metadata = connected_bind(before, after, parts).metadata()
    metadata.pop(field)
    with pytest.raises(ValueError, match="junction"):
        SourceFamily.from_metadata(metadata)


def two_families():
    first_before, first_after, first_parts = connected_fixture()
    second_before, second_after, second_parts = fixture(transform="translate(50 0)")

    def renamed(element):
        return replace(
            element,
            id="two-" + element.id,
            children=tuple(renamed(c) for c in element.children),
        )

    def combined(first, second):
        return replace(
            first,
            root=replace(
                first.root,
                attributes=(("width", "100"), ("height", "40")),
                children=(
                    *first.root.children,
                    *(renamed(c) for c in second.root.children),
                ),
            ),
            geometries=(*first.geometries, *second.geometries),
        )

    initial = combined(first_before, second_before)
    middle = combined(first_after, second_before)
    final = combined(first_after, second_after)
    original = Partition(
        (
            Surface("ink", (1,)),
            Surface("material", (2,)),
            Surface("two-ink", (3,)),
            Surface("two-material", (4,)),
        )
    )
    first = SourceFamily.bind(
        initial,
        middle,
        "ink",
        parse_path(REMOVED),
        first_parts,
        (100, 40),
        Work.start(10),
        junction="bar",
    )
    second_parts = tuple(
        replace(
            p, id="two-" + p.id, material="two-" + p.material if p.material else None
        )
        for p in second_parts
    )
    second = SourceFamily.bind(
        middle,
        final,
        "two-ink",
        parse_path(REMOVED),
        second_parts,
        (100, 40),
        Work.start(10),
    )
    planned = original.with_family(first)
    return middle, final, planned, planned.with_family(second)


@pytest.mark.parametrize("sealed", [False, True])
def test_scheduler_retains_a_second_family_only_with_its_component_seal(
    monkeypatch, sealed
):
    middle, final, original, planned = two_families()
    frontier, evidence, options = setup(
        initial=export_svg(middle), target=export_svg(final), size=(100, 40)
    )
    entry = frontier.baseline
    assert entry is not None
    state = State(
        middle,
        export_svg(middle),
        LocalPolicy(frontier.policy).start(entry.svg, entry.evaluation),
        identity(entry.svg, original),
        {},
        partition=original,
    )
    second = planned.families[1]
    ids = tuple(sorted(second.dependencies))
    component = ComponentEdit.bind(middle, original, "two-body", Work.start(10))
    operators = Operators(evidence, build(evidence), options)
    for name in (
        "families",
        "overlays",
        "paint",
        "geometry",
        "ink",
        "replacements",
        "ridges",
    ):

        def proposals(current, _work, name=name):
            if name == "families":
                yield Proposal(
                    "control",
                    ids,
                    (),
                    current.key,
                    final,
                    bounds(middle, final, ids),
                    partition=planned,
                    component=component if sealed else None,
                )

        monkeypatch.setattr(operators, name, proposals)
    accepted = list(operators(state, Work.start(10)))
    assert len(accepted) == int(sealed)
    if sealed:
        assert accepted[0].partition == planned
        assert planned.families[0] == original.families[0]
        assert component.validate(
            middle, final, original, planned, ids, accepted[0].bounds, Work.start(10)
        )


def test_search_can_resume_native_family_metadata_without_reparsing_editor_ids():
    middle, final, original, _ = two_families()
    frontier, _, options = setup(
        initial=export_svg(middle), target=export_svg(middle), size=(100, 40)
    )
    entry = frontier.baseline
    assert entry is not None
    entry = replace(entry, details={"planning_surfaces": original.metadata()})
    frontier.baseline = entry
    frontier.entries = [entry]
    loaded, _ = load_project(save_project(middle))
    report = search(
        frontier,
        options,
        Work.start(10),
        lambda _state, _work: iter(()),
        seed_document=loaded,
    )
    assert report["status"] == "complete"
    assert report["accepted"] == 0
    with pytest.raises(ValueError, match="selected drawing"):
        search(
            frontier,
            options,
            Work.start(10),
            lambda _state, _work: iter(()),
            seed_document=final,
        )
