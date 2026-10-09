"""Whole-component compaction keeps exact dependencies and checkpoint rules."""

from dataclasses import replace

import numpy as np
import pytest

from tests.helpers import required
from vectrify.document import Editor, Selection, export_svg, import_svg
from vectrify.document.paint import GradientStop, LinearGradient
from vectrify.refine.cel_plan import component_edits as contract
from vectrify.refine.cel_plan import search as beam
from vectrify.refine.cel_plan.component_edits import ComponentEdit
from vectrify.refine.cel_plan.frontier import Frontier
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.proposals import bounds
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import Proposal, Rejections, State, search


def fixture(count=300):
    # Redundant source-owned fragments hidden by the last rectangle. Removing
    # them is an exact, large cost reduction, with unchanged external paint.
    svg = (
        '<svg width="64" height="64"><defs><linearGradient id="shared">'
        '<stop offset="0" stop-color="red"/>'
        '<stop offset="1" stop-color="blue"/></linearGradient></defs>'
        '<g id="body">'
        + "".join(
            f'<path id="p{i}" fill="#904020" d="M8 8H40V56H8Z"/>' for i in range(count)
        )
        + '</g><path id="outside" fill="url(#shared)" '
        'd="M48 8H60V56H48Z"/></svg>'
    )
    partition = Partition(
        (
            *(Surface(f"p{i}", (i,)) for i in range(count)),
            Surface("outside", (count,)),
        )
    )
    frontier = Frontier(Policy(render(svg, (64, 64))))
    assert frontier.add(svg, "Initial", {"planning_surfaces": partition.metadata()})
    frontier.freeze_normalizer()
    entry = frontier.baseline
    assert entry is not None
    state = State(
        import_svg(svg),
        svg,
        LocalPolicy(frontier.policy).start(svg, entry.evaluation),
        entry.key,
        entry.details,
        partition=partition,
    )
    return frontier, state


def compact(state, work):
    ids = tuple(c.id for c in state.document.element("body").children)
    removed = ids[:-1]
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Component compaction") as transaction:
        transaction.delete_objects(frozenset(removed))
    document = editor.snapshot.document
    partition = state.partition.replace(
        ids, (Surface(ids[-1], tuple(range(len(ids)))),)
    )
    return Proposal(
        "component-test",
        removed,
        ("compact",),
        state.key,
        document,
        bounds(state.document, document, removed),
        dependencies=("body",),
        partition=partition,
        component=ComponentEdit.bind(state.document, state.partition, "body", work),
    )


def validate(state, edit, work=None):
    return edit.component.validate(
        state.document,
        edit.document,
        state.partition,
        edit.partition,
        edit.ids,
        edit.bounds,
        work or Work.start(10),
    )


def paint(document, oid, value, attribute="fill"):
    element = document.element(oid)
    attrs = dict(element.attributes)
    attrs[attribute] = value
    return document.replace_element(replace(element, attributes=tuple(attrs.items())))


@pytest.mark.parametrize("sealed", [False, True])
def test_broad_compaction_requires_seal_and_independent_native_checkpoint(sealed):
    frontier, _state = fixture()
    before = frontier.baseline

    def edits(state, work):
        if not state.edits:
            edit = compact(state, work)
            yield edit if sealed else replace(edit, component=None)

    report = search(frontier, Options(refine=False), Work.start(10), edits)
    decision = report["decisions"][0]
    if not sealed:
        assert report["attempted"] == 0
        assert decision["rejections"] == ["local-dependency-limit"]
        assert before is not None
        assert frontier.select(50).svg == before.svg
        return
    assert report["attempted"] == report["accepted"] == report["checkpointed"] == 1
    assert report["score_disagreements"] == 0
    assert decision["component_dependency"]["declared_objects"] == 299
    assert len(decision["ids"]) == 299
    assert len(decision["dependency_revisions"]) == 300
    chosen = frontier.select(50)
    assert chosen.metrics["paths"] == 2
    assert chosen.metrics["nodes"] == 8
    assert before is not None
    assert chosen.metrics["representation_cost"] < before.evaluation.cost
    np.testing.assert_array_equal(
        render(chosen.svg, (64, 64)), render(before.svg, (64, 64))
    )
    required(Partition.from_metadata(chosen.metrics["planning_surfaces"])).validate(
        import_svg(chosen.svg)
    )
    assert frontier.baseline is before


@pytest.mark.parametrize(
    "change",
    ["paint", "geometry", "gradient", "order", "frame", "locks", "pins", "ownership"],
)
def test_sealed_dependencies_detect_hidden_and_metadata_changes(change):
    _frontier, state = fixture(3)
    edit = compact(state, Work.start(10))
    document = state.document
    if change == "paint":
        document = paint(document, "p0", "blue")
        np.testing.assert_array_equal(
            render(export_svg(document), (64, 64)), render(state.svg, (64, 64))
        )
    elif change in {"geometry", "pins"}:
        geometry = document.geometry_for("p0")
        subpath = geometry.subpaths[0]
        node = subpath.nodes[0]
        node = (
            replace(node, pinned=True)
            if change == "pins"
            else replace(node, values=(9, 8))
        )
        geometry = replace(
            geometry, subpaths=(replace(subpath, nodes=(node, *subpath.nodes[1:])),)
        )
        document = replace(
            document,
            geometries=tuple(
                geometry if g.id == geometry.id else g for g in document.geometries
            ),
        )
    elif change == "gradient":
        gradient = document.element("shared")
        stop = gradient.children[0]
        document = paint(document, stop.id, "black", "stop-color")
    elif change == "order":
        group = document.element("body")
        document = document.replace_element(
            replace(group, children=tuple(reversed(group.children)))
        )
    elif change == "frame":
        group = document.element("body")
        document = document.replace_element(
            replace(group, attributes=(("transform", "translate(1 0)"),))
        )
    elif change == "locks":
        element = document.element("p0")
        document = document.replace_element(
            replace(element, locks=frozenset({"geometry"}))
        )
    else:
        assert state.partition is not None
        surfaces = state.partition.surfaces
        state = replace(
            state,
            partition=Partition(
                (
                    replace(surfaces[0], members=(1,)),
                    replace(surfaces[1], members=(0,)),
                    *surfaces[2:],
                )
            ),
        )
    with pytest.raises(ValueError, match="stale source"):
        validate(replace(state, document=document), edit)


@pytest.mark.parametrize(
    "change",
    [
        "undeclared",
        "external",
        "shared-gradient",
        "frame",
        "order",
        "bounds",
        "foreign-id",
        "duplicate-id",
    ],
)
def test_batch_cannot_hide_changes_or_omit_paint_bounds(change):
    _frontier, state = fixture(4)
    edit = compact(state, Work.start(10))
    if change == "undeclared":
        edit = replace(edit, document=paint(edit.document, "p3", "blue"))
    elif change == "external":
        edit = replace(edit, document=paint(edit.document, "outside", "blue"))
    elif change == "shared-gradient":
        gradient = edit.document.element("shared")
        edit = replace(
            edit,
            document=paint(
                edit.document, gradient.children[0].id, "blue", "stop-color"
            ),
        )
    elif change == "frame":
        group = edit.document.element("body")
        edit = replace(
            edit,
            document=edit.document.replace_element(
                replace(group, attributes=(("opacity", "0.5"),))
            ),
        )
    elif change == "order":
        # Keep two undeclared survivors; swapping their order must fail.
        group = state.document.element("body")
        document = state.document.replace_element(
            replace(group, children=tuple(reversed(group.children[2:])))
        )
        edit = replace(edit, document=document, ids=("p0", "p1"))
    elif change == "bounds":
        edit = replace(edit, bounds=replace(edit.bounds, right=20))
    elif change == "foreign-id":
        edit = replace(edit, ids=(*edit.ids, "outside"))
    else:
        edit = replace(edit, ids=(*edit.ids, edit.ids[0]))
    with pytest.raises(ValueError, match="Component replacement"):
        validate(state, edit)


@pytest.mark.parametrize("protection", ["path-lock", "parent-lock", "pin"])
def test_protected_paths_cannot_be_replaced_by_declaring_them(protection):
    _frontier, state = fixture(3)
    edit = compact(state, Work.start(10))
    document = state.document
    if protection == "pin":
        geometry = document.geometry_for("p0")
        subpath = geometry.subpaths[0]
        geometry = replace(
            geometry,
            subpaths=(
                replace(
                    subpath,
                    nodes=(replace(subpath.nodes[0], pinned=True), *subpath.nodes[1:]),
                ),
            ),
        )
        document = replace(
            document,
            geometries=tuple(
                geometry if g.id == geometry.id else g for g in document.geometries
            ),
        )
    else:
        oid = "body" if protection == "parent-lock" else "p0"
        element = document.element(oid)
        document = document.replace_element(
            replace(element, locks=frozenset({"structure"}))
        )
        if protection == "parent-lock":
            edit = replace(
                edit,
                document=edit.document.replace_element(
                    replace(edit.document.element(oid), locks=frozenset({"structure"}))
                ),
            )
    state = replace(state, document=document)
    assert state.partition is not None
    edit = replace(
        edit,
        component=ComponentEdit.bind(document, state.partition, "body", Work.start(10)),
    )
    with pytest.raises(ValueError, match="protected path"):
        validate(state, edit)


def test_private_gradient_changes_are_allowed_and_distinguish_cached_targets():
    _frontier, state = fixture(3)
    edit = compact(state, Work.start(10))
    oid = "p2"
    editor = Editor(edit.document, selection=Selection(whole_document=True))
    with editor.transaction("Private paint") as transaction:
        transaction.set_fill(
            oid,
            LinearGradient(
                (8, 8), (40, 8), (GradientStop(0, "red"), GradientStop(1, "blue"))
            ),
        )
    edit = replace(edit, document=editor.snapshot.document, ids=(*edit.ids, oid))
    first = validate(state, edit)
    cache = Rejections()
    key = cache.key(state, edit, -10, component_revision=first)
    cache.put(key, ("local-objective-regression",))
    assert cache.get(
        cache.key(state, edit, -10, component_revision=validate(state, edit))
    ) == ("local-objective-regression",)
    gradient = next(e for e in edit.document.elements() if e.paint_owner == oid)
    changed = replace(
        edit,
        document=paint(edit.document, gradient.children[0].id, "black", "stop-color"),
    )
    assert (
        cache.get(
            cache.key(state, changed, -10, component_revision=validate(state, changed))
        )
        is None
    )
    with pytest.raises(ValueError, match="validated target"):
        cache.key(state, edit, -10)


def test_rebound_hidden_source_edit_cannot_reuse_a_component_rejection():
    _frontier, state = fixture(3)
    edit = compact(state, Work.start(10))
    cache = Rejections()
    key = cache.key(state, edit, -10, component_revision=validate(state, edit))
    cache.put(key, ("local-objective-regression",))
    changed = replace(state, document=paint(state.document, "p0", "blue"))
    rebound = compact(changed, Work.start(10))
    # The hidden change affects neither full raster nor replacement SVG.
    assert export_svg(rebound.document) == export_svg(edit.document)
    np.testing.assert_array_equal(
        render(export_svg(changed.document), (64, 64)), render(state.svg, (64, 64))
    )
    assert (
        cache.get(
            cache.key(
                changed, rebound, -10, component_revision=validate(changed, rebound)
            )
        )
        is None
    )


def test_component_footprint_and_stop_are_enforced_before_publication(monkeypatch):
    frontier, state = fixture()
    before = frontier.baseline
    edit = compact(state, Work.start(10))
    monkeypatch.setattr(contract, "MAX_OBJECTS", 20)
    with pytest.raises(ValueError, match="object bounds"):
        validate(state, edit)
    monkeypatch.setattr(contract, "MAX_OBJECTS", 8192)
    work = Work.start(10)
    work.stop.set()
    with pytest.raises(StageInterruptedError):
        validate(state, edit, work)
    assert frontier.baseline is before


@pytest.mark.parametrize("failure", ["stop", "incorrect-local-score"])
def test_component_local_acceptance_does_not_bypass_checkpoint_rollback(
    failure, monkeypatch
):
    frontier, _state = fixture()
    before = frontier.baseline
    work = Work.start(10)
    if failure == "incorrect-local-score":
        original = beam.LocalPolicy.update

        def inaccurate(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            return replace(
                result,
                evaluation=replace(
                    result.evaluation, terms={**result.evaluation.terms, "visual": -1.0}
                ),
            )

        monkeypatch.setattr(beam.LocalPolicy, "update", inaccurate)

    def edits(state, local_work):
        if not state.edits:
            yield compact(state, local_work)
            if failure == "stop":
                work.stop.set()

    report = search(frontier, Options(refine=False), work, edits)
    assert report["accepted"] == 1
    assert report["checkpointed"] == (failure == "incorrect-local-score")
    assert before is not None
    assert frontier.select(50).svg == before.svg
    assert report["score_disagreements"] == (failure == "incorrect-local-score")
