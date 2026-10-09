"""Compact faint components retain source support, ownership and edit scope."""

from dataclasses import replace

import numpy as np
import pytest

from tests.helpers import required
from vectrify.document import (
    Editor,
    Element,
    Selection,
    export_svg,
    import_svg,
    load_project,
    save_project,
)
from vectrify.document.paint import GradientStop, LinearGradient
from vectrify.refine import cel
from vectrify.refine.cel_plan import opacity_fields as fields
from vectrify.refine.cel_plan.atoms import Atoms
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Evidence, StageInterruptedError, Work
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import State


def fixture(*, hole=False, frame=(1, 1, 0, 0), stepped=False):
    labels = np.zeros((64, 80), np.int32)
    labels[8:20, 8:14] = 1
    labels[8:20, 14:20] = 2
    if hole:
        labels[11:17, 11:17] = 0
    # Independent component with a one-pixel empty gap.
    labels[8:20, 21:27] = 3
    labels[8:20, 27:33] = 4
    # A sub-four-pixel component also stays present.
    labels[48, 44] = 5
    labels[48, 45] = 6
    labels[32:44, 48:60] = 7
    alpha = np.zeros(labels.shape, np.float32)
    alpha[labels > 0] = 1 / 255
    alpha[labels == 6] = 2 / 255
    if stepped:
        alpha[np.isin(labels, (2, 4))] = 2 / 255
    alpha[labels == 7] = 0.8
    target = np.full((*labels.shape, 3), 255, np.float32)
    target[labels > 0] = (80, 40, 20)
    sx, sy, ox, oy = frame
    source_size = (80, 64)
    outlines = cel.region_outlines(
        labels,
        0,
        fit_boundary=lambda p, _b: [
            ("L", tuple(float(v) for v in q)) for q in cel.simplify(p, 0)[1:]
        ],
    )
    paths = [
        f'<path id="p{i}" d="{outlines[i]}" fill="#502814" '
        f'fill-opacity="{float(alpha[labels == i].max())}"/>'
        for i in range(1, 8)
    ]
    svg = (
        '<svg width="80" height="64"><g id="parent" transform="matrix('
        + " ".join(map(str, (1 / sx, 0, 0, 1 / sy, ox, oy)))
        + ')">'
        + "".join(paths[:6])
        + '<g id="nested">'
        + paths[6]
        + "</g></g></svg>"
    )
    document = import_svg(svg)
    rgba = render(svg, source_size)
    shown = labels > 0
    evidence = Evidence(
        rgba,
        target,
        target,
        target,
        ~shown,
        shown,
        np.zeros_like(shown),
        np.zeros_like(shown),
        np.zeros(shown.shape, np.float32),
        np.zeros(shown.shape, np.float32),
        labels,
        source_size,
        (ox, oy),
        (sx, sy),
        None,
        False,
        alpha,
    )
    graph = build(evidence)
    partition = Partition(tuple(Surface(f"p{i}", (i,)) for i in range(1, 8)))
    policy = Policy(rgba)
    full = policy.evaluate(svg)
    assert full.valid
    policy.establish(full)
    state = State(
        document,
        svg,
        LocalPolicy(policy).start(svg, full),
        "fixture",
        {},
        partition=partition,
    )
    return evidence, graph, state, policy


@pytest.mark.parametrize("hole", [False, True])
@pytest.mark.parametrize("frame", [(1, 1, 0, 0), (1.5, 0.75, 4.25, 5.5)])
def test_connected_fields_keep_gaps_holes_tiny_marks_and_nested_paint(hole, frame):
    evidence, graph, state, policy = fixture(hole=hole, frame=frame)
    work = Work.start(10)
    model = fields.OpacityFields(evidence, graph)
    proposals = list(model(state, work))
    assert len(proposals) == 1
    edit = proposals[0]
    assert edit.partition is not None
    assert state.partition is not None
    assert edit.partition.follows(state.partition)
    assert set(edit.partition.owners) == set(state.partition.owners)
    assert edit.component is not None
    assert edit.component.validate(
        state.document,
        edit.document,
        state.partition,
        edit.partition,
        edit.ids,
        edit.bounds,
        work,
    )
    assert model.diagnostics["fields"] == 3
    assert model.diagnostics["replaced_paths"] == 6
    assert edit.document.element("nested") == state.document.element("nested")
    assert edit.document.geometry_for("p7") == state.document.geometry_for("p7")
    assert edit.document.element("p7") == state.document.element("p7")
    svg = export_svg(edit.document)
    assert policy.evaluate(svg).valid
    actual = render(svg, evidence.source_size)
    restored, _ = load_project(save_project(edit.document))
    np.testing.assert_array_equal(
        render(export_svg(restored), evidence.source_size), actual
    )
    if frame == (1, 1, 0, 0):
        source = evidence.rgba[..., 3] > 0
        np.testing.assert_array_equal(actual[..., 3] > 0, source)
        assert (actual[..., 3][source] >= evidence.rgba[..., 3][source]).all()
        assert actual[8:20, 20, 3].max() == 0
        assert actual[48, 44:46, 3].min() > 0
        if hole:
            assert actual[11:17, 11:17, 3].max() == 0


@pytest.mark.parametrize(
    "protection",
    ["paint", "lock", "pin", "fixed", "gradient", "clip", "parent-alpha", "covered"],
)
def test_incomplete_or_protected_component_keeps_all_its_paths(protection):
    evidence, graph, state, _ = fixture()
    if protection == "paint":
        state = replace(state, details={"paint_constraints": ["p1"]})
    elif protection in {"lock", "clip", "gradient", "parent-alpha"}:
        document = state.document
        if protection == "parent-alpha":
            parent = document.element("parent")
            document = document.replace_element(
                replace(parent, attributes=(*parent.attributes, ("opacity", ".5")))
            )
        else:
            path = document.element("p1")
            if protection == "lock":
                path = replace(path, locks=frozenset({"geometry"}))
            elif protection == "clip":
                clip = Element(
                    "fake",
                    "clipPath",
                    children=(
                        Element(
                            "clip-rect",
                            "rect",
                            attributes=(("width", "80"), ("height", "64")),
                        ),
                    ),
                )
                defs = Element("clip-defs", "defs", children=(clip,))
                document = document.replace_element(
                    replace(document.root, children=(defs, *document.root.children))
                )
                path = replace(
                    path, attributes=(*path.attributes, ("clip-path", "url(#fake)"))
                )
            else:
                editor = Editor(document, selection=Selection(whole_document=True))
                with editor.transaction("Protected gradient") as tx:
                    tx.set_fill(
                        "p1",
                        LinearGradient(
                            (8, 8),
                            (14, 20),
                            (GradientStop(0, "red"), GradientStop(1, "blue")),
                        ),
                    )
                document = editor.snapshot.document
                path = document.element("p1")
            document = document.replace_element(path)
        state = replace(state, document=document)
    elif protection == "pin":
        geometry = state.document.geometry_for("p1")
        contour = geometry.subpaths[0]
        pinned = replace(contour.nodes[0], pinned=True)
        geometry = geometry.replace_node(pinned)
        state = replace(state, document=state.document.replace_geometry(geometry))
    elif protection == "covered":
        partition = Partition(
            tuple(
                replace(s, covered=(7,)) if s.id == "p1" else s
                for s in required(state.partition).surfaces
            )
        )
        state = replace(state, partition=partition)
    else:
        graph = replace(
            graph,
            regions=tuple(
                replace(r, fixed=True) if r.id == 1 else r for r in graph.regions
            ),
        )
    edits = list(fields.OpacityFields(evidence, graph)(state, Work.start(10)))
    if protection == "parent-alpha":
        assert not edits
        return
    assert len(edits) == 1
    edit = edits[0]
    for oid in ("p1", "p2"):
        assert oid not in edit.ids
        assert edit.document.element(oid) == state.document.element(oid)
        assert edit.document.geometry_for(oid) == state.document.geometry_for(oid)


def test_excessive_flat_opacity_retains_original_component():
    evidence, graph, state, _ = fixture()
    assert evidence.opacity is not None
    alpha = evidence.opacity.copy()
    alpha[evidence.labels == 2] = 7 / 255
    alpha[evidence.labels == 1] = 1 / 255
    evidence = replace(evidence, opacity=alpha)
    graph = build(evidence)
    model = fields.OpacityFields(evidence, graph)
    edit = next(iter(model(state, Work.start(10))))
    assert model.diagnostics["excess_flat_mass"] == 1
    assert "p1" not in edit.ids
    assert "p2" not in edit.ids


def test_intentional_faint_opacity_step_still_fails_native_validation():
    evidence, graph, state, policy = fixture(stepped=True)
    edit = next(iter(fields.OpacityFields(evidence, graph)(state, Work.start(10))))
    full = policy.evaluate(export_svg(edit.document))
    assert not full.valid
    assert "translucent-opacity-excess" in full.rejections


def test_split_source_atoms_keep_exact_lineage_and_complete_ownership():
    evidence, graph, state, _ = fixture()
    work = Work.start(10)
    classes = np.zeros(graph.labels.shape, np.uint8)
    classes[:, 11:] = 1
    atoms, groups = Atoms.original(graph).partition(
        graph, (1,), classes, 2, work, compact=True
    )
    assert state.partition is not None
    partition = state.partition.split(
        ("p1",), (Surface("p1", tuple(sorted((*groups[0], *groups[1])))),), atoms
    )
    branch = atoms.graph(evidence, graph, work)
    state = replace(state, partition=partition)
    edit = next(iter(fields.OpacityFields(evidence, branch)(state, work)))
    assert edit.partition is not None
    assert edit.partition.atoms == atoms
    assert edit.partition.follows(partition)
    assert set(edit.partition.owners) == set(partition.owners)
    assert Partition.from_metadata(edit.partition.metadata()) == edit.partition
    assert edit.partition.atoms is not None
    np.testing.assert_array_equal(
        edit.partition.atoms.labels(graph, work), branch.labels
    )


def test_atom_spanning_two_physical_components_cannot_be_collapsed_into_one():
    evidence, graph, state, _ = fixture()
    labels = evidence.labels.copy()
    labels[labels == 5] = 1
    evidence = replace(evidence, labels=labels)
    graph = build(evidence)
    # Keep the mixed owner's original two disjoint contours and full source
    # ledger, so the case cannot be rejected merely as incomplete metadata.
    tiny = state.document.geometry_for("p5")
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Mixed source owner") as tx:
        tx.delete_objects(frozenset({"p5"}))
    document = editor.snapshot.document
    geometry = document.geometry_for("p1")
    document = document.replace_geometry(
        replace(geometry, subpaths=(*geometry.subpaths, *tiny.subpaths))
    )
    document = replace(
        document, geometries=tuple(g for g in document.geometries if g.id != tiny.id)
    )
    partition = Partition(
        tuple(s for s in required(state.partition).surfaces if s.id != "p5")
    )
    state = replace(state, document=document, partition=partition)
    edit = next(iter(fields.OpacityFields(evidence, graph)(state, Work.start(10))))
    assert edit.partition is not None
    assert edit.partition.follows(partition)
    for oid in ("p1", "p2", "p6"):
        assert oid not in edit.ids
        assert edit.document.geometry_for(oid) == document.geometry_for(oid)


def test_stale_source_atom_graph_is_rejected_before_discovery():
    evidence, graph, state, _ = fixture()
    graph = replace(graph, source_atoms="wrong namespace")
    with pytest.raises(ValueError, match="current source atom graph"):
        list(fields.OpacityFields(evidence, graph)(state, Work.start(10)))


@pytest.mark.parametrize(
    "change", ["geometry", "paint", "remove", "declare", "children"]
)
def test_nested_sibling_changes_are_rejected_even_with_declared_parent(change):
    evidence, graph, state, _ = fixture()
    work = Work.start(10)
    edit = next(iter(fields.OpacityFields(evidence, graph)(state, work)))
    document, ids = edit.document, edit.ids
    if change == "geometry":
        geometry = document.geometry_for("p7")
        node = geometry.subpaths[0].nodes[0]
        node = replace(node, values=(node.values[0] + 1, node.values[1]))
        document = document.replace_geometry(geometry.replace_node(node))
    elif change == "paint":
        path = document.element("p7")
        document = document.replace_element(
            replace(
                path,
                attributes=tuple(
                    (k, "red" if k == "fill" else v) for k, v in path.attributes
                ),
            )
        )
    elif change == "remove":
        parent = document.element("parent")
        document = document.replace_element(
            replace(
                parent, children=tuple(c for c in parent.children if c.id != "nested")
            )
        )
    elif change == "children":
        nested = document.element("nested")
        path = replace(document.element("p7"), id="additional-nested-path")
        document = document.replace_element(
            replace(nested, children=(*nested.children, path))
        )
    else:
        ids = (*ids, "nested")
    assert edit.component is not None
    with pytest.raises(ValueError, match="cannot edit nested"):
        edit.component.validate(
            state.document,
            document,
            state.partition,
            edit.partition,
            ids,
            edit.bounds,
            work,
        )


@pytest.mark.parametrize(
    "limit",
    [
        "MAX_PIXELS",
        "MAX_PATHS",
        "MAX_COMPONENTS",
        "MAX_FIELDS",
        "MAX_NODES",
        "MAX_PARENTS",
    ],
)
def test_bounds_publish_no_partial_edit(monkeypatch, limit):
    evidence, graph, state, _ = fixture()
    monkeypatch.setattr(fields, limit, 0)
    model = fields.OpacityFields(evidence, graph)
    assert list(model(state, Work.start(10))) == []
    assert model.diagnostics["bounded"] == 1


@pytest.mark.parametrize("phase", ["before", "contour"])
def test_cancellation_publishes_no_partial_edit(monkeypatch, phase):
    evidence, graph, state, _ = fixture()
    work = Work.start(10)
    if phase == "before":
        work.stop.set()
    else:
        original = cel.region_outlines

        def cancelled(*args, **kwargs):
            value = original(*args, **kwargs)
            work.stop.set()
            return value

        monkeypatch.setattr(cel, "region_outlines", cancelled)
    with pytest.raises(StageInterruptedError):
        list(fields.OpacityFields(evidence, graph)(state, work))
