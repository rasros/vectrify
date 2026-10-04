"""Simultaneous Tidy preserves enclosed layout and the editable project."""

from dataclasses import replace

import pytest

from scripts.bench_tidy import GROUP, OUTLINE, load, properties
from vectrify.document import Editor, Element, Selection
from vectrify.operations import Budget, Job, OperationRequest, Permissions, method
from vectrify.refine.layout import infer


@pytest.mark.parametrize("select_group", [True, False])
def test_sword_tidy_all_paths_meets_geometry_and_rendered_coverage(select_group):
    document, image = load()
    ids = frozenset(e.id for e in document.element(GROUP).children)
    selection = Selection(object_ids=frozenset({GROUP}) if select_group else ids)
    editor = Editor(document, selection=selection)
    before = properties(document)
    assert before["outside_area"] > 1000
    assert before["gap_area"] > 1000
    assert before["symmetry_area"] > 1000
    assert before["transparent_pixels"] > 0
    job = Job(
        method("improve", "nodes"),
        OperationRequest(
            "improve",
            "nodes",
            editor.snapshot,
            editor,
            Permissions(geometry=True, structure=True, paint=True),
            settings={"shape": False},
            budget=Budget(steps=2),
            reference=image,
        ),
    )
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    assert state["result"]["metrics"]["layouts"] == 1
    job.apply()
    after = properties(editor.snapshot.document)
    assert after["outside_area"] < 0.05
    assert after["gap_area"] < 0.05
    assert after["symmetry_area"] < 0.05
    assert after["transparent_pixels"] == 0
    assert all(area < 0.05 for area in after["outside_each"].values())
    # IDs, paint, stacking and unrelated artwork survive one undoable edit.
    final = editor.snapshot.document
    assert [e.id for e in final.elements()] == [e.id for e in document.elements()]
    for oid in ids:
        assert final.element(oid).get("fill") == document.element(oid).get("fill")
    for e in document.elements():
        if e.tag == "path" and e.id not in ids:
            assert final.geometry_for(e.id) == document.geometry_for(e.id)
    assert editor.snapshot.selection == selection
    editor.undo()
    assert editor.snapshot.document == document


@pytest.mark.parametrize(
    "kind", ["partial", "pinned", "held", "opacity", "lock", "gradient", "nonuniform"]
)
def test_layout_inference_respects_artwork_without_the_prerequisites(kind):
    document, _image = load()
    ids = [e.id for e in document.element(GROUP).children]
    held = frozenset()
    if kind == "partial":
        ids = ids[:-1]
    elif kind in {"pinned", "held"}:
        geometry = document.geometry_for(ids[0])
        sub = geometry.subpaths[0]
        node = sub.nodes[0]
        if kind == "held":
            held = frozenset({node.id})
        else:
            document = document.replace_geometry(
                replace(
                    geometry,
                    subpaths=(
                        replace(
                            sub, nodes=(replace(node, pinned=True), *sub.nodes[1:])
                        ),
                        *geometry.subpaths[1:],
                    ),
                )
            )
    else:
        group = document.element(GROUP)
        if kind == "lock":
            group = replace(group, locks=frozenset({"geometry"}))
        else:
            if kind == "gradient":
                gradient = Element(
                    "test-gradient",
                    "linearGradient",
                    children=(
                        Element(
                            "test-stop", "stop", attributes=(("stop-color", "red"),)
                        ),
                    ),
                )
                defs = Element("test-defs", "defs", children=(gradient,))
                document = document.replace_element(
                    replace(document.root, children=(defs, *document.root.children))
                )
            attribute, value = {
                "opacity": ("opacity", "0.5"),
                "gradient": (
                    "fill",
                    "url(#test-gradient)",
                ),
                "nonuniform": ("transform", "scale(2 1)"),
            }[kind]
            if kind == "gradient":
                group = document.element(ids[0])
            group = replace(
                group,
                attributes=tuple({**dict(group.attributes), attribute: value}.items()),
            )
        document = document.replace_element(group)
    assert infer(document, ids, held) == ()


def test_projection_is_stable_and_symmetry_holds_after_uniform_rotation():
    document, _image = load()
    group = document.element(GROUP)
    document = document.replace_element(
        replace(
            group, attributes=(("transform", "translate(30 20) rotate(17) scale(.8)"),)
        )
    )
    ids = [e.id for e in group.children]
    (layout,) = infer(document, ids)
    once = layout.project(document)
    twice = layout.project(once)
    for final in (once, twice):
        stats = properties(final)
        assert stats["gap_area"] < 0.05
        assert stats["outside_area"] < 0.05
        assert stats["symmetry_area"] < 0.05
        assert final.geometry_for(OUTLINE).id == document.geometry_for(OUTLINE).id
