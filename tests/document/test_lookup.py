"""Immutable document lookups stay local during selection and node loading."""

from dataclasses import replace
from unittest.mock import patch

import pytest

from vectrify.document import Document, DocumentError, Element, Selection, import_svg


def test_tree_lookup_preserves_order_ancestry_and_snapshot_isolation():
    document = import_svg(
        '<svg id="root"><g id="layer"><g id="nested">'
        '<path id="p" d="M0 0 L10 10"/></g></g><rect id="r"/></svg>'
    )
    order = document.elements()
    assert [e.id for e in order] == ["root", "layer", "nested", "p", "r"]
    with patch.object(Document, "elements", side_effect=AssertionError("walked tree")):
        assert document.element("p") is order[3]
        assert document.ancestry("p") == order[:4]
        assert document.ancestry("root") == (document.root,)
        with pytest.raises(DocumentError, match="Unknown object"):
            document.element("missing")
        with pytest.raises(DocumentError, match="Unknown object"):
            document.ancestry("missing")

    changed = document.replace_element(replace(document.element("p"), name="New name"))
    assert changed.element("p").name == "New name"
    assert document.element("p").name == ""
    assert changed.ancestry("p")[-1] is changed.element("p")
    # Structural replacement must build the new parents too.
    moved = replace(
        changed, root=replace(changed.root, children=(changed.element("p"),))
    )
    assert [e.id for e in moved.elements()] == ["root", "p"]
    assert [e.id for e in moved.ancestry("p")] == ["root", "p"]
    with pytest.raises(DocumentError, match="Unknown object"):
        moved.element("nested")


def test_object_selection_does_not_read_geometry_without_selected_nodes():
    document = import_svg(
        '<svg id="root"><g id="layer">'
        + "".join(f'<path id="p{i}" d="M0 0 L10 10"/>' for i in range(772))
        + "</g></svg>"
    )
    ids = frozenset(f"p{i}" for i in range(772))
    with patch.object(
        Document, "geometry_for", side_effect=AssertionError("read nodes")
    ):
        assert document.selection_ids(Selection(object_ids=ids)) == ids
        assert document.selection_ids(Selection(object_ids={"layer"})) == ids | {
            "layer"
        }
        assert document.selection_ids(Selection.all()) == ids | {"root", "layer"}
        with pytest.raises(DocumentError, match="Unknown object"):
            document.selection_ids(Selection(object_ids={"missing"}))
    node = document.geometry_for("p0").subpaths[0].nodes[0].id
    assert document.selection_ids(Selection(object_ids={"p0"}, node_ids={node})) == {
        "p0"
    }
    with pytest.raises(DocumentError, match="Selected nodes"):
        document.selection_ids(Selection(object_ids={"p1"}, node_ids={node}))


def test_lookup_keeps_first_match_before_duplicate_ids_are_validated():
    first, second = Element("same", "rect"), Element("same", "circle")
    document = Document(Element("root", "svg", children=(first, second)))
    assert document.element("same") is first
    assert document.elements() == (document.root, first, second)
