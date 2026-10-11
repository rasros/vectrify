"""Geometry proposals stay immutable and enter the transaction's safety guards."""

import math

import pytest

from tests.document.test_document import select
from vectrify.document import (
    DocumentError,
    Editor,
    EditRejectedError,
    Selection,
    import_svg,
)
from vectrify.document.editor_handles import with_node_handles


def test_proposed_closing_handle_keeps_ids_and_requires_structure_authorization():
    document = import_svg('<svg><path id="p" d="M0 0 L20 0 L20 20 Z"/></svg>')
    editor = Editor(document, selection=select("p"))
    geometry = document.geometry_for("p")
    contour = geometry.subpaths[0]
    proposal = with_node_handles(geometry, contour.nodes[0].id, 2)
    assert proposal is not None
    assert editor.snapshot.document is document
    assert proposal.id == geometry.id
    assert proposal.subpaths[0].id == contour.id
    assert [(n.id, n.endpoint) for n in proposal.subpaths[0].nodes[:-1]] == [
        (n.id, n.endpoint) for n in contour.nodes
    ]
    closing = proposal.subpaths[0].nodes[-1]
    assert closing.id not in {n.id for n in contour.nodes}
    assert closing.endpoint == contour.nodes[0].endpoint

    tx = editor.transaction("Handles", allowed=frozenset({"geometry"}))
    with pytest.raises(EditRejectedError, match="Changing structure is not permitted"):
        tx.set_node_handles("p", contour.nodes[0].id, 2)
    assert tx.preview == document
    with pytest.raises(EditRejectedError, match="failed edit"):
        tx.commit()
    assert editor.snapshot.document is document


@pytest.mark.parametrize("constraint", ["lock", "node selection", "shared consumer"])
def test_proposed_handles_cannot_bypass_transaction_constraints(constraint):
    document = import_svg(
        '<svg><path id="p" d="M0 0 L10 0 L20 10"/>'
        '<use id="q" href="#p"/></svg>'
    )
    geometry = document.geometry_for("p")
    target = geometry.subpaths[0].nodes[1]
    selection = select("p", "q")
    if constraint == "node selection":
        selection = Selection(
            object_ids=selection.object_ids, node_ids=frozenset({target.id})
        )
    elif constraint == "shared consumer":
        selection = select("p")
    editor = Editor(document, selection=selection)
    if constraint == "lock":
        editor.set_locks("p", frozenset({"geometry"}))
    before = editor.snapshot
    proposal = with_node_handles(before.document.geometry_for("p"), target.id, 2)
    assert proposal is not None
    assert proposal != geometry
    assert editor.snapshot == before

    message = {
        "lock": "geometry is locked",
        "node selection": "This command requires a whole object selection",
        "shared consumer": "Edit affects unselected objects",
    }[constraint]
    tx = editor.transaction("Handles")
    with pytest.raises(EditRejectedError, match=message):
        tx.set_node_handles("p", target.id, 2)
    assert tx.preview == before.document
    with pytest.raises(EditRejectedError, match="failed edit"):
        tx.commit()
    assert editor.snapshot == before


def test_invalid_handle_coordinates_poison_a_transaction_with_prior_edits():
    editor = Editor(
        import_svg('<svg><path id="p" d="M0 0 C3 1 8 4 10 0"/></svg>'),
        selection=select("p"),
    )
    before = editor.snapshot
    node = before.document.geometry_for("p").subpaths[0].nodes[1]
    tx = editor.transaction("Paint then invalid handle")
    tx.set_attributes("p", {"fill": "red"})
    with pytest.raises(DocumentError, match="Path coordinates must be finite"):
        tx.move_handle("p", node.id, 0, (math.inf, 2.0))
    with pytest.raises(EditRejectedError, match="failed edit"):
        tx.commit()
    assert editor.snapshot == before


def test_straightening_a_retracted_handle_skips_mutation_authorization():
    editor = Editor(
        import_svg(
            '<svg><path id="p" d="M0 0 C3 1 10 0 10 0 C14 2 17 9 20 10"/></svg>'
        ),
        selection=select("p"),
    )
    before = editor.snapshot
    node = before.document.geometry_for("p").subpaths[0].nodes[1]
    with editor.transaction("No movement", allowed=frozenset()) as tx:
        tx.straighten_node_handles("p", node.id)
    assert editor.snapshot == before
