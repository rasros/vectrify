"""Apply an identified history change without replacing unrelated edits."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any

from vectrify.document.model import Document, DocumentError, Selection


class HistoryConflictError(DocumentError):
    """A later edit changed something this history change needs to restore."""

    def __init__(self, path: str):
        self.path = path
        super().__init__(
            f"History conflicts with a later edit at {path}; undo the dependent "
            "change first. Nothing was changed."
        )


class ConcurrentEditError(DocumentError):
    """Concurrent edits need different values for the same part of a drawing."""


def _conflict(path: str) -> HistoryConflictError:
    return HistoryConflictError(path)


def _merge(expected: Any, desired: Any, current: Any, path: str) -> Any:
    if expected == desired:
        return current
    if current == expected:
        return desired
    if is_dataclass(expected) and type(expected) is type(desired) is type(current):
        return replace(
            current,
            **{
                f.name: _merge(
                    getattr(expected, f.name),
                    getattr(desired, f.name),
                    getattr(current, f.name),
                    f"{path}.{f.name}",
                )
                for f in fields(expected)
            },
        )
    if path.endswith(".attributes"):
        old, new, now = dict(expected), dict(desired), dict(current)
        missing = object()
        for key in old.keys() | new.keys():
            if old.get(key, missing) == new.get(key, missing):
                continue
            if now.get(key, missing) != old.get(key, missing):
                raise _conflict(f"{path}.{key}")
            if key in new:
                now[key] = new[key]
            else:
                now.pop(key, None)
        return tuple(now.items())
    if isinstance(expected, tuple) and all(
        hasattr(item, "id") for items in (expected, desired, current) for item in items
    ):
        return _merge_sequence(expected, desired, current, path)
    raise _conflict(path)


def _merge_sequence(
    expected: tuple, desired: tuple, current: tuple, path: str
) -> tuple:
    old, new, now = (
        {item.id: item for item in items} for items in (expected, desired, current)
    )
    missing = object()
    for key in old.keys() | new.keys():
        if key in old and key in new:
            if key not in now:
                if old[key] != new[key]:
                    raise _conflict(f"{path}[{key}]")
            else:
                now[key] = _merge(old[key], new[key], now[key], f"{path}[{key}]")
        elif key in old:
            if now.get(key, missing) != old[key]:
                raise _conflict(f"{path}[{key}]")
            del now[key]
        else:
            if key in now:
                raise _conflict(f"{path}[{key}]")
            now[key] = new[key]

    before = [item.id for item in expected]
    after = [item.id for item in desired]
    present = [item.id for item in current]
    common = old.keys() & new.keys()
    old_order = [key for key in before if key in common]
    new_order = [key for key in after if key in common]
    order = [key for key in present if key in now]
    if old_order != new_order:
        # Reordering is safe only while the sibling membership/order agrees.
        if present != before:
            raise _conflict(f"{path}.order")
        order = after
    else:
        # Restore removed members next to a surviving neighbour, retaining
        # the current order and any independently added members.
        for i, key in enumerate(after):
            if key in order or key not in now:
                continue
            following = next((k for k in after[i + 1 :] if k in order), None)
            preceding = next((k for k in reversed(after[:i]) if k in order), None)
            index = (
                order.index(following)
                if following is not None
                else order.index(preceding) + 1
                if preceding is not None
                else len(order)
            )
            order.insert(index, key)
    return tuple(now[key] for key in order)


def restore_change(
    expected: Document, desired: Document, current: Document
) -> Document:
    """Three-way restoration, checked in full before the editor mutates."""
    remaining = {e.id for e in desired.elements()}
    present = {e.id: e for e in current.elements()}
    # Geometry lives outside the element tree. Removing an otherwise
    # unchanged element must still refuse to hide a later geometry edit.
    for element in expected.elements():
        if element.id in remaining or element.id not in present:
            continue
        if element.geometry_id is not None and (
            current.geometry(element.geometry_id)
            != expected.geometry(element.geometry_id)
        ):
            raise _conflict(f"document.geometry[{element.geometry_id}]")
    result = _merge(expected, desired, current, "document")
    result.validate()
    return result


def kept_selection(selection: Selection, document: Document) -> Selection:
    """Keep existing identities; losing a node filter never selects a whole path."""
    objects = selection.object_ids & {e.id for e in document.elements()}
    nodes = selection.node_ids
    if nodes:
        present = set()
        for oid in objects:
            for element in Document(document.element(oid)).elements():
                try:
                    geometry = document.geometry_for(element.id)
                except DocumentError:
                    continue
                present.update(n.id for sp in geometry.subpaths for n in sp.nodes)
        nodes &= present
        if not nodes:
            return Selection()
    kept = Selection(objects, nodes, selection.whole_document)
    document.selection_ids(kept)
    return kept


def merge_edit(base: Document, edited: Document, current: Document) -> Document:
    """Merge an edit's changes, preserving live constraints and unrelated work."""
    try:
        originals = {e.id: e for e in base.elements()}
        targets = {e.id: e for e in edited.elements()}
        live = {e.id: e for e in current.elements()}
        for oid, element in originals.items():
            target = targets.get(oid)
            geometry_changed = (
                element.geometry_id is not None
                and target is not None
                and target.geometry_id == element.geometry_id
                and base.geometry(element.geometry_id)
                != edited.geometry(element.geometry_id)
            )
            if target == element and not geometry_changed:
                continue
            if oid not in live:
                raise _conflict(f"object[{oid}]")
            coordinate_sensitive = geometry_changed or (
                target is not None
                and (
                    target.geometry_id != element.geometry_id
                    or target.get("transform") != element.get("transform")
                )
            )
            # A preview cannot bypass a lock added while it was running, or
            # use an obsolete coordinate frame after a parent was moved.
            ancestors = base.ancestry(oid)
            if [a.id for a in ancestors] != [a.id for a in current.ancestry(oid)]:
                raise _conflict(f"object[{oid}].parent")
            for ancestor in ancestors:
                now = live[ancestor.id]
                if ancestor.locks != now.locks:
                    raise _conflict(f"object[{ancestor.id}].locks")
                if (
                    coordinate_sensitive
                    and ancestor.id != oid
                    and ancestor.get("transform") != now.get("transform")
                ):
                    raise _conflict(f"object[{ancestor.id}].transform")
            if geometry_changed and element.geometry_id is not None:
                if element.get("transform") != live[oid].get("transform"):
                    raise _conflict(f"object[{oid}].transform")
                if base.geometry_users(element.geometry_id) != current.geometry_users(
                    element.geometry_id
                ):
                    raise _conflict(f"geometry[{element.geometry_id}].users")
        # Pins live in geometry assets, rather than elements. A newly pinned
        # endpoint must not be moved or removed by an older preview.
        proposed_geometries = {g.id: g for g in edited.geometries}
        live_geometries = {g.id: g for g in current.geometries}
        for geometry in base.geometries:
            proposed = proposed_geometries.get(geometry.id)
            if proposed == geometry:
                continue
            now = live_geometries.get(geometry.id)
            if now is None:
                continue
            proposed_nodes = (
                {n.id: n for sp in proposed.subpaths for n in sp.nodes}
                if proposed
                else {}
            )
            live_nodes = {n.id: n for sp in now.subpaths for n in sp.nodes}
            for sp in geometry.subpaths:
                for node in sp.nodes:
                    pinned = live_nodes.get(node.id)
                    changed = proposed_nodes.get(node.id)
                    if (
                        pinned is not None
                        and pinned.pinned
                        and not node.pinned
                        and (changed is None or changed.endpoint != node.endpoint)
                    ):
                        raise _conflict(f"node[{node.id}].pinned")
        return restore_change(base, edited, current)
    except HistoryConflictError as exc:
        raise ConcurrentEditError(
            f"This edit conflicts with another edit at {exc.path}. Nothing was changed."
        ) from None
