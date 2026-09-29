"""The part of a drawing an editor operation may change."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MutationScope:
    """What an editor operation, such as an LLM edit, may touch.

    ``object_ids`` names editable elements by their ``id``; their descendants
    are editable too. ``kinds`` are the permitted edit kinds: ``geometry``,
    ``paint``, ``structure`` and ``transform``. Everything else stays fixed.
    """

    object_ids: frozenset[str]
    kinds: frozenset[str]
