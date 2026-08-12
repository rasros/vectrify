"""Operator tables shared by the format backends.

Every backend declares its mutations as (function, name, weight) triples. The
name is the identifier the engine's policy selects by, so it has to be stable:
it appears in tasks, in results, and in whatever the policy has learned about
the run so far.
"""

import random
from collections.abc import Callable, Mapping

MutationTable = tuple[tuple[Callable[[str], str], str, float], ...]


def operator_weights(table: MutationTable) -> Mapping[str, float]:
    """The table as a name -> weight mapping, for a policy to start from."""
    return {name: weight for _fn, name, weight in table}


def pick_operator(
    table: MutationTable, operator: str | None = None
) -> tuple[Callable[[str], str], str]:
    """Resolve *operator* against *table*, falling back to a weighted draw.

    An unknown name draws at random rather than raising: the caller may be a
    policy carrying state from a run of a different format, and losing one
    mutation is cheaper than losing the task.
    """
    for fn, name, _weight in table:
        if name == operator:
            return fn, name
    fns, names, weights = zip(*table, strict=True)
    return random.choices(
        list(zip(fns, names, strict=True)), weights=list(weights), k=1
    )[0]
