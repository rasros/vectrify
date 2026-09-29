"""The request/result contract every automated operation implements.

An operation is one explicit action (Generate, Improve, Simplify, ...) carried
out by a named method. It reads an immutable editor snapshot and returns
proposals: uncommitted transactions built on that snapshot. Nothing changes the
drawing until one proposal is applied, and applying re-checks the revision, so
a result can never overwrite edits made while the operation ran.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from threading import Event
from typing import Any, ClassVar, Protocol

from PIL import Image

from vectrify.document import DocumentError, EditKind, Editor, Snapshot
from vectrify.document.editor import Transaction

ACTIONS = ("generate", "improve", "simplify", "link")


@dataclass(frozen=True)
class Permissions:
    """What an operation may change, on top of selection and locks."""

    geometry: bool = False
    paint: bool = False
    structure: bool = False
    transform: bool = False

    @classmethod
    def parse(cls, value: Mapping[str, Any] | None) -> Permissions:
        value = value or {}
        unknown = set(value) - {"geometry", "paint", "structure", "transform"}
        if unknown:
            raise DocumentError(f"Unknown edit permission: {sorted(unknown)[0]}")
        if any(type(v) is not bool for v in value.values()):
            raise DocumentError("Edit permissions must be on or off")
        return cls(**value)

    @property
    def allowed(self) -> frozenset[str]:
        return frozenset(
            kind.value for kind in EditKind if getattr(self, kind.value, False)
        )


@dataclass(frozen=True)
class Budget:
    """How much work a method may spend; None leaves it to the method."""

    steps: int | None = None
    seconds: float | None = None

    @classmethod
    def parse(cls, value: Mapping[str, Any] | None) -> Budget:
        value = value or {}
        steps, seconds = value.get("steps"), value.get("seconds")
        if steps is not None and (type(steps) is not int or steps < 1):
            raise DocumentError("Step budget must be a positive whole number")
        if seconds is not None and (
            not isinstance(seconds, int | float)
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise DocumentError("Time budget must be a positive number of seconds")
        return cls(steps, None if seconds is None else float(seconds))


@dataclass(frozen=True)
class OperationRequest:
    action: str
    method: str
    snapshot: Snapshot
    editor: Editor = field(repr=False, compare=False)
    permissions: Permissions = Permissions()
    settings: Mapping[str, Any] = field(default_factory=dict)
    budget: Budget = Budget()
    reference: Image.Image | None = None
    # Document-space viewport for previews: x, y, width, height.
    bounds: tuple[float, float, float, float] | None = None

    def transaction(self, label: str) -> Transaction:
        """An edit of this request's snapshot, limited to its permissions."""
        return self.editor.transaction(
            label,
            selection=self.snapshot.selection,
            allowed=self.permissions.allowed,
            base=self.snapshot,
        )


@dataclass
class Proposal:
    """One candidate outcome. Applying commits its transaction as one edit."""

    transaction: Transaction
    changed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    previews: dict[str, str] = field(default_factory=dict)
    label: str | None = None

    def summary(self, *, previews: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {"changed": self.changed, "metrics": self.metrics}
        if self.label:
            result["label"] = self.label
        if previews:
            result["previews"] = self.previews
        return result


@dataclass
class OperationResult:
    """A recommendation plus alternatives. The unchanged input is always valid."""

    recommended: Proposal
    alternatives: list[Proposal] = field(default_factory=list)
    message: str | None = None

    @property
    def proposals(self) -> list[Proposal]:
        return [self.recommended, *self.alternatives]


class RunContext:
    """Progress and cancellation shared between a running method and its job."""

    def __init__(self, stop: Event | None = None, total: int | None = None):
        self.stop = stop or Event()
        self.total = total
        self.step = 0
        self.message = ""
        self._listeners: list[Callable[[int, str], None]] = []

    def listen(self, listener: Callable[[int, str], None]) -> None:
        self._listeners.append(listener)

    def progress(self, step: int, message: str, *, total: int | None = None) -> None:
        self.step, self.message = step, message
        if total is not None:
            self.total = total
        for listener in self._listeners:
            listener(step, message)

    @property
    def stopped(self) -> bool:
        return self.stop.is_set()


class Method(Protocol):
    """One way of carrying out an action."""

    action: ClassVar[str]
    name: ClassVar[str]
    # Long methods run on a background job; others finish within the request.
    background: ClassVar[bool]
    needs_reference: ClassVar[bool]
    # Shared resources the job must hold while running, e.g. {"gpu"}.
    resources: ClassVar[frozenset[str]]

    def validate(self, request: OperationRequest, /) -> None:
        """Reject an impossible request quickly, before any job starts."""

    def run(self, request: OperationRequest, context: RunContext, /) -> OperationResult:
        """Do the work. Honour context.stop by returning the best result so far."""
        ...


_METHODS: dict[tuple[str, str], Method] = {}


def register(method: Method) -> Method:
    if method.action not in ACTIONS:
        raise ValueError(f"Unknown action {method.action!r}")
    _METHODS[method.action, method.name] = method
    return method


def method(action: str, name: str) -> Method:
    try:
        return _METHODS[action, name]
    except KeyError:
        raise DocumentError(f"Unknown {action} method: {name}") from None


def available(action: str | None = None) -> list[Method]:
    return [m for (a, _), m in _METHODS.items() if action in {None, a}]
