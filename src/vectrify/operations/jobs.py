"""Snapshot-based execution of operation methods, independent of HTTP."""

from __future__ import annotations

import logging
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

from vectrify.document import DocumentError, StaleRevisionError
from vectrify.operations.contract import (
    Method,
    OperationRequest,
    OperationResult,
    RunContext,
)
from vectrify.refine.gpu import gpu_gate

log = logging.getLogger(__name__)

# One gate per shared device. A method naming a resource holds it while it
# runs; the GPU gate is the same one searches hand to their worker processes.
RESOURCES: dict[str, Any] = {"gpu": gpu_gate()}


class Job:
    """One run of a method: running, then ready/failed/cancelled, then applied."""

    def __init__(
        self, method: Method, request: OperationRequest, context_key: Any = None
    ):
        method.validate(request)
        self.method = method
        self.request = request
        self.id = uuid4().hex
        self.context = RunContext(total=request.budget.steps)
        self.context.listen(self._progress)
        self._lock = Lock()
        self.status = "running"
        self.message = (
            "Waiting for " + ", ".join(sorted(method.resources)).upper() + "…"
            if method.resources
            else "Starting…"
        )
        self.result: OperationResult | None = None
        self.error: str | None = None
        # Anything besides the revision a result depends on, e.g. the reference
        # image; the caller compares it again before applying.
        self.context_key = context_key

    @property
    def stop(self):
        return self.context.stop

    def start(self) -> None:
        """Run in the background for long methods, otherwise before returning."""
        if self.method.background:
            Thread(target=self.run, daemon=True, name=f"vectrify-{self.id}").start()
        else:
            self.run()
            if self.status == "failed":
                raise DocumentError(self.error or "Operation failed")

    def _progress(self, _step: int, message: str) -> None:
        with self._lock:
            self.message = message

    def _acquire(self) -> list[Any] | None:
        if self.stop.is_set():
            return None
        held: list[Any] = []
        for name in sorted(self.method.resources):
            resource = RESOURCES[name]
            while not resource.acquire(timeout=0.2):
                if self.stop.is_set():
                    for lock in held:
                        lock.release()
                    return None
            held.append(resource)
        return held

    def run(self) -> None:
        held = self._acquire()
        if held is None:
            with self._lock:
                self.status, self.message = "cancelled", "Stopped before starting"
            return
        try:
            result = self.method.run(self.request, self.context)
            with self._lock:
                self.result = result
                self.status = "ready"
                self.message = result.message or (
                    "Stopped; best result retained"
                    if self.stop.is_set()
                    else "Preview ready"
                )
        except Exception as exc:
            if not isinstance(exc, DocumentError):
                log.exception("%s/%s failed", self.method.action, self.method.name)
            with self._lock:
                self.status, self.error = "failed", str(exc)
        finally:
            for lock in held:
                lock.release()

    def state(self, *, preview: bool = False) -> dict[str, Any]:
        with self._lock:
            state: dict[str, Any] = {
                "id": self.id,
                "action": self.method.action,
                "method": self.method.name,
                "status": self.status,
                "step": self.context.step,
                "steps": self.context.total,
                "message": self.message,
                "error": self.error,
            }
            if self.result is not None:
                state["result"] = self.result.recommended.summary(previews=preview)
                state["alternatives"] = [
                    p.summary(previews=preview) for p in self.result.alternatives
                ]
            return state

    def apply(self, choice: int = 0) -> None:
        """Commit one proposal. The revision check rejects stale results."""
        with self._lock:
            if self.status != "ready" or self.result is None:
                raise DocumentError("Wait for the preview before applying")
            proposals = self.result.proposals
            if type(choice) is not int or not 0 <= choice < len(proposals):
                raise DocumentError("Choose one of the proposed results")
            proposal = proposals[choice]
            if not proposal.changed:
                raise DocumentError("This result leaves the drawing unchanged")
            try:
                proposal.transaction.commit()
            except StaleRevisionError:
                raise StaleRevisionError(
                    "The drawing changed. Run the operation again."
                ) from None
            self.status = "applied"
