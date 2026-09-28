"""Snapshot-based execution of the path-fit mutation, independent of HTTP."""

from __future__ import annotations

import base64
import io
import logging
from threading import Event, Lock, Thread
from uuid import uuid4

from PIL import Image

from vectrify.document import DocumentError, StaleRevisionError
from vectrify.refine.selected import FitOptions, fit_selected_path, validate_selection

log = logging.getLogger(__name__)
GPU_SLOT = Lock()


class PathFitJob:
    def __init__(self, session, payload: dict):
        session.check_revision(payload)
        if not session.reference:
            raise DocumentError("Add a reference image before optimizing a path")
        self.options = FitOptions(**payload.get("options", {}))
        self.snapshot = session.editor.snapshot
        validate_selection(
            self.snapshot.document, self.snapshot.selection, self.options
        )
        self.epoch = session.epoch
        self.reference = session.reference["data_url"]
        self.id = uuid4().hex
        self.lock = Lock()
        self.stop = Event()
        self.status = "running"
        self.step = 0
        self.message = "Waiting for GPU…"
        self.result = None
        self.error = None

    def start(self):
        Thread(target=self.run, daemon=True, name="vectrify-path-fit").start()

    def progress(self, step: int, message: str):
        with self.lock:
            self.step, self.message = step, message

    def run(self):
        acquired = False
        try:
            while not self.stop.is_set():
                if GPU_SLOT.acquire(timeout=0.2):
                    acquired = True
                    break
            if not acquired:
                with self.lock:
                    self.status, self.message = "cancelled", "Stopped before fitting"
                return
            data = base64.b64decode(self.reference.split(",", 1)[1])
            with Image.open(io.BytesIO(data)) as image:
                result = fit_selected_path(
                    self.snapshot.document,
                    self.snapshot.selection,
                    image,
                    self.options,
                    stop=self.stop,
                    progress=self.progress,
                )
            with self.lock:
                self.result = result
                self.status = "ready"
                self.message = (
                    "Stopped; best result retained"
                    if self.stop.is_set()
                    else "Preview ready"
                )
        except Exception as exc:
            log.exception("Path-fit mutation failed")
            with self.lock:
                self.status, self.error = "failed", str(exc)
        finally:
            if acquired:
                GPU_SLOT.release()

    def state(self, *, preview: bool = False) -> dict:
        with self.lock:
            state = {
                "id": self.id,
                "status": self.status,
                "step": self.step,
                "steps": self.options.steps,
                "message": self.message,
                "error": self.error,
            }
            if self.result is not None:
                result = self.result
                state.update(
                    before=result.before,
                    after=result.after,
                    changed=result.changed,
                    size=result.size,
                )
                if preview:
                    state["previews"] = result.previews
            return state

    def apply(self, session):
        if (
            session.epoch != self.epoch
            or session.editor.snapshot.revision != self.snapshot.revision
            or not session.reference
            or session.reference["data_url"] != self.reference
        ):
            raise StaleRevisionError(
                "The drawing or reference changed. Run path fitting again."
            )
        with self.lock:
            if self.status != "ready" or self.result is None:
                raise DocumentError("Wait for a path-fit preview before applying")
            self.result.apply(
                session.editor, self.snapshot.selection, self.snapshot.revision
            )
            self.status = "applied"
        return session.state()
