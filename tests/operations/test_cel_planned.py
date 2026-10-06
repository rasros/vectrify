"""The planned generator uses normal operation transactions and typed settings."""

import time
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from vectrify.document import (
    DocumentError,
    Editor,
    Selection,
    export_svg,
    import_svg,
    load_project,
    save_project,
)
from vectrify.operations import Budget, Job, OperationRequest, Permissions, method
from vectrify.refine.cel_plan.frontier import Frontier
from vectrify.refine.cel_plan.policy import Policy


def request(editor, settings=None):
    pixels = np.zeros((80, 80, 4), dtype=np.uint8)
    pixels[10:70, 10:70] = (210, 60, 40, 255)
    pixels[25:55, 25:55] = (40, 80, 190, 255)
    return OperationRequest(
        "generate",
        "cel-planned",
        editor.snapshot,
        editor,
        Permissions(geometry=True, structure=True, paint=True),
        reference=Image.fromarray(pixels),
        settings=settings or {},
        budget=Budget(seconds=10),
    )


def test_planned_generation_applies_as_one_edit_and_round_trips():
    editor = Editor(
        import_svg('<svg width="80" height="80"/>'),
        selection=Selection(whole_document=True),
    )
    before = export_svg(editor.snapshot.document)
    job = Job(method("generate", "cel-planned"), request(editor, {"gradients": False}))
    job.run()
    assert job.state()["status"] == "ready", job.state()
    assert export_svg(editor.snapshot.document) == before
    job.apply()
    assert editor.undo_labels == ("Generate planned cel drawing",)
    document, _ = load_project(save_project(editor.snapshot.document))
    assert export_svg(document) == export_svg(editor.snapshot.document)
    editor.undo()
    assert export_svg(editor.snapshot.document) == before


@pytest.mark.parametrize(
    "settings",
    [
        {"complexity": 101},
        {"complexity": 3.5},
        {"quality": "unknown"},
        {"refine": 1},
        {"misspelled": 2},
    ],
)
def test_invalid_planner_settings_are_rejected(settings):
    editor = Editor(
        import_svg('<svg width="80" height="80"/>'),
        selection=Selection(whole_document=True),
    )
    with pytest.raises(DocumentError):
        method("generate", "cel-planned").validate(request(editor, settings))


def test_stop_before_first_checkpoint_cancels_without_a_proposal(monkeypatch):
    editor = Editor(
        import_svg('<svg width="80" height="80"/>'),
        selection=Selection(whole_document=True),
    )
    job = Job(method("generate", "cel-planned"), request(editor))
    original = Policy.from_evidence

    def stop_after_evidence(evidence, graph):
        result = original(evidence, graph)
        job.stop.set()
        return result

    monkeypatch.setattr(Policy, "from_evidence", stop_after_evidence)
    job.run()
    assert job.state()["status"] == "cancelled"
    assert job.result is None
    assert not editor.undo_labels


def test_stop_after_checkpoint_returns_validated_one_edit_result(monkeypatch):
    editor = Editor(
        import_svg('<svg width="80" height="80"/>'),
        selection=Selection(whole_document=True),
    )
    job = Job(method("generate", "cel-planned"), request(editor))
    original = Frontier.add

    def stop_after_validation(frontier, svg, label, details=None):
        accepted = original(frontier, svg, label, details)
        if accepted:
            job.stop.set()
        return accepted

    monkeypatch.setattr(Frontier, "add", stop_after_validation)
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    assert state["result"]["metrics"]["stopped"]
    assert state["result"]["metrics"]["validation_rejections"] == []
    job.apply()
    assert editor.undo_labels == ("Generate planned cel drawing",)


def test_unachievable_node_budget_is_reported_without_destroying_features():
    editor = Editor(
        import_svg('<svg width="80" height="80"/>'),
        selection=Selection(whole_document=True),
    )
    job = Job(method("generate", "cel-planned"), request(editor, {"node_budget": 1}))
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    metrics = state["result"]["metrics"]
    assert metrics["budget_unmet"]
    assert metrics["nodes"] > 1
    assert metrics["validation_rejections"] == []


def test_empty_transparent_reference_produces_no_edit():
    editor = Editor(
        import_svg('<svg width="80" height="80"/>'),
        selection=Selection(whole_document=True),
    )
    operation = replace(request(editor), reference=Image.new("RGBA", (80, 80)))
    job = Job(method("generate", "cel-planned"), operation)
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    assert not state["result"]["changed"]
    assert state["result"]["metrics"]["nodes"] == 0
    assert not editor.undo_labels


def test_valid_fallback_survives_rejected_fitted_candidates(monkeypatch):
    from vectrify.refine.cel_plan import pipeline

    original = pipeline.export

    def reject_fitted(evidence, labels, options, work, **kwargs):
        if kwargs.get("conservative"):
            return original(evidence, labels, options, work, **kwargs)
        return '<svg><path d="M0 0L1e999 4Z"/></svg>', {}

    monkeypatch.setattr(pipeline, "export", reject_fitted)
    editor = Editor(
        import_svg('<svg width="80" height="80"/>'),
        selection=Selection(whole_document=True),
    )
    job = Job(method("generate", "cel-planned"), request(editor))
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    metrics = state["result"]["metrics"]
    assert metrics["conservative_geometry"]
    assert metrics["validation_rejections"] == []
    assert any(
        "invalid-candidate" in item.get("rejections", [])
        for item in metrics["candidate_decisions"]
    )
    job.apply()
    assert editor.undo_labels == ("Generate planned cel drawing",)


def test_expired_deadline_does_not_retry_without_a_valid_fallback(monkeypatch):
    from vectrify.refine.cel_plan import pipeline

    attempts = []

    def failed_fallback(_evidence, _labels, _options, work, **kwargs):
        attempts.append(kwargs)
        work.deadline = time.monotonic() - 1
        return '<svg><path d="M0 0L1e999 4Z"/></svg>', {}

    monkeypatch.setattr(pipeline, "export", failed_fallback)
    editor = Editor(
        import_svg('<svg width="80" height="80"/>'),
        selection=Selection(whole_document=True),
    )
    job = Job(method("generate", "cel-planned"), request(editor))
    job.run()
    assert job.state()["status"] == "failed"
    assert len(attempts) == 1
    assert attempts[0]["conservative"]
    assert job.result is None
    assert not editor.undo_labels


def test_interrupted_fit_discards_partial_work_and_returns_the_checkpoint(monkeypatch):
    from vectrify.refine.cel_plan import pipeline
    from vectrify.refine.cel_plan.model import StageInterruptedError

    original = pipeline.export

    def interrupted_fit(evidence, labels, options, work, **kwargs):
        if kwargs.get("conservative"):
            return original(evidence, labels, options, work, **kwargs)
        work.deadline = time.monotonic() - 1
        raise StageInterruptedError("Unfinished shape")

    monkeypatch.setattr(pipeline, "export", interrupted_fit)
    editor = Editor(
        import_svg('<svg width="80" height="80"/>'),
        selection=Selection(whole_document=True),
    )
    job = Job(method("generate", "cel-planned"), request(editor))
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    assert state["result"]["metrics"]["conservative_geometry"]
    assert state["result"]["metrics"]["out_of_time"]
    assert state["result"]["metrics"]["validation_rejections"] == []
    job.apply()
    assert editor.undo_labels == ("Generate planned cel drawing",)
