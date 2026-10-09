"""Operations propose uncommitted edits of a snapshot; applying checks the revision."""

from typing import ClassVar

import pytest

from vectrify.document import (
    DocumentError,
    Editor,
    Element,
    Selection,
    StaleRevisionError,
    import_svg,
)
from vectrify.operations import (
    Budget,
    Job,
    OperationRequest,
    OperationResult,
    Permissions,
    Proposal,
    RunContext,
    available,
    method,
)

SVG = '<svg width="20" height="20"><path id="a" d="M0 0L10 0L10 10Z"/></svg>'


class Recolour:
    action: ClassVar[str] = "improve"
    name: ClassVar[str] = "recolour"
    background: ClassVar[bool] = False
    needs_reference: ClassVar[bool] = False
    resources: ClassVar[frozenset[str]] = frozenset()

    def validate(self, request):
        pass

    def run(self, request, context):
        context.progress(1, "Recolouring", total=1)
        tx = request.transaction("Recolour")
        tx.set_attributes("a", {"fill": "#ff0000"})
        unchanged = request.transaction("Recolour")
        return OperationResult(Proposal(tx, True), [Proposal(unchanged, False)])


def request(editor, **permissions):
    return OperationRequest(
        action="improve",
        method="recolour",
        snapshot=editor.snapshot,
        editor=editor,
        permissions=Permissions(**permissions),
    )


def editor():
    return Editor(import_svg(SVG), selection=Selection(object_ids=frozenset({"a"})))


def test_proposal_applies_as_one_undoable_edit():
    ed = editor()
    job = Job(Recolour(), request(ed, paint=True))
    job.start()
    state = job.state()
    assert state["status"] == "ready"
    assert state["step"] == state["steps"] == 1
    assert state["alternatives"][0]["changed"] is False
    assert state["alternatives"][0]["metrics"] == {}
    assert "diagnostics" in state["alternatives"][0]
    assert ed.snapshot.revision == 0
    job.apply()
    assert ed.undo_labels == ("Recolour",)
    assert ed.snapshot.document.element("a").get("fill") == "#ff0000"


def test_unchanged_alternative_cannot_be_applied():
    ed = editor()
    job = Job(Recolour(), request(ed, paint=True))
    job.start()
    with pytest.raises(DocumentError, match="unchanged"):
        job.apply(1)
    with pytest.raises(DocumentError, match="Choose"):
        job.apply(2)


def test_permissions_are_enforced_by_the_transaction():
    job = Job(Recolour(), request(editor(), geometry=True))
    with pytest.raises(DocumentError, match="not permitted"):
        job.start()
    assert job.state()["status"] == "failed"


def test_edits_made_while_running_merge_with_the_result():
    ed = editor()
    job = Job(Recolour(), request(ed, paint=True))
    job.start()
    with ed.transaction("Move") as tx:
        tx.set_attributes("a", {"opacity": "0.5"})
    job.apply()
    assert ed.snapshot.document.element("a").get("fill") == "#ff0000"
    assert ed.snapshot.document.element("a").get("opacity") == "0.5"


def test_proposal_keeps_a_new_selection_made_after_the_job_started():
    ed = editor()
    job = Job(Recolour(), request(ed, paint=True))
    job.start()
    with ed.transaction("Add", selection=Selection.all()) as tx:
        tx.insert_object(
            tx.preview.root.id,
            Element(
                "b",
                "rect",
                (("width", "5"), ("height", "5")),
            ),
        )
    ed.select(Selection(object_ids=frozenset({"b"})))
    job.apply()
    assert ed.snapshot.selection.object_ids == {"b"}
    assert ed.snapshot.document.element("a").get("fill") == "#ff0000"


def test_transaction_over_an_earlier_snapshot_commits_only_if_current():
    ed = editor()
    base = ed.snapshot
    with ed.transaction("Other") as tx:
        tx.set_attributes("a", {"opacity": "0.5"})
    late = ed.transaction("Late", base=base)
    late.set_attributes("a", {"fill": "#00ff00"})
    assert late.preview.element("a").get("opacity") is None
    with pytest.raises(StaleRevisionError):
        late.commit()


def test_stop_before_start_cancels_without_running():
    class Explodes(Recolour):
        background: ClassVar[bool] = True
        resources: ClassVar[frozenset[str]] = frozenset({"gpu"})

        def run(self, request, context):
            del request, context
            pytest.fail("should not run")

    job = Job(Explodes(), request(editor(), paint=True))
    job.stop.set()
    job.run()
    assert job.state()["status"] == "cancelled"


def test_request_parsing_rejects_unknown_values():
    with pytest.raises(DocumentError, match="Unknown edit permission"):
        Permissions.parse({"colour": True})
    with pytest.raises(DocumentError, match="on or off"):
        Permissions.parse({"paint": "yes"})
    assert Permissions.parse({"paint": True}).allowed == {"paint"}
    with pytest.raises(DocumentError, match="positive whole"):
        Budget.parse({"steps": 0})
    with pytest.raises(DocumentError, match="seconds"):
        Budget.parse({"seconds": float("nan")})
    assert Budget.parse({"steps": 3, "seconds": 2}) == Budget(3, 2.0)
    with pytest.raises(DocumentError, match="Unknown improve method"):
        method("improve", "nope")


def test_built_in_methods_are_registered():
    names = {(m.action, m.name) for m in available()}
    assert {
        ("improve", "path-fit"),
        ("improve", "nodes"),
        ("snap", "edges"),
    } <= names


def test_run_context_reports_progress():
    seen = []
    context = RunContext(total=4)
    context.listen(lambda step, message: seen.append((step, message)))
    context.progress(2, "Half")
    assert (context.step, context.total, seen) == (2, 4, [(2, "Half")])


def test_gpu_jobs_wait_for_the_shared_gate():
    import time

    from vectrify.operations import RESOURCES
    from vectrify.refine.gpu import gpu_gate

    class Gpu(Recolour):
        background: ClassVar[bool] = True
        resources: ClassVar[frozenset[str]] = frozenset({"gpu"})

    assert RESOURCES["gpu"] is gpu_gate()
    job = Job(Gpu(), request(editor(), paint=True))
    gpu_gate().acquire()
    try:
        job.start()
        time.sleep(0.5)
        assert job.state()["status"] == "running"
        assert "GPU" in job.state()["message"]
        job.stop.set()
        for _ in range(50):
            if job.state()["status"] != "running":
                break
            time.sleep(0.05)
        assert job.state()["status"] == "cancelled"
    finally:
        gpu_gate().release()
    job = Job(Gpu(), request(editor(), paint=True))
    job.start()
    for _ in range(100):
        if job.state()["status"] != "running":
            break
        time.sleep(0.02)
    assert job.state()["status"] == "ready"
