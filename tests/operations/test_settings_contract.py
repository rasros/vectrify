"""Schemas and normalized settings are explicit operation capabilities."""

from dataclasses import replace
from typing import ClassVar

import pytest
from PIL import Image

from tests.operations.test_contract import Recolour, editor, request
from vectrify.document import DocumentError, Editor, Selection, import_svg
from vectrify.operations import (
    Budget,
    Job,
    Permissions,
    PreparedOperation,
    RunContext,
    contract,
    method,
)
from vectrify.operations.methods import nodes, path_fit
from vectrify.operations.settings import Setting, method_settings, settings_schema


def test_methods_in_one_module_have_independent_schemas(monkeypatch):
    class Warm(Recolour):
        name: ClassVar[str] = "warm"
        settings_schema: ClassVar[dict[str, Setting]] = {"warmth": Setting(float, 0.5)}

    class Cool(Recolour):
        name: ClassVar[str] = "cool"
        settings_schema: ClassVar[dict[str, Setting]] = {"coolness": Setting(int, 2)}

    monkeypatch.setitem(contract._METHODS, ("improve", "warm"), Warm())
    monkeypatch.setitem(contract._METHODS, ("improve", "cool"), Cool())
    assert set(method_settings("improve", "warm")) == {"warmth"}
    assert set(settings_schema("improve", "cool")["properties"]) == {"coolness"}
    assert Warm().prepare(request(editor())).settings == {"warmth": 0.5}
    with pytest.raises(DocumentError, match="Unknown operation setting: coolness"):
        Warm().prepare(replace(request(editor()), settings={"coolness": 2}))


def test_job_parses_and_normalizes_once_for_validation_execution_and_state(monkeypatch):
    calls = []
    read = contract.read_settings

    def counted_read(*args):
        calls.append("parse")
        return read(*args)

    monkeypatch.setattr(contract, "read_settings", counted_read)
    seen = []

    class Contextual(Recolour):
        settings_schema: ClassVar[dict[str, Setting]] = {"amount": Setting(float, 2.0)}

        def normalize_settings(self, operation, settings, /):
            calls.append("normalize")
            assert "amount" not in operation.settings
            settings["amount"] *= operation.budget.steps
            return settings

        def validate(self, operation):
            prepared = self.prepare(operation)
            seen.append(prepared)
            assert prepared.settings == {"amount": 6.0}

        def run(self, operation, context):
            prepared = self.prepare(operation)
            seen.append(prepared)
            return super().run(prepared, context)

    job = Job(Contextual(), replace(request(editor(), paint=True), budget=Budget(3)))
    job.start()
    assert calls == ["parse", "normalize"]
    assert seen == [job.prepared, job.prepared]
    assert seen[0] is seen[1]
    assert job.state()["effective_settings"] == {"amount": 6.0}
    assert job.request.settings == {}


@pytest.mark.parametrize("with_reference", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_nodes_preserve_omitted_tolerance_semantics(with_reference, explicit):
    operation = replace(
        request(editor()),
        settings={"tolerance": 3.0} if explicit else {},
        reference=Image.new("RGB", (20, 20)) if with_reference else None,
    )
    chosen = method("improve", "nodes")
    prepared = chosen.prepare(operation)
    expected = 3.0 if explicit or with_reference else nodes.NO_REFERENCE
    assert prepared.settings["tolerance"] == expected
    assert ("tolerance" in prepared.request.settings) == explicit
    assert chosen.prepare(prepared) is prepared
    assert prepared.settings["steps"] == nodes.SETTINGS["steps"].default


def test_direct_edge_snap_call_accepts_raw_or_prepared_settings(monkeypatch):
    chosen = method("snap", "edges")
    ed = Editor(
        import_svg(
            '<svg width="20" height="20">'
            '<path id="a" d="M0 0L10 0L10 10L0 10Z"/>'
            '<path id="b" d="M10.3 0L20 0L20 10L10.3 10Z"/></svg>'
        ),
        selection=Selection(object_ids=frozenset({"a", "b"})),
    )
    operation = replace(
        request(ed, geometry=True, structure=True),
        settings={"tolerance": 2},
        bounds=(0, 0, 20, 20),
    )
    prepared = chosen.prepare(operation)
    assert prepared.settings == {"tolerance": 2.0}
    chosen.validate(prepared)
    # After preparation no setting may be read again, even by a direct caller.
    monkeypatch.setattr(
        contract, "read_settings", lambda *_args: pytest.fail("reparsed")
    )
    assert chosen.run(prepared, RunContext()).recommended.changed
    monkeypatch.undo()
    with pytest.raises(DocumentError, match="finite number"):
        chosen.validate(replace(operation, settings={"tolerance": True}))
    assert chosen.run(operation, RunContext()).recommended.changed


def test_path_fit_keeps_budget_override_and_permission_errors():
    chosen = method("improve", "path-fit")
    operation = replace(request(editor(), geometry=True), budget=Budget(3))
    prepared = chosen.prepare(operation)
    assert path_fit.fit_options(prepared).steps == 3
    assert path_fit.fit_options(operation).steps == 3
    assert path_fit.fit_options(replace(operation, budget=Budget())).steps == 8
    assert prepared.settings == {
        key: setting.default for key, setting in chosen.settings_schema.items()
    }
    with pytest.raises(DocumentError, match="Allow geometry changes"):
        path_fit.fit_options(replace(operation, permissions=Permissions()))
    with pytest.raises(DocumentError, match="Allow paint changes"):
        path_fit.fit_options(replace(operation, settings={"color": True}))


def test_cel_auto_palette_is_prepared_before_job_reporting():
    ed = editor()
    ed.select(Selection.all())
    operation = replace(
        request(ed),
        permissions=Permissions(geometry=True, paint=True, structure=True),
        reference=Image.new("RGB", (20, 20)),
    )
    chosen = method("generate", "cel")
    job = Job(chosen, operation)
    assert job.state()["effective_settings"]["regions"] == 50
    assert job.prepared.request.settings == {}
    explicit = chosen.prepare(replace(operation, settings={"regions": 3}))
    assert explicit.settings["regions"] == 3


def test_cleanup_keeps_no_settings_error_and_empty_schema():
    chosen = method("simplify", "cleanup")
    assert chosen.settings_schema == {}
    assert chosen.prepare(request(editor())).settings == {}
    for execute in (chosen.prepare, chosen.validate):
        with pytest.raises(DocumentError, match="Geometry cleanup takes no settings"):
            execute(replace(request(editor()), settings={"tolerance": 1}))
    assert isinstance(chosen.prepare(request(editor())), PreparedOperation)
