"""Typed method settings: reject unknown keys and out-of-range values early."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vectrify.document import DocumentError


@dataclass(frozen=True)
class Setting:
    kind: type
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] | None = None
    label: str = ""

    def read(self, name: str, value: Any) -> Any:
        label = self.label or name.replace("_", " ")
        if self.kind is bool:
            if type(value) is not bool:
                raise DocumentError(f"{label.capitalize()} must be on or off")
            return value
        if self.kind is int:
            if type(value) is not int:
                raise DocumentError(f"{label.capitalize()} must be a whole number")
        elif self.kind is float:
            if type(value) not in {int, float} or not math.isfinite(value):
                raise DocumentError(f"{label.capitalize()} must be a finite number")
            value = float(value)
        elif not isinstance(value, self.kind):
            raise DocumentError(f"{label.capitalize()} has the wrong type")
        if self.choices is not None and value not in self.choices:
            raise DocumentError(f"Choose a supported {label}")
        if self.minimum is not None and value < self.minimum:
            raise DocumentError(f"{label.capitalize()} must be at least {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise DocumentError(f"{label.capitalize()} must be at most {self.maximum}")
        return value


def read_settings(
    values: Mapping[str, Any], schema: Mapping[str, Setting], method: str
) -> dict[str, Any]:
    unknown = set(values) - set(schema)
    if unknown:
        raise DocumentError(f"Unknown {method} setting: {sorted(unknown)[0]}")
    return {
        name: setting.read(name, values[name]) if name in values else setting.default
        for name, setting in schema.items()
    }


def method_settings(action: str, name: str) -> dict[str, Setting]:
    """The same schema used by job validation and discovery."""
    import importlib

    from vectrify.operations.contract import method

    chosen = method(action, name)
    module = importlib.import_module(type(chosen).__module__)
    return dict(getattr(module, "SETTINGS", {}))


def settings_schema(action: str, name: str) -> dict[str, Any]:
    schema = method_settings(action, name)
    types = {
        bool: "boolean",
        int: "integer",
        float: "number",
        str: "string",
        list: "array",
    }
    properties = {}
    for key, setting in schema.items():
        item: dict[str, Any] = {
            "type": types[setting.kind],
            "default": setting.default,
            "description": setting.label or key.replace("_", " "),
        }
        if setting.default is None:
            item["type"] = [item["type"], "null"]
        for attr, output in (
            ("minimum", "minimum"),
            ("maximum", "maximum"),
            ("choices", "enum"),
        ):
            value = getattr(setting, attr)
            if value is not None:
                item[output] = list(value) if attr == "choices" else value
        properties[key] = item
    interactions = {
        "nodes": [
            "Enable at least one of shape, snap, simplify; detail requires snap.",
            "shape and snap require a reference; simplify also works without one.",
            "region restricts changes to points inside the region.",
        ],
        "path-fit": [
            "Enable nodes, handles or color; geometry/paint permissions apply."
        ],
        "cel": [
            "outline, line_width and strokes control line generation.",
            "regions=0 chooses a palette size automatically.",
        ],
        "cel-planned": ["Experimental planner; quality controls runtime."],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "interactions": interactions.get(name, []),
    }
