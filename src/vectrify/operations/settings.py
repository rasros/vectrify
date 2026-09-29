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
