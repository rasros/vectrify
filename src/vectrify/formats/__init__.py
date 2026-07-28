"""Format backend registry.

Plugins are constructed lazily so importing this module does not pull in
graphviz, typst, or cairo for a run that does not need them.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectrify.formats.base import FormatPlugin


def _svg() -> "FormatPlugin":
    from vectrify.formats.svg.plugin import SvgPlugin

    return SvgPlugin()


def _graphviz() -> "FormatPlugin":
    from vectrify.formats.graphviz.plugin import GraphvizPlugin

    return GraphvizPlugin()


def _typst() -> "FormatPlugin":
    from vectrify.formats.typst.plugin import TypstPlugin

    return TypstPlugin()


FORMAT_PLUGINS: dict[str, Callable[[], "FormatPlugin"]] = {
    "svg": _svg,
    "graphviz": _graphviz,
    "typst": _typst,
}

FORMAT_NAMES = tuple(FORMAT_PLUGINS)


def get_plugin(name: str) -> "FormatPlugin":
    try:
        factory = FORMAT_PLUGINS[name]
    except KeyError:
        raise ValueError(f"Unknown format: {name}") from None
    return factory()
