"""Document foundation shared by future UI and operation adapters."""

from vectrify.document.editor import (
    Editor,
    EditRejectedError,
    Snapshot,
    StaleRevisionError,
)
from vectrify.document.hit_test import HitIndex
from vectrify.document.model import (
    Document,
    DocumentError,
    EditKind,
    Element,
    Geometry,
    PathNode,
    Selection,
    Subpath,
)
from vectrify.document.project import load_project, save_project
from vectrify.document.svg import UnsupportedSvgError, export_svg, import_svg

__all__ = [
    "Document",
    "DocumentError",
    "EditKind",
    "EditRejectedError",
    "Editor",
    "Element",
    "Geometry",
    "HitIndex",
    "PathNode",
    "Selection",
    "Snapshot",
    "StaleRevisionError",
    "Subpath",
    "UnsupportedSvgError",
    "export_svg",
    "import_svg",
    "load_project",
    "save_project",
]
