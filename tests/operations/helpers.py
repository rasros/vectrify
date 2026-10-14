"""Shared helpers for the operations tests."""

from typing import ClassVar

from PIL import Image

from vectrify.document import Editor, Selection, import_svg
from vectrify.operations import (
    OperationRequest,
    OperationResult,
    Permissions,
    Proposal,
    SettingsMethod,
)


def render(svg, size=100):
    import io

    import cairosvg

    png = cairosvg.svg2png(
        bytestring=svg.encode(), output_width=size, output_height=size
    )
    assert png is not None
    return Image.open(io.BytesIO(png)).convert("RGB")


def request(editor, name, selection, permissions, reference=None, **settings):
    snapshot = editor.snapshot
    return OperationRequest(
        action="improve" if name == "colours" else "simplify",
        method=name,
        snapshot=type(snapshot)(snapshot.revision, snapshot.document, selection),
        editor=editor,
        permissions=permissions,
        settings=settings,
        reference=reference,
        bounds=(0, 0, 100, 100),
    )


SVG = '<svg width="20" height="20"><path id="a" d="M0 0L10 0L10 10Z"/></svg>'


class Recolour(SettingsMethod):
    action: ClassVar[str] = "improve"
    name: ClassVar[str] = "recolour"
    background: ClassVar[bool] = False
    needs_reference: ClassVar[bool] = False
    resources: ClassVar[frozenset[str]] = frozenset()

    def validate(self, request):
        pass

    def run(self, request, context):
        request = self.prepare(request).request
        context.progress(1, "Recolouring", total=1)
        tx = request.transaction("Recolour")
        tx.set_attributes("a", {"fill": "#ff0000"})
        unchanged = request.transaction("Recolour")
        return OperationResult(Proposal(tx, True), [Proposal(unchanged, False)])


def contract_request(editor, **permissions):
    return OperationRequest(
        action="improve",
        method="recolour",
        snapshot=editor.snapshot,
        editor=editor,
        permissions=Permissions(**permissions),
    )


def editor():
    return Editor(import_svg(SVG), selection=Selection(object_ids=frozenset({"a"})))


# The shape sits in a scaled, skewed group so the fit has to map the ramp
# from root user space into the shape's own.
DOC = (
    '<svg width="100" height="100" viewBox="0 0 100 100">'
    '<rect id="bg" width="100" height="100" fill="#ffffff"/>'
    '<g transform="translate(10 15) scale(1.25 0.8) skewX(10)">'
    '<path id="a" d="M0 0 L56 0 L56 65 L0 65 Z" fill="#808080"{stroke}/></g></svg>'
)


# The reference's ramp, in root user space: dark red to pale blue.
START, END = (0.0, 5.0), (100.0, 80.0)
COLOURS = ((0.7, 0.1, 0.1), (0.3, 0.6, 0.9))


def ramp_reference(size=100):
    """The drawing painted with the known ramp, rendered by cairosvg."""
    (x1, y1), (x2, y2) = START, END
    hexes = ["#" + "".join(f"{round(v * 255):02x}" for v in c) for c in COLOURS]
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" '
        'viewBox="0 0 100 100"><defs><linearGradient id="r" '
        f'gradientUnits="userSpaceOnUse" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}">'
        f'<stop offset="0" stop-color="{hexes[0]}"/>'
        f'<stop offset="1" stop-color="{hexes[1]}"/></linearGradient>'
        '<clipPath id="c"><path transform="translate(10 15) scale(1.25 0.8) '
        'skewX(10)" d="M0 0 L56 0 L56 65 L0 65 Z"/></clipPath></defs>'
        '<rect width="100" height="100" fill="#ffffff"/>'
        '<rect width="100" height="100" fill="url(#r)" clip-path="url(#c)"/></svg>'
    )
    return render(svg, size)
