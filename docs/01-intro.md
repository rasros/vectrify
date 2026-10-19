# Introduction

vectrify is a local SVG editor for turning a raster image into editable
vector artwork.

## What it does

You load an image as a reference, trace it into shapes, then tidy the
result by hand, with automated operations, or through an AI agent over MCP.
Every automated operation previews its result and applies it as one
undoable edit, and every edit, whoever makes it, respects your locks and
pinned points.

## Installing

```bash
uv tool install "vectrify[all]"     # or: pipx install "vectrify[all]"
```

The extras are `desktop` (a native window), `mcp` (the agent server) and
`vision` (PyTorch, for Colour regions and Tidy's shape fit); `all` has all
three. vectrify needs Python 3.10 or newer and Cairo.

## Starting

```bash
vectrify                                         # an empty drawing
vectrify drawing.svg --reference original.png    # a file and its reference
vectrify --serve --port 8765                     # serve it to a browser
```

With the `desktop` extra the editor opens in its own window, and saving
uses native file dialogs. Without it, or with `--serve`, vectrify serves
the editor on loopback and prints the address to open in a browser. From a
source checkout, `PYTHONPATH=src python -m vectrify.ui` works too.

## The window

- Top bar: Commands, New, Open, Restore saved, Save, Save As and Export
  SVG. On a narrow window the less used actions move into a ⋯ menu.
- Left rail: the tools. Select (A), Move (V), Nodes (S), Redraw outline (R)
  and Eyedropper (I) are always in view; More tools holds Draw, Knife (C),
  Region (X), Gradient (G) and Pan (hold Space).
- Left panels: the Reference panel, with the automated operations, above
  the Objects tree.
- Tool strip, above the canvas: undo and redo, the active tool's controls,
  and the selection's level and count, such as "Objects · 3 selected".
  Numbered controls show their digit, and the keys 1-9 press them.
- Right panel: three tabs. Object holds name, paint, transform and locks;
  Points holds coordinates, handles and pins; Actions holds Arrange, Path,
  Clipping, Repair and Object commands.
- Footer: the artboard size, the reference view toggle, Show clips, the
  Agents toggle and zoom.

Right-click opens a short menu of the commands that fit what is under the
pointer, and More actions in it opens the command palette. Ctrl/⌘ K opens
the palette directly: type to find any command by name or keyword, with
its shortcut. A command that cannot run stays in place, dimmed, and its
tooltip says why.

Input made while an edit is running is never lost: it waits for the edit
and then applies in order. Panning and zooming never wait.

## The chapters

1. Introduction, this chapter.
2. [Files and the artboard](02-files.md): opening, saving, projects and the
   page.
3. [Objects and selection](03-objects.md): the Objects tree, selecting,
   hiding, locks and pins.
4. [Transforming](04-transforming.md): move, resize, rotate, flip, align
   and arrange.
5. [Drawing and points](05-drawing.md): the drawing tools, point editing and
   Redraw outline.
6. [Paint](06-paint.md): fill and stroke, gradients, line ends, tapers and
   the eyedropper.
7. [Path operations](07-paths.md): joining, booleans, converting, the knife,
   regions and clipping.
8. [Tracing a reference](08-tracing.md): Generate, Tidy, Fit colours, Clean
   up and Snap edges.
9. [Agents](09-agents.md): the MCP server and its tools.
10. [Shortcuts and limits](10-shortcuts.md).
11. [Operations](11-operations.md): for developers, the contract automated
    operations implement.
12. [Document model](12-document-model.md): for developers, transactions,
    identities and file formats.

Next: [Files and the artboard](02-files.md)
