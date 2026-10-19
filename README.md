# vectrify

[![PyPI](https://img.shields.io/pypi/v/vectrify.svg)](https://pypi.org/project/vectrify/)
[![Python](https://img.shields.io/pypi/pyversions/vectrify.svg)](https://pypi.org/project/vectrify/)
[![License](https://img.shields.io/pypi/l/vectrify.svg)](https://github.com/rasros/vectrify/blob/main/LICENSE)

vectrify is a local SVG editor for turning a raster image into editable vector
artwork. Load the image as a reference, trace it into shapes, then tidy and
edit the result by hand or let an AI agent edit it over MCP. Every automated
operation previews its result and applies it as one undoable edit.

## Start

```bash
uv tool install "vectrify[all]"     # or: pipx install "vectrify[all]"
vectrify                            # open the editor in its own window
vectrify drawing.svg --reference original.png
vectrify --serve --port 8765        # or serve it to a browser
```

Without the desktop extra, or with --serve, vectrify serves the editor on
loopback and prints the address to open.

## Features

- Drawing: pen, pencil, rectangles, ellipses and lines.
- Point editing: handles, snapping, and Redraw outline, which redraws part of
  an outline along the reference's edge in one stroke.
- Path operations: booleans, merge, combine, join, knife, and a Region tool
  that extracts or deletes what lies inside a shape.
- Clipping, linear gradients edited on the canvas, an eyedropper, line ends,
  arrowheads and tapers.
- Move, resize, rotate, flip, align and distribute, with snapping.
- An object tree with visibility and locks.
- SVG import and export, and .vectrify projects that keep the reference.
- Reference overlay and an error view of where the drawing differs from it.

Press ? in the editor for keyboard shortcuts, or Ctrl/⌘ K to find a command.
The [guide](https://github.com/rasros/vectrify/blob/main/docs/01-intro.md) walks through the editor chapter by chapter.

## Tracing an image

1. Load a PNG, JPEG or WebP image in the Reference panel. Press W to cycle
   between the drawing, the overlay, the reference and the error view.
2. Choose Generate with Cel art for outlined illustrations, or Colour regions
   for anything else, and apply the preview.
3. Select paths and choose Tidy to fit them to the reference and simplify
   them.
4. Fix the rest by hand.
5. Save a project, or export SVG.

Other operations are Fit colours, which fits flat fills or gradients, Clean
up, which drops redundant points and merges paths, and Snap edges, which
closes gaps between touching shapes. They change only the selection and
respect locks. See [Tracing a reference](https://github.com/rasros/vectrify/blob/main/docs/08-tracing.md).

## Agents

vectrify is also an MCP server, so an agent such as Claude Code, Codex or
Claude Desktop can look at the drawing and its reference and edit it with the
editor's own commands.

Turn on Agents in the editor's footer to host the server from the open
window. The popover shows the command to add it to your client, for example:

```bash
claude mcp add --transport http --scope user vectrify http://127.0.0.1:8770/mcp \
  --header "Authorization: Bearer <token>"
```

To edit files without a window, run it over stdio:

```bash
claude mcp add vectrify -- vectrify-mcp
```

See [Agents](https://github.com/rasros/vectrify/blob/main/docs/09-agents.md) for the tools.

## Requirements

Python 3.10 or newer and Cairo (`sudo apt install libcairo2` or
`brew install cairo`). Colour regions needs PyTorch and an NVIDIA GPU. Tidy's
shape fit needs PyTorch and runs on the CPU without a GPU, except for mitered
outlines. Everything else works without either.

On Linux the desktop window uses Qt. If it fails to start, install the system
libraries Qt asks for (often libxcb-cursor0 and libegl1) or use --serve.

An optional CUDA extension speeds up Tidy's shape fit. Releases do not
include it; build it locally with:

```bash
VECTRIFY_BUILD_CUDA_RENDERER=1 uv build --wheel --no-build-isolation
uv pip install --force-reinstall --no-deps dist/vectrify-*.whl
```
