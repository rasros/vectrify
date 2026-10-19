# vectrify

[![PyPI](https://img.shields.io/pypi/v/vectrify.svg)](https://pypi.org/project/vectrify/)
[![Python](https://img.shields.io/pypi/pyversions/vectrify.svg)](https://pypi.org/project/vectrify/)
[![License](https://img.shields.io/pypi/l/vectrify.svg)](https://github.com/rasros/vectrify/blob/main/LICENSE)

vectrify is a local SVG editor for turning a raster image into editable vector
artwork. Load the image as a reference, trace it into shapes, then tidy and
edit the result by hand or let an AI agent edit it over MCP. Every automated
operation previews its result and applies it as one undoable edit, limited to
the objects and kinds of change you allow.

## Features

- **Drawing**: pen paths, a smoothing pencil, rectangles, ellipses, circles
  and lines, drawn into the entered group with a remembered fill, stroke and
  width.
- **Point editing**: corner and smooth handles, add, break, join, delete and
  pin points, straighten handles, snapping to points and the artboard, and
  **Redraw outline**, which redraws a stretch of an outline along the
  reference's edge in one stroke.
- **Path operations**: Union, Subtract, Intersect and Exclude; Merge shapes,
  Combine paths, Join line ends, Cut out as hole, Split disconnected parts,
  Convert to path, Fill to centerline and Stroke to path; a Knife; and a
  **Region** tool that extracts or deletes the part of the drawing inside a
  rectangle or polygon.
- **Clipping**: set and remove clips with linked or copied boundaries, and
  show and edit clipping boundaries on the canvas.
- **Paint**: fill, stroke, width and opacity; linear gradients edited in the
  panel or by dragging their handles on the canvas; an eyedropper sampling the
  artwork or the reference; line ends, corner joins, arrowheads and tapers.
- **Arranging**: move, resize, rotate and flip with snapping and guides,
  numeric position and size, align and distribute (to the selection, the
  artboard or a key object), grouping and stacking.
- **Organising**: an object tree with drag-and-drop restacking, visibility
  toggles, names, and geometry, paint, position and structure locks.
- **Files**: open and export SVG, save `.vectrify` projects that keep IDs,
  locks, pins and the reference image, recovery copies of saved versions, and
  an adjustable artboard.
- **Reference comparison**: overlay the reference, view it alone or as an
  error map of where the drawing differs from it.

Press `?` in the editor for the keyboard shortcuts, or Ctrl/⌘ K to find any
command by name. [The editor guide](docs/editor.md) describes every control.

## Start

```bash
uv tool install "vectrify[all]"     # or: pipx install "vectrify[all]"
vectrify                            # open the editor in its own window
vectrify drawing.svg --reference original.png
vectrify --serve --port 8765        # or serve it to a browser
```

The `desktop` extra (included in `all`) opens the editor in a native window.
Without it, or with `--serve`, vectrify serves the editor on loopback and
prints the address to open in a browser. Neither needs a Node build step. From
a source checkout, run `uv run vectrify`.

## Quick tutorial

1. **Open a reference.** In the **Reference** panel at the top left, choose
   **Load…** and pick a PNG, JPEG or WebP image. An empty drawing takes the
   image's size as its artboard. Press `W` to cycle between the drawing, the
   overlay, the reference alone and the error view.
2. **Generate.** Choose **Generate…** with the **Cel art** method, preview the
   trace and press **Apply**. The shapes land in one new group over the
   reference.
3. **Tidy.** Select paths (or zoom in and select nothing) and choose
   **Tidy…** to fit their points to the reference, add missing detail or
   simplify them. Preview, then apply.
4. **Edit.** Fix what remains by hand: **Nodes** (`S`) for points, **Redraw
   outline** (`R`) for a wrong stretch of outline, the Path commands for
   booleans and joins, and the inspector for paint.
5. **Save and export.** **Save** (Ctrl/⌘ S) keeps a `.vectrify` project with
   the reference; **Export SVG** writes plain SVG.

## Operations

| Action | Method | What it does |
| --- | --- | --- |
| Generate | Cel art (`cel`) | Traces flat, outlined illustrations as regions inside their drawn lines, the lines as strokes |
| Generate | Colour regions (`colour-regions`) | Fits a colour palette on the GPU and traces its regions |
| Improve | Tidy (`nodes`) | Fits the selected paths, or what is in view, to the reference and simplifies them, never leaving the match worse than the allowance |
| Improve | Fit colours (`colours`) | Closed-form flat fill colours or linear gradients, geometry locked |
| Simplify | Clean up (`cleanup`) | Drops redundant vertices and merges compatible paths |
| Snap | Snap edges (`edges`) | Snaps the touching edges of the selected paths together (under **Repair**) |

Generated shapes are placed over the artboard exactly where the reference is
shown. The other operations change only the selection, and locks, pins and
permissions are enforced by the backend for every method.
[docs/operations.md](docs/operations.md) describes the operation contract for
writing new methods.

## Agents

Vectrify is an MCP server through which an agent (Claude Code, Claude
Desktop or any MCP client) looks at a drawing and its reference and edits it
with the editor's own commands, one undoable step per call.

**From the editor.** Turn on **Agents** in the editor's footer (installed
with `[mcp]` or `[all]`): the editor itself hosts the MCP server over
Streamable HTTP at `http://127.0.0.1:8770/mcp` (the next free port if that
one is taken), and the footer's popover shows the command that adds it to
Claude Code, with a Copy button:

```bash
claude mcp add --transport http --scope user vectrify http://127.0.0.1:8770/mcp \
  --header "Authorization: Bearer <token>"
```

Add it once: the token is kept across restarts (regenerate it from the same
popover), and the client reaches that window whenever Agents is on, so you
watch each edit happen.

The same popover's *Add to an app* section also has text to copy for
**Codex** (the ChatGPT desktop app, CLI and IDE extension: a
`[mcp_servers.vectrify]` table for `~/.codex/config.toml`, or the
`codex mcp add` command) and **Claude Desktop** (an `mcpServers` entry for
`claude_desktop_config.json`). Both start this install's `vectrify-mcp`,
which attaches to the window with Agents on; restart the app after adding
it. Vectrify never edits these files itself.

**Headless, over stdio.** `vectrify-mcp` edits a file with no window, or
`connect()`s to a running editor that allows agents:

```bash
claude mcp add vectrify -- vectrify-mcp                 # installed with [mcp] or [all]
claude mcp add vectrify -- uvx --from "vectrify[mcp]" vectrify-mcp
```

[docs/mcp.md](docs/mcp.md) lists the tools and how the live channel works.

## Requirements

Python 3.10 or newer, and Cairo for SVG rendering. The `vision` extra
installs PyTorch, which Colour regions and Tidy's shape fit need; `all`
installs it with the `desktop` and `mcp` extras.

## Troubleshooting

- **Cairo is missing** (an `OSError` that no Cairo library was found): install it
  with `sudo apt install libcairo2` on Debian/Ubuntu or `brew install cairo`
  on macOS.
- **No PyTorch or no CUDA.** Without PyTorch, Colour regions is unavailable
  and Tidy's **Fit path** falls back to edge-seeking; Cel art, Fit colours,
  Clean up, Snap edges and all manual editing work. With PyTorch but no
  NVIDIA GPU, Colour regions is still unavailable and Tidy's shape fit runs
  on the CPU, except for miter-joined outlines, which need CUDA.
- **No desktop window on Linux.** The `desktop` extra uses pywebview's Qt
  backend. Without the extra, vectrify says so and serves to a browser. If
  the Qt window fails to start, install the system libraries Qt asks for
  (often `libxcb-cursor0` and `libegl1`), or run `vectrify --serve`.
- **Port in use.** `--serve` listens on 8765; pick another with `--port`, or
  `--port 0` for any free one (the address is printed). The MCP server moves
  to the next free port by itself, and `--mcp-port` chooses one.

## Development

```bash
uv sync --extra dev --extra vision
uv run pytest -n 2
uv run ruff check src tests
uv run ruff format --check src tests
uv run pyrefly check
uvx vulture
```

The UI tests run the editor's scripts with Node.js (and some with
Chrome/Chromium); they are skipped when those are not installed.

## CUDA extension

The native CUDA extension for the filled-path fit is built only on request.
PyPI releases are portable Python wheels and do not bundle it. Build and
install a local wheel with it:

```sh
VECTRIFY_BUILD_CUDA_RENDERER=1 uv build --wheel --no-build-isolation
uv pip install --force-reinstall --no-deps dist/vectrify-*.whl
```

`uv run pytest tests/refine/test_filled_paths.py` exercises it.
