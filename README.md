# vectrify

[![PyPI](https://img.shields.io/pypi/v/vectrify.svg)](https://pypi.org/project/vectrify/)
[![Python](https://img.shields.io/pypi/pyversions/vectrify.svg)](https://pypi.org/project/vectrify/)
[![License](https://img.shields.io/pypi/l/vectrify.svg)](https://github.com/rasros/vectrify/blob/main/LICENSE)

vectrify is a local SVG editor for turning a raster image into editable vector
artwork. Alongside manual tools it offers automated operations that work
against a reference image: generate new shapes, improve existing ones, and
simplify the result. Every operation previews its result and applies it as one
undoable edit, limited to the objects and kinds of change you allow.

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
a source checkout, run `uv run vectrify`. See [the editor guide](docs/editor.md)
for every control.

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

## Operations

| Action | Method | What it does |
| --- | --- | --- |
| Generate | Cel art | Traces flat, outlined illustrations as regions inside their drawn lines, the lines as strokes |
| Generate | Colour regions | Fits a colour palette on the GPU and traces its regions |
| Improve | Tidy | Tidies the selected paths, or what is in view, in seconds: snaps their points to the reference, fits their shape and simplifies within an error budget, never leaving the match worse, moving shared edges together and stroked lines onto their ink; simplifies without one |
| Improve | Fit colours | Closed-form flat fill colours, geometry locked |
| Simplify | Clean up | Drops redundant vertices and merges compatible paths |

Generated shapes are placed over the artboard exactly where the reference is
shown. Improve and Simplify change only the selection, and locks, pins and
permissions are enforced by the backend for every method.
[docs/operations.md](docs/operations.md) describes the operation contract for
writing new methods.

## Requirements

Python 3.10 or newer. SVG rendering needs Cairo; on Debian/Ubuntu install it
with `sudo apt install libcairo2`.

The `vision` extra installs PyTorch, which colour regions
and the shape fit of Tidy need; `all` installs it with the desktop and MCP
extras. Colour regions need an NVIDIA GPU with CUDA. The shape fit runs on the
GPU with the optional native CUDA extension (below) and on the CPU otherwise,
except for outlined fills, which need the GPU.

## CUDA extension

The native CUDA extension for the filled-path fit is built only on request.
Build a local wheel with it, then time it:

```sh
VECTRIFY_BUILD_CUDA_RENDERER=1 uv build --wheel --no-build-isolation
uv pip install --force-reinstall --no-deps dist/vectrify-*.whl
.venv/bin/python scripts/check_cuda_renderer.py
```

PyPI releases are portable Python wheels and do not bundle the CUDA extension.

## Scripts

`scripts/` holds standalone tools run from a checkout:
`bench_colour_regions.py` runs colour regions on one image,
`bench_cuda_renderer.py` and `check_cuda_renderer.py` time the filled-path
fit, `bench_trace.py` benchmarks Generate (Cel art by default) and Tidy on
fixed references, `bench_lines.py` scores the Cel art tracer's lines
against vector originals (clean, noisy and stretched), `bench_shadows.py`
measures the shading it leaves out (each of these three takes `--heldout`
to run a held-out set of images never used for tuning instead of the
default tuning set; each set holds generated cartoon images, mostly anime
and then several other styles, kept in `~/.cache/vectrify-bench/generated`,
hard drawings made for the bench, in `scripts/bench_data`, and the line
bench's Wikipe-tan drawings; `bench_trace.py --photos` runs a set of
photographs from Wikimedia Commons instead, kept in
`~/.cache/vectrify-bench/photos` and attributed in
`scripts/bench_data/photos.json`), and `analyze_profile.py` summarises a
py-spy profile.

Tidy is benchmarked as it is used, a touch-up after tracing:
`bench_trace.py --tidy` tidies the 20 largest traced paths (`--paths N`,
or `--tidy-all`) and reports, before and after, the whole image's error,
the error over the tidied paths' own area, their points, Tidy's time per
path, how many paths it changed, left alone, refused or stopped at its
time limit, the self-crossings it left, and the gaps and overlaps between
fills where it acted; each `--tidy-steps
snap,simplify[,detail][,shape]` tidies the same trace once more,
`--trace-cache DIR` reuses traces across runs (keyed by the image, settings
and tracing code), `--tidy-crops DIR` saves before/after crops, and `--summary RUN...`
compares the configurations with sign tests across images and paths.
`bench_lines.py --tidy STEPS` scores a tidied trace's lines (F, width)
beside the trace's own.

`bench_tidy.py` benchmarks the fixed editable sword project, selecting all ten
paths in its `blade` group in one operation. It measures each fill outside
Path 80, uncovered interior, reflected outline area, and partially transparent
interior pixels with the blade rendered alone. It also reports transparent
pixels in the complete drawing, distinguishing
visible gaps from gaps covered by later artwork. The original quality gate uses
the blade-only count.
The fixture includes the reference and stays independent of the working
`sword.vectrify` file. These are benchmark
targets; the generic Tidy algorithm does not impose a blade layout or symmetry.
Geometric checks use double-precision regions flattened with a 0.0001-square-unit
error budget per path, below the 0.05-square-unit pass threshold. This avoids
unstable cubic intersections when the outline nearly overlaps its reflection.
`--check` fails when any target is unmet. For a longer fit:
`uv run python scripts/bench_tidy.py --nodes '{"seconds":30,"rounds":8,"steps":40,"movement":4}' --out .bench/sword.json`.
