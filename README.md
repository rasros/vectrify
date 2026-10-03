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
| Generate | SAMVG | Segments the reference with SAM and traces each region |
| Generate | Colour regions | Fits a colour palette on the GPU and traces its regions |
| Improve | Tidy | Tidies the selected paths in seconds: snaps their points to the reference and simplifies, with Add detail and a gradient shape fit on request; simplifies without one |
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

The `vision` and `samvg` extras install PyTorch and transformers, which SAMVG,
colour regions and the shape fit of Tidy need; `all` installs both.
Colour regions need an NVIDIA GPU with CUDA; SAMVG uses it when available. The
shape fit runs on the GPU with the optional native CUDA extension (below) and
on the CPU otherwise, except for outlined fills, which need the GPU.

## SAMVG

SAMVG is inspired by the SAMVG paper, not an installation of the unreleased
research code. It uses SAM ViT-H by default (ViT-B is faster), keeps masks only
when they materially improve a flat-colour reconstruction, and traces them into
layered SVG paths. It is a general-purpose tracer for photos and painterly
images; the editor's Generate dialog chooses the model, the resolution SAM
segments at and the maximum number of shapes.

The native CUDA extension for the filled-path fit is built only on request.
Build a local wheel with it, then time it:

```sh
VECTRIFY_BUILD_SAMVG_CUDA=1 uv build --wheel --no-build-isolation
uv pip install --force-reinstall --no-deps dist/vectrify-*.whl
.venv/bin/python scripts/check_cuda_renderer.py
```

PyPI releases are portable Python wheels and do not bundle the CUDA extension.

## Scripts

`scripts/` holds standalone tools run from a checkout:
`bench_colour_regions.py` runs colour regions on one image,
`bench_samvg_renderer.py` and `check_cuda_renderer.py` time the filled-path
fit, `bench_trace.py` benchmarks Generate with SAMVG and Tidy on
fixed references, `bench_lines.py` scores the Cel art tracer's lines
against vector originals (clean, noisy and stretched), `bench_shadows.py`
measures the shading it leaves out (each of these three takes `--heldout`
to run a held-out set of images never used for tuning instead of the
default tuning set; each set also holds generated cartoon images in several
styles and drawings made for the bench, in `scripts/bench_data`), and
`bench_generate.py` remakes the generated images with the OpenAI image API
(`uv run --no-project --with openai python scripts/bench_generate.py`; the
images go in `~/.cache/vectrify-bench/generated`, not the repository),
`subtle_screen.py`
checks that the scorers in `vectrify.score` order graded path damage
correctly, and
`analyze_profile.py` summarises a py-spy profile.
