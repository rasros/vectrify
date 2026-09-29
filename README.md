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

## Operations

| Action | Method | What it does |
| --- | --- | --- |
| Generate | SAMVG | Segments the reference with SAM and traces each region |
| Generate | Colour regions | Fits a colour palette on the GPU and traces its regions |
| Generate | LLM | Asks a multimodal model to draw the reference |
| Improve | Optimize nodes | Moves, adds or removes the selected paths' points to follow the reference; simplifies without one |
| Improve | Edit with LLM | Sends the drawing and an instruction to a model |
| Improve | Fit colours | Closed-form flat fill colours, geometry locked |
| Simplify | Clean up geometry | Drops redundant vertices and merges compatible paths |

Generated shapes are placed over the artboard exactly where the reference is
shown. Improve and Simplify change only the selection, and locks, pins and
permissions are enforced by the backend for every method, including LLM edits.
[docs/operations.md](docs/operations.md) describes the operation contract for
writing new methods.

## Requirements

Python 3.10 or newer. SVG rendering needs Cairo; on Debian/Ubuntu install it
with `sudo apt install libcairo2`.

The `vision` and `samvg` extras install PyTorch and transformers, which SAMVG,
colour regions and the GPU engine of Optimize nodes need; `all` installs both.
Colour regions and the GPU engine need an NVIDIA GPU with CUDA; SAMVG uses it
when available. The GPU engine also needs the optional native CUDA extension
(below); without it Optimize nodes uses its CPU search.

The LLM methods need an OpenAI, Anthropic or Gemini API key, or a local
server with an OpenAI-compatible API (Ollama, LM Studio, llama.cpp, vLLM) and a
vision model, entered under **Settings** in the editor. They are saved to
`~/.config/vectrify/settings.json` (owner-readable only). With the provider set
to automatic, the hosted providers are tried in that order and the local server
last.

## SAMVG

SAMVG is inspired by the SAMVG paper, not an installation of the unreleased
research code. It uses SAM ViT-H by default (ViT-B is faster), keeps masks only
when they materially improve a flat-colour reconstruction, and traces them into
layered SVG paths. SAM inputs default to a 1024px maximum side
(`VECTRIFY_SAMVG_MAX_SIDE`) and decode 64 prompts per CUDA batch
(`VECTRIFY_SAMVG_POINTS_PER_BATCH`). `VECTRIFY_SAMVG_MODEL` changes the
default checkpoint.

The native CUDA extension is built only on request. Build a local wheel with
it, then run the two-phase measurement (initial fit, residual prompts and
recovery fit) on an image:

```sh
VECTRIFY_BUILD_SAMVG_CUDA=1 uv build --wheel --no-build-isolation
uv pip install --force-reinstall --no-deps dist/vectrify-*.whl
.venv/bin/python scripts/bench_samvg_two_phase.py --target image.png
```

PyPI releases are portable Python wheels and do not bundle the CUDA extension.

## Scripts

`scripts/` holds standalone tools run from a checkout:
`bench_colour_regions.py` runs colour regions on one image,
`bench_samvg_renderer.py` and `check_cuda_renderer.py` time the filled-path
fit, `bench_samvg_two_phase.py` runs SAMVG's two phases, `subtle_screen.py`
checks that the scorers in `vectrify.score` order graded path damage
correctly, and
`analyze_profile.py` summarises a py-spy profile.
