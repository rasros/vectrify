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
vectrify                            # open the editor
vectrify drawing.svg --reference original.png --port 8765
```

Open the printed localhost address in a browser. The server listens on loopback
only and needs no Node build step. From a source checkout, run
`uv run vectrify`. See [the editor guide](docs/editor.md) for every control.

## Operations

| Action | Method | What it does |
| --- | --- | --- |
| Generate | SAMVG | Segments the reference with SAM and traces each region |
| Generate | Colour regions | Fits a colour palette on the GPU and traces its regions |
| Generate | LLM | Asks a multimodal model to draw the reference |
| Improve | Optimize path | GPU gradient fitting of one path's nodes, handles and colour |
| Improve | Search improvements | Hill climbing over the selection |
| Improve | Edit with LLM | Sends the drawing and an instruction to a model |
| Improve | Fit colours | Closed-form flat fill colours, geometry locked |
| Simplify | Smooth / simplify | Refits contours with fewer lines and curves |
| Simplify | Clean up geometry | Drops redundant vertices and merges compatible paths |

Generated shapes are placed over the artboard exactly where the reference is
shown. Improve and Simplify change only the selection, and locks, pins and
permissions are enforced by the backend for every method, including LLM edits.
[docs/operations.md](docs/operations.md) describes the operation contract for
writing new methods.

## Requirements

Python 3.10 or newer. SVG rendering needs Cairo; on Debian/Ubuntu install it
with `sudo apt install libcairo2`.

The `vision` extra enables the perceptual scorer and colour regions; the
`samvg` extra enables SAM segmentation; `all` installs both. Optimize path and
colour regions need an NVIDIA GPU with PyTorch CUDA; SAMVG and the search use
it when available. Optimize path also needs the optional native CUDA extension
(below).

The LLM methods need one provider key: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
or `GEMINI_API_KEY`. With the provider set to automatic, the first key set is
used in that order.

## SAMVG

SAMVG is inspired by the SAMVG paper, not an installation of the unreleased
research code. It uses SAM ViT-H by default (ViT-B is faster), keeps masks only
when they materially improve a flat-colour reconstruction, and traces them into
layered SVG paths. SAM inputs default to a 1024px maximum side
(`VECTRIFY_SAMVG_MAX_SIDE`) and decode 64 prompts per CUDA batch
(`VECTRIFY_SAMVG_POINTS_PER_BATCH`). `VECTRIFY_SAMVG_MODEL` changes the
default checkpoint.

For the dissertation-style two-phase measurement (initial fit, residual prompts
and recovery fit), build a local wheel with the optional native CUDA renderer
and run:

```sh
VECTRIFY_BUILD_SAMVG_CUDA=1 uv build --wheel --no-build-isolation
uv pip install --force-reinstall --no-deps dist/vectrify-*.whl
.venv/bin/python scripts/bench_samvg_two_phase.py --cat
```

`--all` also evaluates the connect-the-dots duck.
PyPI releases are portable Python wheels and do not bundle the CUDA extension.

## Benchmarks

`scripts/bench_colour_regions.py` runs colour regions on one image.
