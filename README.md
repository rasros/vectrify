# vectrify

[![PyPI](https://img.shields.io/pypi/v/vectrify.svg)](https://pypi.org/project/vectrify/)
[![Python](https://img.shields.io/pypi/pyversions/vectrify.svg)](https://pypi.org/project/vectrify/)
[![License](https://img.shields.io/pypi/l/vectrify.svg)](https://github.com/rasros/vectrify/blob/main/LICENSE)

vectrify turns a raster image into editable vector code. It asks an LLM for
candidate drawings, compares their renders with the input image, and refines
the strongest candidates over several search epochs using NSGA-II multi-
objective evolutionary search. For SVG output, a path optimizer provides
additional geometric refinement.

It currently writes SVG, Graphviz DOT, or Typst. SVG is the default.

## Install

Python 3.10 or newer is required. Install the CLI with `pipx` or `uv`:

```bash
pipx install "vectrify[vision]"  # recommended
# or: uv tool install "vectrify[vision]"
```

The vision extra enables the perceptual scorer. Use
`pipx install "vectrify[all]"` to also install the Graphviz and Typst output
backends.

A GPU is optional, but it speeds up both the SVG path optimizer and the
perceptual scorer. A compatible PyTorch installation can use NVIDIA CUDA for
both; the perceptual scorer can also use Apple MPS. Both components fall back
to CPU, and the simple scorer does not require a GPU.

Graphviz output also needs the Graphviz system package. SVG rendering needs
Cairo. On Debian/Ubuntu, install both with `sudo apt install graphviz libcairo2`.

Set one LLM provider key before running: OPENAI_API_KEY,
ANTHROPIC_API_KEY, or GEMINI_API_KEY.

With `--provider auto` (the default), vectrify uses the first configured key
in this order: OpenAI, Anthropic, Gemini. Select one explicitly when more than
one key is set.

## Usage

Convert an image to SVG with `vectrify input.png -o output.svg`.

Supported input formats are PNG, JPEG, WEBP, and GIF. The default run uses up
to 50 epochs, stops after two unimproved epochs, or ends at the one-hour wall
clock limit. LLM calls are bounded by epochs x seed (50 and 5 by
default, respectively).

Here are some common options:

```bash
# Give the LLM extra direction
vectrify logo.png -o logo.svg \
  --goal "Use thick strokes and avoid gradients"

# Spend more or less on each epoch
vectrify photo.jpg -o sketch.svg --seeds 10 --epochs 4 \
  --max-wall-seconds 1800

# Retain the best candidates for local parts of the target.
# Eight disjoint target tiles is the default; increase this for detail-heavy art.
vectrify mascot.png -o mascot.svg --segment-count 12

# Choose a provider, model, or scorer explicitly
vectrify input.png --provider anthropic --model MODEL_NAME
vectrify input.png --scorer simple

# Write another vector format
vectrify diagram.png -o diagram.dot --format graphviz
vectrify page.png -o page.typ --format typst

# Disable optional per-node artifacts
vectrify input.png -o output.svg --no-save-raster --no-write-lineage
```

Run `vectrify --help` for every option, including resolution, worker count,
epoch stopping criteria, and logging controls.

## Resume a run

By default, a new run starts from scratch. Continue from the latest run for the
same output path with `vectrify input.png -o output.svg --resume`.

Resume only the best N saved candidates with `--resume-top N`. To refine saved
candidates without making new LLM calls, use `--seeds 0 --resume`.

## Output files

The selected result is written to the path passed with `-o`. Run artifacts are
stored beside it:

```text
output.svg
output/
└── runs/
    └── 2026-08-22_13-00-00/
        ├── lineage.csv
        ├── segments/
        │   ├── manifest.json
        │   └── segment-00.png
        └── nodes/
            ├── 1.svg
            ├── eval0.123456_2.svg
            └── ...
```

Lineage is enabled by default. Rendered PNGs are saved alongside node files
by default; add `--save-heatmap` for perceptual difference maps.

## Segment elites

`--segment-count N` controls both the number of disjoint target tiles and the
maximum number of locally retained elites (one champion per tile). It defaults
to `8`. The search remembers the candidate with the best masked colour-and-edge
match for each tile, then reserves part of later LLM seed batches for those
local champions. The generated masks and their manifest are saved in each run's
`segments/` directory.
