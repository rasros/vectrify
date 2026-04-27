# vectrify

[![PyPI](https://img.shields.io/pypi/v/vectrify.svg)](https://pypi.org/project/vectrify/)
[![Python](https://img.shields.io/pypi/pyversions/vectrify.svg)](https://pypi.org/project/vectrify/)
[![License](https://img.shields.io/pypi/l/vectrify.svg)](https://github.com/rasros/vectrify/blob/main/LICENSE)

While LLMs are powerful they still struggle to generate perfect vector
images from reference raster images in one shot. That is where vectrify
can help. It turns raster images into editable vector code by treating
vectorization as a search problem: an LLM proposes candidate
SVG/Graphviz/Typst code, a vision scorer ranks how close each candidate
looks to the source, and an optimization algorithm iteratively refines
the best candidates.

The results are quite good and produces human-readable code.

## Features

Three output formats are supported out of the box: SVG (default),
Graphviz DOT, and Typst (HTML and TikZ are planned). API keys for OpenAI,
Anthropic, and Google Gemini are auto-detected from environment
variables. Two search strategies are available: NSGA-II for
diversity-preserving multi-objective optimization that weighs in
complexity, and beam search for a budget-friendly singular solution.
Perceptual scoring uses a local vision model with embeddings, with a
pixel-level fallback or LLM-as-judge as alternatives. Runs are resumable,
so you can pick up where you left off or fork from the top-N nodes of a
previous run. A live dashboard shows pool stats, scoring, and convergence
criteria.

## Install

The recommended way to install a CLI tool is `pipx` or `uv tool`, which
each put `vectrify` in its own isolated environment and on your PATH:

```bash
pipx install vectrify           # or: uv tool install vectrify
```

Plain `pip install vectrify` works too, but installs into whatever Python
environment is active. If you use `pip install --user`, add
`~/.local/bin` to your PATH so the `vectrify` command resolves.

The base install includes SVG output and the simple pixel-difference
scorer. For everything else, pick the extras you need:

| Extra      | What it adds                                                   |
|------------|----------------------------------------------------------------|
| `vision`   | torch + transformers for the perceptual (CLIP/SigLIP) scorer   |
| `graphviz` | the graphviz Python bindings (system Graphviz still required)  |
| `typst`    | the typst Python compiler                                      |
| `all`      | vision + graphviz + typst                                      |

```bash
pipx install "vectrify[vision]"          # recommended for best quality
pipx install "vectrify[all]"             # everything
```

System dependencies: Cairo is needed for SVG → PNG rendering (`apt
install libcairo2` or `brew install cairo`), Graphviz is needed only when
using `--format graphviz` (`apt install graphviz` or `brew install
graphviz`), and a CUDA-capable GPU is optional since the vision scorer
falls back to CPU/MPS.

## Provider setup

Set exactly one of the following environment variables:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

Override with `--provider {openai,anthropic,gemini}` if you have multiple
keys set.

## Quickstart

```bash
vectrify input.png -o output.svg
```

That's it. The defaults run up to 5 NSGA-II epochs and stop early once
the search stops finding new improvements (see [Convergence](#convergence)
below). Worst case it runs for an hour and gives up.

A few useful variations:

```bash
# Bigger budget, longer runs
vectrify photo.jpg -o sketch.svg --epoch-patience 60 --max-wall-seconds 1800

# Steer the search with a goal
vectrify logo.png --goal "Use thick strokes only and avoid gradients"

# Output Graphviz DOT instead of SVG
vectrify diagram.png -o out.dot --format graphviz

# Resume from a previous run, keeping only the 20 best nodes
vectrify input.png --resume --resume-top 20
```

Run `vectrify --help` for the full flag reference, organized into LLM
provider, scoring, search strategy, epoch control, resume, output
artifacts, and runtime sections.

## How it works

vectrify runs an evolutionary loop over a pool of candidate vector
representations. The pool is seeded with `--seeds` LLM-generated
candidates, then on each iteration a parent is sampled from the pool.
With probability `1 - llm_rate` the parent is mutated locally (color
tweaks, path nudges, crossover); otherwise the LLM is called to produce
a refined edit. The new candidate is scored against the source image
(perceptual via vision transformer embeddings, pixel-space, or
LLM-as-judge) and either replaces a worse pool member or is dropped.

Two search strategies decide how the pool is managed and how parents are
picked. The default `nsga` strategy uses NSGA-II with non-dominated
sorting and crowding distance, which keeps diverse Pareto-optimal
candidates around and shines when you have time for multiple epochs. The
`beam` strategy runs beam search over `--beams` parallel hill-climbers,
with `--cull-keep` controlling how aggressively low-ranked beams are
pruned, and converges faster on a single good answer. NSGA-only flags
are `--epoch-diversity`, `--epoch-variance`, and `--epoch-seeds`;
beam-only flags are `--beams` and `--cull-keep`. The CLI rejects mixed
usage.

NSGA-II minimizes two normalized objectives in parallel: visual error
(scorer distance to the source) and content complexity (code size / token
cost). The variant used here is constraint-first (Deb 2000): only
candidates whose visual error is in the top 25% of the active pool are
considered feasible and compete on the Pareto frontier of (error,
complexity); everything else is automatically dominated. In practice that
means visual quality is the primary objective; complexity acts as a
tiebreaker among the quality-leaders, biasing the search toward small,
clean renderings instead of accreting detail forever once the image is
already close.

### Convergence

Each epoch ends as soon as one of these triggers fires; the next epoch
re-seeds from the current Pareto front. The search stops once
`--max-epochs` is reached, `--max-wall-seconds` runs out, or the global
`--max-llm-calls` cap (if set) is hit.

| Flag                 | Default | Triggers when…                                                 |
|----------------------|--------:|----------------------------------------------------------------|
| `--max-epochs`       |       4 | hard cap on epoch count                                        |
| `--epoch-patience`   |      20 | this many LLM calls in a row produce no improvement            |
| `--epoch-steps`      |      50 | this many LLM calls have run in the current epoch              |
| `--epoch-variance`   |       0 | (NSGA-only) score std-dev in the active pool drops below value |
| `--epoch-diversity`  |       0 | (NSGA-only) mean pairwise genome diversity drops below value   |
| `--max-wall-seconds` |    3600 | global wall-clock budget; ends the run, not just the epoch     |
| `--max-llm-calls`    |       0 | global hard cap on total LLM calls; 0 disables                 |

Most tasks are cheap local mutations (controlled by `--llm-rate`, default
10% LLM). They run constantly and only rarely produce a new best score,
so counting every task toward `--epoch-patience` would burn it through in
seconds. Patience and step counters therefore tick only on LLM-driven
exploration tasks, which is what you actually pay for and what drives
meaningful progress. A new best from any source, LLM or local, still
resets `--epoch-patience`. Set `--epoch-variance` and `--epoch-diversity`
to non-zero values to add NSGA-specific stop criteria; their right
thresholds depend on your scorer and image, so they're off by default.

### Bounding the API bill

The defaults give an upper bound on LLM calls per run, computed as:

```
max LLM calls ≈ max_epochs × epoch_steps + epoch-0 seeds + drain overhead
              = 4 × 50 + ~10 + a few ≈ 220
```

That's the worst case; typical runs end earlier on `--epoch-patience`.
If you need a strict ceiling, e.g. for cost-sensitive automation, set
`--max-llm-calls 200` and the engine will halt the run as soon as the
counter hits that value, regardless of which epoch it's in.

Each edit call sends three images (target, current render, diff heatmap)
plus the current code as input (typically a few thousand tokens), and
returns small search/replace diff blocks rather than rewriting the whole
file, so output is usually only a few hundred tokens. A full default run
is on the order of a US dollar on flagship models. Verify against the
[OpenAI](https://openai.com/api/pricing/),
[Anthropic](https://www.anthropic.com/pricing), or
[Google AI](https://ai.google.dev/pricing) pricing pages.

### Output layout

Given `--output sketch.svg`, vectrify writes:

```
sketch.svg                       # the best final candidate
sketch/
└── runs/
    └── 2026-04-26_14-30-21/     # one directory per run, timestamped
        ├── lineage.csv          # accepted node history (score, parent, ops)
        └── nodes/
            ├── 0.0421_0001.svg  # one file per accepted node, prefixed by score
            ├── 0.0421_0001.png  # rendered preview (--save-raster)
            └── ...
```

Disable artifacts you don't need with `--no-write-lineage` or
`--no-save-raster`, or enable `--save-heatmap` to also dump perceptual
diff maps next to each node.
