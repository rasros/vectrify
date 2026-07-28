# vectrify

[![PyPI](https://img.shields.io/pypi/v/vectrify.svg)](https://pypi.org/project/vectrify/)
[![Python](https://img.shields.io/pypi/pyversions/vectrify.svg)](https://pypi.org/project/vectrify/)
[![License](https://img.shields.io/pypi/l/vectrify.svg)](https://github.com/rasros/vectrify/blob/main/LICENSE)

LLMs still struggle to generate perfect vector images from a reference
raster in one shot. vectrify turns raster images into editable vector
code by treating vectorization as a search problem: an LLM proposes
candidate SVG/Graphviz/Typst code, a vision scorer ranks how close each
candidate looks to the source, and an optimization loop iteratively
refines the best candidates.

The results are quite good, and the output is human-readable code.

## Features

- Output formats: SVG (default), Graphviz DOT, Typst. HTML and TikZ planned.
- LLM providers: OpenAI, Anthropic, Google Gemini, auto-detected from env vars.
- Search strategies: NSGA-II for diversity-preserving multi-objective
  optimization, or beam search for a cheaper single-best run.
- Scoring: local vision-model embeddings (perceptual), with pixel-diff
  and LLM-as-judge as alternatives.
- Resumable runs: pick up where you left off, or fork from the top-N
  nodes of a previous run.
- Live dashboard: pool stats, scoring, and convergence criteria.

## Install

The recommended way to install a CLI tool is pipx or uv tool, both of
which put vectrify in its own isolated environment and on your PATH:

```bash
pipx install vectrify           # or: uv tool install vectrify
```

Plain pip works too, but it installs into whatever Python environment is
active. With `pip install --user`, make sure `~/.local/bin` is on your
PATH.

The base install includes SVG output and the pixel-difference scorer.
For everything else, pick the extras you need:

| Extra    | What it adds                                                   |
|----------|----------------------------------------------------------------|
| vision   | torch + transformers for the perceptual (CLIP/SigLIP) scorer   |
| graphviz | the graphviz Python bindings (system Graphviz still required)  |
| typst    | the typst Python compiler                                      |
| all      | vision + graphviz + typst                                      |

```bash
pipx install "vectrify[vision]"          # recommended for best quality
pipx install "vectrify[all]"             # everything
```

System dependencies:

- Cairo (required for SVG): `apt install libcairo2` or `brew install cairo`
- Graphviz binaries (for `--format graphviz`): `apt install graphviz` or `brew install graphviz`
- GPU is optional; the vision scorer falls back to CPU/MPS.

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

The defaults run up to 4 NSGA-II epochs and stop early once the search
stops finding improvements (see [Convergence](#convergence)). Worst case,
it runs for an hour and gives up.

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
representations. The pool is seeded with a few LLM-generated candidates.
On each iteration a parent is sampled, and:

- with probability 1 − `--llm-rate`, mutated locally (color tweaks, path
  nudges, crossover);
- otherwise, sent to the LLM for a refined edit.

The new candidate is scored against the source image (perceptual via
vision-transformer embeddings, pixel-space, or LLM-as-judge), then
either replaces a worse pool member or is dropped.

### Search strategies

The default NSGA-II uses non-dominated sorting and crowding distance to
keep a diverse Pareto front, which is best when you have time for
multiple epochs. Beam search instead runs several parallel
hill-climbers with pruning, converging faster on a single
good answer. NSGA-only flags: `--epoch-diversity`, `--epoch-variance`,
`--epoch-seeds`. Beam-only flags: `--beams`, `--cull-keep`. The CLI
rejects mixed usage.

### NSGA-II objectives

Three normalized objectives are minimized in parallel:

| Objective                | Measure                                        |
|--------------------------|------------------------------------------------|
| visual error             | scorer distance to the source image            |
| visual complexity        | JPEG-compressed size of the render             |
| structural complexity    | code size (whitespace-stripped source length)  |

Each is scaled by its own maximum across the current pool, so the three
are directly comparable and no weighting between them is needed — NSGA
trades them off by dominance alone. The selection machinery itself is
arity-agnostic: dominance, crowding distance, and the Pareto helpers all
read the objective count off the data rather than assuming it. Adding a
fourth objective still means threading the new measure through the
worker, node model, and run artifacts, but the algorithm needs no
changes.

Be sparing about it, though. Dominance dilutes as objectives multiply —
on a 12-node pool the first front grows from 8 nodes at three objectives
to 11 at five — so past about four, nearly everything is non-dominated
and the front stops discriminating.

The constraint-first variant (Deb 2000) gates on visual error: a
candidate is feasible only while its error is better than the pool
median, and infeasible candidates are automatically dominated by
feasible ones. So visual quality stays the primary objective and the two
complexity measures act as tiebreakers among the quality-leaders,
biasing toward small, clean renderings instead of accreting detail
forever once the image is already close.

The median split (`FEASIBLE_FRACTION`) is chosen rather than something
stricter because a tighter gate is not automatically a stronger one: if
the feasible group is very small, most binary-tournament comparisons are
between two infeasible candidates, where the gate contributes nothing.
Splitting at the median maximises the share of comparisons the gate
actually decides.

Structural complexity is deliberately format-agnostic, so it means the
same thing for SVG, DOT and Typst and no backend is scored as free. It
counts source characters rather than compressed size: every crossover
operator injects elements from a *related* parent, so near-duplicate
elements accumulate, and a compressed measure discounts exactly that
kind of bloat by around 80%.

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

Most tasks are cheap local mutations, with only a small fraction sent to
the LLM (the llm-rate setting defaults to min(2/workers, 0.2), so roughly
two LLM calls stay in flight regardless of how many workers you run). They
run constantly and only rarely produce a new best score, so counting every
task toward patience would burn it through in seconds.
Patience and step counters therefore tick only on LLM-driven exploration
tasks, which is what you actually pay for and what drives meaningful
progress. A new best from any source, LLM or local, still resets the
patience counter. Set `--epoch-variance` and `--epoch-diversity` to
non-zero values to add NSGA-specific stop criteria; their right
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
sketch.svg                       # the best final candidate (written at the end)
sketch/
└── runs/
    └── 2026-04-26_14-30-21/     # one directory per run, timestamped
        ├── lineage.csv          # accepted node history (all three objectives, parent, ops)
        └── nodes/
            ├── 0.0421_0001.svg  # one file per accepted node, prefixed by score
            ├── 0.0421_0001.png  # rendered preview (--save-raster)
            └── ...
```

Disable artifacts you don't need with `--no-write-lineage` or
`--no-save-raster`, or enable `--save-heatmap` to also dump perceptual
diff maps next to each node.
