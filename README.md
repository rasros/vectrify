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

The output is human-readable code you can keep editing by hand.

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

pipx or uv tool keeps vectrify in its own environment and on your PATH:

```bash
pipx install vectrify                    # or: uv tool install vectrify
pipx install "vectrify[vision]"          # recommended for best quality
pipx install "vectrify[all]"             # everything
```

Plain pip install works too, into whatever environment is active; when
using --user, check that ~/.local/bin is on your PATH.

The base install covers SVG output and the pixel-difference scorer. The
extras add the rest:

| Extra    | What it adds                                                   |
|----------|----------------------------------------------------------------|
| vision   | torch + transformers for the perceptual (CLIP/SigLIP) scorer   |
| graphviz | the graphviz Python bindings (system Graphviz still required)  |
| typst    | the typst Python compiler                                      |
| all      | vision + graphviz + typst                                      |

Cairo is required for SVG output (apt install libcairo2 or brew install
cairo), and the graphviz format additionally needs the Graphviz binaries
(apt install graphviz or brew install graphviz). A GPU is optional; the
vision scorer falls back to CPU or MPS.

Finally, set an API key for one provider:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

The provider is auto-detected from whichever key is set; override it with
--provider {openai,anthropic,gemini} if you have several keys.

## Quickstart

```bash
vectrify input.png -o output.svg
```

The defaults run up to 4 NSGA-II epochs and stop early once the search
stops finding improvements (see
[Convergence and cost](#convergence-and-cost)). Worst case,
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

Run vectrify --help for the full flag reference, organized into LLM
provider, scoring, search strategy, epoch control, resume, output
artifacts, and runtime sections.

## How it works

vectrify runs an evolutionary loop over a pool of candidate vector
representations. The pool is seeded with a few LLM-generated candidates.
On each iteration a parent is sampled, and:

- with probability 1 − llm-rate, mutated locally (color tweaks, path
  nudges, crossover);
- otherwise, sent to the LLM for a refined edit.

The new candidate is scored against the source image (perceptual via
vision-transformer embeddings, pixel-space, or LLM-as-judge), then
either replaces a worse pool member or is dropped.

### Search strategies

The default NSGA-II uses non-dominated sorting and crowding distance to
keep a diverse Pareto front, worth it when you have time for several
epochs. Beam search runs parallel hill-climbers with pruning instead, and
converges faster on one good answer. NSGA-only flags are --epoch-diversity, --epoch-variance and
--epoch-seeds; beam-only flags are --beams and --cull-keep. The CLI
rejects mixed usage.

### NSGA-II objectives

Three normalized objectives are minimized in parallel:

| Objective                | Measure                                        |
|--------------------------|------------------------------------------------|
| visual error             | scorer distance to the source image            |
| visual complexity        | JPEG-compressed size of the render             |
| structural complexity    | code size (whitespace-stripped source length)  |

Each is scaled by its own maximum across the current pool, so they are
directly comparable and NSGA trades them off by dominance alone, with no
weighting to tune. Structural complexity is format-agnostic, so no output
format is scored as free.

The constraint-first variant (Deb 2000) gates on visual error: only
candidates better than the pool median are feasible, and infeasible ones
are automatically dominated. Visual quality stays the primary objective
while the complexity measures act as tiebreakers among the
quality-leaders, biasing toward small, clean output instead of accreting
detail once the image is already close.

The strongest lever on that bias is --tournament-size. Parents are chosen
by tournament, so raising it converges faster on visual quality but
spends pool diversity, which also brings --epoch-diversity transitions
forward.

Metrics live in one registry, METRICS in score/complexity.py. The
objective vector, node model, lineage.csv columns, and analysis scripts
all derive from it, so adding one is a single entry. Keep the count low:
dominance weakens as objectives multiply, and past four or so almost
nothing is dominated.

### Convergence and cost

Each epoch ends as soon as one of these triggers fires; the next epoch
re-seeds from the current Pareto front. The search stops once it reaches
--max-epochs, runs out of --max-wall-seconds, or hits the global
--max-llm-calls cap, if one is set.

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
Patience and step counters tick only on LLM-driven exploration tasks, the
ones you pay for. A new best from any source, LLM or local, still resets
the patience counter. Set --epoch-variance and --epoch-diversity to
non-zero values to add NSGA-specific stop criteria; their right
thresholds depend on your scorer and image, so they're off by default.

Those defaults also bound the API bill: LLM calls cannot exceed
max_epochs × epoch_steps plus the epoch-0 seeds and a little drain
overhead, so 4 × 50 + ~10 lands near 220, and most runs stop earlier on
--epoch-patience. For a hard ceiling, --max-llm-calls halts the run the
moment the counter reaches it.

Each edit call sends three images (target, current render, diff heatmap)
plus the current code as input (typically a few thousand tokens), and
returns small search/replace diff blocks rather than rewriting the whole
file, so output is usually only a few hundred tokens. A full default run
is on the order of a US dollar on flagship models. Verify against the
[OpenAI](https://openai.com/api/pricing/),
[Anthropic](https://www.anthropic.com/pricing), or
[Google AI](https://ai.google.dev/pricing) pricing pages.

### Output layout

Given --output sketch.svg, vectrify writes:

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

Disable artifacts you don't need with --no-write-lineage or
--no-save-raster. Enable --save-heatmap to also dump perceptual diff maps
next to each node.
