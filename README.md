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

### Scoring resolution

Vision models take a fixed input size, so scoring a whole image downscales it
and fine detail drops below the model's patch size. The scorer instead cuts the
raster into crops of exactly that input size and scores each unresampled.

resolution sets the raster and everything follows from it: it is rounded up to a
whole number of crops so they tile exactly, and it fixes the coordinate space
candidates are written in (SVG viewBox, Typst page). It only downscales, so a
700px source stays 700px however high you set it.

| resolution    | raster | crops per candidate |
|--------------:|-------:|--------------------:|
| 384           | 384    | 1                   |
| 768 (default) | 768    | 4                   |
| 1000          | 1152   | 9                   |
| 1500          | 1536   | 16                  |

resolution-llm sizes the images sent to the LLM and has no effect on scoring.
Vision pricing tiles at 512px, the default, so raising it triples the cost of
every prompt image.

### Search strategies

NSGA-II, the default, keeps a diverse Pareto front and does better when
you can afford several epochs. Beam search converges faster on a single
answer. NSGA-only flags are epoch-diversity, epoch-variance and
epoch-seeds; beam-only flags are beams and cull-keep. The CLI rejects
mixed usage.

### NSGA-II objectives

The search minimizes four objectives at once:

| Objective             | Measure                                        |
|-----------------------|------------------------------------------------|
| visual error          | scorer distance to the source image            |
| visual complexity     | JPEG-compressed size of the render             |
| structural complexity | code size (whitespace-stripped source length)  |
| worst region          | distance over the worst crops of the render    |

Visual error is the primary objective; the complexity measures only break
ties among the best-scoring candidates, biasing toward small, clean output
once the image is already close. Worst region counters visual error being an
average, under which a small defect in a mostly-correct image is too cheap to
be worth fixing; it reads the same crops the score is built from, so the region
it names is always one the score measured. Raising tournament-size pushes
harder toward visual quality at the cost of pool diversity.

### Convergence and cost

Each epoch ends when one of these fires; the next re-seeds from the
current Pareto front. The run stops at max-epochs, max-wall-seconds, or
the max-llm-calls cap.

| Flag             | Default | Triggers when…                                          |
|------------------|--------:|---------------------------------------------------------|
| max-epochs       |       2 | hard cap on epoch count                                 |
| epoch-patience   |     200 | this many tasks in a row produce no improvement         |
| epoch-steps      |      50 | this many LLM calls have run in the current epoch       |
| epoch-variance   |       0 | (NSGA-only) score std-dev in the pool drops below value |
| epoch-diversity  |       0 | (NSGA-only) mean pairwise diversity drops below value   |
| max-wall-seconds |    3600 | wall-clock budget; ends the run, not just the epoch     |
| max-llm-calls    |       0 | hard cap on total LLM calls; 0 disables                 |

Patience counts every task, so epoch length does not move with llm-rate;
epoch-steps still counts LLM calls, since it caps the expensive resource. A
new best resets patience. The two NSGA stop criteria are off by default;
good thresholds depend on your scorer and image.

The defaults cap LLM calls near max_epochs × epoch_steps, so around 110, and
most runs stop well before that. Later epochs return little — on the
reference image epoch 0 produced 82% of the total gain and epochs 2-3 only
3.7% — so max-epochs defaults to 2. Set max-llm-calls for a hard ceiling.

### Output layout

Given --output sketch.svg, vectrify writes:

```
sketch.svg                       # the best final candidate (written at the end)
sketch/
└── runs/
    └── 2026-04-26_14-30-21/     # one directory per run, timestamped
        ├── lineage.csv          # accepted node history (all four objectives, parent, ops)
        └── nodes/
            ├── 0.0421_0001.svg  # one file per accepted node, prefixed by score
            ├── 0.0421_0001.png  # rendered preview (--save-raster)
            └── ...
```

Disable artifacts you don't need with no-write-lineage or no-save-raster.
Enable save-heatmap to also dump perceptual diff maps next to each node.
