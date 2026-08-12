# vectrify

[![PyPI](https://img.shields.io/pypi/v/vectrify.svg)](https://pypi.org/project/vectrify/)
[![Python](https://img.shields.io/pypi/pyversions/vectrify.svg)](https://pypi.org/project/vectrify/)
[![License](https://img.shields.io/pypi/l/vectrify.svg)](https://github.com/rasros/vectrify/blob/main/LICENSE)

LLMs still struggle to generate perfect vector images from a reference
raster in one shot. vectrify turns raster images into editable vector
code by treating vectorization as a search problem: an LLM proposes
candidate SVG/Graphviz/Typst code, a scorer ranks how close each candidate
looks to the source, and an optimization loop iteratively refines the best
candidates.

The output is human-readable code you can keep editing by hand.

## Features

- Output formats: SVG (default), Graphviz DOT, Typst. HTML and TikZ planned.
- LLM providers: OpenAI, Anthropic, Google Gemini, auto-detected from env vars.
- Search: NSGA-II for diversity-preserving multi-objective optimization,
  with LLM proposals and local refinement split into separate phases.
- Scoring: pixel distance in the search loop, local vision-model
  embeddings (perceptual) at each epoch boundary.
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

The defaults run up to 2 NSGA-II epochs and stop early once the search
stops finding improvements (see
[Convergence and cost](#convergence-and-cost)). Worst case,
it runs for an hour and gives up.

A few useful variations:

```bash
# Bigger LLM batch per epoch, longer runs
vectrify photo.jpg -o sketch.svg --seeds 20 --max-wall-seconds 1800

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
representations, split into epochs of two phases:

- **Seed.** The LLM produces seeds candidates — generated from scratch in
  epoch 0, edited from the previous epoch's Pareto front afterwards.
  Their children become the epoch's pool; from epoch 1 on, whatever was
  in the pool before is discarded.
- **Refine.** Only local operators run — color tweaks, path nudges,
  crossover — until the epoch stops improving.

Every candidate is scored in pixel space during the round and joins the
next generation only if the pool does not dominate it. The vision model is
reserved for the converged front at each epoch boundary, where it decides
which candidates the next batch of LLM edits starts from — it costs roughly
300x a pixel score, which is worth paying a few times per epoch and not once
per candidate. The best candidate of the whole run is tracked separately, so
an epoch restart never loses it.

The phases are separate because the operators are not interchangeable. An
LLM edit degrades the median parent about four times as much as a local
mutation and costs roughly a thousand times more per attempt, so mixing
them into every task spent the expensive operator competing against local
moves. As restart points they instead do the one thing local search cannot:
leave the basin it is stuck in. Total LLM calls are therefore bounded by
epochs × seeds.

### Resolution

resolution sets the long side every candidate is rendered at, and with it the
coordinate space candidates are written in (SVG viewBox, Typst page). It only
downscales, so a 700px source stays 700px however high you set it.

Both scorers work from that raster at their own fixed size — the pixel score
downscales it, the vision model resizes to its input edge — so raising
resolution buys finer geometry in the output rather than a finer-grained
score, and costs proportionally more to rasterize.

resolution-llm sizes the images sent to the LLM and has no effect on scoring.
Vision pricing tiles at 512px, the default, so raising it triples the cost of
every prompt image.

### NSGA-II objectives

The search minimizes five objectives at once:

| Objective     | Measure                                                    |
|---------------|------------------------------------------------------------|
| score         | pixel distance to the source image                         |
| worst quarter | distance over the worst of 4 regions                       |
| worst 16th    | distance over the worst of 16 regions                      |
| zip ratio     | compressed size of the render, per unit of error removed   |
| node ratio    | element count, per unit of error removed                   |

No objective is privileged: dominance compares the whole vector, which is what
lets a complexity measure actually shape the front rather than break ties among
candidates already sorted by score. The two region scales counter score being
an average, under which a small defect in a mostly-correct image is too cheap
to be worth fixing — quarters catch a whole area being wrong, sixteenths catch
a localised defect.

The complexity measures are ratios rather than raw counts because an empty
canvas beats everything on a raw count and so is never dominated, which would
park it on the front for the whole run. Charging complexity against the error
it removes puts its ratio at the ceiling instead. A candidate that removes less
than half the available error is pinned there too, which rules out the other
degenerate winner: a flat rectangle of the average colour.

Parent selection is a tournament on non-dominated rank then crowding distance;
survival is the same comparison applied to parents and children together, once
per generation. Raising tournament-size pushes harder toward the front at the
cost of pool diversity.

### Mutation operators

Local refinement draws from a handful of operators — nudge a number, tweak a
color, change a stroke, reorder siblings, drop an element, graft a subtree.
Which of them pays off is not fixed: it depends on the image, and it changes
within a single run as structural edits give way to nudges.

So the search learns the mix as it goes. Each mutation records whether its
child survived the generation, and EXP3.S over those outcomes decides what to
try next. EXP3 rather than a stochastic bandit because survival is not an
i.i.d. draw per operator: a child competes against the pool the policy itself
just filled, so the payoff for nudging a number depends on what the other
operators have been producing, and it drifts as the drawing gets closer.

Rewards are weighted by the probability the operator was drawn with, so
evidence about a rarely-picked one is not diluted by how often the others were
picked. A share of the weight is redistributed uniformly each update, which
both keeps the policy tracking the current phase and holds a floor under every
operator so none is starved before the phase it is good for arrives.

Pass --no-adaptive-operators to draw from one fixed weight table instead.

### Convergence and cost

An epoch's refine phase ends when one of these fires; the next epoch then
re-seeds from the current Pareto front. The run stops at epochs or
max-wall-seconds.

| Flag             | Default | Triggers when…                                        |
|------------------|--------:|-------------------------------------------------------|
| epochs           |       2 | hard cap on epoch count                               |
| epoch-patience   |     200 | this many local tasks in a row produce no improvement |
| epoch-variance   |       0 | score std-dev in the pool drops below value           |
| epoch-diversity  |       0 | mean pairwise diversity drops below value             |
| max-wall-seconds |    3600 | wall-clock budget; ends the run, not just the epoch   |

Patience counts local tasks only — a seed batch is not a hill-climb and
cannot go stale — and a new best resets it. The variance and diversity
criteria are off by default; good thresholds depend on your scorer and image.

LLM spend is exactly epochs × seeds — 20 at the defaults — so the two flags
that set it are the whole cost model. More epochs give diminishing returns, so
epochs defaults to 2.

### Output layout

Given --output sketch.svg, vectrify writes:

```
sketch.svg                       # the best final candidate (written at the end)
sketch/
└── runs/
    └── 2026-04-26_14-30-21/     # one directory per run, timestamped
        ├── lineage.csv          # accepted node history (every objective, parent, ops)
        └── nodes/
            ├── 0.0421_0001.svg  # one file per accepted node, prefixed by score
            ├── 0.0421_0001.png  # rendered preview (--save-raster)
            └── ...
```

Disable artifacts you don't need with no-write-lineage or no-save-raster.
Enable save-heatmap to also dump perceptual diff maps next to each node.
