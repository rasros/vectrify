# Local-search benchmark

A fixed corpus and harness for measuring changes to the **non-LLM** search:
mutation, crossover, micro-search and Pareto selection. No LLM call is made and
no API key is needed.

## Running it

```bash
uv run python scripts/bench_search.py run --out before.json
# change the search
uv run python scripts/bench_search.py run --out after.json
uv run python scripts/bench_search.py compare before.json after.json
```

`compare` pairs runs by (case, seed) and reports the mean paired delta with a
bootstrap 95% CI. Lower is better; a CI entirely below zero is an improvement.

## How a case runs

Each case is a `target.png` and a `seed.svg`. The harness plants the seed as a
previous run in a temp dir and invokes `vectrify --seeds 0 --resume`, so the
pool starts from that one candidate and only local operators touch it.

`--workers 1 --random-seed N` makes a run reproducible. Above one worker the
task interleaving varies and it is not, which is why the default is one worker
and several seeds rather than one seed and many workers.

## Metrics

| Metric | Meaning |
|--------|---------|
| `final` | best score at the end of the budget |
| `auc`   | mean of the running best over the run — rewards getting there sooner |
| `gain`  | fraction of the seed's error removed |

`auc` is the one to watch for operator changes: two searches can end at the
same score with very different convergence, and a change that only helps at the
very end of a long budget usually will not survive a shorter one.

## The corpus

`bench/generate.py` regenerates it. Targets are drawn as raster at 4x and
downsampled, so they carry anti-aliased edges, gradients and a translucent haze
that no shipped SVG encodes. There is deliberately no ground-truth SVG.

| Case | Target | What it stresses |
|------|--------|------------------|
| `leaf`         | bezier silhouette | path geometry only — every shape is a `d` attribute |
| `mascot`       | cartoon character | flat fills, thick outlines, small facial features |
| `wordmark`     | company logo      | glyph sizing, overlapping rounded geometry |
| `emblem`       | circular badge    | concentric rings, star geometry, small caps |
| `sunset`       | gradient scene    | linear + radial gradients, opacity, large regions |
| `connect-dots` | numbered puzzle   | 22 dots and 22 numerals at glyph scale |

**Seeds are structurally complete.** Every element the target needs is already
in the seed, with the wrong colours, sizes, positions, stroke widths and font
sizes. That is the whole design: local search can move a number and shift a
colour but cannot invent a shape, so a seed missing an element would measure an
unreachable ceiling instead of the operators' actual job.

Every perturbed attribute is one an operator can reach, which constrains how
the seeds may be written:

- geometry goes in `<path d="...">`, never `<polygon points="...">` — `d` is
  mutable by `mutate_path`, `points` is mutable by nothing;
- gradients use `stop-color`, which `mutate_color` treats as a colour attribute;
- `font-size`, `opacity`, `stroke-width` and the usual coordinates are all in
  `mutate_numeric`'s attribute set.

`font-family` is *not* mutable, so seeds keep the target's family and get the
size wrong instead. If you add a case, check its perturbations against
`_NUMERIC_ATTRS` and `_COLOR_ATTRS` in `formats/svg/operations.py` — an
unreachable gap silently caps the whole case.

## Caveats

`--scorer simple` is the default because it is fast and CPU-only. It is a
pixel-space proxy for the vision scorer that real runs use, so confirm anything
you intend to keep with `--scorer vision` before believing it.
