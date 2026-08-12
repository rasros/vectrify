# Local-search benchmark

A fixed corpus and harness for measuring changes to the **non-LLM** search:
mutation, crossover, operator selection and Pareto survival. No LLM call is
made and no API key is needed.

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

Each case is a `target.png` and five seeds in `seeds/`. The harness plants all
of them as a previous run in a temp dir and invokes `vectrify --seeds 0
--resume`, so the pool starts from those candidates and only local operators
touch it.

Five, not one, because crossover grafts subtrees *between* candidates: a pool
descended from a single ancestor gives it nothing to recombine, which is not
the regime a real epoch runs in. Measured on this corpus crossover is worth
-0.00087 on final error, 95% CI [-0.00149, -0.00019]; measured against a single
seed it looked worthless.

`--workers 1 --random-seed N` makes a run reproducible. Above one worker the
task interleaving varies and it is not, which is why the default is one worker
and several seeds rather than one seed and many workers.

## Metrics

| Metric   | Meaning |
|----------|---------|
| `vision` | the vision model's distance to the target, on the artifact the run wrote |
| `final`  | best pixel score at the end of the budget |
| `auc`    | mean of the running best over the run — rewards getting there sooner |
| `gain`   | fraction of the seed's pixel error removed |

`vision` is the one that decides whether a change is worth keeping. The other
three measure pixel L1, which is what the round optimises: fair for comparing
two searches, but a search can lower it without the result looking any better.

Among the pixel metrics `auc` is the informative one for operator changes: two
searches can end at the same score with very different convergence, and a
change that only helps at the very end of a long budget usually will not
survive a shorter one.

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
in every seed, with the wrong colours, sizes, positions, stroke widths and font
sizes. That is the whole design: local search can move a number and shift a
colour but cannot invent a shape, so a seed missing an element would measure an
unreachable ceiling instead of the operators' actual job.

**The five seeds of a case are different decompositions**, not one drawing
re-jittered — a ring as one stroked circle or as two filled discs, a word as
one `<text>` or one per letter, different grouping and element counts. No
single seed can win either: each is deliberately unable to converge on some
part (a straight-`<line>` mouth that never bows, a `<rect>` with no `rx` whose
corners never round), so the best reachable drawing is a merge taking each part
from whichever lineage drew it right. Seeds live in `bench/seeds.py`.

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

`vision` is scored on whatever the run wrote as its artifact, and the run picks
that by pixel score — so a change can improve the pool without the pick
following.

`--scorer` selects the evaluator that ranks a converged front during a run, and
at `--epochs 1` the run ends before any front is handed over, so it changes
nothing here. The `vision` metric is scored by the harness afterwards either
way.

`--no-adaptive-operators` pins the operator mix to the fixed weight table,
which is how the adaptive policy was measured against it.
