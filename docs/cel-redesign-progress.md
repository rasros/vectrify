# CEL redesign implementation evidence

The requirements remain those in [the complete plan](cel-redesign-plan.md).
This record distinguishes working infrastructure from release evidence. The
method remains experimental and the existing CEL default is unchanged.

## Current implementation

- The compressed human fixture loader, native frozen-mask benchmark, paired
  degradation generator and artwork-family manifests are in place. The sword
  is a development fixture, excluded from ranker training and held-out claims.
- A paired tuning runner now records clean/input/mask hashes, local evaluation
  features, line scores, every exactly evaluated candidate, and clean-target
  oracles for the complete pool and the selected representation ceiling. Its
  observer copies diagnostic values and cannot supply scores to generation.
  Diagnostic SVG retention has independent byte and entry bounds; incomplete
  pools are reported and do not produce an oracle claim.
- CPU evidence collection, canonical region/boundary graph, finite merge
  protection and SVG export are implemented as an initial prototype.
- The experimental operation validates settings, previews without mutation,
  applies as one undo entry and supports project save/reload.
- Score version 2 measures robust multi-scale premultiplied color, contrasting
  backdrops/alpha, ink distances and fixed local feature supports. It reports
  each term. Its weights are initial engineering values, not corpus-calibrated.
- Exact checkpoint validation rejects new self-crossings, opaque interior
  gaps, distant silhouette spill and filled protected holes. The first valid
  candidate establishes immutable coverage limits for subsequent proposals.
- A bounded nondominated frontier selects by representation cost and visual
  score. One frozen frontier gives ordered total-cost choices as complexity
  rises. Explicit node budgets select feasible candidates or report infeasibility.
  A lower-cost drawing with more nodes cannot erase a feasible lower-node
  alternative; frontier bounds also protect the minimum-node candidate.
- Stop before the first validated candidate cancels the operation. Stop after
  validation retains a preview that can be applied normally. A fully transparent
  reference returns no edit.
- A conservative CEL checkpoint precedes fitted proposals, using canonical
  linear fill boundaries and separate ink runs. It first tries a 0.75 native
  pixel polygon bound (or a smaller explicit tolerance), then 0.25 and zero
  after rejection. Exact native checks determine which fallback is retained;
  this bound does not establish a quality or runtime pass. Optional fitting
  checks interruption between chains and discards unfinished exports. Exhausting
  the deadline without a checkpoint cannot trigger repeated candidate attempts.
  Dense drawings can still make this checkpoint expensive; runtime, compactness
  and partial-alpha coverage remain open gates.
- The representation normalizer now comes from the exactly validated detailed
  candidate rather than the first pixel fallback, when that candidate finishes.
  Coverage limits stay tied to the independent fallback. The scale freezes once
  before fitting and selection; diagnostics identify fallback normalization
  when the detailed candidate is unavailable.
- Region statistics use bounding boxes instead of one full-canvas scan per
  region. Tests compare the resulting area, paint and texture statistics with
  full-canvas sampling, including an absent label slot.
- Structured competitors now include bounded straight chains and ellipse paths,
  with exact endpoints, persistent-corner checks and one canonical fit for both
  adjacent fills. The legacy fitter is the unrestricted competitor.
- Paired dark-ridge evidence can propose a continuous stroke along a color
  boundary, measuring its local ink and width. Shade discontinuities and long
  blank gaps are excluded; self-crossing ink proposals are discarded.
- A local layer operator can continue adjacent surfaces beneath an isolated
  compact overlay. Small same-hue shade families can compete as one surface
  only with outline evidence. Shapes with holes or silhouette contacts remain
  unrestricted. This is an initial layer operator, not general layer inference.
- Automatic refinement now has a bounded CPU foundation: simplify, fit flat or
  gradient paint, propose edge positions and measured widths, then refit paint.
  Fixed complexity anchors feed the common frontier independently of the
  requested slider. Each retained edit must improve or retain its anchor's full
  native objective and pass coverage/topology checks. Initial frontier tradeoffs
  survive pruning unless a new candidate dominates them at every complexity;
  a replacement rejected for pool storage restores the previous entries.
- CPU fitting holds accepted straight/ellipse paths, shared junctions, corners
  and explicit widths. Whole-path model holds are conservative; parameterized
  constrained joint fitting remains unfinished. A wrapped shared-edge redraw
  preserves its canonical endpoint ID and pin state across the closing segment.
- Search reserves 25% of the requested time for refinement when enabled and at
  least 10% for final validation. Separate stage slices and per-path geometry
  limits give fitting stages opportunities; oversized compounds are reported
  as bounded work. Native rendering and some proposal helpers can still overrun
  a deadline. Metrics distinguish search expiration, fitting status and total
  deadline overshoot. CPU fitting works without Torch.

## Sword experiment: shared frontier

Run on the completed compressed fixture with an explicit 20-second budget:

```sh
PYTHONPATH=src python scripts/bench_cel_planned.py \
  --methods cel-planned --seconds 20 --out .bench/planned-frontier
```

The observed operation took 15.47 seconds. It selected 24 paths, 61 contours,
329 nodes, seven gradients and 39 stroke contours. Its human-reference MSE was
816.46, above both the legacy baseline (approximately 663.31) and the proposed
497.39 gate. Reference MSE was 542.90. There were zero self-crossings in the
selected result and no hard validation rejections. Four nondominated candidates
were retained; the complexity-0 proposal was rejected for a new self-crossing.

The experiment's algorithm source SHA-256 was
`d61e04443ab7fca56a22c44b6f9db48b5c8662fb9b6e6d943881808fbb2740ec`.
The frozen mask SHA-256 was
`f2e692b86e2814f5958c0ca6cc19800a34c12891a527449e68e1624c8bdfe514`.
The benchmark writes SVGs, native previews, feature crops, term diagnostics,
source hash and settings under the chosen output directory. Generated previews
are ignored development artifacts, not bundled release fixtures.

This is a failed sword quality gate, not evidence of improved human likeness.
It demonstrates exact rejection and selection plumbing. The current model
still needs better boundary/primitive proposals, variable ink interpretations,
feature evidence and tuning-corpus calibration.

## Structural-model experiment

The structured competitor is exact-validated alongside the existing traced
plans. In the 20-second sword run under `.bench/planned-layer-proof`, it scored
0.053309 with representation cost 1,003, compared with the traced complexity-50
plan's 0.052620 and cost 893. It was subsequently dominated, so the default
selection remained 329 nodes and human-reference MSE 816.46. The operation
took 14.93 seconds with no reported deadline overshoot in this run.

An isolated structured complexity-50 probe measured human-reference MSE
744.49, compared with approximately 749.68 for the previous traced competitor
at that level. This small development-case change does not pass the sword
gate or establish a broader quality gain. No whole-shape overlay was eligible
on this sword segmentation under the current geometric/protection bounds.

The model tests cover complete noisy round contours, rejection of cornered
shapes and short arcs, fixed endpoints, shared fill coverage, pointed corners,
paired ink versus shade, supported versus blank gaps, opaque overlays across
multiple base shades, hole/silhouette exclusion, stopped layer planning, and
recovering a supported outline even when the initial detector missed it.
The last broad relevant run passed 156 tests. These local checks do not replace
the human-match, line-corpus, resizing or held-out gates.

The latest recorded sword source hash was
`7ad94cf5d976d3fe8f88b0f36a136f9cdc51bc5b6ef2dbce0cc06929008ec115`.
The frozen mask is unchanged. Next required work includes local proposal
selection rather than accepting operators as one bundle, calibration on clean
paired tuning artwork, and automatic fitting under the same validation policy.

A final 20-second run under `.bench/planned-geometry-final` retained only two
valid candidates before the scheduler's reserve stopped further search. It
selected the structured candidate: 449 nodes, 87 contours, human-reference MSE
744.49, 18.10 seconds for the operation and no reported pipeline overshoot.
Its source hash was
`cbe5eec0567af62d61c55f7ceeff7840f5fc58ac5ba7d3b6c0fc605646447d3e`.
The sword gate still failed. Selection differs when the time-limited search
finishes additional anchors; this run does not establish a quality improvement
at matched completed search effort. Scheduling/cache stability and score
calibration therefore remain open requirements.

## Paired tuning and initialization experiment

The tuning runner saves native-sized clean and degraded references, generated
SVGs, fixed masks, feature crops, candidate SVGs and exact score terms. The
clean SVG and manually selected evaluation rectangles are scorer inputs only.
It reuses the existing line benchmark against the clean geometry, including
when the generator receives degraded pixels. Candidate-pool oracles exclude
hard-invalid proposals; one oracle also limits representation cost to the
selected drawing's cost. Raw clean MSE remains a diagnostic, not the planning
objective or a claim of human preference.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python scripts/bench_cel_pairs.py --degradations clean noise --seconds 20 \
  --method-settings '{"cel-planned":{"quality":"balanced","refine":false}}' \
  --out .bench/cel-pairs-bounded
```

The first run found a real deadline defect: all six noisy anime-girl candidates
failed the crossing check, and without a valid checkpoint the pipeline kept
attempting candidates after its deadline. That failed run took 124.62 seconds.
The conservative checkpoint, boundary-stage interruption checks and loop
termination now prevent that retry sequence. An isolated rerun returned a
validated result in 17.12 seconds. Region statistics also avoid full-canvas
scans for every label. These are reliability changes, not a quality pass.

The subsequent complete run covered four tuning families, clean and RGB-noisy
inputs, at a 1,000-pixel long side. Seven of eight planned runs returned a valid
drawing; the noisy anime-face run exhausted its limit before a checkpoint.
Five retained drawings were only the conservative fallback. The requested
budget was 20 seconds; native checkpoint/export work still caused measured
overshoot, as shown below. Legacy CEL does not enforce this operation budget
and took 11.60–41.84 seconds, so this is not a matched-runtime ablation.

| Tuning case | Input | Legacy nodes | Planned nodes | Legacy clean MSE | Planned clean MSE | Line F1 legacy / planned | Planned seconds |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| Anime girl | Clean | 4,136 | 2,354 | 117.20 | 263.31 | 0.966 / 0.948 | 22.17 |
| Anime girl | Noise | 6,735 | 2,913 | 127.65 | 322.10 | 0.950 / 0.917 | 20.30 |
| Anime face | Clean | 6,812 | 106,614 | 149.22 | 303.69 | 0.965 / 0.970 | 30.85 |
| Anime face | Noise | 11,817 | No checkpoint | 168.79 | — | 0.954 / — | 20.01 |
| Western park | Clean | 4,495 | 42,872 | 188.89 | 389.68 | 0.973 / 0.976 | 26.13 |
| Western park | Noise | 6,801 | 49,292 | 213.95 | 580.04 | 0.927 / 0.922 | 23.52 |
| Rubberhose band | Clean | 5,311 | 66,752 | 100.41 | 243.25 | 0.953 / 0.954 | 22.02 |
| Rubberhose band | Noise | 10,651 | 73,171 | 117.12 | 335.97 | 0.925 / 0.886 | 26.40 |

The run source SHA-256 was
`59a3191e718a3af07281ad5b2dbdd14a485800a792998cd161f01ca69e6cc9da`.
All input, clean-render, manifest and mask hashes are recorded in its summary.
No held-out artwork was evaluated. The benchmark exited with failure because
one generation failed; returning every drawing would still not establish the
release quality gates.

Most candidate pools contain only the fallback, so this evidence does not
justify learned ranking. Next work must bound initialization cost, keep time
for compact geometry and paint proposals, and implement automatic refinement.
The raw fallback's cost also now establishes the fixed cost normalizer, which
is much larger than the earlier traced baseline; calibrate that structural
estimate before interpreting slider budgets as complete. Native rendering,
validation and dense boundary/stroke extraction still need tighter scheduling.
Local feature and line regressions remain explicit failures rather than being
averaged into a complexity reduction claim.

The broad relevant test run passed 177 tests, covering copied observer values,
invalid/dominated proposal logging, bounded/incomplete diagnostic pools,
clean-target isolation, planner versus operation pixels, stale-output removal,
fallback apply/undo, expired deadlines and interruption between shared-boundary
fits. These checks do not replace the failed quality and runtime gates.

A subsequent normal-operation sword run with a 20-second limit and refinement
explicitly off selected the first fitted competitor after the fallback. It
returned 1,181 nodes, 106 contours and 19 gradients in 14.25 seconds, with
human-reference MSE 861.49 and zero self-crossings. The frozen mask is unchanged.
Its source hash was
`181790a6acb7fe70846cd5df95da4d4539736b0e3e29cd925375baed5f2770e7`.
The sword gate failed. Only the fallback and complexity-100 traced candidate
were retained before the scheduler's prediction stopped more search. This
confirms that adding a safe checkpoint has not solved initialization cost,
candidate availability or score calibration; it must not be described as a
human-match improvement.

## CPU refinement and compact fallback experiment

Run the native sword benchmark in separate processes with refinement off and
on, and the dense anime-face tuning subset without refinement. All runs used
complexity 50, balanced quality, a 20-second operation limit and four OpenBLAS/OMP
threads. The paired subset used a 1,000-pixel long side. Reproduce with:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python scripts/bench_cel_planned.py --methods cel-planned --seconds 20 \
  --settings '{"complexity":50,"quality":"balanced","refine":false}' \
  --out .bench/planned-cpu-final-off
PYTHONPATH=src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python scripts/bench_cel_planned.py --methods cel-planned --seconds 20 \
  --settings '{"complexity":50,"quality":"balanced","refine":true}' \
  --out .bench/planned-cpu-final-on
PYTHONPATH=src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python scripts/bench_cel_pairs.py --methods cel-planned --cases anime-face \
  --degradations clean noise --long-side 1000 --seconds 20 \
  --method-settings '{"cel-planned":{"complexity":50,"quality":"balanced","refine":false}}' \
  --out .bench/planned-cpu-dense
```

| Run | Nodes | Contours | Seconds | Foreground MSE | Line F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sword, refinement off | 1,181 | 106 | 18.92 | 861.49 against human | — |
| Sword, refinement on | 1,180 | 106 | 17.89 | 861.64 against human | — |
| Anime face, clean | 3,555 | 600 | 19.34 | 352.62 against clean | 0.951 |
| Anime face, noise | 4,802 | 728 | 19.39 | 416.59 against clean | 0.953 |

The algorithm source hash for all four final runs was
`de5471da24b10ee8fdc8a9abd6bd8e482dc4169db0c75a9c7213d98f052ba2c8`.
The sword mask remains
`f2e692b86e2814f5958c0ca6cc19800a34c12891a527449e68e1624c8bdfe514`.
The paired summary records input, clean-render, manifest and native-mask hashes.
There was no measured pipeline deadline overshoot in these four runs. Their
different completed search effort and runtime variation do not establish a
speedup or a full matched-runtime ablation.

Both sword runs retained the conservative fallback and complexity-100 traced
candidate before fitting. The detailed candidate fixed the cost scale at 1,967,
while the conservative candidate cost 11,828. CPU fitting attempted two native
checkpoints, accepted one node-removal edit and rejected a paint refit for an
objective regression. The selected objective changed from 0.088193278 to
0.088183281; human error increased slightly. Fitting was reported as bounded,
not complete. The sword still fails the 800-node and 497.39 human-error gates.
Guard and jewel crops still show fragmented shade boundaries and interrupted
ink. This result supports improving structural proposals and calibrating their
score, not substituting fitting or learned ranking for missing interpretations.

Both dense inputs now returned validated drawings. Previously, the clean case
retained a 106,614-node raw fallback and the noisy case had no checkpoint. Each
new pool contains the compact conservative candidate and a traced competitor;
the traced competitor was selected. This is initialization progress. Clean
MSE remains worse than the recorded legacy values, and clean line F1 is more
than one percentage point below legacy's 0.965. Neither result establishes the
broader quality gates; no held-out artwork was used.

The relevant check run passed 199 tests. It covers CPU execution with Torch
unavailable, real paint improvement, gradient competition and ownership during
refits, injected flat-color noise, explicit-width preservation, shared
edges/corners, stop after fitting,
fixed cost scale, objective rejection, storage rollback and all 101 slider
choices under pruning. A subsequent node-budget regression check also keeps a
feasible drawing when a cheaper competitor has more nodes. Existing CEL,
simplify/snap, operation apply/undo/reload
and benchmark-isolation checks are included. Ruff passed and Pyrefly reported
zero errors. Full runtime/memory, partial-alpha and release evaluation remain
open requirements.

## Remaining requirements

None of the eight complete deliveries is claimed finished yet. In particular:

| Delivery | Remaining evidence or behavior |
| --- | --- |
| 1 | Full synthetic/curated-human coverage, frozen broader-suite tolerances and calibrated score terms |
| 2 | Dense-input fallback/runtime bounds; partial-alpha, transformed-scope and difficult-hole coverage |
| 3 | Local exact acceptance of individual graph edits, beam search, split/paint operators, content-normalized soft budgets, bounded shared-frontier cache and resizing invariance |
| 4 | Variable-width/fill ink alternatives, full join/feature checks, parameterized primitive fitting beyond whole-path holds and passing sword/line/feature gates |
| 5 | Broader local-layer/order inference, joint geometry/width fitting, geometric regularization, complete spatial scheduling, memory/runtime gates and optional acceleration ownership |
| 6 | UI/MCP controls, browser/API round trips, invalidation and documentation |
| 7 | Expanded paired evaluation/corpus coverage, ablations and conditional learned-ranker experiment |
| 8 | Independent blind review, fresh held-out results and default migration only after release gates pass |

The operation still reports `refinement_complete: false`. The bounded CPU
foundation implements real fitting behavior; it does not complete joint fitting
or the required runtime/memory and quality gates. The frontier still receives
complete merge/tolerance proposals rather than the planned local beam search.
These gaps must be resolved before completion is claimed.
