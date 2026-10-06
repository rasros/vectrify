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
- Stop before the first validated candidate cancels the operation. Stop after
  validation retains a preview that can be applied normally. A fully transparent
  reference returns no edit.
- A conservative CEL checkpoint precedes fitted proposals, using unsmoothed
  canonical fill boundaries and separate ink runs. Optional boundary fitting
  checks interruption between chains and discards unfinished exports. Exhausting
  the deadline without a checkpoint cannot trigger repeated candidate attempts.
  Dense drawings can still make this checkpoint expensive; runtime, compactness
  and partial-alpha coverage remain open gates.
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

## Remaining requirements

None of the eight complete deliveries is claimed finished yet. In particular:

| Delivery | Remaining evidence or behavior |
| --- | --- |
| 1 | Full synthetic/curated-human coverage, frozen broader-suite tolerances and calibrated score terms |
| 2 | Dense-input fallback/runtime bounds; partial-alpha, transformed-scope and difficult-hole coverage |
| 3 | Local exact acceptance of individual graph edits, beam search, split/paint operators, content-normalized soft budgets, bounded shared-frontier cache and resizing invariance |
| 4 | Variable-width/fill ink alternatives, full join/feature checks, primitive constraints during fitting and passing sword/line/feature gates |
| 5 | Broader local-layer/order inference, automatic refinement, geometric regularization, paint refits, spatial scheduling, memory/runtime measurements and optional acceleration ownership |
| 6 | UI/MCP controls, browser/API round trips, invalidation and documentation |
| 7 | Expanded paired evaluation/corpus coverage, ablations and conditional learned-ranker experiment |
| 8 | Independent blind review, fresh held-out results and default migration only after release gates pass |

The operation still reports `refinement_complete: false`; accepting `refine`
in its schema does not constitute implementing automatic refinement. The
frontier currently receives complete merge/tolerance proposals, not the planned
local beam search. These gaps must be resolved before completion is claimed.
