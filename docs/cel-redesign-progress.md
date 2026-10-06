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
- Individual flat-paint, shared-boundary fitting and supported ink proposals now
  run through a bounded local beam. Working states share immutable native raster
  history. Local acceptance uses the complete policy's fixed denominators and
  feature aggregates; publication independently verifies the entire raster,
  score and document checks. Connected surface families now also have individual
  merge proposals with retained source ownership. Owned ink replacement now
  compares filled unions with strokes over restored neighboring paint. Region
  splits, broader ink/layer interpretations and order edits remain unfinished.
- The experimental operation validates settings, previews without mutation,
  applies as one undo entry and supports project save/reload.
- Score version 3 measures robust multi-scale premultiplied color, contrasting
  backdrops/alpha, ink distances and fixed local feature supports. It reports
  each term. Its weights are initial engineering values, not corpus-calibrated.
- Translucent evidence keeps original RGBA and uses connected color/opacity
  surfaces, with flat alpha or supported gradient stop opacity. Relative alpha
  bands and smooth surface colors limit byte-level fragmentation; native checks
  validate the resulting approximation. Low-opacity marks and holes have
  separate hard safeguards. The frozen human benchmark mask is unchanged.
- Exact checkpoint validation rejects new self-crossings, opaque interior
  gaps, distant silhouette spill and filled protected holes. The first valid
  candidate establishes immutable coverage limits for subsequent proposals.
- A bounded nondominated frontier selects by representation cost and visual
  score. One frozen frontier gives ordered total-cost choices as complexity
  rises. Explicit node budgets select feasible candidates or report infeasibility.
  A lower-cost drawing with more nodes cannot erase a feasible lower-node
  alternative; frontier bounds also protect the minimum-node candidate.
  It now reports the initial soft representation target, its simplest retained
  validated floor, achieved cost and whether the target is unmet, separately
  from a hard node ceiling. This reporting does not yet drive local beam search.
- Stop before the first validated candidate cancels the operation. Stop after
  validation retains a preview that can be applied normally. A fully transparent
  reference returns no edit.
- A conservative CEL checkpoint precedes fitted proposals, using canonical
  linear fill boundaries and separate ink runs for opaque evidence. It first tries a 0.75 native
  pixel polygon bound (or a smaller explicit tolerance), then 0.25 and zero
  after rejection. Exact native checks determine which fallback is retained;
  this bound does not establish a quality or runtime pass. Optional fitting
  checks interruption between chains and discards unfinished exports. Exhausting
  the deadline without a checkpoint cannot trigger repeated candidate attempts.
  RGBA evidence starts with unsimplified canonical boundaries; its paint still
  approximates each relative-alpha partition and must pass native checks.
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
- Search reserves up to 25% of the requested time for refinement when enabled and at
  least 10% for final validation. Separate stage slices and per-path geometry
  limits give fitting stages opportunities; oversized compounds are reported
  as bounded work. A fallback above the CPU seed limit gives that fitting time
  to structural search until a validated compact seed exists. Refinement orders
  bounded eligible paths across spatial cells, starting with larger shapes;
  this does not yet guarantee a fitting opportunity for every component.
  Native rendering and some proposal helpers can still overrun
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

## Translucent evidence and native coverage experiment

The opacity-aware path now retains content below 50% alpha, including marks at
one byte of opacity. Original RGBA reference crops are passed to the planned
method; reconstructing color from an eight-bit white preview loses information
at low alpha. The existing preview and legacy CEL inputs retain their contracts.
Four-connected horizontal-run graphs label the joint color/alpha partitions
without a full-image component pass for each possible paint value. Ink receives
its own palette so a one-color body palette cannot erase it. Region statistics
use actual RGBA surface colors rather than a smoothed background under ink.

Initial opacity bands have a ratio of 1.125, giving 49 possible levels, rather
than one region class per alpha byte. Paint fitting uses original samples and
compares flat RGBA against a shared-axis color/opacity gradient. Native
validation checks premultiplied color on black and white, opacity excess and
loss, translucent holes and connected-component retention. Thin components
without an eroded interior need 95% retained opacity; broad components need
75%, alongside interior pixel checks. These are engineering safeguards, not
calibrated release tolerances. Alpha-only transitions do not count as ink.
New opacity diagnostics include the full visible support without changing the
frozen human-scoring mask.

Long narrow translucent crops retain native samples when they fit the same
1536-squared analysis pixel allowance. Downsampling larger content remains an
open native-feature requirement. Positive widths generate fractional coverage
for filled ink and protect its geometry and paint from merges/refinement.
Floating ink with no underlying filled surface still needs a complete width
override model. RGB-only CPU paint proposals also still need joint RGBA fitting.

The transformed-group round-trip test found that insertion applied the selected
group's transform twice. Shared generation now compensates for the container's
coordinate frame. Translated/scaled group placement, project save/reload and one
undo entry pass. An original RGBA crop is retained beside the white preview.

The sword reference itself is mostly slightly translucent: 127,214 pixels have
alpha 253, while only 627 have alpha 255. Treating every byte as a separate paint
class created a very large fallback. Relative bands and smoother color evidence
reduce that fragmentation, but do not solve structural compaction. Fitted curves
can still lose low-opacity marks or holes, so rejected candidates retain the
conservative checkpoint.

Reproduce the final native development run with:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python scripts/bench_cel_planned.py --methods cel-planned --seconds 20 \
  --settings '{"complexity":50,"quality":"balanced","refine":true}' \
  --out .bench/planned-opacity-final
```

The final source hash was
`2b279d0a8428aa8573ccf192088451ae39203941eb55e800bdc4c10be183e87d`.
The frozen sword mask remains
`f2e692b86e2814f5958c0ca6cc19800a34c12891a527449e68e1624c8bdfe514`.
The run retained 8,147 paths, 8,217 contours, 54,320 nodes and no gradients, with
human MSE 446.79 and zero self-crossings. Pipeline time was 14.09 seconds;
operation/apply/benchmark time was 16.59 seconds. It reported no pipeline
overshoot. Native opacity MSE over visible support and its margin was
0.00002838. Only the conservative candidate entered this pool. CPU refinement
now bounds seeds above 32,000 nodes before expensive parsing/path visits; this
run skipped one dense seed in 0.024 seconds and attempted no edits.

The sword gate fails by a wide margin on nodes and contours. Human error meets
its numerical ceiling because this is a detailed trace. This is not a cleaner
editable drawing or a structural quality improvement. Complexity cannot offer
meaningful alternatives on this one-candidate pool. Runtime variation and
different completed proposal sets prevent a matched-effort ablation claim.
The next required work is opacity-aware compact structural proposals with
native topology preservation, followed by local acceptance and score calibration.
No held-out artwork was used and the experimental method remains gated.

The relevant suite passed 231 tests, including native flat/ramp opacity, a
one-byte thin mark, holes, palette-one ink retention, filled-width constraints,
relative bands, cancellation during boundary extraction, transformed placement
and project/undo round trips. Existing CEL, shared boundaries, simplify/snap and
benchmark isolation checks are included. Ruff passed and Pyrefly reported zero
errors. These checks establish the tested fidelity behavior, not the release
quality, calibration, resizing or memory gates.

## Compact RGBA coverage and complexity-budget experiment

The compact opacity path now distinguishes a thin alpha component from a
narrow color partition inside a broad component. Native exterior boundaries,
holes, isolated thin components and large relative-opacity discontinuities
retain their canonical polygon interpretation. Ordinary internal color edges
can compete as fitted boundaries. A self-crossing fitted region restores all
of its canonical edges, including the neighbor's copy, before full validation.
Merge proposals carry observed opacity ranges; a flat interpretation cannot
merge a large relative-alpha discontinuity merely because its absolute
difference is small.

For connected near-uniform translucent components, a native core base competes
inside an ordinary SVG opacity group. Child fills and gradient stops normalize
their opacity by the parent value. The core excludes intentional holes and
weak fringes; thin components stay independent. Modal source alpha avoids
fragmenting a nearly opaque material into many holes from byte variation.
The current 5% core tolerance and 2% maximum-alpha allowance are engineering
proposal parameters, subject to full native scoring and hard rejection.
Broad variable-alpha components retain the adjacent RGBA interpretation.
The first detailed normalization seed now includes this supported coverage
interpretation for translucent evidence, while the conservative fallback
remains independent. No human geometry or annotated benchmark crop guides it.

Paint proposals now compare estimated gradient improvement with its extra
representation price, using the fixed cost normalizer and visible support.
Padding does not inflate that support. This inexpensive test orders/chooses
paint models inside a proposal; complete native validation still determines
candidate retention. Opaque paint pricing remains unfinished. The subsequent
local-search stage adds individual acceptance for paint/boundary/ink operators.

An exploratory 60-second run in `.bench/planned-alpha-bases-final`, before the
detailed seed included the base interpretation, selected 22,883 nodes, 4,078
contours and 25 gradients, with human-reference MSE 500.33. Pipeline time was
50.86 seconds and operation/apply time was 59.70 seconds. Its source hash was
`51293269dbe829d5eb8af6891ddbd40328580e05a8fbe222544edfc7d8b59d5c`.
The adjacent detailed candidate failed translucent-interior coverage; the
structured base candidate passed. This is a failed sword milestone, although
it establishes a usable coverage interpretation. Guard and jewel crops still
show missing ink and excessive subdivision.

Final-source short runs used complexity 50, balanced quality, a 20-second
operation limit and four OpenBLAS/OMP threads, in separate processes:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python scripts/bench_cel_planned.py --methods cel-planned --seconds 20 \
  --settings '{"complexity":50,"quality":"balanced","refine":false}' \
  --out .bench/planned-alpha-core-off
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python scripts/bench_cel_planned.py --methods cel-planned --seconds 20 \
  --settings '{"complexity":50,"quality":"balanced","refine":true}' \
  --out .bench/planned-alpha-core-on
```

| Run | Nodes | Contours | Gradients | Human MSE | Pipeline seconds | Operation/apply seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Refinement off | 54,320 | 8,217 | 0 | 446.79 | 19.33 | 22.41 |
| Refinement on | 23,212 | 4,147 | 544 | 439.75 | 19.89 | 23.90 |

Both runs used source hash
`2e41ce79865c4b874fee1a72b7c895a068ffa4969b28afa343635c96ebcc5081`.
The frozen mask remains
`f2e692b86e2814f5958c0ca6cc19800a34c12891a527449e68e1624c8bdfe514`.
Neither reported pipeline overshoot, but operation/apply exceeded the requested
20 seconds by 2.41 and 3.90 seconds. That outer cost remains a runtime gap.
No held-out artwork was used.

The first run finished only the conservative fallback. The second finished a
validated detailed base candidate, fixing its cost scale at 54,564, and reclaimed
fitting time after compaction. CPU fitting visited three paths, attempted zero
edits and reported interruption after 2.18 seconds. The different finished
search prefixes and runtime variation explain the different outputs; these
runs do not establish a refinement benefit or a matched-effort speedup.
Both still fail the node and contour gates. The initial soft-budget floor is
the cheapest available validated candidate, so a reported feasible target
cannot establish that useful simple alternatives exist.

A final-source 60-second run completed the six fitted competitors:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python scripts/bench_cel_planned.py --methods cel-planned --seconds 60 \
  --settings '{"complexity":50,"quality":"balanced","refine":false}' \
  --out .bench/planned-alpha-core-coverage
```

It selected the same 23,212-node detailed base drawing as the short refinement-on
run, with 4,147 contours, 544 gradients and human MSE 439.75. Pipeline time was
42.81 seconds; operation/apply time was 47.75 seconds, with no pipeline
overshoot. The source and mask hashes are the same as the final short runs.
The pool contained five retained candidates, including the fallback; two
adjacent traced competitors failed translucent-interior coverage. The simpler
structured candidate cost 47,517 and scored 0.043413, versus cost 54,564 and
score 0.037217 for the selected detailed candidate. Its initial soft target and
validated floor were 46,365; achieved cost was 54,564 and the budget was reported
unmet. Meeting the human-error ceiling still leaves a very large structural
failure. Guard and jewel feature MSEs were 1,015.14 and 1,233.07, respectively;
their crops show irregular shade fragments and insufficiently coherent ink.
This is development evidence for better surface/ink operators and selection
calibration, not a release or held-out quality pass.

The final relevant suite passed 254 tests:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m pytest -q tests/refine/test_cel*.py tests/refine/test_shared.py \
  tests/refine/test_simplify.py tests/refine/test_snap.py \
  tests/operations/test_cel_planned.py tests/operations/test_generate.py \
  tests/test_bench_cel_planned.py tests/test_bench_cel_pairs.py
```

New coverage includes diagonal subdivision seams, normalized gradient opacity,
holes, weak fringes, disconnected materials, thin byte-opacity marks, variable
alpha, modal cores, native project round trips and interrupted core extraction.
It also checks all 101 soft-budget values, hard-node-ceiling separation, padding,
spatial path limits and fitting-time reclamation after validated compaction.
Existing CEL and operation/benchmark isolation checks are included. Ruff passed;
Pyrefly reported zero source errors. These checks do not complete the remaining
runtime, memory, local-search, calibration or release gates.

## Native local scoring and bounded operator search

`local.py` now maintains an immutable native RGBA raster with shared root data
and bounded patches. Changed color, blur and ink-distance contributions retain
the complete policy's denominators. The 16-pixel output halo and additional
input halo cover the current finite filters. Feature errors, their maximum,
hole opacity and component retention update alongside the global terms.
Changing the first or last observed ink pixel preserves the full policy's
empty-edge semantics. A declared edit that changes pixels outside its bounds
is rejected locally; publication compares the entire accumulated raster byte
for byte against an independently rendered candidate and compares score terms
within `2e-7`. Document validity and topology checks still run at publication.

Renderer experiments found that a cropped Cairo viewport could change gradient
and opacity-group rounding by one or two bytes. Those cases retain the original
native viewport and crop before conversion to floating-point pixels. Named
paths outside conservative support bounds can be omitted, while containers,
definitions, unknown geometry and all overlapping hidden layers remain. Uses,
imported primitives and long miters have conservative full-canvas dependencies.
This reduces irrelevant drawing work without claiming a crop-only renderer.
Native rendering still allocates a full native buffer within the stage limit.

`search.py` implements working beams of one, four and eight states, with exact
evaluation caps of 16, 48 and 128. The available operators run round-robin:
gradient-to-flat paint, shared-boundary simplification and locally supported
continuous ink. They use the same proposal settings and complexity anchors for
every requested slider value. Advanced feature protection also scales the
fixed policy's feature weight. Individual improvements enter working states;
only independent full checkpoints enter the published frontier. Cancellation
or optional-stage failure preserves the previous validated drawing.

Rejection proofs include operator parameters, geometry/paint and visibility
dependencies, the native context and relevant global score aggregates. The
per-run LRU retains at most 256 proofs and eight document dependency indexes.
Hidden overlapping paint remains a dependency even when it contributes no
current visible pixel. Independent remote color edits can reuse a rejection.
Scan limits also bound stale, cached and otherwise unevaluated proposals.
Current seed limits are 32,000 nodes and 8,192 painted objects; changed IDs and
explicit dependencies are capped at 256. The local raster allowance is 16 MiB,
and score input crops are capped at 262,144 pixels. The 64 MiB state accounting
covers retained SVG bytes and unique shared raster/patch buffers, including
parents and additions during expansion. It does **not** account for every
document object, metadata allocation or process RSS; the complete memory gate
remains open.

Reproduce the final native development run with:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python scripts/bench_cel_planned.py --methods cel-planned --seconds 60 \
  --settings '{"complexity":50,"quality":"balanced","refine":false}' \
  --out .bench/planned-local-context
```

| Measurement | Result | Sword target |
| --- | ---: | ---: |
| Nodes | 23,194 | At most 800 |
| Contours | 4,147 | At most 140 |
| Gradients | 544 | No fixed ceiling |
| Human foreground MSE | 438.92 | At most 497.39 |
| Pipeline seconds | 53.86 | Explicit 60-second limit |
| Operation/apply seconds | 59.12 | Outer runtime gate remains separate |

The source hash was
`7d4f474c5f39c8e662fcfc93618527ef432f8ab09af5cce35e660540038788d9`.
The frozen mask remains
`f2e692b86e2814f5958c0ca6cc19800a34c12891a527449e68e1624c8bdfe514`.
The stage used 11.86 seconds, including 2.58 seconds of independent full
validation. It attempted 11 exact evaluations, accepted six working edits,
bounded one oversized crop, and published one checkpoint with zero raster or
score disagreements. All six accepted edits were alternatives from the same
parent, not six cumulative improvements. The selected boundary edit removed
18 nodes and reduced the fixed visual score by approximately 0.000541. The
requested soft representation target was 46,365; achieved cost was 54,546 and
the target remained unmet. The hard node ceiling was automatic. Peak accounted
SVG/raster state bytes were 12,956,991. No pipeline deadline overshoot was
reported. These measurements do not establish the full RSS or memory gate.

Before support-based rendering omission, an exploratory run in
`.bench/planned-local-beam` used source hash
`42e75356fc715f7753a2884f1744673a93b486c17498579a146c1b0a5904ef27`.
It attempted seven local evaluations and published one paint edit, selecting
23,212 nodes, 4,147 contours and human MSE 442.34. Its local stage took 12.73
seconds against a 12-second slice. The final run completed a different search
prefix; neither this comparison nor comparison with earlier runs proves a
matched-effort speedup or a calibrated quality improvement.

A separate final-source 20-second run in `.bench/planned-local-short` used the
same settings, mask and source hash. It retained the detailed seed with 23,212
nodes, 4,147 contours, 544 gradients and human MSE 439.75. The pipeline took
21.21 seconds, reporting 1.21 seconds of deadline overshoot; operation/apply
took 26.15 seconds. Local search was unavailable and attempted no edits because
initialization and validation consumed the search allowance. Only two
candidates were retained, so its soft floor/target of 54,564 is not evidence of
useful low-complexity choices. This short-budget runtime failure remains open;
the successful 60-second local checkpoint does not resolve it.

The sword's structural gates still fail by a wide margin. Guard and jewel
human MSEs remain approximately 1,015.70 and 1,233.07. Adding continuous ink
alone does not remove the fragmented surface/ink fills. Family surface models,
ink replacement with restored underlayers, and individually accepted graph
merge/split/order edits remain necessary. A long facet edit also exceeded the
crop allowance; bounded tiling remains unfinished. The complete plan now gives
these structural compaction steps explicit state, operator and validation
contracts. No held-out artwork was used; the method remains experimental.

The final relevant suite passed 287 tests using the command above in the
previous experiment section. New checks cover native local/full score and
raster agreement, gradients/transforms/opacity, immutable overlapping edits,
holes and one-byte marks, hidden-layer dependencies, cache epochs, evaluation
and memory caps, stale proposals, failed full checkpoints, cancellation,
operator acceptance and actual advanced protection. All 101 choices on a
common frontier remain ordered by cost. A supported continuous-ink proposal
is retained at complexity 100; cheaper drawings can correctly win at 50 under
the initial weights. Ruff passed and Pyrefly reported zero source errors.
These tests do not complete score calibration, structural quality, resizing,
cross-job caching or the runtime/memory gates.

## Owned region families and search depth

The planner now retains immutable ownership of original source regions through
label renumbering and SVG export. Primary fill/overlay paths partition those
regions; opacity bases have an explicitly secondary coverage role. Canonical
source boundary IDs remain stable: a merge removes internal edges from the
active ownership view and changes both sides of exterior ownership together.
Atomic regrouping supports membership-preserving merges and splits without
mutating sibling states. A partition supplied independently that splits an
original graph atom reports incomplete ownership; it requires new graph atoms
before owned edits can proceed. The runtime split geometry operator is still
unfinished. These values are temporary planning metadata, not a project-format
or SVG geometry extension.

`families.py` uses that membership to propose connected surfaces within a common
component and coordinate/opacity group. Each proposal joins current curves,
retains holes, refits flat RGBA or a linear color/opacity gradient, and removes
the superseded paths. It retains the frontmost participating path identity and
preserves unrelated geometry and accepted model holds. Gradient endpoints map
through native reference coordinates into the retained path's user space;
group opacity normalizes the child paint. Explicit-width ink and paint holds
remain excluded from these surface edits.

Family generation is bounded to 128 paths and 6,000 input nodes per group, 96
provisional groups and 24 ranked paint alternatives per expansion. Relative
premultiplied color/opacity thresholds and weaker boundary evidence propose
families; they are engineering parameters, not acceptance exemptions. At most
4,096 samples per fit compare approximate RGBA error with the current native
raster. A cheap cost/error estimate ranks the alternatives before curve union
and exact evaluation, using at most a quarter of the remaining local-stage
time. It omits visibility/filter/feature terms and cannot accept an edit.
Oversized native crops retain the existing bound pending tiled evaluation.

Local decisions now record term deltas for rejected edits as well as accepted
ones. Working states retain their edited ownership independently, and missing
source ownership rejects a proposal before native evaluation. The same global
beam/evaluation limits remain, with per-state exact evaluation limits of four,
eight and sixteen for fast, balanced and high quality. This allows cumulative
edits before one parent consumes the entire allowance. Stale/cached scans still
have the independent global cap. Truncated expansions are reported explicitly.
These bounds do not prove the complete process-RSS or candidate-metadata gate.

An initial unranked 60-second sword run in `.bench/planned-families-first` used
source hash
`7939c974a3f99d8a90c61e95b9908b7228d7e936735953faeff72b241dc49cfa`.
The two evaluated family proposals joined two and six paths, saving estimated
geometry costs of 149 and 241 after export, but both failed the exact common
objective. The selected paint edit retained 23,212 nodes, 4,147 contours and
human MSE 442.34. The stage took 14.67 seconds against its 12-second slice,
including 4.87 seconds of independent validation; pipeline time was 51.45
seconds and operation/apply time was 57.87 seconds. This motivated bounded
paint-fit ordering and better rejection diagnostics rather than relaxing the
objective or assuming every family should merge.

A subsequent ranked run in `.bench/planned-families-ranked` used source hash
`5b5ad4941ed5441df894aeb06905be5b05299582a9bc14d8da52ca598463a705`.
It selected one fully validated flat family edit replacing 128 paths with one
surface. The result had 22,635 nodes, 4,026 contours, 3,991 paths, 544 gradients
and human MSE 439.75. That edit saved 577 nodes, 121 contours and 1,315 units
of representation cost. Its fixed visual score fell by approximately 0.000039;
there were zero local/full raster or score disagreements. The edit also added
47 pixels to the opacity-loss diagnostic within the existing hard allowance.
This is not exact source-alpha equality or calibrated feature preservation.
The run attempted ten exact evaluations and accepted six sibling alternatives;
only one entered a full checkpoint. Stage time was 12.16 seconds, including
2.30 seconds of full validation. Pipeline and operation/apply times were 50.14
and 55.16 seconds, with no pipeline overshoot. The soft cost target was 46,365,
achieved cost was 53,249, and the target remained unmet. Accounted SVG/raster
peak bytes were 12,386,125; metadata/document/RSS remain separate open gates.

The frozen sword mask is unchanged:
`f2e692b86e2814f5958c0ca6cc19800a34c12891a527449e68e1624c8bdfe514`.
The runs completed different proposal prefixes and do not prove a matched-effort
speedup or a score-calibration improvement. The ranked run passes the numerical
human-error ceiling but fails the node and contour targets by a wide margin.
Guard and jewel MSEs remain approximately 1,015.14 and 1,233.07. No held-out
artwork was used. Ink replacement, richer surface interpretations, new-region
splits, large-area tiling, calibration and release evaluation remain necessary.

The final-source native run enables the per-state expansion limits and reports
current region counts after replacement. Reproduce it with:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python scripts/bench_cel_planned.py --methods cel-planned --seconds 60 \
  --settings '{"complexity":50,"quality":"balanced","refine":false}' \
  --out .bench/planned-families-depth
```

Its source hash was
`c2ce7e383975006f7a1300c5cef407020ee22fa657ac64c23d3c3306fd6103bc`.
The selected drawing has 22,097 nodes, 3,901 contours, 3,864 paths, 538 gradients
and human MSE 439.64. Two cumulative family edits each replaced 128 paths with
one; together they removed 1,115 nodes, 246 contours and 254 paths from the
detailed seed. Representation cost fell from 54,564 to 51,885, still above the
reported soft target/floor of 46,365. The second edit increased the fixed visual
score by 0.0000951 but reduced cost by 1,364, improving the common objective at
all five complexity anchors. This is an intended tradeoff under the initial
weights, not evidence of corpus calibration. The edits added 47 and nine
opacity-loss pixels within the existing hard allowance; the second also added
15 opacity-excess pixels. Full validation accepted those measured differences.

The stage evaluated ten proposals from two parents, accepted seven working
alternatives, bounded two crops and one parent expansion, and published one
full checkpoint. There were zero raster or score disagreements. Stage time was
11.15 seconds, including 1.90 seconds of full validation; pipeline time was
49.01 seconds and operation/apply time was 53.46 seconds, without pipeline
overshoot. Accounted SVG/raster peak bytes were 12,040,628. Metadata reported
3,863 primary regions, matching the retained ownership rather than the seed's
4,117. These measurements do not establish a full memory gate or a matched-effort
performance comparison.

Inspection of the native crops confirms the remaining quality gap: blade facets
still contain small shade fragments and irregular edges, guard shading remains
subdivided, and the jewel lacks the human drawing's coherent circular ink rim.
Guard and jewel errors remain 1,015.14 and 1,233.07. The frozen sword structural
gates fail despite the passing numerical human-error ceiling. More coherent
surface/outline interpretations and ink replacement remain the next work;
learned ranking cannot supply those missing alternatives.

The final relevant suite passed 302 tests using the previously recorded broad
command. New coverage includes label renumbering, canonical ownership on both
sides of edges, merge/split regrouping and sibling rollback, secondary opacity
bases, incomplete atom subdivisions, exact family alpha/hole preservation,
RGBA-gradient alternatives, scaled/translated gradient frames, retained neighbor
geometry, rejection of missing members, fixed-width/component exclusions and
cumulative edits within the fast evaluation cap. Native raster samples also
agree after overlapping history patches. Ruff passed; Pyrefly reported zero
source errors. Broader paired calibration, original-atom split geometry, joint
ink/underlayer fitting, resizing, runtime/memory and release gates remain open.

## Bounded native tiles and streamed family samples

Long proposals no longer fail solely because their score crop exceeds 262,144
pixels. The evaluator partitions output ownership into at most 32 disjoint
tiles, with overlapping 16-pixel input halos bounded by that crop allowance.
Every tile compares the original before-state with the same completed candidate
render. Color/ink halo contributions and alpha/feature ownership are accumulated
once using the complete policy's fixed denominators. Immutable raster patches
are published together only after the entire edit finishes. Stop/deadline checks
between rendering and tiles discard incomplete work, preserving the independent
checkpoint. Search reports the tile count and invalid or bounded native areas.

For multiple tiles, one native uint8 RGBA render retains the original viewport
and opacity-group rounding. Only bounded tile windows convert to floats. The
existing 16 MiB native raster ceiling remains; this is not arbitrary-resolution
streaming or a complete process-RSS bound. Full checkpoint validation remains
independent and includes a complete render and score.

Family paint estimates now scan bounded source-label chunks, preserving the
dense mask's row-major sample stride with at most 4,096 paired position/RGBA
samples. No full family mask or float crop is needed. The same flat/gradient
fitter accepts those samples in the analysis frame. Interrupted sampling returns
no partial model. Rejection proofs hash bounded uint8 chunks and include the
declared affected bounds, avoiding a spatially different proposal reusing the
same pixel proof.

New tests compare tile statistics, hard rejections and exact native pixels with
full evaluation for gradients, transformed opacity groups, holes, one-byte
marks, first/last ink, spill and opacity loss. They exercise disjoint/overlapping
cumulative edits and sibling histories, stop between tiles, tile limits before
rendering, invalid bounds and rejection-key chunk limits. Streaming family
samples and paint models match dense sampling across chunk boundaries. A real
160×2,000 gradient edit exceeds the default crop limit; a 192×2,048 RGBA surface
family reaches both tiled acceptance and an independent full checkpoint.

Reproduce the native development check with:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python scripts/bench_cel_planned.py --methods cel-planned --seconds 60 \
  --settings '{"complexity":50,"quality":"balanced","refine":false}' \
  --out .bench/planned-native-tiles
```

Source SHA-256 was
`a9bf113637628776e4e22cc652844e25945beb404a98a3edfc7f214ed4fd6a4c`;
the frozen mask remains
`f2e692b86e2814f5958c0ca6cc19800a34c12891a527449e68e1624c8bdfe514`.
The former oversized blade boundary at native bounds `(311, 78, 417, 1558)`
is now evaluated in two tiles. It enters a working state with four fewer nodes
and visual delta −0.00005715. That alternative was not the published checkpoint;
the final selection still retains the two cumulative family edits. Synthetic
long-edit tests supply the independent complete agreement evidence for this
new path, rather than claiming full validation of an unpublished sword sibling.

The final drawing remains 22,097 nodes, 3,901 contours, 3,864 paths and 538
gradients, with human MSE 439.64097. Search attempted ten evaluations, scored
11 tiles, accepted seven working alternatives, bounded one parent expansion
and published one checkpoint. No proposals hit the old crop-size rejection;
there were zero checkpoint score/raster disagreements. Stage time was 11.15
seconds, including 1.89 seconds of full validation. Pipeline time was 49.20
seconds and operation/apply time 53.56 seconds, without pipeline overshoot.
Accounted retained SVG/raster peak bytes were 12,662,465. The soft target remains
46,365 against achieved cost 51,885, reported unmet. This is not a matched-effort
speedup or a quality improvement over the previous development run.

Native guard and jewel crops still show fragmented shading and a missing
coherent jewel ink rim. Structural count gates remain failed. Compact opacity
interpretations and ink replacement with restored underlayers are the next
quality work; tiling enables their evaluation but does not supply those models.
No held-out artwork was used. The final relevant suite passed 325 tests with
the previously recorded broad command. Ruff passed; Pyrefly reported zero
source errors with 61 existing warnings. Broader paired calibration and complete
runtime, memory and release evidence remain open.

## Owned ink replacement and restored neighboring paint

The new `cel_plan/ink_replace.py` operator connects owned dark regions with
substantial detected-ink support, respecting component, parent, transform and
color compatibility. A paired dark-ridge proof distinguishes ink from an
ordinary shade step. It compares the current fragments with a filled union and,
where width and coverage support it, a fitted stroke. The filled alternative
retains the current exterior, holes and width variation; it does not yet fit a
new variable-width outline. Constant-width strokes are excluded when measured
width spread exceeds the initial 1.6 ratio, unless a positive user width supplies
the explicit competing interpretation. These thresholds are engineering
proposal rules, not calibrated human preferences.

A stroke restores adjoining surface paint inside the old ink footprint before
replacing the fragments. Neighbor labels extend by nearest supported exterior
samples; their existing paint and gradient frames are retained. A dominant
neighbor covers the complete old union, with other continuations clipped to
that footprint. The stroke follows in the survivor's drawing position. Opaque
material, matching isolated opacity, or a geometrically verified opaque core
is required for overlap; membership alone cannot prove core coverage. Uncovered
variable translucency retains the filled competitor. Silhouette/hole contacts,
different parents/transforms and unsupported neighbor context exclude the
initial stroke interpretation. General occlusion/order inference is still open.

The surviving ink path owns all original members as an overlay. Restored paths
have secondary underlay ownership; sibling states and the source graph stay
unchanged. Geometry and explicit-width holds propagate, and ordinary SVG/project
serialization remains the editing format. Exact local checks and independent
full checkpoints remain the only acceptance path. No human geometry or sword
recognition is supplied to the generator.

Discovery is bounded by 16 families, 128 source paths and 6,000 source nodes;
restoration has at most four neighbors and 6,000 total generated underlay nodes.
Source masks retain the 262,144-pixel ceiling. At most 16 centerline runs are
measured per family, with a cached immutable luminance field. Ink discovery has
one quarter of the remaining search time and a slot after surface, paint,
boundary and existing ink proposals. Diagnostics distinguish eligibility,
missing ridge support, variable width, missing restoration, bounded areas,
proposed models and time expiration. These bounds do not complete candidate
metadata accounting or the full process-RSS gate.

An initial run with ink discovery first in the cycle, without its own time
slice, used source SHA-256
`93d46d03dc23bf5917ef2912234b1faadbde3c606ac2c7d593b33796ff83dc26`
in `.bench/planned-ink-replacement`. Its checkpoint retained one filled ink
replacement and one family merge: 22,616 nodes, 4,023 contours and human MSE
439.75101. The ink edit saved 37 cost units with visual change below 1e-9, but
the completed search prefix lost the second larger family edit. That is a
denser final drawing, not a quality improvement or reason to accept additive
ink indiscriminately. It motivated the bounded discovery slice and scheduling
change above.

Reproduce the final-source native development run with:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python scripts/bench_cel_planned.py --methods cel-planned --seconds 60 \
  --settings '{"complexity":50,"quality":"balanced","refine":false}' \
  --out .bench/planned-ink-bounded
```

Source SHA-256 was
`86f43bff68480441865501b3454f1992fcd3efc15abf6ddfc6b631c5fd83ff86`;
the frozen mask remains
`f2e692b86e2814f5958c0ca6cc19800a34c12891a527449e68e1624c8bdfe514`.
The final drawing again contains 22,097 nodes, 3,901 contours, 3,864 paths and
538 gradients, with human MSE 439.64097. The two family edits remain its full
checkpoint. An additional filled ink alternative entered the working beam,
saving 37 cost units, but did not enter that checkpoint. No stroke replacement
was offered in this completed prefix: nine families were scanned, seven lacked
the required ridge proof, one source area was bounded, and one variable-width
filled proposal was offered before discovery expired. This is not proof that
all sword strokes are unsupported or that the full candidate set was searched.

Search attempted ten evaluations, accepted seven working alternatives and
published one checkpoint, with zero raster/score disagreements. Stage time was
11.13 seconds including 1.92 seconds of independent validation. Pipeline time
was 49.05 seconds; operation/apply time was 53.41 seconds without pipeline
overshoot. Accounted retained SVG/raster peak bytes were 12,604,397. The soft
cost target remains 46,365 against achieved cost 51,885, reported unmet. These
different completed prefixes do not establish a matched-effort speedup. Guard
and jewel MSEs remain 1,015.14 and 1,233.07, and the structural gates still fail.

Tests cover flat/stroke replacement, exact alpha at 255/128/64, local/full
agreement, restored two-shade paint, retained gradient coordinates, tapered
filled marks, blank gaps, unsupported shade steps, component boundaries,
uncovered translucency, transformed native widths, explicit width/hold
propagation, project reload and cancellation/independent discovery deadlines.
The broad relevant suite passed 338 tests; all 14 focused ink-replacement tests
passed, including the additional discovery-deadline case. Ruff passed; Pyrefly
reported zero source errors with 61 existing warnings. No held-out artwork was
used. Broader joined/variable-width ink fitting and coverage remain open.

The dense drawing also reports 3,986 native-alpha geometry holds across its
regions. Source alpha is mostly byte 253, with weaker fringes and low-opacity
pixels; this does not justify discarding them indiscriminately. The next
compaction work must offer coherent surfaces and refinable internal chains
while preserving supported native contacts. In particular, investigate region
boundaries excluded by coarse line evidence when a paired ridge is absent,
and compare those alternatives under the same full validation policy. A learned
ranker cannot recover surface or outline models that are never proposed.

## Remaining requirements

None of the eight complete deliveries is claimed finished yet. In particular:

| Delivery | Remaining evidence or behavior |
| --- | --- |
| 1 | Full synthetic/curated-human coverage, frozen broader-suite tolerances and calibrated score terms |
| 2 | Dense-input fallback/runtime and memory bounds; broader partial-alpha, transformed-scope and difficult-hole coverage beyond the new native cases |
| 3 | Region splits, richer surface/paint interpretations beyond the initial owned family merges, calibrated content normalization and budget-directed search, bounded shared-frontier cache, large-input fallback beyond the bounded native tile kernel and resizing invariance |
| 4 | Fitting variable-width filled ink beyond retained unions, broader stroke replacement/underlayer coverage, full join/feature checks, parameterized primitive fitting beyond whole-path holds and passing sword/line/feature gates |
| 5 | Broader local-layer/order inference, joint RGBA/geometry/width fitting, geometric regularization, complete spatial scheduling, memory/runtime gates and optional acceleration ownership |
| 6 | UI/MCP controls, browser/API round trips, invalidation and documentation |
| 7 | Expanded paired evaluation/corpus coverage, ablations and conditional learned-ranker experiment |
| 8 | Independent blind review, fresh held-out results and default migration only after release gates pass |

The operation still reports `refinement_complete: false`. The bounded CPU
foundation implements real fitting behavior; it does not complete joint fitting
or the required runtime/memory and quality gates. Owned family merges and
individual paint/boundary/ink edits, including initial owned replacements, have
a bounded working beam and independent full checkpoints. Complete merge/tolerance
anchor proposals also remain; region splits, broader ink replacement, richer
models and order edits are unfinished.
These gaps must be resolved before completion is claimed.
