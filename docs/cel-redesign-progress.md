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
- Surface families now test bounded paired-ridge evidence before treating
  coarse line support as an exclusion. Supported and unresolved strong edges
  remain barriers even when a weak alternate adjacency route connects their
  owners. Missing ridge evidence permits a surface proposal, not acceptance.
- RGBA export can now carry bounded interior-chain permissions alongside a
  whole-path native hold. Geometry and CPU refinement bind them to the exact
  current path/frame, preserve all other segments and both shared-edge copies,
  and refresh them after accepted fitting. Structural replacement discards
  permissions and keeps its conservative whole-path hold.
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
  from a hard node ceiling. Initial operator slots now use the fixed shared-pool
  nominal target, independently of the unproven observed floor. Broader risk
  ranking and spatial scheduling remain open.
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
- Owned closed overlays now also compete in structural search, including RGBA
  material inside a geometrically verified opacity core. Neighboring fills
  continue beneath the replacement; primary region ownership and hidden paint
  coverage are recorded separately. Bounded geometry proofs permit crossing a
  disjoint sibling whose bounding box overlaps, while actual overlaps remain
  barriers. Richer local order inference and nested surface interpretations are
  still unfinished.
- Closed contours now have a coupled interior-material competitor. Complete
  owner residuals retain independent highlights, ink and constrained paint;
  explicit coverage mixtures near their boundaries explain antialias samples.
  Flat/gradient material and ellipse/contour geometry compete under unchanged
  native local/full checks. Shared checkpoint budgeting,
  complementary operator opportunities and bounded enclosure scheduling hints
  improve access to cumulative edits. They do not establish a sword quality win.
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

## Bounded shade-versus-ink family evidence

The family operator previously excluded every canonical boundary with coarse
line support above 0.5. It now checks a paired dark ridge on sufficiently long,
bounded chains before excluding them. This offers a surface competitor for a
shade discontinuity falsely marked as ink. The existing native/full objective
and hard checks still decide whether any resulting merge can be retained.

Each operator caches at most 512 immutable evidence decisions. A proof uses at
most 2,048 original chain points, minimum chain length eight analysis pixels,
and the existing ridge estimator at width 1.5 analysis pixels. Smoothed
luminance is computed lazily once. Short or oversized chains, exhausted proof
capacity and interruption retain the conservative exclusion. Cancelled proofs
do not enter the cache. This is a bounded initial interpretation, not a complete
multi-width or junction classifier.

Family growth also respects every supported or unresolved strong-edge barrier
between its members. Merely omitting that edge from adjacency was insufficient:
a weak route through a third region could still absorb both sides. A family
cannot take that detour around a protected pair. Pairs farther than twice the
maximum family color threshold cannot belong to the same family and are
excluded before ridge work. Existing ownership, alpha and native acceptance
requirements are unchanged.

The native sword development command was:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/rasmus/Workspaces/vectrify/.venv/bin/python scripts/bench_cel_planned.py \
  --methods cel-planned --seconds 60 \
  --settings '{"complexity":50,"quality":"balanced","refine":false}' \
  --out .bench/planned-shade-ridges
```

Its source SHA-256 is
`0eaaea2fe269f368fd6cd0808f9fe3058d5e986bd954ffce096334fb9de21b7b`.
The mask SHA-256 remains
`f2e692b86e2814f5958c0ca6cc19800a34c12891a527449e68e1624c8bdfe514`.
The result contains 22,122 nodes, 3,901 contours, 3,864 paths and 536 gradients,
with representation cost 51,886 and human MSE 439.56173. It still fails the
800-node/140-contour gates. The preceding bounded ink run had 22,097 nodes,
cost 51,885 and human MSE 439.64097. This comparison does not establish a
material quality gain, a structural improvement or a matched-effort speedup.

The new evidence check evaluated 107 unique chains: 101 supported a ridge and
six did not. There were 5,490 unresolved strong-edge encounters, counted across
parent scans rather than as unique boundaries. The cache capacity was not
reached. Diagnose short versus oversized chains and their cost contributions
before changing these bounds. This result supports finer evidence/geometry
diagnosis rather than assuming that most coarse ink is misclassified shading.

Search attempted ten evaluations, accepted seven working alternatives and
published one full checkpoint with zero raster/score disagreements. The final
checkpoint contains two 128-path family replacements, with cost deltas -1,304
and -1,374. Stage time was 11.15 seconds including 1.88 seconds of independent
validation. Pipeline time was 50.25 seconds; operation/apply time was 54.72
seconds with zero pipeline overshoot. Accounted retained SVG/raster peak bytes
were 12,607,690; this is not process peak RSS. The soft target remains 46,365,
reported unmet. Guard and jewel human MSEs remain 1,015.14 and 1,233.07.

A small tuning diagnostic used the four declared tuning families, clean inputs,
192-pixel long sides, complexity 50, balanced quality, refinement disabled and
20-second operation limits:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/rasmus/Workspaces/vectrify/.venv/bin/python scripts/bench_cel_pairs.py \
  --methods cel-planned \
  --cases anime-girl anime-face western-park rubberhose-band \
  --degradations clean --long-side 192 --seconds 20 \
  --method-settings '{"cel-planned":{"complexity":50,"quality":"balanced","refine":false}}' \
  --out .bench/planned-shade-ridges-pairs
```

| Tuning case | Nodes | Contours | Clean-target MSE | Line F1 | Generation seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Anime girl | 2,677 | 150 | 895.39 | 0.678 | 5.95 |
| Anime face | 5,467 | 270 | 703.60 | 0.664 | 7.59 |
| Western park | 4,394 | 226 | 1,280.81 | 0.670 | 6.75 |
| Rubberhose band | 900 | 123 | 994.56 | 0.651 | 7.77 |

All four returned a validated ready result and complete diagnostic candidate
storage. Their clean-target oracle at the selected cost found the selected
candidate; unconstrained pools offered lower MSE at higher cost. This is neither
an independent preference assessment nor proof that selection is correct.
Important feature errors remain large, the first three normalizers used the
conservative fallback, and only clean small renders were examined. No matched
old/new or legacy comparison was run here, no degradation or held-out artwork
was evaluated, and these are not calibrated release results.

The 241-test planner/operation/benchmark suite passed, including 21 focused
family tests. A separate 75-test legacy CEL/shared-boundary suite passed. New
checks cover coarse false ink on shade steps with native acceptance, cached
evidence, a protected ridge across an alternate route, conservative short/large
and proof-bound behavior, and cancellation without caching an incomplete
decision. Ruff passed; Pyrefly reported zero source errors with 61 existing
warnings. Its configured exclusions do not type-check tests.

The next concrete experiment in the plan diagnoses hold reasons, carries
constraints at canonical-chain granularity and compares interior fitting and
larger coherent surfaces separately. Whole-path native holds remain the safe
fallback until that finer correspondence is proved. None of these results
justifies exposing unfinished controls, training a ranker or changing defaults.

## Interior-chain permissions and native hold diagnosis

The RGBA exporter already fit safe interior chains separately from native
alpha contacts, but its whole-path hold blocked later fitting of both. It now
records the actually emitted interior segments on eligible held paths. Binding
requires a matching geometry fingerprint, reference-to-document matrix and
complete segment correspondence. Every segment outside those permissions,
including an implicit closure or an unrecorded canvas contact, remains exact.

`cel_plan/constraints.py` freezes protected endpoints, checks protected controls
and segment multiplicity, and verifies the matrix again after editing. Existing
corner/junction anchors remain active. A shared edit updates both copies and
checks their agreement. Geometry simplification enters the native local beam;
CPU refinement also consumes the permissions and independently validates its
whole checkpoint. Accepted geometry gets an independently forked fingerprint
and permissions, retaining export-chain identities and source membership.

The bounds are 8,192 recorded canonical export chains, 32,768 emitted segments,
4,096 source points per recorded chain and 512 nodes per eligible current path.
Canonical records may be referenced by both owners. Missing, oversized or stale
correspondence retains the whole-path restriction. Union and ink replacement
explicitly discard affected permissions; a sibling's metadata is unchanged.
This is initial export-chain correspondence, not original-atom subdivision or
complete canonical-edge reconstruction. Compact primitive holds are unchanged.

Twelve new tests cover a shared interior simplification inside transformed
groups, exact protected segments, refreshed/sibling metadata, project reload,
whole-held neighbors, stale geometry and ancestor transforms, native acceptance
with holes and one-byte-alpha marks, CPU refinement, cubic handles,
cancellation, replacement invalidation and each independent recording cap.
The complete relevant suite passed **328 tests**; Ruff passed; Pyrefly reported
zero source errors with 61 existing warnings. Tests remain excluded by its
configured ignore rules.

Before the native sword check, the four small clean tuning cases were rerun:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/rasmus/Workspaces/vectrify/.venv/bin/python scripts/bench_cel_pairs.py \
  --methods cel-planned \
  --cases anime-girl anime-face western-park rubberhose-band \
  --degradations clean --long-side 192 --seconds 20 \
  --method-settings '{"cel-planned":{"complexity":50,"quality":"balanced","refine":false}}' \
  --out .bench/planned-chain-constraints-pairs
```

Their selected geometry, clean MSE and line scores exactly match the preceding
shade/ridge diagnostic. Selected drawings do not carry these RGBA chain
permissions, so that equality demonstrates no new fitting or quality benefit
on this subset. All four returned ready, retained complete diagnostic pools and
had zero checkpoint disagreements. Completed evaluation prefixes differed;
no speedup is inferred. No held-out or degradation cases were run.

The native development command was:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/rasmus/Workspaces/vectrify/.venv/bin/python scripts/bench_cel_planned.py \
  --methods cel-planned --seconds 60 \
  --settings '{"complexity":50,"quality":"balanced","refine":false}' \
  --out .bench/planned-chain-constraints
```

Both experiments used source SHA-256
`2a9a38c79d4813007842d3376013870e860b5610d1c0b384616da05dcaae2aa5`.
The native mask remains
`f2e692b86e2814f5958c0ca6cc19800a34c12891a527449e68e1624c8bdfe514`.
The sword result contains 22,646 nodes, 4,026 contours, 3,991 paths and 544
gradients, with cost 53,260 and human MSE 439.75132. It is denser than the
preceding 22,122-node/cost-51,886 result and still fails both structural gates.
Its selected checkpoint contains one family merge, saving 1,304 cost units.
The final large-family evaluation was interrupted; the different completed
prefixes do not isolate a geometry-quality or runtime effect.

Search attempted nine evaluations, accepted six working alternatives and
published one full checkpoint with zero score/raster disagreements. Stage time
was 11.33 seconds including 1.93 seconds of independent validation. Pipeline
time was 49.13 seconds; operation/apply time was 53.60 seconds without pipeline
overshoot. The accounted retained SVG/raster peak was 12,607,690 bytes and does
not include all metadata or process RSS. The soft target remains 46,365 and is
reported unmet. Guard and jewel human MSE remain 1,015.14 and 1,233.07; inspected
crops still show the same poor contour/shading interpretations.

The selected generation metadata contains 700 permission paths and 1,207
refinable segments, with no omitted chain-cap records. JSON encoding occupies
369,955 bytes before additional copies. These are planning IDs: operation Apply
allocates editor IDs normally, so they are not directly bindable against the
applied drawing using those original names. Planning constraints do not become
a new persistent editor/project geometry model or a post-Apply fitting cache.

The new hold-reason diagnostics count canonical callback chains/segments:

| Reason | Chains | Emitted segments |
| --- | ---: | ---: |
| Transparent contact | 4,215 | 5,827 |
| Thin component | 545 | 890 |
| Explicit width | 0 | 0 |
| Alpha step | 6,155 | 6,897 |
| Repaired crossing | 0 | 0 |

Reasons overlap and shared segments can appear in both visible owners; this is
not an additive decomposition of all nodes. Canvas-border chains bypass this
callback and are not included. The numbers do establish that protected native
contacts dominate the recorded chain population, while few segments are
eligible for interior fitting. Simply removing whole-path holds would neither
solve the surface partitioning nor justify moving those contacts.

The next compaction work must offer larger connected RGBA surface/coverage
interpretations and schedule them usefully, while preserving intentional holes,
thin marks and supported fringes through exact validation. Complete original
atom splits/edge reconstruction and primitive-parameter fitting remain open.
This implementation is a useful fitting foundation, not a sword improvement or
a completed delivery.

## Budget-directed operator slots and resumable parent discovery

Local search now retains one bounded proposal cursor per retained parent, with
a maximum equal to the quality's beam width. After its four/eight/sixteen exact
evaluation slice, a parent can resume later if it remains in the beam. A rejected
prefix with no accepted child no longer ends discovery. The slice check occurs
before pulling the next proposal, so the first unevaluated tail edit is not
silently consumed. Pruned/completed cursors and their operator generators close
normally; stop after discovery prevents scoring or publication of that edit.
Evaluation, scan, dependency, native raster and checkpoint bounds remain active.

The initial state carries a fixed shared-pool budget context: complexity-50
anchor, nominal representation target `C₀ / 2`, detailed normalizer and the
separate explicit node ceiling. Requested complexity still selects from the
common frontier and does not alter this context. Above 1.25 times either target,
balanced/high quality offers two/three family proposals before the reserved
operator round; fast keeps one. Each round reserves paint, shared geometry,
additive ink and owned ink opportunities, rotating their order by edit depth
and cycle. Existing sampled paint residuals rank family alternatives; exact
native evaluation alone accepts them.

Budget reports now include `nominal_target` and `floor_proven: false`. The
existing clamped target/observed floor is still reported, but using the observed
floor to schedule exploration would make the cheapest current drawing its own
lower bound. It is neither a mandatory-feature proof nor a feasibility claim.
The new ordering is an initial deterministic schedule, not calibrated risk
prediction, full spatial scheduling or a completed complexity delivery.

The full relevant suite passed **333 tests**, including a useful fifth proposal
after four rejected fast-quality edits, cursor closure/bounds, reserved operator
slots under pressure, a node ceiling that cannot admit visual regression,
cancellation during discovery and nominal-versus-observed budget reporting.
Ruff passed; Pyrefly reported zero source errors with 61 existing warnings.

The final source was checked on the same small tuning subset before the sword:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/rasmus/Workspaces/vectrify/.venv/bin/python scripts/bench_cel_pairs.py \
  --methods cel-planned \
  --cases anime-girl anime-face western-park rubberhose-band \
  --degradations clean --long-side 192 --seconds 20 \
  --method-settings '{"cel-planned":{"complexity":50,"quality":"balanced","refine":false}}' \
  --out .bench/planned-budget-resume-pairs
```

| Tuning case | Nodes | Cost | Clean MSE | MSE change from chain run | Line F1 | Generation seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Anime girl | 2,681 | 3,603 | 891.84 | -3.55 | 0.678 | 5.81 |
| Anime face | 5,394 | 7,032 | 703.29 | -0.31 | 0.664 | 7.60 |
| Western park | 4,295 | 5,661 | 1,304.12 | +23.31 | 0.670 | 6.72 |
| Rubberhose band | 907 | 1,571 | 982.09 | -12.46 | 0.651 | 7.08 |

All four returned ready with complete diagnostic storage and zero checkpoint
disagreements. Different completed pools/evaluation counts prevent a
matched-effort speedup or general quality claim. No held-out or degradation
cases were evaluated. Western park is a concrete selection conflict: the same
hard-valid pool contains clean MSE 1,287.24 at cost 5,658, better raw error and
slightly lower cost than production selection. The selected face-region error
also rises from 5,785.03 to 6,001.01. This calls for the declared score/feature
calibration, not a claim that more compaction universally improves quality or
that raw MSE alone establishes human preference.

The native command was:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/rasmus/Workspaces/vectrify/.venv/bin/python scripts/bench_cel_planned.py \
  --methods cel-planned --seconds 60 \
  --settings '{"complexity":50,"quality":"balanced","refine":false}' \
  --out .bench/planned-budget-resume
```

Both final runs used source SHA-256
`c99187ce156f4e60aeff5ce48e788f7807e92b31f836a01076b226906e52578c`.
The native mask remains
`f2e692b86e2814f5958c0ca6cc19800a34c12891a527449e68e1624c8bdfe514`.
The sword again selects two family replacements (-1,304/-1,374 cost units):
22,122 nodes, 3,901 contours, 3,864 paths, 536 gradients, cost 51,886 and human
MSE 439.56173. This restores the preceding shade/ridge drawing after the denser
chain-permission prefix, not a new quality or structural-target pass. Guard and
jewel human errors remain 1,015.14 and 1,233.07.

Search evaluated ten proposals, accepted eight working alternatives and
published one full checkpoint with zero disagreements. Six generated proposals
were families and four were reserved operators. Two compaction parents were
visited. No parent cursor resumed on this native prefix; the resumed-tail
behavior is demonstrated by the synthetic test, not attributed to this result.
Stage time was 11.17 seconds including 1.89 seconds of validation. Pipeline
time was 48.99 seconds; operation/apply time was 53.42 seconds with zero pipeline
overshoot. Accounted retained SVG/raster peak was 12,607,690 bytes, excluding
some metadata and process RSS. Nominal search target was 27,282; reported target
46,365 includes the observed, unproven floor. Achieved cost remains above both.

An independent native-reference inventory found 136 alpha-positive connected
components: two with a two-pixel-eroded core and 134 thin components. There are
110 components of fewer than four pixels, totaling 166 pixels; all have peak
alpha at most 7/255 (median 1/255). This confirms a population of faint source
fragments worth interpreting, but does not classify every fragment as noise or
authorize dropping intentional low-opacity marks. The earlier thin-hold count
545 measures canonical boundary encounters, not 545 distinct components.

The next surface work also needs owned RGBA compact closed overlays: the
existing `layers.continued` competitor is in the opaque export branch, while
RGBA export currently supplies opacity cores and individual chain models.
Offer compact shape families with restored surrounding paint and retained
primitive constraints under the same native policy. Source-fragment ambiguity,
richer RGBA surfaces and the measured tuning selection conflict remain open;
this scheduling change does not complete any of the eight deliveries.

## Owned RGBA closed overlays and neighboring paint continuation

`cel_plan/overlays.py` adds ellipse and anchored closed-contour competitors for
small connected owned families and individual surfaces. Native boundary and
paint samples supply the geometry and flat/linear RGBA models. Replacing the
fragments retains their original source members on one overlay. It does not
receive human geometry, evaluation rectangles or artwork-specific labels.

The first implementation added separate underpaint paths. Synthetic tests
showed that this duplicated the old outline and raised representation cost.
The retained implementation instead partitions the old footprint by supported
neighbor paint, clips continuation against the current exact footprint, and
unions it into those neighboring fills. Extending assignment cells beyond the
source mask before clipping avoids tiny gaps between a fitted contour and the
pixel-grid mask. Existing gradient frames stay unchanged.

Primary `Surface.members` remain a partition. Optional `covered` members record
hidden overlap beneath other owned regions, without claiming complete geometric
containment. Metadata round trips retain that distinction, and ordinary primary
replacement cannot silently discard it. Family/ink regrouping currently skips
surfaces with hidden coverage until those operators support layered replacement.
The continued neighbors and accepted compact shape keep conservative geometry
holds; replaced export-chain permissions are discarded.

For translucent evidence, an existing opaque core inside the same isolated
material group must geometrically contain both the old footprint and the fitted
shape. Group-relative fill/gradient opacity applies translucency once. Holes,
silhouette contacts and unproved cores retain their existing interpretation.
Draw order places the overlay above its continued bases only across those bases
or provably disjoint siblings. Overlapping bounds trigger a bounded exact fill
intersection test; an actual overlap, unsupported stroke/clip, or exhausted
proof limit excludes that order change.

Discovery alternates family and individual-shape opportunities; it no longer
spends all slots on hole-bearing families before reaching individual shapes.
Limits are 32 eligible groups, 4,096 perimeter samples, a 262,144-pixel source
crop, up to 64 raw neighboring paths and 6,000 total neighbor/continuation nodes.
The pre-union neighbor count is checked as well as the resulting geometry.
Order changes allow at most 16 intersection proofs per candidate, each bounded
to 6,000 nodes. Discovery has its own quarter of remaining wall time. Existing
dependency, raster, scan, evaluation and independent checkpoint limits still
apply. The shared operator schedule now reserves closed overlays alongside
paint, geometry, additive ink and owned ink replacement (schedule version 2).

The final relevant suite passed **351 tests**. New cases cover alpha 253/128/64,
flat/gradient continuation, actual versus merely bounding-box overlap, source
label order, faint intentional marks, holes and irregular outlines, partial
core exclusion, a missing core, bounded discovery, fragmented neighboring paint,
source ownership and offset/scaled scope save/reload. A four-fragment translucent
fixture goes from 61 nodes/cost 103 to 34 nodes/cost 58 for the ellipse, or 32
nodes/cost 56 for the contour. Local/full score terms and native pixels agree.
Ruff passed with 28 formatted files; Pyrefly reported zero source errors and
61 existing warnings.

The final tuning command was:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/rasmus/Workspaces/vectrify/.venv/bin/python scripts/bench_cel_pairs.py \
  --methods cel-planned \
  --cases anime-girl anime-face western-park rubberhose-band \
  --degradations clean --long-side 192 --seconds 20 \
  --method-settings '{"cel-planned":{"complexity":50,"quality":"balanced","refine":false}}' \
  --out .bench/planned-overlay-order-pairs
```

Source SHA-256 was
`cdbebfaea222a79f6a0f90b1574e5965775eecec327409fb6d42bb76182658b6`.

| Tuning case | Nodes | Cost | Clean MSE | MSE change from budget-resume run | Line F1 | Generation seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Anime girl | 2,901 | 3,867 | 889.48 | -2.36 | 0.678 | 6.61 |
| Anime face | 5,394 | 7,032 | 703.29 | 0 | 0.664 | 9.33 |
| Western park | 4,295 | 5,661 | 1,304.12 | 0 | 0.670 | 8.15 |
| Rubberhose band | 907 | 1,583 | 963.78 | -18.31 | 0.651 | 8.48 |

All four returned ready with complete diagnostic storage and zero checkpoint
disagreements. Closed-contour proposals now reach exact evaluation in three
cases. Western park accepts working contour replacements of -54/-71 cost
units; Rubberhose accepts a higher-cost visual alternative. None is in the
final selected drawing, and no ellipse was offered on this subset. The final
changes are selected family proposals from different completed pools. These
results do not establish a general improvement, matched-effort speedup, or
quality benefit attributable to the closed-overlay operator.

The Western park same-cost oracle gap remains 16.88 raw MSE. Selection
calibration, broader compact-family coverage, nested opaque-color holes versus
intentional alpha holes, and richer supported local order remain open. No
held-out or degradation evaluation was used for this iteration.

The final native command was:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/rasmus/Workspaces/vectrify/.venv/bin/python scripts/bench_cel_planned.py \
  --methods cel-planned --seconds 60 \
  --settings '{"complexity":50,"quality":"balanced","refine":false}' \
  --out .bench/planned-overlay-order
```

It used the same final source hash and frozen mask
`f2e692b86e2814f5958c0ca6cc19800a34c12891a527449e68e1624c8bdfe514`.
The selected sword has 22,646 nodes, 4,026 contours, 3,991 paths, 544 gradients,
cost 53,260 and human MSE 439.75132. It retains one family replacement, the
same drawing as the earlier chain-permission prefix. This is denser than the
budget-resume prefix and does not improve the sword or pass its structural
targets. Guard/jewel error remains 1,015.14/1,233.07; blade-facet error is 230.02.

Search evaluates seven proposals, accepts six working alternatives and
publishes one independent checkpoint with zero disagreements. It visits nine
family and sixteen single-shape overlay groups. Nine fail topology, one exceeds
the perimeter bound and fourteen lack restoration; discovery then reaches its
time slice. Restoration records four neighbor-count exclusions, nine
silhouette/hole contacts and one unproved core. The shared restoration helper's
peak neighbor count is 244. No closed overlay reaches native exact evaluation,
so neither learned selection nor different score weights can select one from
this prefix.

Pipeline time is 51.21 seconds, operation/apply time 56.95 seconds, with zero
pipeline overshoot. Structural search takes 11.46 seconds including 2.08 seconds
of full validation. Accounted retained SVG/raster peak remains 12,607,690 bytes,
excluding some metadata and process RSS. Nominal target is 27,282, reported
target includes the unproven observed floor 46,365, and achieved cost remains
above both. These are completed-prefix measurements, not a matched-effort
speedup or a memory-limit pass.

Next, provide coherent neighboring material and nested coverage competitors
before widening raw path limits further. Diagnose whether hole-bearing families
contain protected alpha holes or separately owned opaque marks that could stay
above a continuing base. Keep genuine holes and source marks intact and prove
the local order and complete footprint under native scoring. The initial closed
operator does not complete richer layer inference, constrained primitive fitting,
the sword gates or any of the eight deliveries.

## Continuing material beneath owned opaque marks

`cel_plan/nested.py` distinguishes enclosed RGB paint from an intentional alpha
hole using source ownership and actual SVG paint. Enclosed primary owners must
be wholly inside the source cavity; a surface that also owns a distant mark is
excluded. Their current geometry, attributes and gradient definitions remain
unchanged. A continuing base records their regions as secondary `covered`
members while their primary ownership remains intact.

Closed overlays fit the filled outer mask and prove containment of every
retained mark. Neighbor restoration excludes those inside owners. The coherent
family operator now offers two independent interpretations: its previous
adjacent union, and a union continuing beneath eligible enclosed marks. The
second can remove hole contours without flattening the marks into base paint.
Both enter the ordinary exact evaluator and independent full checkpoint path.

`cel_plan/layer_order.py` shares the bounded order proof between these operators.
Marks retain their relative order above the base; continued outside material
stays beneath the closed overlay. An unrelated sibling can be crossed only
when actual filled geometry is disjoint. Unsupported stroke/clip/non-path
crossings, actual overlap and exhausted intersection limits exclude the edit.
The affected IDs include retained marks whose order changes, so dependency and
local-render bounds cover them as well as edited geometry.

The first diagnostic used source SHA-256
`f0c63262a51d2807e3ac17f7c687fd5b5d3fc68e158a221d4479b9a793ebd1aa`.
Its 60-second sword run under `.bench/planned-nested-material` retained the
same 22,646-node, 4,026-contour drawing, human MSE 439.75132. No closed or nested
surface candidate reached native scoring. Its overlay exclusions included
three source-alpha mismatches and one alpha hole; restoration still encountered
neighbor-count failures. This result led to a compositing investigation rather
than a weight change or a larger raw-path limit.

The unapplied raw structured export inventory in
`.bench/nested-core-diagnosis.json` inspected 17 hole-bearing groups. None of
those source cavities contains alpha-empty pixels. A jewel-area ring surrounds
960 pixels in 56 owned regions; its parent opacity is 253/255 and source alpha
ranges from 253/255 to 254/255. Its actual child fills are opaque, and the
existing core contains the family. Several guard/jewel families have the same
small source-byte variation. Other handle families exceed the 64-mark bound
and lack a proved core. This is a raw-export diagnostic, not the production
selected drawing or a feature box supplied to generation.

The retained implementation therefore permits source-alpha variation only when
the existing marks' actual paints are opaque and an actual same-group core
geometrically contains the entire old family and the retained marks. The
proposed continued fill also needs its own complete core proof. This preserves
the already rendered alpha through overlap; the native policy still scores
changed edge colors and checks the source. Source alpha emptiness, current
translucent fill/gradient paint, unproved cores and partly enclosed owners stay
excluded. Merely matching an alpha byte or declaring hidden coverage is not a
compositing proof.

Nested discovery allows 64 marks and 6,000 total selected/mark nodes. Family
source crops and continued geometry have separate pixel/node bounds; actual
core geometry is bounded before intersection. The shared order helper retains
the 16-proof, 6,000-node sibling limit. Existing deadline, evaluator, dependency
and independent checkpoint limits apply. Metadata reports nested proposal
counts and specific source ownership, style, alpha/core, geometry and order
exclusions for both operator routes.

The native synthetic tests cover alpha 253/128/64, source label order, retained
opaque flat/gradient marks, faint/translucent marks, true holes, distant members,
unrelated covering paint, source-alpha variation with and without a real core,
discovery bounds, scaled/offset scope and save/reload. Geometry and paint remain
identical for retained marks; opaque interior pixels stay unchanged. Mixed
antialiased edge pixels may change when their underlying material changes, so
the complete native visible context is scored instead of demanding byte
equality across that edge. Local/full score terms and independent checkpoint
pixels agree. On the four-fragment marked fixture, the initial 68 nodes/cost 116
become 58/cost 92 for the adjacent family, 55/cost 85 for its continued version,
40/cost 74 for the nested ellipse, or 38/cost 72 for the nested contour.

The final relevant suite passed **383 tests in 37.39 seconds**. Ruff passed
with 28 formatted files; Pyrefly reported zero source errors and 61 existing
warnings. The final source SHA-256 is
`71eb40b5b1208f1330500672a3bb56f863dbf621f95259b1919e57c8302b4932`.

The follow-up raw-export proof in `.bench/nested-core-full-proof.json` uses this
final source and checks the family **plus the actual retained mark geometries**.
The jewel ring now qualifies with all 56 opaque mark owners and a proved core;
its continued union has 39 nodes. A second jewel-area surface qualifies with
three marks and a 43-node union. An inspected guard surface remains excluded
because a mark's gradient is not opaque. These are availability and geometry
proofs on the raw export, not accepted production proposals or a sword-quality
pass.

The final paired tuning command was:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/rasmus/Workspaces/vectrify/.venv/bin/python scripts/bench_cel_pairs.py \
  --methods cel-planned \
  --cases anime-girl anime-face western-park rubberhose-band \
  --degradations clean --long-side 192 --seconds 20 \
  --method-settings '{"cel-planned":{"complexity":50,"quality":"balanced","refine":false}}' \
  --out .bench/planned-nested-core-pairs
```

| Tuning case | Nodes | Cost | Clean MSE | MSE change from owned-overlay run | Line F1 | Generation seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Anime girl | 2,883 | 3,845 | 889.48 | 0 | 0.678 | 6.22 |
| Anime face | 5,427 | 7,047 | 703.60 | +0.31 | 0.664 | 8.13 |
| Western park | 4,435 | 5,791 | 1,281.92 | -22.20 | 0.670 | 7.11 |
| Rubberhose band | 849 | 1,505 | 966.66 | +2.88 | 0.651 | 8.42 |

All four return ready with complete diagnostic candidate storage and zero
checkpoint disagreements. Search attempts 24/14/21/38 proposals, accepts
20/11/19/31 working alternatives and publishes four independent checkpoints per
case. All four selected drawings contain a continued family; anime face selects
two. Nested family proposal counts are 2/2/3/4. One nested closed contour reaches
evaluation in anime face, but it is not selected; no ellipse is offered on this
subset. The learned branch is still unnecessary to demonstrate these operators'
availability.

Counts and errors move in both directions. Anime girl removes 18 nodes at the
same clean error. Western park improves its raw error while using 140 more
nodes; its face-region error drops by 266.74, but lettering/animal-face errors
rise by 81.05/183.46. Anime face's star-clip error rises by 18.51; Rubberhose's
banjo error rises by 89.32. These are measured tuning regressions to address in
the declared feature/score calibration, not a feature-gate or general-quality
pass. The new pools' same-cost oracle gaps are zero, but their differing
completed prefixes cannot establish that the earlier selection conflict was
calibrated away. No held-out or degradation evaluation was used.

These initial nested interpretations do not complete any of the eight deliveries
or the sword gate. The final native comparison is recorded next.

The final native command was:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/rasmus/Workspaces/vectrify/.venv/bin/python scripts/bench_cel_planned.py \
  --methods cel-planned --seconds 60 \
  --settings '{"complexity":50,"quality":"balanced","refine":false}' \
  --out .bench/planned-nested-core
```

It uses the final source hash above and frozen mask
`f2e692b86e2814f5958c0ca6cc19800a34c12891a527449e68e1624c8bdfe514`.
The selected sword still has 22,646 nodes, 4,026 contours, 3,991 paths, 544
gradients, cost 53,260 and human MSE 439.75132. One -1,304-cost family replacement
survives. Guard/jewel error remains 1,015.14/1,233.07; blade-facet error is 230.02.
It passes only the numerical human-error ceiling, failing the node/contour
targets and the combined structural milestone.

Search attempts seven proposals, accepts six working alternatives and publishes
one independent checkpoint, with zero score disagreements. Overlay discovery
visits 28 groups (ten families/eighteen singles), but no closed or nested family
candidate reaches native exact evaluation. Nested overlay exclusions are three
nonopaque gradients and seven alpha holes; family exclusions include three
alpha holes. Sixteen overlay models lack restoration. The shared restoration
helper records four neighbor-count exclusions, eleven silhouette/hole contacts
and one unproved core; peak neighbor count is 244. These production starting
families differ from the raw-export probe, whose positive proofs must not be
presented as accepted production results.

Pipeline time is 53.77 seconds and operation/apply time is 60.67 seconds.
Reported pipeline overshoot is zero, but operation/apply exceeds the requested
60-second budget by about 0.67 seconds; end-to-end reservation remains a runtime
requirement. Structural search takes 11.99 seconds, including 2.68 seconds of
independent validation. Accounted retained SVG/raster peak is 12,607,690 bytes,
excluding some metadata and process RSS. Nominal representation target is
27,282; the reported target remains 46,365 including an unproven observed floor.
Achieved cost is still above both. None of these timings or storage counts
establishes the broader runtime/memory gates or a matched-effort speedup.

Next provide coherent neighboring material and compact initialization under the
same alpha/feature policy. The raw ring can now pass the ownership/compositing
proof, but the production starting partition still prevents it from reaching
evaluation. Retain exclusions for actual translucent gradient marks and alpha
holes until richer representations are supported. Score/feature calibration is
also required by the measured tuning regressions. These findings support work
on proposal coverage and scheduling before learned selection.

## Coherent material initialization before SVG export

The next bounded proposal operates on original graph atoms before SVG export,
instead of requiring thousands of individual SVG family edits. The new
`cel_plan/materials.py` streams weighted moments of position and premultiplied
RGBA, then compares a flat fit with a linear fit using one shared spatial axis.
Two independent color axes cannot be represented as one SVG gradient and retain
a residual. Discovery grows connected families using fit change and estimated
boundary savings. Supported/unresolved ink ridges, explicit-width atoms,
different components and strong ink-class changes remain barriers. Blocked
contacts propagate through merges, so a weak alternate route cannot erase a
supported ridge.

A broad alpha range requires a linear model explaining at least 98% of alpha
variance; a continuous ramp is eligible and an abrupt opacity step is excluded.
These are proposal checks, not replacements for native coverage validation.
Growth uses an optimistic fit without charging a complete gradient to every
small merge. Charging that activation repeatedly caused a local minimum on a
perfect banded ramp with a hole. Actual export still compares flat and gradient
paints using their price, and the native frontier charges all actual contours,
nodes, paths and gradients. `linear_estimate_families` describes the growth
estimate, not the final paint count.

Discovery caps the graph at 16,384 regions and 65,536 boundaries, and edge-model
evaluations at 32,768. Pixel moments use two-dimensional chunks of at most
65,536 pixels, including unusually wide inputs, with stop checks inside moment
and adjacency loops. The orchestrator reserves a discovery slice, uses the
fixed detailed normalizer and complexity-50 shared-pool context, exports one
optional competitor and validates it before local search. Failed, interrupted
or rejected work retains the independently validated frontier. No human data
enters generation. This initial route runs only on partial-alpha evidence.

Thirteen new cases check independent dense-versus-streamed moments, a shared
gradient axis, band/alpha ramp compaction, holes, original ownership and
save/reload, abrupt alpha steps, supported-ridge alternate routes, explicit
width, discovery bounds, interruption and optional-fit failure. A synthetic
export can select an actual gradient with a suitable fixed normalizer; at a
small fixture's production cost scale it can select a flat instead. Neither
result establishes score calibration on illustration artwork.

The relevant evidence/policy/frontier/operator, planned operation, benchmark,
legacy CEL and shared-boundary suite passed **396 tests in 30.74 seconds**.
Ruff and formatting checks passed. Pyrefly reported zero source errors and 61
existing warnings; project configuration excludes tests from its type-check
scope.

The native comparison command was:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/rasmus/Workspaces/vectrify/.venv/bin/python scripts/bench_cel_planned.py \
  --methods cel-planned --seconds 60 \
  --settings '{"complexity":50,"quality":"balanced","refine":false}' \
  --out .bench/planned-coherent-materials
```

Source SHA-256 is
`99786f0f15a3aa97ba737959b6c2f962fd1ddd9a2a1ad1b972394c8752874c2b`,
with the unchanged frozen mask
`f2e692b86e2814f5958c0ca6cc19800a34c12891a527449e68e1624c8bdfe514`.
The material route completes 2,933 merges and 14,931 edge-model evaluations
from a graph of 8,502 atoms, leaving 5,214 visible result regions. Source graph
counts include hidden atoms; subtracting merges from the total is therefore not
the visible count. It records 4,133 alpha exclusions, 2,345 linear-model family
estimates, 255 ridge proofs (252 supported, three shade interpretations) and
6,556 unresolved boundary encounters. Discovery reaches neither graph nor model
caps. Total proposal/export/validation time is 6.99 seconds, including 2.36
seconds of validation.

The native policy rejects this whole-drawing candidate for
`translucent-component-lost`. The operation benchmark itself does not retain
rejected SVGs. A follow-up observer capture is recorded below; its rejected
counts must not be treated as an accepted result.

Selected output remains **22,646 nodes, 4,026 contours, 3,991 paths, 544
gradients, cost 53,260 and human MSE 439.75132**, pixel-identical to the prior
nested-core run. Guard/jewel MSE remains 1,015.14/1,233.07. It passes the
numerical human-error ceiling but fails the structural targets and combined
milestone. The nominal cost target is 27,282 and the reported target 46,365
including the unproven observed floor; achieved cost exceeds both.

Pipeline time is 53.13 seconds and operation/apply time 57.61 seconds, within
this requested 60-second run. This single run does not resolve runtime
reservation across workloads or establish a speedup: completed candidate
prefixes differ. Local search attempts five proposals, accepts four working
alternatives and publishes one checkpoint, with zero score disagreements.
Accounted retained SVG/raster peak is 11,660,344 bytes, not total process RSS.
Refinement is disabled, so the result cannot establish automatic fitting's
benefit.

A diagnostic-only rerun with the same source/settings and 60-second budget
captures the material SVG, then stops after its native validation. Artifacts
are in `.bench/planned-coherent-materials-diagnosis`, with the capture command
saved as `.bench/diagnose-coherent-materials.py`. Discovery reproduces the same
merge/model/alpha/ridge counts. The rejected SVG key is
`820271c4b4c07a6ca520851634f96acdbe576091d3315cc0e1e2f1334dad9469`.
It has 29,540 nodes, 5,275 contours, 5,215 paths, 34 gradients and cost 61,478.
The estimated 2,345 linear families therefore produce only 34 actual gradients.
Its cost is below the conservative fallback's 103,482 but above the validated
detailed seed's 54,564 and selected drawing's 53,260. Source-atom merge counts
cannot stand in for improvement over an already merged SVG competitor.

Independent native alpha checks identify eight failing components totaling 88
pixels. None has an eroded core; sizes range from four to 21 pixels and peak
alpha from 2/255 to 5/255. Their retained alpha mass ranges from 61.54% to 93.75%,
below the existing 95% thin-component requirement. The first occupies
`[361,74,365,77]` in source coordinates and retains 61.54%. The diagnostic uses
the policy's summed-mass comparison, rather than comparing a rounded ratio at
the 95% boundary. These measurements establish thin/faint coverage loss here;
they do not authorize deleting those marks, identify its paint-versus-geometry
cause, or prove the route would be selected after coverage repair. Full renders
and per-component boxes/masses are retained in the diagnostic bundle. This run
deliberately stops early and is not an end-to-end performance comparison.

The opaque tuning controls were rerun using:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/rasmus/Workspaces/vectrify/.venv/bin/python scripts/bench_cel_pairs.py \
  --methods cel-planned \
  --cases anime-girl anime-face western-park rubberhose-band \
  --degradations clean --long-side 192 --seconds 20 \
  --method-settings '{"cel-planned":{"complexity":50,"quality":"balanced","refine":false}}' \
  --out .bench/planned-coherent-materials-pairs
```

| Opaque tuning control | Nodes | Cost | Clean MSE | Generation seconds |
| --- | ---: | ---: | ---: | ---: |
| Anime girl | 2,883 | 3,845 | 889.48 | 5.43 |
| Anime face | 5,427 | 7,047 | 703.60 | 7.18 |
| Western park | 4,435 | 5,791 | 1,281.92 | 6.43 |
| Rubberhose band | 842 | 1,492 | 966.05 | 7.25 |

All four report material initialization unavailable because the input is opaque;
they are controls, not real-artwork tests of the new RGBA route. The first three
selected counts/errors match the previous run. Rubberhose removes seven nodes
and lowers MSE by 0.61 with a longer completed search prefix; this cannot be
attributed to the new material route. Search attempts 35/15/25/45 proposals,
accepts 30/12/23/38 working alternatives and publishes four checkpoints each,
with zero score disagreements. Diagnostic pools are complete without omissions.
Same-cost oracle gaps are zero; unconstrained gaps remain
22.66/45.09/34.35/79.93. No score calibration, held-out, degradation or blind
review gate is claimed.

Next distinguish paint and geometry causes of the native alpha rejection,
offer component/family competitors with preserved validated coverage, and add
real partial-alpha tuning cases. Do not relax the retained-alpha gate or use a
ranker to admit the rejected drawing. This proposal foundation completes none
of the eight deliveries by itself.

## Screening thin paint and growing from the detailed checkpoint

The rejected material SVG established a paint mismatch. Growth could fit a
linear alpha model across 1–5-byte marks, but cost-aware export could select a
median flat instead. An initial analytic screen still rejected the sword:
Cairo renders a 1.5-byte flat alpha as one byte, whereas ordinary numerical
rounding predicted two. That intermediate run is retained in
`.bench/planned-coherent-checkpoint` (source
`d4720fed2e58674063192ffedcd72b6e8277055b71957704eca112ca141a4c03`).
It restored nine families but did not repair native coverage.

`materials.retain_thin_paint` now screens the actual fitted flat/linear paint
with a native SVG rectangle in its analysis-coordinate frame. Each rectangle
has at most 8,192 pixels; a larger bound restores source atoms instead of
starting an oversized render. Thin families whose paint loses more than 5% of
source alpha mass, or exceeds the existing opacity allowance, return to their
original complete atoms. Supported uniform and gradient marks still compact.
The screen checks paint only; it cannot prove geometry, transformed native
coverage or compositing. Full candidate validation remains authoritative, and
stop is checked between fitting/render calls.

The optional seed also grows from the **validated detailed partition** rather
than rebuilding a denser alternative from original atoms. Its graph statistics
are recomputed, but exported ownership still names original evidence atoms.
Repair can therefore restore atoms inside a coarser starting cohort without
splitting or duplicating their primary ownership. A rejected/failed discovery
leaves the detailed checkpoint available. No production score weights changed.

Six additional cases cover faint-paint loss, native half-byte rounding,
supported thin flat/gradient compaction, bounds/stop and growth from a coarser
checkpoint with original owners. The expanded relevant suite passed **410
tests in 45.15 seconds**. Ruff/formatting passed; Pyrefly reported zero source
errors and 61 existing warnings.

Final source SHA-256 is
`e28f16a2013b9e7b647599aa082efad32dc01244fe9b818c99a297ec89abb354`.
The native comparison command was:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/rasmus/Workspaces/vectrify/.venv/bin/python scripts/bench_cel_planned.py \
  --methods cel-planned --seconds 60 \
  --settings '{"complexity":50,"quality":"balanced","refine":false}' \
  --out .bench/planned-coherent-native-paint
```

The frozen mask is unchanged. Material discovery starts with 4,472 graph atoms
(including hidden atoms), completes 888 merges/3,990 edge-model evaluations and
records 2,244 alpha exclusions. It checks 198 thin paints and restores 25
families, leaving 3,256 visible regions versus 3,229 before paint repair. No
paint rectangle hits the pixel bound. The seed passes full native validation
with cost 40,096; subsequent local search selects cost 38,868. Seed stage time
is 4.97 seconds, including 1.59 seconds of full validation.

| Native selected result | Previous material run | Native paint screen |
| --- | ---: | ---: |
| Nodes | 22,646 | 19,528 |
| Contours | 4,026 | 3,176 |
| Paths | 3,991 | 3,138 |
| Gradients | 544 | 30 |
| Representation cost | 53,260 | 38,868 |
| Human MSE | 439.75 | 472.73 |

Cost falls by about 27%, but human error rises by 32.98. Blade tip/facets,
guard/wrapping/jewel MSE becomes 580.57/241.84/1,106.97/680.73/1,370.95;
all five regress. The result passes only the numerical human-error ceiling,
failing the ≤800-node/≤140-contour targets and combined milestone. Visual
inspection of the jewel crop shows fragmented shading and an irregular ring
compared with the clean human outline. The current score's cost/error tradeoff
is not a human-quality pass.

Pipeline time is 51.33 seconds; operation/apply takes 54.77 seconds within this
60-second run. No speedup at equal quality or completed proposal effort is
claimed. Search attempts 12 proposals, accepts seven working alternatives and
publishes one independent checkpoint with zero score disagreements. Accounted
retained SVG/raster peak is 10,107,827 bytes, not process RSS. The observed floor
and reported target fall to 38,868, so the clamped report says unmet false;
the **nominal 27,282 target is still exceeded**, and the floor is unproven.
Do not interpret that report as achieving the desired nominal budget.

Remaining native constraints include 4,809 alpha-step and 3,710
transparent-contact chain encounters, with 5,715/5,466 emitted segments.
There are 3,027 geometry-constrained paths. These overlapping counts still
point toward alpha-fringe/geometry interpretations rather than treating every
alpha change as a distinct surface. No closed contour or ellipse reaches
evaluation in this prefix. Six nested overlays are excluded (four mark
style/frame, two alpha holes); restoration records four neighbor-count and one
silhouette/hole exclusions. One filled-ink proposal becomes available.

The paired benchmark now has version 2 and `--composition-opacity` (default
one). It isolates the repository-authored fixtures' rendering children so the
opacity multiplier applies once, including overlaps; definitions, holes and
root attributes remain. Default one preserves exact original SVG bytes. Both
clean and input renders derive from that target variant before deterministic
input corruption. Reported variant/opacity fields and distinct artifact paths
prevent confusing it with alpha degradation. Variants inherit the original
family and split, and add no independent artworks to the corpus.

New tests verify composition opacity, overlapping paint, existing root opacity,
gradient definitions, holes, invalid values, actual pair construction and that
clean pixels/geometry still never reach the generator. The final RGBA tuning
command was:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/rasmus/Workspaces/vectrify/.venv/bin/python scripts/bench_cel_pairs.py \
  --methods cel-planned \
  --cases anime-girl anime-face western-park rubberhose-band \
  --degradations clean --composition-opacity 0.5 --long-side 192 --seconds 20 \
  --method-settings '{"cel-planned":{"complexity":50,"quality":"balanced","refine":false}}' \
  --out .bench/planned-coherent-native-paint-rgba-pairs
```

A previous-production control was run from an isolated archive of `ce89446`,
backporting only the same two version-2 paired benchmark scripts to construct
identical targets. Its actual source hash is
`c1535dd17d39210fc3a5b36c8ceb51a717cba7da9de501c61c85927ad5a5d06d`;
results are in `.bench/planned-coherent-checkpoint-rgba-pairs-before`.
The archive is not a Git checkout, so its report's revision is unavailable;
the source hash identifies the executed combination. Input/clean SVG/clean
pixel/mask hashes, families, splits, dimensions, settings and deadlines match
the final run. Completed search prefixes differ; these are matched-deadline
controls, not identical-pool or equal-quality speed comparisons.

| Half-opacity tuning case | Previous nodes/cost | New nodes/cost | Previous → new clean MSE | New line F1 | New generation seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Anime girl | 266 / 454 | 175 / 319 | 139.48 → 177.70 | 0.270 | 8.83 |
| Anime face | 433 / 825 | 176 / 272 | 133.69 → 206.86 | 0.600 | 15.93 |
| Western park | 452 / 834 | 532 / 936 | 195.44 → 193.63 | 0.652 | 8.73 |
| Rubberhose band | 527 / 895 | 407 / 765 | 162.90 → 167.23 | 0.641 | 10.36 |

All four new material seeds are retained, with 29/30/26/43 merges from
50/46/92/88 starting atoms. These opaque-canvas compositions have a translucent
uniform background and no thin component requiring paint repair; they test
material growth and selection, while synthetic/native sword cases test repair.
New search reaches 48 evaluations each, accepts 14/22/22/21 alternatives and
publishes four checkpoints per case, without score disagreements. Every
diagnostic pool is complete with zero omissions. The earlier control reaches
48/38/37/25 evaluations, so attribution cannot ignore scheduling.

Three cases regress in global clean error. Anime girl's bag-clasp MSE rises by
182.10 despite improvements in eyes/bow/hairclip. Anime face's eyes/necklace/
star-clip errors rise by 166.93/313.67/366.77; its inspected eye crop loses iris
and highlight structure. Rubberhose notes rise by 105.47 and left-eye error by
31.34. Western park's reported feature errors improve. Same-cost oracle gaps
are 15.60/0.25/0/0.39; unconstrained gaps are
159.41/190.68/165.04/134.03 and favor much denser conservative drawings.
These are explicit score/proposal-quality conflicts, not calibrated-away
regressions or a broader feature gate pass. No held-out or blind-review data
was used.

Next produce faithful alpha-fringe/coherent-surface interpretations and preserve
small ink details; prepare the declared same-pool calibration replay using
these selection conflicts. Native validation establishes safety and exact
agreement, not sufficient resemblance. None of the eight deliveries is complete.

## Reset after comparison with legacy CEL

The owner challenged the practical gain over baseline. The appropriate
comparison is 19,528 nodes versus legacy CEL's 2,312 and the human fixture's
523, not only the previous 22,646-node development result. The latest global
human-reference MSE is approximately 29% lower than legacy's, but geometry is
8.45 times larger. The most recent compaction also regresses all five feature
crops. This is not the intended overhaul or a release-quality improvement.

A source-only coverage experiment decomposed connected native opacity into
15 nested envelope bands for the main component, and used legacy CEL
(`regions=50`, `tolerance=1`, filled ink) on its near-modal opaque paint core.
Other faint/unsupported components retained native pixel-level alpha paths.
The generator never received the human geometry. This is an isolated diagnostic,
without complete planning ownership, deadline/memory proofs or integration.

Artifacts are in `.bench/alpha-envelope-diagnosis`, with the experiment source
in `.bench/diagnose-alpha-envelope.py`. Executed production source hash is
`e28f16a2013b9e7b647599aa082efad32dc01244fe9b818c99a297ec89abb354`, and
the diagnostic script hash is
`13ebde101942a8db758538bc5e72b761eaa6cbc25061e7292b569f1e57f22eeb`.
The final diagnostic retains faint components; an earlier interrupted loop and
two unsuccessful construction attempts did not yield selected drawings.

The resulting 273 paths, 1,294 contours, 6,769 nodes and nine gradients cost
12,599. Human MSE is 645.40, failing the 497.39 ceiling, with tip/facets/guard/
wrapping/jewel errors 901.88/496.42/1,087.05/677.12/933.32. Native policy rejects
five self-crossings, excess opacity and a lost protected hole. The inspected
jewel remains irregular and its surrounding shades fragmented. Reducing alpha
partitions alone, then tracing the interior as before, is insufficient. This
result is not published through the operation or treated as an accepted proposal.

### Native admission audit

`.bench/diagnose-human-validation.py` reconstructed source evidence and graph
with default options and refinement disabled. It exported and validated the
actual conservative RGBA fallback, established that baseline, then evaluated
the human, legacy and latest development SVGs only as benchmark diagnostics.
The baseline has 54,320 nodes and zero missing/excess opacity pixels. The fixed
native pixel allowance is 85. Its score uses source graph features and ink,
matching production policy construction; a preliminary unbased audit did not
include those fields and is retained separately.

The authoritative report is `.bench/human-native-validation-baseline.json`.
Its source hash is
`672cc945d986665e6d75a683e324228d121bf7a049b1b90c9ffdcc0afdbbef77`.
This differs from the envelope/source comparison because only frontier budget
reporting changed before the audit; the admission policy and drawing algorithms
were unchanged.

| Diagnostic drawing | Missing / excess opacity pixels | Self-crossings | Native rejection |
| --- | ---: | ---: | --- |
| Human | 1,234 / 86 | 2 | Crossings, translucent gap/excess, protected hole and component loss |
| Legacy CEL | 833 / 279 | 0 | Translucent gap/excess and component loss |
| Latest development result | 80 / 38 | 0 | None |

This does not establish that each rejected human/legacy discrepancy is visually
acceptable or that the target is mathematically unattainable. It establishes
that the current safeguards exclude the exact human drawing that motivates
the requested abstraction. Locate and classify the failures before altering
the policy. Tiny/faint source components cannot be presumed intentional or
discardable solely from their area or alpha; true faint marks and meaningful
holes need explicit regression coverage.

With the latest detailed normalizer 54,564 at complexity 50, the audited
human objective would be approximately 0.04627 versus 0.06698 for the latest
development drawing. The fixed score can prefer that human drawing, but hard
admission prevents its consideration. A learned ranker or reweighting does not
remove this feasibility conflict. These values are diagnostic evaluations,
not generated candidates, training examples or release gate passes.

### Explicit nominal budget shortfalls

Selected drawings and alternatives now share one budget-reporting helper.
`representation_budget` adds `nominal_unmet`, `nominal_overrun` and
`search_floor_clamped`. Existing effective `target`/`unmet` semantics and the
separate explicit `node_budget` flag remain. An observed floor may clamp the
effective target, but cannot hide an unmet nominal slider target. No selection,
score weight, admission check or geometry changed.

Regression coverage exercises all 101 slider positions, selected/alternative
consistency, a clamped target with a genuine nominal shortfall, and a detailed
target without a shortfall. The targeted frontier and operation suite passes
35 tests; Ruff lint and formatting pass. Existing drawing benchmark numbers
are not presented as fresh results from this metadata-only change.

The plan now prioritizes admission compatibility, direct structural proposals
before palette/alpha segmentation, then joint coverage/geometry/paint fitting.
The sword gate and broader release criteria remain unchanged. Full delivery
remains open.

## Material coverage and an ink-aware silhouette competitor

The admission audit was localized before changing production policy. The
source's 69 raw alpha holes are mostly narrow gaps in weak perimeter coverage;
the eight holes lost by the human are of that kind. The 25 omitted native
components total approximately 2.35 fully opaque pixels of alpha mass. Two
human self-crossings belong to filled shade shapes near the lower guard. These
observations explain the conflicts, but do not justify deleting every faint
mark or permitting arbitrary crossings.

`coverage.py` now derives material interiors from each component's modal alpha
and persistent holes from the half-modal support. A coherent weak enclosing
plateau retains an intentional hole even when attached to a stronger body.
Source color/alpha scoring still includes the entire fringe. Independent
component mass retention, opaque interior, exterior spill and crossing checks
remain. Reduced-opacity holes gain their own source-derived ceilings, and
local/native evaluation uses the same fixed supports and component retention.
The coverage interpretation is version 1; the admission/score version is now
4. These development thresholds still require broader tuning and freezing.

`core_materials.py` offers a competing fitted silhouette for a native-scale,
near-uniform material. It accepts complete original source atoms, retains
unsupported opacity and thin components, and records fringe ownership separately
from secondary core coverage. The new evidence interpretation does not change
the original pixels, labels or atom namespace. A promoted primary base's
secondary coverage can support a replacement only when actual transformed
geometry proves containment. Native independent validation admits the candidate;
ownership metadata alone is insufficient.

The hypothesis is bounded to 1,536² native analysis pixels and eight material
components. Variable intrinsic alpha, protected partial-opacity holes, resized
analysis and explicit filled width retain the existing interpretations. This
does not implement joint coverage/geometry/paint fitting or primitive fitting.

### Native sword comparisons

All runs use complexity 50, balanced quality, refinement disabled, the same
native mask and a 60-second generation limit. They complete different candidate
pools and do not establish equal-quality speedups.

| Drawing / control | Nodes | Contours | Human MSE | Generation seconds |
| --- | ---: | ---: | ---: | ---: |
| Legacy CEL baseline | 2,312 | 339 | 663.31 | See frozen baseline |
| Previous coherent native-paint search | 19,528 | 3,176 | 472.73 | 60-second limit |
| Coverage v4 with core initialization disabled | 19,105 | 3,078 | 472.70 | 57.87 |
| Initial color-only material silhouette | 3,818 | 296 | 683.61 | 58.43 |
| Ink-aware material silhouette, separate export budget | 12,618 | 2,229 | 517.60 | 50.60 |
| Frozen development gate | ≤800 | ≤140 | ≤497.39 | Balanced quality |

The color-only run and disabled-core control use source hash
`6bf4b7877b8d22b9b1169cae610b44ae4fa8286fc1dcdc2d79cf4eb2d77cb315`.
Artifacts are `.bench/planned-material-silhouettes` and
`.bench/planned-persistent-alpha-only`; the latter records its no-op monkeypatch
and diagnostic script hash. The large compaction worsens all five human feature
crops and erases much of the jewel's dark rim. It is a rejected development
approach, despite passing native policy.

The ink-aware run uses source hash
`599c4758f6030d858830f497d4c65e63abf2c5c9283525cd85422015d654509c`
and artifacts `.bench/planned-material-silhouettes-ink-live-export`. It retains
the jewel rim but still produces patchy surrounding paint and jagged blade
shading. Tip/facets/guard/wrapping/jewel human errors are
542.09/259.79/1,164.70/896.53/1,308.24. It costs 26,176 and has zero
self-crossings or native policy rejections. The core model uses alpha 253/255,
2,608 core atoms and 5,240 fringe atoms. Ink-aware growth performs 762 merges,
but 2,497 ridge encounters remain unresolved. These counts are not unique
semantic edges. The candidate is still 5.46 times the legacy node count and
fails every combined sword milestone; this is not a practical baseline win.

An earlier ink-aware run discarded the optional candidate after its growth
deadline also expired the export budget. It selected 20,078 nodes and human
MSE 472.74; source hash
`3aa6dee88ef65b66d092b183c7ac41a76fdd014b0d369d50a368cdf735415998`,
artifacts `.bench/planned-material-silhouettes-ink`. Growth now has a separate
bounded slice: a complete partition may export under the remaining search
budget. Stop still discards it. Tests cover both cases.

The first color-only operation crashed on an optional native curve boolean.
The scheduler now catches only `pathops.PathOpsError`, records the failed
operator cursor and continues other operators from validated states. Six
regressions exercise each operator slot; programming exceptions remain visible.
The completed color-only run records four such failures. Its initial crashed
attempt did not produce a benchmark drawing.

### Updated admission audit

`.bench/human-native-validation-material-support.json` repeats the native audit
with the original independently validated conservative baseline. Its source
hash is
`7868624b1c222ac1b5276293c185e14a9aef3e8b4b6befc46a704c0f538026f6`;
the only source change after the successful ink-aware sword run adds a return
type annotation. The policy has 148,322 material-interior pixels and no
protected holes of at least four pixels on this sword. The 85-pixel allowance
and original 26 component supports remain.

| Diagnostic | Missing / excess opacity pixels | Remaining rejection |
| --- | ---: | --- |
| Human | 14 / 0 | Two crossings and faint component loss |
| Legacy CEL | 8 / 0 | Faint component loss |
| Ink-aware material candidate | 0 / 0 | None |

This resolves the fringe/hole admission conflict without granting the human
fixture an exception. The human is still diagnostic input only. Synthetic
regressions preserve clear/reduced-opacity holes, weak attached rings, faint
independent marks, opacity steps, closed dark ink and full/local agreement.
Passing admission does not establish likeness, and the human drawing remains
inadmissible under the unchanged crossing/component rules.

The next implementation priority is whole contours and coherent shade surfaces
across palette fragments, with bounded restoration of underlying paint. Diagnose
candidate availability and selection at the frozen cost/feature gates. A dense
trace's nominal complexity target is too weak to serve as a product milestone.
No UI rollout, learned-model benefit or delivery completion is claimed here.

### Translucent tuning controls

The same four half-opacity, clean tuning variants were rerun at a 192-pixel long
side, complexity 50, balanced quality, refinement disabled and a 20-second limit:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/rasmus/Workspaces/vectrify/.venv/bin/python scripts/bench_cel_pairs.py \
  --methods cel-planned \
  --cases anime-girl anime-face western-park rubberhose-band \
  --degradations clean --composition-opacity 0.5 --long-side 192 --seconds 20 \
  --method-settings '{"cel-planned":{"complexity":50,"quality":"balanced","refine":false}}' \
  --out .bench/planned-material-silhouettes-rgba-pairs
```

The summary records source hash
`7868624b1c222ac1b5276293c185e14a9aef3e8b4b6befc46a704c0f538026f6`.
Input, clean SVG, clean pixel and mask hashes match
`.bench/planned-coherent-native-paint-rgba-pairs`. All four selected PNGs are
byte-identical to that previous production run, although candidate keys/pools
change. The new core candidates are retained, but do not improve selection.

| Tuning variant | Nodes / cost | Clean MSE | Line F1 | Generation seconds |
| --- | ---: | ---: | ---: | ---: |
| Anime girl | 175 / 319 | 177.70 | 0.270 | 11.60 |
| Anime face | 176 / 272 | 206.86 | 0.600 | 17.97 |
| Western park | 532 / 936 | 193.63 | 0.652 | 10.19 |
| Rubberhose band | 407 / 765 | 167.23 | 0.641 | 13.41 |

Every run reaches 48 local evaluations and retains a complete diagnostic pool,
without score disagreements. Same-cost clean oracle gaps remain
15.60/0.25/0/0.39; the existing feature and line shortcomings remain. These
controls establish neither a quality improvement nor independent corpus
expansion. No held-out artwork or blind review was used.

The focused planner, operation and benchmark suite passes 370 tests, including
the new source-coverage, ownership, closed-ink, deadline/stop and native-boolean
regressions. Ruff lint/format checks pass; Pyrefly reports zero errors with 61
existing warnings. A separate legacy CEL/shared fitting/generation regression
batch passes 112 tests, for 482 relevant tests across the two batches. Numerical
acceptance and these tests do not complete delivery.

## Owned ink, short shade contacts and closed-mark checkpoints

The preceding material candidate still depended on thousands of partitions.
This work removes three specific proposal/publication barriers without changing
the native admission policy, score weights, detailed normalizer or release
targets. All changes remain in the experimental method.

### Evidence and implementation

The source-only audit in `.bench/material-boundary-audit.json` reconstructs the
committed evidence and graph at source hash
`7868624b1c222ac1b5276293c185e14a9aef3e8b4b6befc46a704c0f538026f6`.
The core-material adjacency has 2,497 short high-line-support contacts and 211
testable long contacts. Every such contact joins two ink-majority atoms. These
are graph contacts, not unique semantic edges; coarse line support alone cannot
distinguish a continuous mark's internal palette divisions from its boundary.

`boundary_evidence.py` now requires complete monotone source cross-sections
before a short contact permits a competing shade merge. Dark troughs, pale
marks, empty support and large opacity discontinuities keep the contact
protected. At most 64 segment profiles are sampled per proof, with a four-pixel
reach. `families.py` caps short proofs at 4,096 and permits compatible internal
dark-ink contacts only when both complete, bounded native atom samples agree
with their paint models. Each sample rectangle is capped at 65,536 pixels;
fixed-width atoms keep their existing protection. Interrupted proofs are never
cached as successful. Eligibility does not replace native geometry/paint
acceptance or complete source ownership.

The closed-mark availability diagnosis at source hash
`9f68962db83f575966603f1b40a5a0e683aac8833e973c8c8bc0b2abd3b22424`
finds a source-backed ring with 76 nodes, a 239-pixel owned rim, a 952-pixel RGB
cavity and fit residual 1.7293 against a 2.125 tolerance. Its 35 interior paths
are geometrically contained in the proposed ellipse. Direct proposal generation
exhausts the previous 16 individual order proofs. This is an order-proof barrier,
not an ellipse containment failure. Other larger cavities protrude outside
their material base and remain correctly excluded. Diagnostic artifacts include
`.bench/closed-material-ellipse-availability.json`,
`.bench/closed-ellipse-containment.json` and
`.bench/closed-ellipse-direct-proposals.json`. Human geometry is not an operator
input, and the generator has no sword-specific region rules.

`layer_order.py` now proves the union of objects actually crossing each
unrelated child disjoint once, reusing that proof for the remaining crossings.
Stationary enclosing surfaces are excluded from that child's crossing union.
This preserves the strict geometric disjointness requirement; overlaps still
require individual proofs and are rejected when real. The independent bounds
are 64 group proofs, 16 fallback pair proofs and 6,000 moving geometry nodes.
The caches are local to one immutable order operation and stop is checked
between native unions. Native tests cover transformed half-opacity scenes,
more than 16 moved marks, actual overlaps and unchanged rendered pixels.

`overlays.py` prioritizes bounded, source-backed RGB cavities ahead of ordinary
complex patches. Per-cavity inspection is capped at 65,536 pixels and aggregate
inspection at 262,144 pixels. A cavity is not treated as an alpha hole or an
admission exception. All restoration, core support, containment, topology and
order checks remain.

`search.py` accepts a separate live checkpoint budget from `pipeline.py`.
Useful local states can receive independent full validation after their local
discovery slice expires, using remaining global search time. The pipeline's
final validation/fitting reserve remains separate. Explicit cancellation still
prevents publication, and observed full-validation time limits subsequent
checkpoint attempts. Tests cover phase expiration and shared cancellation.

### Native sword experiments

All rows use complexity 50, balanced quality, refinement disabled and a
60-second requested limit. Times are actual generation times, not matched
completed proposal effort. Each source stayed fixed during its benchmark.

| Experiment | Nodes / contours | Human MSE | Seconds | Full local checkpoints |
| --- | ---: | ---: | ---: | ---: |
| Prior ink-aware material candidate | 12,618 / 2,229 | 517.60 | See preceding record | 0 |
| Monotone short contacts | 11,086 / 1,917 | 513.11 | 54.18 | 0 |
| Compatible internal ink | 10,286 / 1,711 | 512.81 | 54.30 | 0 |
| First group-order proofs and cavity priority | 10,286 / 1,711 | 512.81 | 59.37 | 0 |
| Actual crossing unions and live checkpoints | 10,121 / 1,660 | 514.67 | 51.17 | 4 |
| Legacy operation baseline | 2,312 / 339 | 663.31 | See baseline record | — |
| Human fixture | 523 / 93 | 0 | — | — |
| Frozen development gate | ≤800 / ≤140 | ≤497.39 | — | — |

Artifact directories and source hashes, in experiment order:

- `.bench/planned-monotone-fragments`:
  `5fdcc619e925c5e9c3bafbc3663c3f5b51505f7532ed20a366d2063cd5143e85`.
- `.bench/planned-compatible-ink`:
  `9f68962db83f575966603f1b40a5a0e683aac8833e973c8c8bc0b2abd3b22424`.
- `.bench/planned-closed-mark-order`:
  `038af0b9f6071b72f393760b8a0e203117fb2a5c3ad73696387cb03a467edae1`.
- `.bench/planned-closed-mark-checkpoints`:
  `5b4c010af121892dd76aee987788fea89ecb30c281672d43f49a07cf8c386d29`.

Reproduce the final operation/export comparison with:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/rasmus/Workspaces/vectrify/.venv/bin/python scripts/bench_cel_planned.py \
  --methods cel-planned --seconds 60 \
  --settings '{"complexity":50,"quality":"balanced","refine":false}' \
  --out .bench/planned-closed-mark-checkpoints
```

The final source has 1,281 core-material merges, 1,657 visible regions, 432
compatible internal-ink contacts, 148 proven short shade contacts and 1,917
protected short contacts. No encountered short contact remains unresolved.
Local search reaches five evaluations, accepts four states and fully checks
four checkpoints, with zero local/full score disagreements. Closed overlays
emit one nested ellipse, with 25 group proofs, 519 proof reuses and no order
proof limit. The isolated closed-overlay state saves 55 representation cost
and passes local checks; its full checkpoint is dominated. It is not part of
the selected drawing. The selected filled-ink replacement saves 471 cost and
slightly improves native visual score. Full frontier cost is 20,347, with 31
gradients, zero crossings and no native policy rejections.

The final human tip/facets/guard/wrapping/jewel errors are
560.66/254.12/1,166.23/882.61/1,332.64. The jewel crop remains patchy and its
human error worsens from the compatible-ink initialization. All three numerical
sword targets still fail; meeting the nominal 27,282 slider budget is not a
product quality win. The result has 4.38 times the legacy node count and more
than twelve times the gate's node limit. `refinement_complete` remains false.

### Final-source translucent controls

The four existing half-opacity tuning variants use the same input/clean/mask
hashes as `.bench/planned-material-silhouettes-rgba-pairs`. The summary in
`.bench/planned-closed-mark-checkpoints-rgba-pairs` records final source hash
`5b4c010af121892dd76aee987788fea89ecb30c281672d43f49a07cf8c386d29`.
Reproduce with:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/rasmus/Workspaces/vectrify/.venv/bin/python scripts/bench_cel_pairs.py \
  --methods cel-planned \
  --cases anime-girl anime-face western-park rubberhose-band \
  --degradations clean --composition-opacity 0.5 --long-side 192 --seconds 20 \
  --method-settings '{"cel-planned":{"complexity":50,"quality":"balanced","refine":false}}' \
  --out .bench/planned-closed-mark-checkpoints-rgba-pairs
```

| Tuning variant | Nodes / cost | Clean MSE | Line F1 | Generation seconds |
| --- | ---: | ---: | ---: | ---: |
| Anime girl | 175 / 319 | 177.70 | 0.270 | 9.49 |
| Anime face | 176 / 272 | 206.86 | 0.600 | 16.06 |
| Western park | 547 / 987 | 178.72 | 0.652 | 8.62 |
| Rubberhose band | 407 / 765 | 167.23 | 0.641 | 11.17 |

Selected PNGs remain byte-identical for anime girl, anime face and rubberhose
band. Western park changes from 532 nodes/cost 936 and MSE 193.63. Lettering
error improves from 684.38 to 483.85, ball from 258.69 to 254.46 and face from
522.42 to 521.55; animal-face error is unchanged. No measured feature worsens.
Its line F1 is unchanged. All four reach 48 local evaluations and four full
checkpoints, retain complete diagnostic pools and have zero score disagreements.
Same-cost clean-oracle gaps remain 15.60/0.25/0/0.39. This small tuning benefit
does not resolve existing feature/line failures, expand the corpus or pass
held-out/release evaluation. Runtime differences are observations, not a
separate matched-effort performance claim.

### Verification

The full relevant planner, legacy CEL, shared fitting, operation and benchmark
suite passes 503 tests in 47.47 seconds. A subsequent focused run passes all
five layer-order tests, including an added regression where a stationary outer
surface overlaps an unrelated child but the moving inner marks are disjoint.
That brings coverage to 504 distinct relevant tests across these runs; no
production source changed after the benchmark or full suite.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/rasmus/Workspaces/vectrify/.venv/bin/python -m pytest -q \
  tests/refine/test_cel*.py tests/refine/test_shared.py \
  tests/refine/test_simplify.py tests/refine/test_snap.py \
  tests/operations/test_cel_planned.py tests/operations/test_generate.py \
  tests/test_bench_cel_planned.py tests/test_cel_pairs.py tests/test_bench_cel_pairs.py
```

Ruff lint/format and `git diff --check` pass. Pyrefly, explicitly using the
workspace virtualenv interpreter, reports zero errors and 61 existing warnings.
The final production source hash still matches both final benchmark summaries.
These checks validate the implementation slice, not completion of a delivery.

### Next structural experiment

Couple compact closed contours with coherent enclosed paint and retained
highlights; propose long supported facet boundaries with their paint models.
Measure candidate availability separately from selection and whether useful
edits can compose within the beam's bounded time, rather than remaining isolated
alternatives. Keep complete atom ownership and native local/full checks. The
current fixes prove that one ellipse can reach evaluation, not that the pool
contains a faithful compact sword. A learned ranker and slider UI remain
dependent on that missing structural milestone.

## Coupled enclosed material and cumulative search

`enclosed_paint.py` extends a proved closed-overlay proposal with a competing
interior material. The largest eligible owner supplies the initial flat or
linear RGBA model. Every analyzed source pixel of each eligible whole owner is
checked against it; an owner with a strong unexplained residual stays separate.
Fixed source atoms and constrained paint remain separate even when their color
matches. Complete ownership is retained; no majority mask can claim an entire
source atom. The operation has no human fixture or sword-specific region input.

The material proposes a compact ellipse or fitted contour from the cavity's
source boundary, with flat/gradient paint alternatives. Actual retained mark
geometry must lie inside it, the material must lie inside the proposed rim, and
its full new footprint must have an actual opaque core when RGBA requires one.
Retained mark geometry and paint remain unchanged. Secondary covered membership
is explicit and local order still uses strict disjointness proofs. A composite
candidate is independently scored before it can become a published checkpoint.

### Source availability diagnosis

`.bench/enclosed-paint-inventory.json` at source hash
`5b4c010af121892dd76aee987788fea89ecb30c281672d43f49a07cf8c386d29`
finds 35 interior owners in the source-backed jewel cavity. The largest owns
705 pixels across 16 original atoms; a shadow owner has 107 pixels and two
highlight owners have 35 and 33 pixels. This identifies a real whole-material
opportunity rather than a contour-only replacement.

The first native combined-material run emits no combined candidate and retains
the previous 10,121-node drawing. Its source-only availability audit,
`.bench/enclosed-material-availability.json`, uses source hash
`1580bd58b070da76baa56fb7b63d2786bb212c51e4d4b877e08e04f87aaf473a`.
The dominant owner's maximum native color residual is 81.1 bytes, while its
95th percentile is 26.1. A maximum-only material model cannot account for its
edge samples, and relaxing that maximum would also swallow independent marks.

The retained operator instead tests explicit convex color mixtures between the
material and its enclosing rim or distinct neighboring owned paint. This is
restricted to a 1.5-native-pixel boundary band, with compatible modeled alpha.
An interior residual remains protected; a lone outlier in a matching owner
cannot serve as its own paint context. All source samples still participate in
the screen, and all changed native pixels participate in acceptance. Distance
sampling follows the evidence scale. This changes proposal evidence, not the
hard policy, score weights, normalizer or frozen sword targets.

Inspection is capped at 65,536 analysis pixels, bounded paint fits at 4,096
samples and source perimeters at 4,096 vertices. Existing enclosure limits cap
the owner count at 64 and its original geometry at 6,000 nodes. Each proposed
material also has a 6,000-node limit. Checks observe stop/deadline between
contexts, models and native proofs; incomplete work is never published.

### Search composition

Balanced search retains the existing eight evaluated alternatives per parent
and the 48-evaluation total cap. Following an ink replacement, a closed interpretation receives the first complementary
opportunity; following a closed interpretation, ink replacement does. Every
other operator remains in that cycle. These are shared-pool scheduling choices,
independent of the requested complexity anchor.

A previously feasible enclosed ellipse also supplies a scheduling hint keyed
by its complete original members. Storage is bounded to 64 hints with at most
256 members each. Changed parents repeat every ownership, style, core,
restoration, containment and order proof; no admission decision is cached.
Tests invalidate an actual core after a hint is recorded and require rejection.

When the pipeline supplies a separate live full-checkpoint budget, local
discovery now uses its complete allotted phase. Standalone search still keeps
its own 25% validation reservation. This removes the duplicated reservation;
the global search deadline, final validation/fitting reserves, minimum full
checkpoint estimate and cancellation behavior remain. A controlled-clock
regression proves that a useful proposal in the last quarter is evaluated only
under the supplied shared reserve and then independently validated.

### Experiments that did not improve the drawing

All native trials below use the frozen sword mask, complexity 50, balanced
quality, refinement disabled and a 60-second operation budget. No human
geometry enters proposal generation. The initial three trials isolate proposal
availability; none selects the combined material.

| Local experiment directory under `.bench/` | Source SHA-256 | Selected nodes / contours | Human MSE | Generation seconds |
| --- | --- | ---: | ---: | ---: |
| `planned-enclosed-material` | `1580bd58b070da76baa56fb7b63d2786bb212c51e4d4b877e08e04f87aaf473a` | 10,121 / 1,660 | 514.67 | 53.51 |
| `planned-enclosed-material-coverage` | `bfd6f4d7d7ec64a46f74e3465235e18339a963a162080e224984a6edabe762ad` | 10,121 / 1,660 | 514.67 | 52.07 |
| `planned-enclosed-material-mark-coverage` | `6c1c0463607eb3840a0e75993c8a4ad477eeab8f8661a04d4dded13b78062276` | 10,121 / 1,660 | 514.67 | 52.33 |
| `planned-enclosed-material-composition` | `b85c122dda54a896c4706b3eb553a44095f577072d4de5dc4febc0b63f7d4586` | 10,009 / 1,630 | 514.85 | 51.82 |
| `planned-enclosed-material-revisits` | `e59339fd8f688b77b3a7ebb6969e666e4a0e9609dc78fa450cc47f741d94c4ac` | 10,009 / 1,630 | 514.85 | 54.49 |
| `planned-enclosed-material-shared-reserve` | `6c1b192674c16a6b7745cb764f13af756dc7e7540ec1a255982976f3eca52dc3` | 10,009 / 1,630 | 514.85 | 54.11 |
| `planned-enclosed-material-priority-depth` | `89b5c21da059b7ccf010a7b75342a4d15f43050364c83fc4b239dd402a5704c6` | 10,009 / 1,630 | 514.85 | 51.50 |

The mark-coverage interpretation allows one combined material candidate to
reach native local evaluation. It saves 387 representation cost and increases
native visual error by about 0.000248, improving the objective at anchors
0/25/50. Hints and the shared reserve allow another such candidate on a parent
with one ink replacement. The selected five-expansion trial instead contains
two ink replacements, costs 20,055 and has 31 gradients. Its human
blade-tip/facets/guard/wrapping/jewel errors are
560.66/254.12/1,167.78/882.61/1,332.64. The jewel remains fragmented; this is
not a whole-surface quality win. It has zero crossings and native policy
rejections and still fails every numerical sword target.

The depth-priority experiment immediately bounded the beam after expanding one
parent, usually following the fixed anchor 50 and rotating other fixed anchors
every third turn. A controlled-clock native test reached three cumulative
edits, and another retained a useful faithful branch behind a rejected prefix.
The full 523-test relevant suite passed in 48.55 seconds. This verifies
mechanics, not quality. At native sword size the four parent expansions still
exhausted discovery before a combined material entered the selected drawing;
16 evaluations led to 12 local acceptances and four independent checkpoints,
with zero score disagreements. The output PNG is unchanged from the previous
two-ink result.

At the same existing four half-opacity tuning inputs, 192-pixel long side,
20 seconds, complexity 50, balanced quality and refinement disabled, the
priority schedule regresses western-park clean MSE from 178.72 to 204.43 and
lettering from 483.85 to 683.50, while saving 63 representation cost. Its
same-cost oracle gap grows from zero to 9.00. The other outputs are anime-girl
177 nodes/cost 327/MSE 178.09, anime-face 175/271/206.73 and rubberhose-band
407/765/167.85. Every case reaches 48 evaluations and four full checkpoints,
with complete diagnostic pools and zero score disagreements. Line F1 remains
0.270/0.600/0.652/0.641; measured feature failures persist. The prior
five-expansion schedule also worsened anime-girl eye error from 842.31 to
928.92. These regressions are recorded, not treated as acceptable progress.

The priority experiment and the five-evaluation per-parent change were removed.
Final production search retains the established eight-evaluation parent
expansion. The coupled material, complementary operator opportunity, bounded
source-membership hints and corrected checkpoint reservation remain for final
validation. Scheduling changes cannot substitute for useful interpretations or
calibrated feature selection. No policy threshold, score weight, normalizer,
release gate or default method was changed.

### Final source and candidate audit

The retained source hash is
`84fe778a9e7da258a8ed9b73f445422bc4b4da5ed4052f6fc6d89e1050739c02`.
The final native run in `.bench/planned-enclosed-material-original-width`
uses the same settings and mask as the table above and takes 52.45 seconds.
It selects the same two-ink drawing: 10,009 nodes, 1,630 contours, 1,577 paths,
31 gradients, cost 20,055 and human MSE 514.85. There are zero crossings or
native policy rejections. Twelve local evaluations lead to nine acceptances
and four full checkpoints with zero score disagreements. Local search and
validation take 21.04 seconds, including 4.15 seconds for the full checkpoints.
Only one combined material is emitted before discovery expires. The drawing
has 4.33 times the legacy node count and more than twelve times the gate's
node limit. The nominal slider budget and all frozen targets are unchanged.

Reproduce the retained native and paired runs with the commands above, using
`.bench/planned-enclosed-material-original-width` and
`.bench/planned-enclosed-material-original-width-rgba-pairs` respectively.
The four final half-opacity tuning inputs have identical source, clean and
mask hashes to the prior controls, and all selected drawing PNGs are
byte-identical to `.bench/planned-closed-mark-checkpoints-rgba-pairs`.

| Final tuning variant | Nodes / cost | Clean MSE | Same-cost oracle gap | Generation seconds |
| --- | ---: | ---: | ---: | ---: |
| Anime girl | 175 / 319 | 177.70 | 15.60 | 10.36 |
| Anime face | 176 / 272 | 206.86 | 0.25 | 15.85 |
| Western park | 547 / 987 | 178.72 | 0 | 9.19 |
| Rubberhose band | 407 / 765 | 167.23 | 0.39 | 11.77 |

All four again reach 48 local evaluations and four full checkpoints, with
complete diagnostic pools and zero score disagreements. Feature scores and
line F1 are restored exactly. These unchanged controls remove the observed
scheduling regressions; they do not establish a quality or runtime improvement.

A separate source-only availability audit in
`.bench/enclosed-material-candidate-audit/summary.json` reconstructs immutable
evidence, the owned initialization and the native policy using a 180-second
diagnostic limit. It evaluates the first combined material independently and
only then compares it with the human fixture. This is an availability check,
not a matched-effort selection result. The candidate is native-valid and
compacts 32 interior paths into an ellipse with a gradient, retaining the two
highlight owners. It reduces the core initialization from 10,286 to 10,115
nodes and cost 20,818 to 20,431. Human MSE decreases from 512.81 to 511.84;
jewel error decreases from 1,302.62 to 1,279.26. Other measured feature errors
are unchanged. The raster agrees exactly with the native local update; the
largest full/local score-term difference is 2.31e-10, within the existing
checkpoint tolerance. The candidate still fails all numerical sword targets
and makes only a small local improvement. It is not the published drawing.

The audit also tests the possible straight-chain corner blocker without
changing generation. Of 21,454 bounded open source chains, 5,283 already meet
the current straight-distance and monotonicity bounds. Fourteen are rejected
by the raw corner classifier; smoothing removes only seven exclusions, all
with endpoint spans under twelve native pixels. This does not substantiate a
claim that staircase-corner filtering blocks long facets. Leave that classifier
unchanged until a representative supported case demonstrates the need.

The next structural milestone therefore needs broad material/shade proposals
and supported boundaries across fragmented graph junctions, with explicit
whole-atom ownership or canonical atom splits. Continue comparing available
compact candidates with production selection and protected feature crops.
Neither a larger ranker, a deeper beam, nor a slider mapping supplies that
missing drawing. General joint fitting, calibration and release evaluation
remain open.

### Retained-source verification

After removing the unsuccessful scheduling experiments, the full relevant
planner, legacy CEL, shared fitting, operation and benchmark suite passes
521 tests in 49.57 seconds. This includes 12 new enclosed-material cases and
five new composition/checkpoint-reservation cases. Ruff lint passes and all
59 checked files are formatted. Project Pyrefly reports zero errors and 61
existing warnings; its configured exclusions omit tests, which are exercised
by pytest. The final native benchmark, paired controls and source-only
candidate audit all record the same retained source hash above.

## Broad material models and measured checkpoint time

The broad-model experiment adds `cel_plan/surface_models.py`, integrates it
within the existing family slot in `families.py`, and reports its exclusions
in `pipeline.py`. It offers coherent paint across complete source owners rather
than requiring every neighbor to resemble one seed color. Existing adjacency,
ink, geometry and enclosed-material operators keep their slots. It does not
change score weights, coverage gates, slider normalization or the legacy default.

### Model and safety bounds

A source-supported seed proposes a flat or linear material model. Its linear
hypothesis can extend beyond the seed's observed extent; the exported paint is
refitted over the complete proposed family and screened with actual SVG
clamping. Both gradient and flat competitors reach the same native objective.
Every source pixel of every included owner must meet the paint residual screen,
including distant atoms and isolated outliers. Fitting samples are bounded, but
the eligibility proof streams complete owned source support in 65,536-pixel
chunks. Interruption never publishes a partially screened owner.

The model considers at most eight material seeds, 128 paths and 6,000 geometry
nodes per family, sixteen emitted proposals per parent, 4,096 eligible owners
and a 1,536-squared analysis grid. Residual proposal parameters are 24 and 48
RGB byte values; these are eligibility screens, not relaxed native acceptance
thresholds. Fixed atoms, chosen ink overlays, paint constraints, translucent
current paints, strokes and clips are excluded. A partially translucent source
requires proof that the entire actual replacement geometry lies in the existing
opaque core of its supported opacity group.

Coarse ink classification is evidence for an alternative interpretation, not a
blanket veto of compatible shade owners. Coarse ink does not seed the material
model; any such owner included in a family must satisfy the complete source RGB
screen. A distinct dark mark stays independent when it fails that screen, and
fixed atoms or already selected ink overlays cannot be absorbed. Tests explicitly
exercise compatible coarse labels beside an unchanged black ink owner.

Exact contour unions retain the exterior geometry. They prove local order by
intersecting the actually moving earlier-fragment prefix with each intervening
sibling union. Siblings crossing the same prefix share one exact geometric
proof; the stationary last fragment is not part of the moving prefix. The
128-proof and 6,000-node temporary geometry bounds remain. A real overlap,
unproved stroke/clip, core failure or bounded proof rejects the proposal.
Whole geometry holds remain held after union. This is paint/model availability,
not fitted facet geometry or canonical source atom splitting.

### Exclusion experiments

Separate 180-second source-only audits explain the initial lack of candidates.
They use immutable source evidence, owned initialization and native policy;
the human render is consulted only after generating and evaluating a proposal.

| Eligibility/order experiment | Eligible owners | Emitted proposals | Interpretation |
| --- | ---: | ---: | --- |
| Veto any source atom classified as coarse ink | 118 | 0 | Coarse labels prevent material connectivity |
| Veto only owners with at least 60% coarse ink support | 118 | 0 | A whole-owner majority veto still blocks the same connectivity |
| Allow compatible coarse labels, prove order per sibling | 1,403 | 1 | A larger family reaches the unchanged 128-proof limit |

The audit directories are `.bench/broad-material-candidate-audit-atom-veto`,
`.bench/broad-material-candidate-audit-owner-veto` and
`.bench/broad-material-candidate-audit-before-group-proofs`. Their source hashes
are respectively `5c6b66795cc23660a9cbd08af859ef5792db3a936a9f597f718e2dccd220a069`,
`d36b3adeaa03f532612f2c6fe42e99f2bce8fedde9b910ffa124adf0faa03570` and
`faf70bad17934a8068fdb91a5b6689c3005adb78e4c8b996e63bf261cc8a5499`.
The per-sibling version emits one native-valid gray surface, removes 34 paths
and lowers initialization cost from 20,818 to 20,530. Its human MSE rises from
512.81 to 513.47. It is availability evidence with a negative visual tradeoff,
not a quality improvement or the selected drawing.

### Publication and final native comparison

The first matched 60-second run with grouped order proofs, before the checkpoint
time fix, is `.bench/planned-broad-material-surfaces` with source hash
`d4cbefea36b8249bdcdd8ff23226579e4d369b019ed07ca22988179b1eb94649`.
Nineteen evaluations and thirteen local acceptances receive no full checkpoints.
It returns the 10,286-node initializer with human MSE 512.81 in 53.15 seconds.
Local search could consume the time required by its own minimum full-check
estimate, even with a separate shared validation deadline.

`search.py` now caps discovery at the earlier of its local deadline and the
shared validation deadline minus the measured minimum checkpoint duration.
It does not subtract another fixed percentage when the caller already has a
longer validation window. Two deterministic clock tests reproduce the short
window failure and preserve full useful discovery with a longer window. Stop
still prevents publication; incomplete working states remain unpublished.

The retained source hash is
`4999e6c766974d418a488784941aaf6ac7b354ce1fa2d971cbdd3cecaba89999`.
The final matched run, `.bench/planned-broad-material-checkpoints`, uses the same
native source/mask, complexity 50, balanced quality, refinement disabled and
60-second time limit. It takes 53.24 seconds, evaluates 23 proposals, accepts
sixteen locally and publishes four independently validated checkpoints. There
are zero score disagreements, crossings or published native policy rejections.
Local search and validation take 16.81 seconds, including 4.84 seconds for the
full checkpoints. Three broad material proposals reach local evaluation.
Initialization timings differ between runs, so this rerun does not isolate
how much publication improvement comes from the deadline fix alone.

| Native sword drawing | Nodes | Contours | Paths | Cost | Human MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Human fixture | 523 | 93 | 80 | 1,127 | 0 |
| Legacy CEL | 2,312 | 339 | 137 | 4,230 | 663.31 |
| Previous retained material/ink run | 10,009 | 1,630 | 1,577 | 20,055 | 514.85 |
| Broad models with full checkpoints | 10,009 | 1,630 | 1,577 | 20,055 | 514.85 |
| Frozen sword gate | At most 800 | At most 140 | — | — | At most 497.39 |

The selected drawing still contains the same two ink replacements and 31
gradients. Tip/facets/guard/wrapping/jewel errors remain
560.66 / 254.12 / 1,167.78 / 882.61 / 1,332.64. Out-of-time, search-out-of-time
and stopped are false, and deadline overshoot is zero. Refinement remains
incomplete. The detailed normalizer is still 54,564 and the nominal complexity
50 representation target is 27,282. The selected result is more than four times
the legacy node count and exceeds every numerical sword target. Broad paint
models provide no improvement to the published sword.

### Final paired controls and reproduction

The four final half-opacity tuning controls in
`.bench/planned-broad-material-checkpoints-rgba-pairs` use the same source,
clean-target and mask hashes as
`.bench/planned-enclosed-material-original-width-rgba-pairs`. Every selected PNG
is byte-identical; all global, feature and line scores are unchanged. All four
reach 48 local evaluations and four full checkpoints, with complete diagnostic
pools and zero score disagreements.

| Tuning variant | Nodes / cost | Clean MSE | Same-cost oracle gap | Generation seconds |
| --- | ---: | ---: | ---: | ---: |
| Anime girl | 175 / 319 | 177.70 | 15.60 | 11.55 |
| Anime face | 176 / 272 | 206.86 | 0.25 | 17.94 |
| Western park | 547 / 987 | 178.72 | 0 | 10.29 |
| Rubberhose band | 407 / 765 | 167.23 | 0.39 | 13.38 |

These are unchanged quality controls, not runtime improvements. No timed CPU
tests overlap the native benchmark or paired controls. Reproduce with:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/rasmus/Workspaces/vectrify/.venv/bin/python scripts/bench_cel_planned.py \
  --methods cel-planned --seconds 60 \
  --settings '{"complexity":50,"quality":"balanced","refine":false}' \
  --out .bench/planned-broad-material-checkpoints

PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/rasmus/Workspaces/vectrify/.venv/bin/python scripts/bench_cel_pairs.py \
  --methods cel-planned \
  --cases anime-girl anime-face western-park rubberhose-band \
  --degradations clean --composition-opacity 0.5 --long-side 192 --seconds 20 \
  --method-settings '{"cel-planned":{"complexity":50,"quality":"balanced","refine":false}}' \
  --out .bench/planned-broad-material-checkpoints-rgba-pairs
```

### Final source-only availability audit

The final audit in `.bench/broad-material-candidate-audit/summary.json` records
the retained source hash above, checks that it stays unchanged, and evaluates
the first four broad proposals from the independently validated initializer.
Its 180-second diagnostic limit is not matched-budget production selection.
The human geometry never enters generation, screening, fitting or native policy.

| Independent candidate | Removed paths | Nodes / contours | Cost | Human MSE |
| --- | ---: | ---: | ---: | ---: |
| Initialization | — | 10,286 / 1,711 | 20,818 | 512.81 |
| Broad blue-gray gradient | 50 | 10,098 / 1,661 | 20,342 | 513.22 |
| Broad blue-gray flat | 50 | 10,098 / 1,661 | 20,330 | 512.06 |
| Broad gray gradient | 34 | 10,202 / 1,677 | 20,530 | 513.47 |
| Broad gray flat | 34 | 10,202 / 1,677 | 20,518 | 511.70 |

All four pass the unchanged native policy and agree exactly with the independent
local raster. The maximum full/local score-term difference is 2.32e-10, within
the existing checkpoint tolerance. The audit inspects 579,894 source pixels,
fits three seeds and uses 144 exact interval order proofs across the four
proposals, with no core/order exclusion or proof-limit failure. It deliberately
stops after four proposals; this is not an exhaustive pool or oracle.

Flat blue-gray paint slightly improves the blade facet score from 254.12 to
252.51 but worsens guard error from 1,166.23 to 1,169.40. Flat gray paint lowers
guard error to 1,162.86 but raises facet error to 255.23. Wrapping and jewel scores are unchanged in all four candidates; the blue-gray
models slightly lower tip error. Neither independent candidate meets the
combined sword target. The best global MSE gain here is only 1.11 points,
about 0.22%, while the largest node reduction is 188, about 1.83%. This is far
below the required abstraction. The production selected PNG is byte-identical
to the previous retained drawing. Candidate availability has expanded, but
there is no material whole-drawing improvement to report.

### Verification and remaining structural work

The relevant planner, legacy CEL, shared fitting, operation and benchmark suite
passes 541 tests in 62.55 seconds. This includes eighteen broad-material cases
and two measured-checkpoint-time cases. Ruff lint passes and all 61 checked
files are formatted. Project Pyrefly reports zero errors and 61 existing
warnings; tests are excluded from that project check and are exercised by pytest.

No complete delivery or sword gate passes because of this change. The next
structural work must fit supported exterior boundaries across graph junctions,
combine compact geometry with broad paint/retained marks, and introduce exact
canonical source splits when whole atoms cannot represent the proposed drawing.
Retaining jagged unions and merely recoloring them cannot deliver the intended
human abstraction. General joint fitting, content normalization, UI controls,
corpus expansion, conditional ranking and independent release review remain open.

## Structured interior fitting and union correspondence

The previous goal turn was progress: `b56761b` added broad material competitors
and independent availability evidence. This turn addresses the geometry blocker
found in the source-only initialization rather than claiming a sword quality gain.
The eight complete deliveries and their frozen gates remain open.

### Diagnosed and changed behavior

The inventory in `.bench/coherent-boundary-inventory-before-permissions.json`
records source hash
`4999e6c766974d418a488784941aaf6ac7b354ce1fa2d971cbdd3cecaba89999`.
All 1,658 primary paths of the owned material initializer have whole geometry
holds and no usable chain permissions, totaling 10,286 nodes. Some large
surfaces nevertheless lie entirely inside the actual opaque core. Structured
RGBA export held every path and omitted permissions even for generic interior
curve models; this also blocked ordinary CPU fitting of those curves.

`opacity.py` now records bounded chain permissions for generic curve models in
structured export. Straight and ellipse primitives remain protected. Native
transparent contacts, thin components, explicit widths, alpha steps, repaired
crossings and unrecorded segments retain their exact complement. A path with
both generic and protected chains can fit only its proved generic segments.
The existing path/node/chain/segment caps and fingerprint/frame checks apply.

`constraints.merged` transfers permissions through an owned exact contour union
in `families.py` and `surface_models.py`. It matches surviving complete segment
controls in the same current frame, retires internal edges, fingerprints the new
geometry and keeps original chain identities as lineage. A protected coincident
copy vetoes a free copy. Unknown, stale, differently framed, over-bound or
interrupted paths cannot authorize new fitting. New or boolean-subdivided
segments stay protected. Continued-under-mark geometry still discards replaced
permissions because it changes more than an ordinary union.

The shared fitter also redrew already identical neighbor runs, creating new
node identities on a held underlayer and making a legal interior fit appear to
edit that protected path. `shared.follow` now skips only an exactly identical
run. It still follows an actual change as small as 0.000001 native coordinate
units; rounded shared-edge equality is not used for this no-op check. Native
regressions verify both adjacent surfaces change together while the underlayer
geometry and complete alpha raster remain unchanged. Full/local scores and
reload permissions agree.

These changes preserve existing supported curves; they do not prove arbitrary
boolean subcurves or rebuild the canonical graph after splitting source atoms.
Parameterized primitive fitting and richer variable-width ink remain unfinished.

### Discovery completion guard

The first matched run with the geometry fixes but without a completion guard,
`.bench/planned-structured-boundary-permissions`, records source hash
`dd688a3928baa51b5906dbf4ff5b76787fbae1172400ea3894e110cb75370be0`.
Eight evaluations and seven local acceptances include two boundary fits, but
receive zero full checkpoints. It returns the 10,286-node initializer with human
MSE 512.81 in 53.24 seconds. Reserving exactly the estimated full-check duration
was insufficient when a proposal finished after its last deadline poll.

Search now additionally leaves the longest observed proposal opportunity before
the shared validation deadline, with a 0.05-second minimum guard when an explicit
full-check estimate is supplied. This changes only the time bound; anchors,
beam order, evaluation caps, native acceptance and score weights are unchanged.
The report includes `checkpoint_guard_seconds`. A deterministic-clock test
reproduces completion just after a deadline poll and verifies that the admitted
state receives a full checkpoint. An observation-based guard is not a proof of
worst-case latency for an unseen renderer/boolean; hardware runtime gates remain
open.

### Final matched native and paired evidence

The retained source hash is
`69f704de9bb688c279822826e008a273922e7ad9c1825602069b5f96173183f8`.
The final sword run in `.bench/planned-structured-boundary-guarded` uses the same
native source/mask and 60-second, complexity-50, balanced, refinement-disabled
settings. It takes 53.61 seconds. Nine proposals are evaluated, seven are locally
accepted and three receive independent full checkpoints, with zero score
disagreements. The completion guard is 2.90 seconds. Local search and validation
take 12.13 seconds, including 3.77 seconds for full checks. Two boundary fits
are admitted locally; neither is selected.

| Sword result | Nodes | Contours | Paths | Cost | Human MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Human fixture | 523 | 93 | 80 | 1,127 | 0 |
| Legacy CEL | 2,312 | 339 | 137 | 4,230 | 663.31 |
| Previous retained broad-material run | 10,009 | 1,630 | 1,577 | 20,055 | 514.85 |
| Structured permissions and guarded checkpoints | 10,121 | 1,660 | 1,607 | 20,347 | 514.67 |
| Frozen gate | At most 800 | At most 140 | — | — | At most 497.39 |

The selected drawing contains one ink replacement and 31 gradients. It has zero
crossings or published native policy rejections. Tip/facets/guard/wrapping/jewel
errors are 560.66 / 254.12 / 1,166.23 / 882.61 / 1,332.64. Out-of-time,
search-out-of-time and stopped are false; deadline overshoot is zero and
refinement remains incomplete. The normalizer stays 54,564 and the nominal
complexity-50 cost target stays 27,282. Against the prior retained run, error
decreases by just 0.18 while nodes increase by 112 and cost increases by 292.
This is not a useful drawing improvement. Initialization times vary between
runs, so the comparison does not isolate the guard's runtime effect.

The final four half-opacity tuning controls in
`.bench/planned-structured-boundary-guarded-rgba-pairs` have identical source,
clean-target and mask hashes to `.bench/planned-broad-material-checkpoints-rgba-pairs`.
All selected PNGs are byte-identical, and all global, feature and line scores
remain unchanged. Each reaches 48 evaluations and four full checkpoints with
complete diagnostic pools and zero score disagreements.

| Tuning variant | Nodes / cost | Clean MSE | Same-cost oracle gap | Generation seconds |
| --- | ---: | ---: | ---: | ---: |
| Anime girl | 175 / 319 | 177.70 | 15.60 | 14.18 |
| Anime face | 176 / 272 | 206.86 | 0.25 | 18.02 |
| Western park | 547 / 987 | 178.72 | 0 | 10.16 |
| Rubberhose band | 407 / 765 | 167.23 | 0.39 | 13.02 |

The paired completion guards range from 0.11 to 0.25 seconds. This establishes
unchanged quality on these controls, not a speedup or broader corpus coverage.
Tests do not overlap timed benchmarks. Reproduce with the native and paired
commands in the preceding section, replacing output directories with
`.bench/planned-structured-boundary-guarded` and
`.bench/planned-structured-boundary-guarded-rgba-pairs` respectively.

### Final source-only boundary availability

The final inventory in `.bench/coherent-boundary-inventory.json` uses the retained
source hash. It finds 1,233 paths with usable permissions, containing 8,650 nodes
and 5,794 free segments; 425 paths with 1,636 nodes retain their whole holds.
The underlying initializer geometry is unchanged. The first broad material
union merges 51 paths; its surviving `cel-fill-4295` retains exact permissions.
Its whole metadata fork has 1,197 permission records. Retired source chains are
lineage only, not evidence of rebuilt canonical boundaries.

The 180-second diagnostic audit in
`.bench/coherent-boundary-candidate-audit/summary.json` tests the first four
boundary fits on the initializer and then on that broad union. It records the
same source hash before and after. It uses only source evidence for proposals,
and consults the human render afterward. All eight proposals pass native policy,
match the independent local raster exactly and have a maximum full/local
score-term difference of 2.32e-10. This bounded prefix is not an exhaustive
candidate pool or matched-budget production result.

| Fitting prefix | Resulting node counts | Human MSE range | Largest node saving within the prefix |
| --- | --- | --- | ---: |
| Initializer (10,286 nodes, MSE 512.81) | 10,285 / 10,286 / 10,286 / 10,264 | 512.25–512.83 | 22 |
| Broad gradient union (10,098 nodes, MSE 513.22) | 10,097 / 10,098 / 10,098 / 10,098 | 513.18–513.27 | 1 |

The best initializer fit reduces nodes by about 0.21% and MSE by about 0.11%.
A usable permission is therefore not proof of a compact interpretation. The
next geometry work must rebuild supported current boundary correspondence
through fragmented junctions, including proved subdivisions and true canonical
source atom splits, and compose those boundaries with coherent paint and ink.
Do not substitute more generic curve fitting or a learned ranker for that
missing representation.

### Retained verification

The full relevant planner, legacy CEL, shared fitting, operation and benchmark
suite passes 554 tests in 61.44 seconds. New coverage includes structured export
and CPU refinement with holes/faint marks; exact surviving union permissions
with native joint fitting and reload; stale frames, bounds, stop and unproved
boolean subdivisions; protected ellipses; exact no-op shared runs and tiny real
changes; and the completion guard. Ruff lint passes and all 63 checked files
are formatted. Project Pyrefly reports zero errors and 61 existing warnings;
tests are excluded from that project check and exercised by pytest. No complete
delivery or release gate is claimed finished.

## Remaining requirements

None of the eight complete deliveries is claimed finished yet. In particular:

| Delivery | Remaining evidence or behavior |
| --- | --- |
| 1 | Full synthetic/curated-human coverage, frozen broader-suite tolerances and calibrated score terms |
| 2 | Dense-input fallback/runtime and memory bounds; broader partial-alpha, transformed-scope and difficult-hole coverage beyond the new native cases |
| 3 | General curved source splits and boundary rebuilding beyond the bounded straight-cut operator, broader coherent surface/paint interpretations beyond owned family and enclosed-material proposals, calibrated content normalization and broader budget/risk priorities beyond the initial slot schedule, bounded shared-frontier cache, large-input fallback beyond the bounded native tile kernel and resizing invariance |
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
anchor proposals also remain; general curved source splits and supported boundary
rebuilding, broader ink replacement, richer models and order edits are unfinished.
These gaps must be resolved before completion is claimed.

## True source atom splits and branch graphs

The structural family slot now also proposes bounded source-supported straight
shade cuts. These cuts classify source pixels directly, including inside an
original color region. They do not use a majority assignment or merely regroup
original region IDs. An immutable ledger records each retired parent and its
children's exact disjoint row runs, source-label hash, shape and pixel counts.
Replay against the original graph rejects foreign support, incomplete support,
other-owner runs, hidden/fixed roots and malformed or oversized ledgers.

Ownership expands atomically for every primary surface, covered membership and
underlay. The canonical region graph, statistics and edge owners are rebuilt
for each admitted namespace. All downstream family, paint-restoration, closed
contour and ink operators use that branch's graph and label image. Original
native policy masks and source evidence remain fixed. Search state identities
and rejection proofs include the atom namespace. Existing structural operators
preserve the ledger through their immutable ownership replacements.

Candidate generation inspects at most eight complete material owners, 128 atoms
per owner and 262,144 pixels per owner bounding box. Thirty-two normal angles and
integer offset votes propose at most four lines per owner; edge observations
are capped at 4,096. Flat/linear models fit at most 4,096 samples, then screen
**every** owned pixel with the exported gradient clamping. Both sides need at
least 16 pixels and five percent of the owner, maximum RGB-byte residual 48 and
at least ten percent lower source-paint squared error than the whole-owner fit.
These are proposal restrictions, not relaxed native acceptance rules.

Current contours are clipped into complementary children in their actual SVG
frame. They retain the parent's exterior and holes. Partial-alpha proposals
require actual opaque-core geometry beneath both children. Exact surviving
interior permissions can rebind; new cut segments and unproved subdivisions
stay protected. Real translucent paint, fixed atoms, chosen overlays, covered
owners, strokes and clips remain excluded. Nominally opaque gradient stops can
have numerical roundoff within 1e-9 of one after parent-opacity conversion;
actual geometry and native coverage checks still apply.

The ledger is capped at 64 cuts and 16,384 runs on at most a 1536-square analysis
grid. The branch cache retains two contexts with a conservative 32 MiB graph
charge, including live contexts held by proposal cursors after cache eviction.
Oversized branches cannot enter the search. Labels and canonical boundary point
arrays are read-only. Ledger objects and serialized atom metadata contribute to
the existing working-state memory charge. These explicit restrictions and
synthetic cancellation checks do not establish hardware runtime/memory gates.

Verification covers a real split through a single source region, different
editable side paints, canonical separating edges, holes, partial alpha,
rotated child frames, independent native/local agreement, direct project
save/reload, successive splits/merges, secondary support, sibling rollback,
namespace-separated rejection proofs, active branch memory restrictions,
foreign/stale/malformed ledgers, complete-pixel paint outliers, fixed atoms,
missing actual cores and stop/bounds behavior. The full relevant suite passes
576 tests in 52.50 seconds; the subsequent focused file passes 14 tests,
including the added sibling-cache test (577 distinct tests across the two runs).
Ruff and project Pyrefly pass; Pyrefly reports zero errors and the existing 61
warnings. Tests do not overlap the timed benchmarks below.

This is a first bounded straight-cut operator and a complete source-ownership
mechanism for it. It is not general canonical boundary simplification, curved
splitting, joint variable-opacity/width fitting, a compact whole-sword drawing,
a calibrated complexity slider or completion of delivery 3. Numerical sword
and paired evidence follows below.

### Matched evidence and source-only availability

The retained production source hash is
`03e479ac9a16482307787e278e18f22b3f9819ff3356c12641c7d469ec66b6a9`.
The native sword run in `.bench/planned-source-atom-splits` uses the same source,
mask, 60-second budget, complexity 50, balanced quality and refinement disabled.
It produces 10,009 nodes, 1,630 contours, 1,577 paths and 31 gradients, cost
20,055 and human MSE 514.8500919869 in 52.24 seconds. It evaluates 26 alternatives,
locally admits 22 and publishes four independent checkpoints with zero score
disagreements. The selected drawing contains two existing ink replacements.
There are no crossings or published native rejections. Out-of-time,
search-out-of-time and stopped are false; refinement remains incomplete.

Against the preceding 10,121-node result, this removes 112 nodes but increases
human MSE by 0.18. It matches the earlier broad-material drawing. Against legacy
CEL it still has over four times as many nodes and fails all frozen numerical
sword gates. This is **not a new sword quality improvement**. Native operators
can now compose across nominally opaque gradient stops after numerical
roundoff, and the run gets more evaluations; the measured outcome does not
isolate runtime effects or establish a quality benefit from that correction.

The production report records zero split-owner visits and zero graph rebuilds:
the new third family cursor is not reached before the bounded discovery window
ends. The independent diagnostic audit in
`.bench/source-split-candidate-audit/summary.json` explicitly reaches that
operator on the 10,286-node initializer and the first broad material union.
Its source hash is identical before and after. It constructs proposals from
source evidence only, and does not feed the human rendering into the model.
For each phase it examines the bounded eight-owner prefix. Initial inspection
finds 28 line hypotheses and 20 complete-paint exclusions; post-union inspection
finds 32 and 24. The other lines fail minimum side support. Neither phase yields
a candidate, so there is no new native admission or human score to report.
This distinguishes missing interpretations in the inspected prefix from a
selection error; it is not an exhaustive search or evidence that a different
split model cannot work.

The next structural experiment must couple coherent whole-family geometry and
piecewise paint directly. Requiring a broad single-paint intermediate owner
prevents useful shade partitions from reaching the split stage. Exact contour
booleans alone also retain the dense exterior. Combine supported compact
boundary geometry, flat/linear side paint, source atom ownership and local order
in one proposal, then measure native admission and the best faithful compact
candidate against the selected drawing. Keep all frozen sword, feature and
line gates. Do not substitute a larger ranker or an exposed slider for this
missing representation.

The four paired half-opacity tuning controls in
`.bench/planned-source-atom-splits-rgba-pairs` have identical source, clean-target
and mask hashes to `.bench/planned-structured-boundary-guarded-rgba-pairs`.
All selected PNGs and global/feature/line scores remain identical. Each reaches
48 evaluations, four checkpoints and zero score disagreements. Anime girl,
anime face, western park and rubberhose band retain 175/176/547/407 nodes,
costs 319/272/987/765 and clean MSE 177.70/206.86/178.72/167.23. Their observed
generation times are 9.58/15.67/9.25/12.44 seconds. No speedup is claimed.

Reproduce with the native and paired commands above, using output directories
`.bench/planned-source-atom-splits` and
`.bench/planned-source-atom-splits-rgba-pairs`. The ignored diagnostic driver is
`.bench/diagnose-source-split-candidates.py`, with an independent 180-second
budget. All tests, timed benchmarks and the diagnostic audit run sequentially.
No complete delivery or release gate is claimed finished.
