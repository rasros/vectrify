# CEL redesign merge checkpoint

PR #315 adds an experimental `cel-planned` operation, owned structural planning,
CPU refinement and source-only editable-stroke experiments. The existing CEL
method remains unchanged. This checkpoint can merge independently of completing
[the release plan](cel-redesign-plan.md): **all eight deliveries remain open**.
The complete measurements, failed hypotheses, commands and historical source
hashes remain in the [experiment archive](archive/cel-redesign-experiments.md).
Earlier work-package proposals remain in the
[plan history](archive/cel-redesign-plan-history.md).

## Implemented behavior

- `operations/methods/cel_planned.py` exposes complexity 0–100 (default 50),
  fast/balanced/high quality, refinement, gradients and advanced overrides.
  The planner combines CEL line/silhouette evidence with color/opacity regions,
  considers coherent surfaces and local layers, and retains a validated common
  frontier. The slider selects representation cost from available alternatives;
  safe plateaus and unmet budgets remain possible.
- Immutable source-atom partitions and component edits prove complete ownership
  and dependencies. Candidate acceptance uses native rendering, hard validity,
  local/full agreement and the common objective. Bounded search/refinement keeps
  validated checkpoints on stop; broader latency/memory gates remain open.
- Explicit `Operators(filled_bands=True)` co-plans actual editable strokes with
  retained materials. `SourceBands`, `BandFit`, `SourceCaps`, `PaintContinuation`
  and `SourceJunctions` use copied original source observations, preserve real
  gaps and physical terminals, and keep exact residual shadow/mark geometry.
  Old filled outlines are removed within complete owned replacements.
- `StrokeInventory` and its replay CLI inspect actual exported stroke bodies,
  native styles/transforms/group opacity and literal shared ports. Missing source
  samples locate remaining work. Its report explicitly does not prove outline
  completion, valid junctions, ownership or removal of former filled outlines.

The band co-planner remains an explicit experiment. Ordinary experimental High
leaves that flag off; the accepted offline candidates are not a claim that the
Generate operation reliably delivers this stroke quality. No learned model or
new reference corpus ships at this checkpoint.

## Verified stroke/material checkpoint

The owner inspected the fitted Detailed sword and judged its shadows clean and
comparable to the human rendering, with different choices. The remaining priority
is clean connected editable strokes while preserving those materials. The human
rendering repairs a source defect; generation must not copy that repair.

The terminal source-only captured-proposal replay produces twelve complete
candidates. The first eight retain the preceding artifact sets byte-for-byte;
four further siblings add a fitted handle to the editable blade and guard.
All twelve pass original-graph/component ownership, native hard validity,
local/native RGBA equality and project reload. Maximum score disagreement is
6.943e-10. The pool remains within the existing 64-cut / 128-child limits.

The inspected candidate has a connected 35-node blade stroke, a four-node guard
stroke and a four-node fitted handle stroke at native width 1.678925366437946.
Both handle endpoints are literally shared with adjacent strokes. A separate
source-only audit verifies the supported handle connection, exact retained
shadow/control geometry and no increased stroke-body coverage at all 404
original gap queries. This does not establish all junctions in the drawing.
The complete drawing has 2,920 nodes, 415 contours and 33 stroke contours.

The inventory finds 11 stroke objects, 33 contours and 143 centerline nodes.
Of 13,145 qualified nongap source sample positions, 2,287 lack actual editable
stroke-body support at alpha 0.05: 82.60175 percent are supported. This is sample
coverage, not the percentage of complete outlines. Its 15 literal shared ports
are a diagnostic count, not 15 proved source junctions. Missing support is
concentrated around the handle exterior and attached ink/shadow sections.

The unchanged objective still selects earlier blade/guard alternatives at sampled
slider settings. The offline shared-junction replay took 97.86 seconds and peaked
at 706,532 KiB RSS. These measurements do not satisfy operation runtime/memory or
release-quality gates. The sword's 800-node / 140-contour targets remain unmet.
Attached-band prototypes are excluded from the merge checkpoint: native locality
and complete ownership replay remain unresolved.

## Reproduction and validation

The replay artifacts below are local ignored `.bench/` outputs, not repository
fixtures. The band replay requires captured ancestor/planned project documents,
partition/details JSON and a proposal specification containing ids, parent,
component parent, parameters and operator; its CLI help documents that format.
The preserved capture is a development hypothesis, not a reproducible automatic
Generate result. Replaying it diagnoses proposal availability independently of a
bounded operation run.

```sh
PYTHONPATH=src:. python scripts/bench_cel_band_plans.py --case sword \
  --captured .bench/cel-band-capture --out .bench/cel-source-junction-band-plans-final
PYTHONPATH=src:. python scripts/bench_cel_stroke_inventory.py --case sword \
  --candidate .bench/cel-source-junction-band-plans-final/candidate-11.project.json \
  --out .bench/cel-stroke-inventory-final --crop 310 1640 420 1870
```

The historical terminal band replay used helper/benchmark source SHA-256
`85712bdfb1e334babe924397b9bb19035d0e160bdd8d644072b63252cb51e62e`.
The inventory used source SHA-256
`25958416b4337782e536fb43c9c24263d4bef77a93e0a356869abb10725d00f3`,
driver SHA-256
`0a26d22af086beedfb0e52477eec0802d431a186ffc639ad6c68340a7926af06`,
and inspected project SHA-256
`b606759c1237cab277e5ba383e418839467c68aeb29e857cf15dd1fc82f3bb1f`.
Hashes identify those historical artifacts; documentation/test cleanup does not
constitute a new replay or operation-quality result.

Earlier checkpoints ran targeted tests and a local type check that accidentally
skipped tests when interpreter discovery excluded this worktree. CI exposed the
missing optional-result assertions and stale fixture contracts. The merge cleanup
checks both `src/` and `tests/`, asserts required fixture results before use,
updates mocks to current APIs and removes redundant assertions. Explicit checker
exclusions prevent an editable install in another worktree from hiding the local
project. No test directory or diagnostic rule is disabled.

Ruff, formatting and whitespace checks pass locally. The complete source/test
Pyrefly check reports zero errors (two existing suppressions, 69 warnings).
The merge gate is GitHub's `Lint and test` job: Ruff, the complete Pyrefly check
and `pytest -q`. Final full-suite and commit-specific CI results are recorded on
[PR #315](https://github.com/rasros/vectrify/pull/315).

## Remaining release work

| Delivery | Status and remaining evidence |
| --- | --- |
| 1 · Evaluation foundation | Open: larger licensed corpus, frozen broader tolerances and score calibration |
| 2 · Operation and graph | Open: dense/large-input fallback, difficult scopes/holes and runtime/memory evidence |
| 3 · Complexity and materials | Open: calibrated content normalization, richer coherent interpretations and meaningful five-level tradeoffs |
| 4 · Editable ink and geometry | Open: attached ink/shadow decomposition, complete exterior chains, supported joins and sword/line/feature gates |
| 5 · Fitting and runtime | Open: broader order inference, joint fitting, geometric regularization, complete scheduling and memory/cancellation limits |
| 6 · Product integration | Open: Generate UI/MCP controls, browser/API round trips, invalidation, undo and editing identity binding |
| 7 · Broad evaluation | Open: expanded paired suite, ablations and conditional learned-ranker decision |
| 8 · Rollout | Open: fresh held-out results, independent blind review and gated default migration |

Resume with complete attached ink/material decomposition and source-supported
exterior strokes. Preserve the inspected shadows, remove each replaced filled
outline atomically, and verify complete ownership, native alpha/locality and real
gaps before accepting a candidate. Then improve selection and reliable operation
delivery. Broader references and learned ranking follow a useful generator.

The unfinished attached-band prototype is saved locally in
`.bench/cel-attached-band-wip/`, with its base commit and file hashes in
`manifest.json`. It is not part of this merged checkpoint.
