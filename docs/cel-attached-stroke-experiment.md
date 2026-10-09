# Attached outline stroke experiment

This draft follows merged PR #315. It explores replacing a complete thin source
chain attached to a filled shadow with an editable centerline, while preserving
the original shadow controls and removing the former filled outline. The
[release plan](cel-redesign-plan.md) and its open gates remain unchanged.
Ordinary CEL and the ordinary experimental High route leave the band flag off.
This work is available only through the explicit filled-band co-planner.

`source_slice` clips a bounded complete nongap original source chain from its
compound owner. Directed winding subtracts the selected field without rebuilding
the original shadow curves. Independent Boolean difference/intersection checks
verify the geometry. The caller must still prove ownership, native alpha,
source gaps and painted locality; construction alone proves none of those.

`BandPlans` offers these interpretations after its existing complete fitted
parents, replaying the original ancestor's source classes within the existing
ledger limits. Prior fitting witnesses remain available to the replay audit.
The original proposal prefix remains available when an attached route fails.

## Atomic construction and acceptance

An attached subtraction temporarily removes coverage that the new material and
stroke may restore. Its raw-cut locality and continuation-versus-complement
checks are construction diagnostics, including explicit `null` quantization
results when they fail. Neither intermediate drawing can be published. Ordinary
source routes retain their earlier continuation-complement requirements.

For attached routes, restoration uses the original filled parent's fully covered
native pixels. The removed fill can supply that coverage itself. The native
opacity bound is measured through the same ordered group-opacity stack, rather
than using an unrounded analytic product.

The geometric removed-field domain is tried first. If that complete attempt
fails, a second bounded attempt can continue material into whole opaque parent
pixel cells touched by the removed field, including its antialias fringe. The
result is ordinary vector contours under the existing material paint and frame;
it embeds no raster, clip or mask. The residual shadow still subtracts the whole
original selected field. This fallback addresses internal coverage seams; it
cannot restore partially covered silhouette cells.

The final complete drawing is compared with its actual material parent. Global
native alpha equality is mandatory. Changes are bounded by the removed field and
old/new affected stroke bodies; retained shadow bounds do not authorize unrelated
changes. `color_quantization` permits at most one premultiplied color byte outside
that support, with alpha unchanged exactly. This is explicitly weaker than
byte-exact RGBA locality, is reported separately and is not a release policy.

Attached fitting retains only parameter vectors with exact parent alpha in the
independent native fitting crop, then rechecks the complete drawing. Alpha
penalties guide the search but cannot make an infeasible vector publishable. The
constraint and actual butt/round cap are part of the fit-cache key. White-body
source-gap checks and ownership coverage use the actual stroke cap as well.
Physical source ports and supported corners remain fixed.

## Current evidence and limits

- Synthetic controls publish two-node editable strokes for opaque, translucent
  and nested-group cases. Material holes beneath the former outline, including
  holes crossing a fractional terminal, require restoration rather than retained
  old ink. The terminal cases exercise the native-cell fallback. Original shadow
  commands/values, saturated ancestor ledgers and native project round trips
  remain exact. Independent Boolean checks verify the published residual has no
  intersection with the removed field.
- A control permits failed intermediate locality diagnostics but requires the
  final complete-parent proof. Rejecting that final proof excludes the candidate.
  Background paint changes, changed group alpha and displaced retained shadows
  also fail the final locality check.
- Straight and rotated support controls verify that native-cell restoration uses
  exactly the opaque cells touched by the removed field in the original frame.
  Butt and round fitting controls verify the actual cap, source ports, native
  fitting context and round trip.
- The source-only direct sword replay on accepted fitted candidate 11 still
  publishes **no attached-band candidate**, without interruption. Five of 13
  eligible profiles reach both bounded fitting attempts; no exactly feasible
  native-alpha vector is found. Round caps alone also yielded no candidate in an
  exploratory runtime-only replay; there is no production round-cap generator.
- A residual diagnostic records the least alpha discrepancy observed during each
  bounded search. Profile 43 loses coverage at 44 partial silhouette pixels on
  the left handle. Profile 54 differs at 34 pixels, including right-side spill.
  Pommel profiles 67, 85 and 97 also change partial silhouette coverage. These
  diagnostic vectors are rejected drawings, not accepted strokes or a quality
  improvement. The observations do not prove that every possible vector is
  infeasible.
- A separate-material diagnostic restores the removed field as a distinct paint
  object using the neighboring material color. For profile 43, widths 0.8–1.2
  reduce alpha discrepancies to 12 pixels and 15 total alpha bytes, but still
  fail exact equality. It is an XML diagnostic without a published ownership
  interpretation, not a supported production restoration mode.
- A tightly bounded vector-clip diagnostic keeps the original owner outside the
  removed field and the subtracted remainder inside it. With no box margin,
  profiles 43 and 67 pass off-field alpha/color locality. Other profiles still
  fail, and clipped material restoration changes alpha inside the field. No
  production clip dependencies or clipped ownership proof are implemented.
  Equivalent direct and even-odd cut representations do not repair locality.
- Profile 43's complete subtraction also changes a remote alpha byte at native
  (397, 1722), from 10 to 11. Equivalent contour ordering, exact subdivision and
  cut representations have not repaired it. Restricting subtraction to opaque
  cells leaves part of the old field behind and is not an accepted complete cut.
- Broad shadow interiors, internal source gaps, ambiguous chains, unsupported
  winding and bounded/interrupted work remain excluded. Source observations never
  copy the human rendering's repaired handle connection.

The focused source-slice, band-plan, source-band, source-junction, source-cap,
material-continuation and line-fidelity suite passes: 130 tests in 7.04 seconds.
Ruff and the full source/test Pyrefly check pass at the restoration checkpoint
(0 type errors, two suppressions and 70 warnings). The changed Python files pass
formatting; the global formatter flags pre-existing blank lines in unchanged
`tests/ui/test_server.py`. Draft CI is green at `8bf5367` (lint and test, 6m05s).
The full bounded captured-proposal replay completes with 12 proposals, all
native-valid and with complete ownership/component and save/reload checks. All
12 SVGs are byte-for-byte unchanged from the previous accepted pool. Its attached
suffix adds no candidate.

## Active goal

Make complete source-supported attached outline chains on the fitted material
candidate into clean editable strokes, removing their former filled outlines
atomically while preserving the inspected shadows and real source gaps.
Validate complete-candidate pixel locality, exact native alpha, original
ownership ledgers and save/reload. Demonstrate a visible sword improvement with
focused synthetic controls, then present a stroke preview for review. Learned
ranking and broader reference collection remain deferred.

Completed ignored diagnostics:

- `.bench/cel-attached-native-cells-probe/`: geometric and native-cell restoration
  on fitted candidate 11, with the captured original ancestor and source ledger.
- `.bench/cel-attached-alpha-residual-probe/`: source-only bounded search residuals
  and explicitly rejected project/SVG drawings; no human target.
- `.bench/cel-attached-restoration-full-pool/`: completed 12-proposal replay with
  byte-exact prefix verification.
- `.bench/cel-attached-separate-material-probe/`: distinct field paint and bounded
  width diagnostics, with a rejected left-handle stroke inspection.
- `.bench/cel-attached-tight-clip-probe/` and
  `.bench/cel-attached-cell-split-material-probe/`: off-field and inside-field
  coverage diagnostics for clipped constructions. They are not production
  clip support or accepted candidates.

Each replay records its algorithm source hash and checks it at completion. Saved
drivers and logs provide diagnostic provenance, not automated-operation release
or reference-corpus evidence.

## Resume here

1. Co-plan the continued material's partial silhouette and editable stroke. The
   opaque interior restoration is insufficient at these edges. Preserve physical
   source ports and real gaps; do not waive final alpha to obtain a candidate.
2. Preserve exact residual shadow controls and complete removed-field geometry.
   The remote profile-43 alpha change and silhouette changes still require a
   complete-parent proof.
3. Replay complete ownership, original ledgers, residual controls, native body
   gaps, save/reload and the exported editable stroke inventory, then show the
   actual strokes for visual feedback.

No new reference corpus, learned ranking, automatic-operation quality result or
release gate is claimed by this draft. The active goal remains unfinished.
