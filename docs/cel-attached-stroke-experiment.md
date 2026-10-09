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

After the geometric and native-cell stroke-only attempts fail, the attached
route can jointly fit exclusive material restoration and the new centerline.
The original material commands, user-space paint server/frame and sealed shadow
remain fixed. A native-tolerance flattening seeds only the added material;
projecting failed pixels onto its edges selects controlling endpoints, including
long edges whose endpoints are farther away. At most 24 parameters move material
endpoints 0.0625 pixels normal to the physical port chord and 0.625 pixels along
it, stroke controls two pixels normally, and width within the existing bounds.
The search has at most 600 evaluations. It retains only crop-alpha-exact,
source-line/gap-feasible, exclusive-material vectors, then independently checks
the whole drawing's alpha, crop consistency and complete original source bank.
Ownership, sealed residuals and final locality still require the enclosing planner.

If the complete zero-extension attached attempt publishes no candidate, planning
tries a half-pixel removal extension at the end, then at the start. The first
domain that publishes candidates stops the fallback sequence. These are cut
alternatives with unchanged physical stroke ports, not source-gap completion.
The explicit API still bounds any requested extension to two native pixels.

Attached fitting retains only parameter vectors with exact parent alpha in the
independent native fitting crop, then rechecks the complete drawing. Alpha
penalties guide the search but cannot make an infeasible vector publishable. The
constraint and actual butt/round cap are part of the fit-cache key. White-body
source-gap checks and ownership coverage use the actual stroke cap as well.
Physical source ports and supported corners remain fixed.

Fitting also retains only vectors whose native white stroke body leaves the
observed source-gap positions empty within the existing absence tolerance. A
gap penalty can guide search but cannot select a cheaper invalid body over a
valid one. The independent tiled absence check still runs after assembly. A
tapered-terminal regression evaluates a cheaper wider stroke that covers a real
source-negative point and verifies that the valid narrower stroke is retained.

Before the final native checks, assembly must also preserve the exact proved
residual element and geometry. The new centerline must remain a single open
subpath with `fill="none"` and a positive painted stroke width. Native pixels
alone cannot establish these properties: same-colored old ink beneath a stroke,
or a fill attribute on a straight open path, can be invisible to raster checks.

## Current evidence and limits

- The unmodified automatic attached proposal search on fitted candidate 11
  completes in 358.55 seconds and generates profile 43 through the joint fitter
  and half-pixel terminal-cut fallback. No optimizer/proposal filters, saved
  fitting parameters or acceptance overrides are used. Complete native alpha,
  static SVG subset, original ownership/component replay and native reload pass,
  with 64 cuts and 81 allocated children. This is one complete attached-chain
  improvement; it does not complete every sword outline or the CEL release plan.
- A cold production joint fit generates profile 43 without saved parameters.
  Its zero-extension attempt reaches crop feasibility but fails full alpha;
  the automatic half-pixel terminal cut then produces an open two-node stroke
  of width 1.27222. Independent checks verify full native alpha, byte-exact RGBA
  outside the literal 167-pixel footprint, unchanged original shadow/material
  commands, paint and frames, complete former-field removal, source ports/body
  gaps, painted source lines and exact reload. Original ownership/component
  replay passes with 64 cuts and 81 allocated children, and editable-stroke
  missing samples fall from 2,287 to 2,191. The targeted replay skips preceding
  stroke-only optimization and unrelated profiles for work bounds; it does not
  bypass any acceptance gate. The successful joint search takes 243 optimizer
  evaluations plus its initial seed evaluation, and the replay completes in
  45.07 seconds. The complete unmodified search is reported above.
- A later source-only construction passes every attached acceptance check on
  profile 43. Extending only the removal domain half a native pixel beyond the
  terminal preserves full-viewport alpha exactly, with byte-exact RGBA outside
  the literal removed-field and actual-stroke support (167 pixels). The actual
  two-node, 1.24556-pixel open stroke keeps both physical source ports. Independent
  Boolean checks prove complete removal of the former field and no residual ink;
  original shadow commands, material commands, paint and frames stay exact.
  Original ownership/component replay passes with 64 cuts and 81 allocated
  children; source-body gaps, painted source lines and native save/reload pass.
  Editable-stroke missing samples fall from 2,287 to 2,191. No acceptance guard is
  bypassed. This earlier construction supplies saved source-only material/stroke
  fitting parameters; the subsequent cold production fit above removes that
  dependency.
- The removal recipe is now available through explicit `port_extension` on the
  attached proposal API. Each extension is bounded to two native pixels and
  independently contains the complete original field. It extends only the cut,
  never the physical source anchors or the stroke. The first attempt remains
  zero, with the half-pixel automatic fallbacks described above. Forward/reversed
  synthetic controls verify containment, untouched ports/shadow commands and rejection of
  invalid bounds and real source gaps. Independent sword replay uses this
  production cut recipe with all acceptance/ownership checks intact.
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
- Two corruption controls deliberately resurrect the former filled owner beneath
  the new stroke or give the straight centerline a fill attribute. Both pass the
  native pixel checks but are rejected by the final structural checks. Accepted
  attached witnesses explicitly record exact residual geometry and paint.
- Straight and rotated support controls verify that native-cell restoration uses
  exactly the opaque cells touched by the removed field in the original frame.
  Butt and round fitting controls verify the actual cap, source ports, native
  fitting context and round trip.
- The earlier source-gap retention replay on fitted candidate 11 publishes
  **no attached-band candidate**, without interruption. Five of 13
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
- The earlier tight-clip locality result is withdrawn. The native renderer ignores
  even-odd clip rules, so a same-winding inner contour did not remove the old ink
  inside the cut. Correct opposite winding and disjoint rectangular clips both
  remove that outside ink, but change seven remote alpha bytes on profile 43.
  An independent ink-only raster now checks that the outside piece contributes
  no old ink inside the removed field. No production clip support is implemented.
- Fitting a separate 29-node material patch derived from the native flattening of
  the original shadow makes profile 43's fitting-crop alpha exact. Six bounded
  material parameters are fitted; the physical stroke ports and original shadow
  controls stay fixed. Complete-viewport alpha still changes one remote byte,
  so the drawing remains rejected. No ownership interpretation or production
  material-patch fitter is claimed by this diagnostic.
- Selecting material vertices by their distance from failed pixels misses long
  edges whose endpoints are farther away. A source-only joint diagnostic instead
  projects those pixels onto material edges and fits their endpoints. It reaches
  exact fitting-crop alpha in 455 evaluations with fixed physical ports and exact
  retained shadow commands. Independent checks verify an open `fill="none"`
  stroke, no crossings, source-body absence, no painted source-line rejection,
  exact residual subtraction and save/reload pixels. Actual editable-stroke
  support gains 96 of 13,145 qualified source samples (missing samples fall from
  2,287 to 2,191). Complete alpha still differs at (397, 1722), from 10 to 11.
  The distinct restored material object has no published ownership interpretation;
  the drawing remains rejected and is not an accepted quality result.
- The same projected-edge construction can append exclusive restoration to the
  existing material owner instead of introducing another painted object. On
  profile 43 it reaches exact crop alpha in 249 evaluations, preserving original
  material contours, paint, frame and private gradient. Independent checks show
  zero patch intersection with existing material and unchanged residual shadow
  commands. An explicitly rejected construction reaches the original ownership
  and component replay: 64 cuts and 81 allocated children, within the existing
  64-cut/128-child limits. Save/reload and the final assembled pixels match the
  fitted trial exactly. The one remote alpha byte remains; the diagnostic bypass
  only lets that rejected drawing reach independent ledger checks, and is not
  production acceptance or a new published candidate.
- Both separate and existing-owner edge fits complete on all five eligible
  sword chains. Only profile 43 reaches exact crop alpha; none reaches complete
  alpha equality. The other four fits exhaust their evaluation budgets. Native
  stroke-only checks show that the pommel fits already spill into transparent
  parent pixels, including shared physical ports: added material cannot repair
  those excess pixels. Profiles 67 and 97 also lose support for adjacent source
  chains when cut independently. Co-planning must account for those interactions.
- On profile 54, refitting only material normals with a fixed gap-valid stroke
  leaves the same 65 total alpha-byte discrepancy. A wider material range and
  a 0.8-pixel stroke minimum reduce that discrepancy to 35, but the source-line
  guard then rejects lost support on profile 54. Lower alpha error alone is not
  a usable stroke improvement; neither diagnostic changes production bounds.
- Resolving the retained shadow as a full Boolean difference preserves its
  filled region exactly but changes 11 native alpha pixels, rather than one.
  It does not repair the remote discrepancy and loses the original command
  prefix; no resolved-shadow construction is adopted.
- A vector-mask feasibility diagnostic also leaves one remote alpha discrepancy.
  Masks are unsupported by the editor's static SVG subset; this is not a native
  candidate or a proposed mask-support implementation.
- Profile 43's complete subtraction also changes a remote alpha byte at native
  (397, 1722), from 10 to 11. Equivalent contour ordering, exact subdivision and
  cut representations have not repaired it. Restricting subtraction to opaque
  cells leaves part of the old field behind and is not an accepted complete cut.
- Broad shadow interiors, internal source gaps, ambiguous chains, unsupported
  winding and bounded/interrupted work remain excluded. Source observations never
  copy the human rendering's repaired handle connection.

The focused source-slice, band-plan, source-band, source-junction, source-cap,
material-continuation, joint material/stroke and line-fidelity suite passes:
144 tests in 9.48 seconds at the cold joint-fit checkpoint. New synthetic controls
restore fractional silhouette coverage which a bounded stroke alone cannot
preserve, including translucent groups and user-space gradients. A control reaches
crop fitting but rejects an unrelated remote alpha change at the full native
check; another preserves fixed width and excludes bounding-box material paint.
Ruff and the full source/test Pyrefly check pass at this checkpoint
(0 type errors, two suppressions and 70 warnings). The changed Python files pass
formatting; the global formatter flags pre-existing blank lines in unchanged
`tests/ui/test_server.py`. Draft CI is green at the source-gap retention commit
`f378494` (lint and test, 6m13s), and documentation head `e847f0b` is also green
(lint and test, 8m13s). Subsequent changes need their own CI.
The full bounded captured-proposal replay at `8bf5367` completes with 12 proposals, all
native-valid and with complete ownership/component and save/reload checks. All
12 SVGs are byte-for-byte unchanged from the previous accepted pool. Its attached
suffix adds no candidate.

The new captured-proposal replay at the source-gap retention checkpoint reaches
its 600-second limit and is recorded as interrupted. Its 12 accepted prefix SVGs
are byte-for-byte unchanged, with native, complete original ownership/component
and save/reload checks passing. The interrupted suffix cannot establish a
completed attached search.

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
- `.bench/cel-attached-tight-clip-probe/`: invalid even-odd clip-hole diagnostic;
  its apparent locality proof is withdrawn.
- `.bench/cel-attached-corrected-local-shadow-probe/` and
  `.bench/cel-attached-rect-local-shadow-probe/`: corrected clip diagnostics and
  independent old-ink visibility checks; both fail remote alpha locality.
- `.bench/cel-attached-native-flat-fit-probe/`: exact fitting-crop alpha for a
  separate material patch, with one remaining complete-viewport alpha byte.
- `.bench/cel-attached-material-edge-fit-probe/`: generic projected-edge endpoint
  selection and independent checks of the saved rejected stroke/material fit;
  exact crop alpha and 96 additional supported source samples, one remote alpha
  byte, and no published original-source ownership proof.
- `.bench/cel-attached-all-edge-fit-probe/` and
  `.bench/cel-attached-owned-material-edge-fit-probe/`: completed five-chain
  comparisons, saved rejected drawings and independent source, residual,
  material-prefix, round-trip and stroke-only alpha checks.
- `.bench/cel-attached-owned-fit-ownership-replay/`: the existing-owner profile-43
  drawing passes original ledger/component and round-trip checks while remaining
  explicitly rejected for complete alpha. No diagnostic rejection bypass is
  present in production.
- `.bench/cel-attached-resolved-shadow-probe/`: six exact-region Boolean residual
  representations; all change 11 alpha pixels and remain rejected.
- `.bench/cel-attached-material-normal-refit/` and
  `.bench/cel-attached-thin-owned-edge-fit-probe/`: profile-54 width/material
  feasibility controls; the thinner fit loses source support and is rejected.
- `.bench/cel-attached-masked-cell-material-probe/`: unsupported vector-mask
  diagnostic with one remaining remote alpha byte; no ownership or subset proof.
- `.bench/cel-attached-remote-layer-probe/`: independent native layer isolation
  attributes profile 43's remote alpha change to the retained shadow (3 to 4),
  while the underlay remains unchanged at 7.
- `.bench/cel-attached-source-port-field-probe/`: 81 bounded source-only removal
  domains; four retain full native alpha with unchanged physical stroke ports.
- `.bench/cel-attached-superset-owned-replay/`: one complete runtime construction
  passes all original native, gap, locality, ownership/component and reload gates,
  with independent literal-footprint and complete-field validation.
- `.bench/cel-attached-production-cut-replay/`: the same construction through the
  production `port_extension` API; saved offline material/stroke fit parameters
  remain explicit. `validation.json` checks full-image alpha, byte-exact outside
  RGBA, original controls/material, complete removal, source ports/gaps and reload.
  `stroke-preview.png` shows the parent and actual editable stroke bodies. Cut and
  independent validation use source hash
  `cf0300a4a4bd4d8541af3b816a6491795d7318df3486e8eab277a83e7b6c39b2`;
  the saved offline fit input retains its original `c1c4379…` source provenance.
- `.bench/cel-attached-source-constrained-owned-fit-probe/`: profile 54's source-
  feasible joint fit reduces alpha-byte discrepancy from 65 to 45. Independent
  body/painted-source, material and reload checks pass, but alpha remains unequal;
  that drawing is rejected.
- `.bench/cel-attached-cold-joint-sword-replay/`: cold production joint fitting and
  automatic terminal-cut fallback on profile 43, with preceding stroke-only
  optimization and unrelated profiles excluded solely for work bounds. No saved
  parameter vector or acceptance override is used. Independent `validation.json`
  and `stroke-preview.png` check and show the generated drawing. Its algorithm
  source hash is `4b9d78e850be9557af70041e4d8d54270c979f92cdaf0bb3eb54a9603eae76fb`.
- `.bench/cel-attached-full-joint-sword-replay/`: unmodified automatic attached
  planning on fitted candidate 11, complete in 358.55 seconds with one generated
  attached candidate. Its source hash matches the cold replay. Original native,
  source, locality, ownership/component and reload gates stay intact; independent
  validation and a stroke/body preview accompany the exported drawing.

Each replay records its algorithm source hash and checks it at completion. Saved
drivers and logs provide diagnostic provenance, not automated-operation release
or reference-corpus evidence. Dependent diagnostic inputs retain their own source
revision; a current source hash does not retroactively validate an earlier input.

## Resume here

1. Co-plan the continued material's partial silhouette and editable stroke using
   the edges that control failed pixels, rather than only nearby vertices. The
   opaque interior restoration is insufficient at these edges. Preserve physical
   source ports and real gaps; do not waive final alpha to obtain a candidate.
   Before fitting material, identify native alpha excess already supplied by the
   stroke and retained context. A positive underpaint cannot subtract that excess.
   Plan interacting source chains together when one field removes another chain;
   a nominal profile is not automatically an independent replacement.
2. Extend complete planning to the remaining attached chains, including closing
   material edges and interactions with adjacent source support. The unmodified
   search now generates a fully verified profile-43 stroke without saved
   parameters; the other chains still lack complete native-feasible proposals.
   Keep exact residual controls, complete removal and all final native,
   locality/source/ownership gates.
3. Replay complete ownership, original ledgers, residual controls, native body
   gaps, save/reload and the exported editable stroke inventory, then show the
   actual strokes for visual feedback.

No new reference corpus, learned ranking, automatic-operation quality result or
release gate is claimed by this draft. The active goal remains unfinished.
