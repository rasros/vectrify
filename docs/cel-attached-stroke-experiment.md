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
long edges whose endpoints are farther away, including implicit closing edges.
Final crop verification restores the retained stroke width after optimization,
rather than leaving the last evaluated trial in the crop. At most 24 parameters
move material endpoints 0.0625 pixels normal to the physical port chord and 0.625 pixels along
it, stroke controls two pixels normally, and width within the existing bounds.
The search has at most 600 evaluations. It retains only crop-alpha-exact,
source-line/gap-feasible, exclusive-material vectors, then independently checks
the whole drawing's alpha, crop consistency and complete original source bank.
Ownership, sealed residuals and final locality still require the enclosing planner.

The joint fit also requires the new stroke itself to support every qualified
sample on its original profile. A dark material or retained filled path cannot
satisfy this contract. Its normal matching windows and permitted raw-positive
queries are copied when the original source bank is prepared. The fitting crop
uses a positive-body penalty; final verification renders the isolated actual
stroke at the full native viewport under its original opacity hierarchy.
The witness reports supported and missing samples separately. This complements
painted-line and negative-gap checks without authorizing endpoint movement.

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

- Closing-edge selection and retained-width verification have regression controls.
  A cold targeted profile-43 replay after the closing-edge fix produces an SVG
  byte-identical to the accepted automatic result. Independent alpha, locality,
  source, original material/shadow and reload checks still pass. This preserves
  the existing improvement; it does not establish another completed chain.
  A second cold replay after both fixes completes in 47.05 seconds, again with
  byte-identical exported SVG and all original native/ownership/reload gates intact.
- Profile 54 still produces no accepted candidate with the corrected edge
  selection. Runtime diagnostics with wider material motion, additional local
  edge nodes and a thinner stroke also remain rejected. At its fixed terminal,
  isolated retained context and stroke coverage explain why positive material
  restoration alone can fail: with the unextended cut, context alpha is 117 and
  the 0.8-pixel seed stroke body is 104 where parent alpha is 162. Continuous
  composition estimates excess coverage; these measurements do not prove every
  possible native vector infeasible. Complete native assembly remains the gate.
- A runtime joint stroke-opacity experiment on profile 54 also emits no candidate
  across the three automatic cut domains. It fits opacity within 0.25–1 with
  local material edge nodes and wider material movement; every final acceptance
  gate remains intact. Neither opacity fitting nor these wider bounds are adopted.

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
146 tests in 11.09 seconds after the closing-edge and retained-width fixes.
The broader CEL planning suite also passes: 1,467 tests in 209.36 seconds.
Synthetic controls restore fractional silhouette coverage which a bounded stroke alone cannot
preserve, including translucent groups and user-space gradients. A control reaches
crop fitting but rejects an unrelated remote alpha change at the full native
check; another preserves fixed width and excludes bounding-box material paint.
Ruff and the full source/test Pyrefly check pass at this checkpoint
(0 type errors, two suppressions and 70 warnings). The changed Python files pass
formatting; the global formatter flags pre-existing blank lines in unchanged
`tests/ui/test_server.py`. The automatic joint-fit head `d4be68d` is
[green in GitHub CI](https://github.com/rasros/vectrify/actions/runs/37983035298).
The documented closing-edge and retained-width checkpoint `099a081` is also
[green in GitHub CI](https://github.com/rasros/vectrify/actions/runs/37986998409).
The positive-body contract checkpoint `08d9431` is
[green in GitHub CI](https://github.com/rasros/vectrify/actions/runs/37996879559).
The earlier `5187445` run was cancelled after it was superseded.
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

- `.bench/cel-attached-closing-edge-43-replay/`: the cold targeted production
  replay after closing-edge selection was corrected. Exported SVG is byte-identical
  to the accepted automatic profile-43 result. Independent validation passes.
- `.bench/cel-attached-closing-edge-54-replay/`,
  `.bench/cel-attached-wide-material-54-replay/`,
  `.bench/cel-attached-local-material-54-replay/` and
  `.bench/cel-attached-local-thin-material-54-replay/`: corrected-edge and runtime
  material/width feasibility diagnostics. None emits an accepted candidate;
  runtime subdivisions and wider bounds are not production changes.
- `.bench/cel-attached-port-alpha-floor/`: native isolated context/stroke coverage
  measurements at profile 54's source ports. Analytic composition is explicitly
  diagnostic, not evidence of a complete native candidate.

- `.bench/cel-attached-retained-width-43-replay/`: cold production replay at the
  corrected-width head; full exported SVG matches the accepted automatic result
  byte for byte. Native, source/locality, ownership/component and reload gates pass.
- `.bench/cel-attached-opacity-material-54-replay/`: bounded runtime opacity,
  local-material and wider-movement trial; all three cut attempts return no
  accepted candidate. Driver, prototype and source hashes are recorded separately.

Each replay records its algorithm source hash and checks it at completion. Saved
drivers and logs provide diagnostic provenance, not automated-operation release
or reference-corpus evidence. Dependent diagnostic inputs retain their own source
revision; a current source hash does not retroactively validate an earlier input.

## Source terminal and retained-context diagnosis

The following ignored diagnostics use source hash
`f9d361c9f8780328af0c6d3d23a73170f05a1c0c0143ac9788e28e3274f3198d`.
They do not add accepted candidates or change production endpoint constraints.

- `.bench/cel-attached-source-port-evidence/` measures qualification and raw
  alpha at stored source endpoints. Both profile-85 estimates are unqualified
  with raw alpha at most 2/255. The shared profile-67/97 estimate and profile-97
  far endpoint are also unqualified and transparent. A transparent cluster mean
  does not prove a real source gap: connectivity around the junction still needs
  positive raw-source evidence. Profile 67's qualified start and profile 54's
  high-alpha attached start remain protected observations.
- `.bench/cel-attached-source-body-parent-bound/` checks all qualified positions
  on profiles 43, 54, 67, 85 and 97. Each has parent coverage at at least one of
  its nine raw-positive normal queries. This necessary condition does not prove
  that a complete feasible stroke exists, but excludes a simplistic argument
  based only on transparent stored endpoint estimates.
- `.bench/cel-attached-bounded-port-source-body-85-replay/` stages a cold compact
  single-C body before material restoration. Only unsupported low-alpha terminal
  tails are trimmed to copied qualified observations; uncertain endpoint poses
  stay within three pixels of their original estimates. Every original qualified
  body query retains its original matching tolerance. The body covers all 63
  qualified samples and respects raw-positive endpoints and source gaps, but the
  atomic material fit emits no complete candidate. The final fitted-material
  stage still requires its own body-support verification; staging is not proof
  of the subsequently changed stroke.
- `.bench/cel-attached-bounded-port-alpha-guide-85/` records the same cold search's
  best rejected material crop: 19 alpha discrepancies, with summed byte error 68.
  Its pruned SVG is explicitly a rejected crop, not a complete drawing or preview.
- `.bench/cel-attached-retained-context-alpha-85/` isolates the original and
  subtracted shadow. At native pixel `(329, 1996)`, shadow alpha changes from 3
  to 4 while unchanged underlay alpha remains 12; complete alpha changes from 15
  to 16. All three standard cuts reproduce this change, with pruned/full crop
  equality. A separate single-path replay reproduces it without source-cap
  recovery or new stroke paint, including when unrelated paths are removed.
- `.bench/cel-attached-source-domain-alpha-85/` completes all 81 quarter-pixel
  start/end cut combinations from zero to two pixels. Each geometrically valid
  subtraction still increases that same native alpha byte. This is a limitation
  of the current directed-winding construction under the native renderer, not a
  successful source-domain fallback. Positive material restoration cannot remove
  an excess in untouched retained context.

Opacity, thinner-width, local-material, underlay-hole and terminal-control trials
on profile 54 also emit no accepted candidate. Their driver/prototype hashes are
separate from the production source hash. None enables opacity, base holes,
subdivision, port correction or enlarged fitting bounds in production.

The positive-body contract checkpoint passes all 1,476 CEL planning tests in
210.83 seconds, including frozen source-window, raw-bright-query, opacity,
displacement, crop-edge and final isolated-native-body controls. Ruff passes
across source/tests/scripts; full Pyrefly reports zero errors, two existing
suppressions and 70 warnings. Changed Python files pass formatting.

`.bench/cel-attached-body-support-43-replay/` records a cold targeted production
replay: one candidate in 51.18 seconds, byte-identical to the earlier accepted
automatic handle SVG. Its isolated new stroke supports all 95 qualified samples.
Complete alpha, native subset, original ownership/component and native reload
pass with 64 cuts and 81 allocated children. The independent validation also
passes the literal 167-pixel footprint, exact original shadow/material controls
and paint, full former-field removal, source ports, gaps and crossings.
Generation uses source hash
`139a00b989afed2d0e8254e00d7f338317c4356a0f26f6676e23d0e11599e2d3`;
independent validation after import/type cleanup uses
`bc9c3f55cc3d00d814a181086f7d07b9af071a006380dedb48fdaad7d56808aa`.
No additional accepted sword chain is claimed by this checkpoint.

## Complete primitive spans and separate restoration diagnosis

The following ignored diagnostics use production source hash
`bc9c3f55cc3d00d814a181086f7d07b9af071a006380dedb48fdaad7d56808aa`.
They change runtime constructors explicitly; none adds a production candidate.
Primitive indices remain fixture-specific and are not a general discovery rule.

Removing a whole bounded pair of opposing original curve spans resolves the
profile-85 subtraction artifact. The six-node left-pommel field contains the
entire previous profile field, stays inside the original fill, preserves all
original residual controls, and passes exact Boolean difference/intersection
checks. Its native subtraction has no alpha gains and is byte-exact outside the
actual field footprint. Extending just the former partial cut through another
96 domains, or adding small intersection chips, does not resolve the artifact.
See `.bench/cel-attached-primitive-field-pommel-probe/` and the explicitly rejected
reach/field-completion diagnostics.

The larger field requires a longer source-supported body. Measuring the raw dark
ridge across the complete left-pommel span supplies 248 qualified samples and no
observed gaps. Cold source-body planning supports all of these and all 63
qualified samples on original profile 85. The original 13,145-sample source bank
remains separate and unchanged. Full atomic material fitting still emits no
candidate. The source trace, cold crop replay and rejected curved-material fit
are saved in `.bench/cel-attached-primitive-source-trace-pommel/`,
`.bench/cel-attached-primitive-source-left-atomic-crop-replay/` and
`.bench/cel-attached-primitive-curved-material-replay/`. An earlier oversized-crop
driver fails its area assertion and supplies no valid replay evidence.

Separate restoration preserves the original material geometry exactly. A saved
input comparison improves from 30 alpha-discrepant pixels for appended material
to 27 for a separate surface, but neither drawing is accepted. Planning the
stroke in the complete separate-restoration context reduces the discrepancy to
six pixels. Independent full-native audit confirms:

- A real open, four-node `fill="none"` stroke, width 1.2148100829712065.
- Isolated actual-body support for all original 63 and extended 248 samples.
- Exact original shadow-prefix and material controls, original material frame,
  complete former-field removal and zero residual-field intersection.
- Byte-exact RGBA outside the literal removed field and actual stroke body,
  native subset validity and byte-exact native save/reload.
- Six remaining alpha discrepancies, no original painted-line/gap rejection,
  and one missing painted sample on the added raw profile within its existing
  comparison allowance. Body support and painted support remain distinct checks.

This is `.bench/cel-attached-separate-stroke-independent-audit/`; ownership and
component acceptance are explicitly unproved. An earlier material diagnostic's
body isolation omitted the ancestor transform and reports zero support; that
incorrect isolation is superseded by this independent actual-root-frame audit.
Do not use its body counts. Bounded material coordinate fitting and a 23-parameter
joint least-squares trial do not produce exact complete alpha. The joint trial
retains actual body support but has 11 alpha-discrepant pixels. Solid-versus-gradient
paint and exact contour ordering/orientation controls reproduce the same original
two restoration deficits; changing paint or serialization does not solve them.

A longer whole-primitive field extends to the lower handle shoulder. It has 11
nodes, retains exact original curve commands and passes geometric subtraction,
no native alpha gains and byte-exact outside-field RGBA. Its raw source trace
fits as four nodes, supplies 239 qualified samples and has no measured gaps.
See `.bench/cel-attached-complete-left-neck-primitive-probe/`. Cold body planning
in its separate-restoration context reaches a rejected guide with one byte of
stroke-associated alpha excess. Restoration already contributes two additional
excess pixels with the stroke hidden, so a body-only optimizer cannot repair the
complete drawing. The independent full drawing audit at
`.bench/cel-attached-long-source-stroke-independent-audit/` finds five alpha
discrepancies, exact locality and reload, exact original material/residual
controls, and actual-body support for all original 63 and extended 239 samples.
Its added painted profile has one missing sample and no comparison rejection.
Original ownership/component proof is still absent. This is a diagnostic drawing,
not a newly accepted chain or a release preview. Its native comparison and
isolated-stroke panel are saved as `stroke-preview-rejected.png` with separate
preview provenance in that audit directory.

The extended trace's endpoint coordinates are construction guides: `ink.measure`
retains its supplied endpoints while centering interior samples. Averaging the
old field boundaries does not itself measure a physical source port. Any future
endpoint uncertainty policy must use raw evidence, keep the original source bank
and gaps fixed, and distinguish these new guides from protected original ports.
No such policy, new restoration surface, enlarged fitting bounds or alpha
exception is enabled in production.

## Generic primitive discovery and complete U restoration

`source_spans.discover` now constructs bounded alternatives around the complete
former field without fixture primitive indices. It enumerates pairs of short
transverse cuts on one original closed contour, copies the opposing original
curve primitives, and keeps the original retained command prefix. It proves
geometric containment, exact subtraction and empty retained intersection. Up to
256 cut candidates and 32 retained alternatives are allowed, within the existing
native extent/node bounds; interruption is checked during enumeration. The
read-only midpoint guide supplies a raw-source search domain, not physical ports
or positive stroke proof. **The helper is not enabled in production search.**

Eleven focused constructor tests cover curved bands attached to broad material,
complete U-shaped rails attached to a broad pad,
reversed contours, rotated SVG starts crossing the implicit closing edge,
transformed and translucent owners, complete original curve preservation,
fragment/gap/low-support/broad-field exclusions, deterministic bounds and
interruption. Constructor tests deliberately make no native body, ownership or
full-candidate acceptance claim. All 1,486 CEL planning tests pass in 200.07
seconds before the final U constructor control was added; all eleven current
constructor tests pass in 0.50 seconds. Ruff and formatting pass for the new files,
and Pyrefly reports zero
errors with the two existing suppressions and 70 warnings.

A cold sword diagnostic found 31 alternatives. Selecting by complete restored
native alpha, literal outside RGBA locality and raw nongap evidence finds a
20-node field covering both sides of the pommel U. The retained context has no
alpha gains; separate restoration reproduces **every original alpha byte before
adding a stroke** and preserves outside-field RGBA exactly. The full raw guide
measures 448 qualified samples, no source gaps and support 1.0, and fits a
six-node open curve. This is construction evidence only: the Gold restoration
paint still needs a right-face material proof and original ownership/component
declaration. The single accepted production handle chain remains unchanged.

The first cold U body trial was rejected. Requiring the new U alone to reproduce
all profiles touched by its removal field mistakenly included the complete
horizontal collar profile. Its 185 qualified samples belong predominantly to an
existing genuine stroke: isolated native rendering supports 184, with one
pre-existing missing sample. The original source bank is unchanged; that missing
sample is recorded, not waived. The existing collar's physical endpoints are
(333.5, 1941.5) and (393.5, 1941.5), under the same component parent as the U. Its
geometry, width, round caps, paint, opacity and order must remain intact. A new
construction guide may bind to those actual source-supported ports before fresh
raw measurement; original physical ports and frozen observations may not move.
The U needs its own body proof for the original side profiles and the additional
raw trace; the collar junction needs a separate native proof from the combined
**genuine strokes** under their actual shared opacity hierarchy.

The discovery, raw measurement, full native floors, rejected body trial and
existing collar observations are recorded separately under
`.bench/cel-attached-generic-primitive-full-u-body-replay/`,
`.bench/cel-attached-full-u-profile-scope-probe/` and
`.bench/cel-attached-full-u-existing-port-probe/`, at algorithm SHA
`806df7da36fb5d43259d852bb7d369d32d852194c6e94a50cf9ff9d42f342b14`.
The earlier floor probe used `cKDTree`; the checked-in helper uses its typed public
`KDTree` equivalent. Its cold U replay repeats discovery and all complete floors.
These artifacts are excluded from version control and are not accepted examples.

The next cold trial binds only the new construction guide to the genuine collar
ports before raw measurement, giving a separately frozen 449-sample profile.
It renders the new U alone for the original side profiles and the additional
trace, and the new U plus the unchanged collar together under their actual shared
opacity hierarchy for the collar obligation. A 21-parameter planar-control fit
supports all 449 new samples and all 185 collar samples, but misses two original
right-tip samples and changes five complete alpha pixels. Its independent saved-
guide audit also finds painted-source regressions on original profiles 67 and 97.
The Gold restoration creates a visible right-edge fringe; exact restoration
alpha therefore does not establish the correct material face.

A fresh 14-parameter fit instead moves each control and noncorner fitted vertex
along its **local** curve normal. Original source observations, existing physical
junction ports and the measured raw corner remain fixed. At 600 evaluations the
new U supports all 70, 63 and 46 qualified samples on the original side profiles,
all 449 added qualified samples, and all 185 collar samples through the genuine
joint body. It remains rejected: ten alpha pixels gain one byte. No saved body
parameters seed either cold trial, and neither changes production generation.

The planar trial, independent audit/preview, and local-normal trial are recorded
under `.bench/cel-attached-generic-primitive-full-u-junction-replay/`,
`.bench/cel-attached-full-u-junction-guide-audit/` and
`.bench/cel-attached-generic-primitive-full-u-local-normal-replay/`, with the same
algorithm hash above and separate driver/helper hashes. The independent planar
preview audit verifies exact original material geometry, retained original shadow
commands, complete field removal, exact native reload, true source-gap exclusion,
and exact RGBA outside the literal field **plus actual body**. Outside the field
alone is not exact because the new body extends beyond it; the report records
both rather than treating the field as the entire allowed footprint. Reassigned
node IDs differ, so original command/value prefix equality is reported separately
from full dataclass identity. Original ownership/component acceptance remains
unproved. Its separate full-scene audit under
`.bench/cel-attached-full-u-local-normal-guide-audit/` confirms all original and
added isolated-body obligations, the ten one-byte alpha gains, exact original
material geometry and retained shadow commands, complete field removal, unchanged
gaps, native reload and outside field-plus-body RGBA locality. Painted original
profiles 67 and 97 still regress despite the complete isolated-body support; the
additional 449-sample painted profile passes. Both the visible material fringe
and painted-source contracts need restoration/placement work before acceptance.

## Adjoining original material face checkpoint

`source_faces.discover` constructs a bounded partition of a complete primitive
removal field using a complete shared edge of an original adjoining material.
It copies original cubic controls and matches native edges in either direction.
The base restoration keeps the complete field's command prefix and appends the
selected contour with whichever sign proves exact geometric subtraction. Both
fields are exclusive, and their union must equal the complete removal field.
Material identities come from the original partition, within the same component;
this constructor does not invent source observations, restoration ownership or
painted coverage. It remains disabled in every production search.

Eleven synthetic controls cover straight and curved shared boundaries, reversed
original owners and removal winding, rotated/translucent components, nearby but
unshared curves, material overlap, unsupported paint, actual source gaps,
interruption, bounded inputs and deterministic alternatives.
All 1,498 CEL tests pass in 199.67 seconds. Ruff passes across source, tests and
scripts; the changed Python files pass formatting. Pyrefly reports zero errors
with the two existing suppressions and 70 warnings.

Cold checked-in constructor replay repeats all 31 whole-span alternatives,
selects the complete U through full restored alpha and raw nongap support, then
selects the adjoining face from 21 geometric alternatives. Source observations
come from the unchanged original bank; 13 nearby material identities come from
the original partition and component. There is no saved body or optimizer seed
and no fixture primitive/face index. The smallest shared-edge result identifies
material 26 through original profile 67. Its two fields are exclusive and have
zero union XOR against the complete field. Literal outside-field RGBA remains
exact; the restoration alone gains one alpha byte at (396, 1998). Thus the
correct geometric/material interpretation still needs a feasible native floor.
The replay is `.bench/cel-attached-generic-full-u-source-faces-replay/`, at SHA
`fb2ae34bb9f3003949f9d9582ce27df5f828d3996c1f2d72b62bc820e5cf5a1b`.
It is construction evidence, not a published or accepted drawing.

The original right-side painted-source failures exposed the wrong restoration
paint. The lower-right part of the U shares an original curve with material 26,
a dark `#584022` face. Restoring that part with its original face paint removes
the gold fringe. Merely placing the U above the original dark face does not
repair the gold-only restoration, and reducing restoration opacity does not
improve complete alpha.

An independent saved-guide audit of the mixed-face drawing supports all 179
original U body samples and all 449 additional samples. The actual U plus the
unchanged genuine collar supports all 185 collar samples. Every original
painted-source comparison passes, as does the added painted profile and source
absence. Original material geometry/attributes and retained shadow command
values are exact; complete former-field removal, byte-exact native reload and
RGBA outside the literal field plus actual stroke body pass. Reassigned node
IDs are not original command changes. The drawing remains **rejected**: eleven
pixels gain one alpha byte, and original ownership/component acceptance is
unproved. This saved fitted guide is diagnostic evidence, not cold generation.

The mixed drawing and preview are under
`.bench/cel-attached-full-u-mixed-face-fixed-body-audit/`. Related source-scope,
shared-face discovery, winding-floor, order-only, opacity and cold joint-fit
probes retain separate driver/helper hashes and their original algorithm SHA
`806df7da36fb5d43259d852bb7d369d32d852194c6e94a50cf9ff9d42f342b14`.
Its corrected command/locality audit separately records the current revision;
it does not retroactively change those earlier generation claims.

A fresh cold shared-face/body diagnostic starts at the measured raw width,
prioritizes frozen original/added positive-body and painted-source feasibility,
and normalizes the tiny restoration parameters without increasing their
0.0625-pixel physical bounds. It retains a separate best source-feasible guide,
so a lower numeric score with missing strokes cannot be presented as improved
stroke output. At 600 evaluations and 24 parameters it supports all 179 original
and 449 added U samples plus the genuine 185-sample collar, with zero fitting
painted-source penalty, but leaves 387 total alpha bytes different. It generates
no candidate. This is
`.bench/cel-attached-generic-full-u-source-first-replay/`, under the current SHA
above; this runtime fitter is not adopted. Starting at a feasible wider stroke
and weighting source penalties more heavily does not solve the coupled fit.

Appending each geometrically exclusive restoration field to its original
material owner was also tested independently on both the stroke-hidden floor and
the saved positive-body drawing. Original paint/frame and command prefix remain
unchanged, but native alpha discrepancies increase: the separate floor's one
pixel becomes eleven for either individual append and twenty-one for both.
The body drawing also worsens. This is
`.bench/cel-attached-full-u-shared-face-existing-owner-probe/`; it establishes
neither source/ownership acceptance nor a better restoration mode. Exact
geometric union does not imply native equivalence of compound path rendering.

## Alpha-exact complete U and remaining ownership gate

A different restoration interpretation preserves the complete former field as
base material and restricts the adjoining dark-face overlay to fully covered
original parent pixel cells. Those cells are measured under the actual opacity
hierarchy. The dark overlay is an ordinary 108-node vector path, not an image,
clip or mask. It overlaps base underpaint; this is a separate interpretation
from the earlier exclusive two-face partition. Complete stroke-hidden native
alpha is now byte-exact. Lowering stroke opacity alone still loses qualified
source-body support and is not adopted.

Increasing the complete source guide's curve resolution helps more than fitting
the earlier six-node centered trace. `source_centerline.construct` fits the
original opposing-rail midpoint guide while inserting every freshly measured raw
corner as an exact segment endpoint. Its ports must already equal the measured
ports; it does not infer a connection. It selects the most resolved tolerance
that fits within the existing 24 local-normal/width parameters. The nearest guide
point may change at most two native pixels to meet a raw corner. Construction
returns immutable points and unique node IDs; source/gap/paint/native/ownership
proof remains separate. **The helper is disabled in production search.**

Ten focused controls cover exact raw corners and ports, transformed native points,
input immutability, far-corner and unbound-port rejection, invalid/oversized
inputs, fitting bounds, interruption and straight paths. All 1,508 CEL tests pass
in 206.20 seconds; source/tests/scripts Ruff and changed-file formatting pass.
Pyrefly reports zero errors, two existing suppressions and 70 warnings.

A fresh 23-parameter, 600-evaluation body fit yields a genuine nine-node open U
stroke of width 0.9304574064110984. It uses no saved optimizer vector. Its physical
collar ports and measured raw corner stay exact. An independent complete drawing
audit verifies:

- Every full native alpha byte, literal outside field-plus-actual-body RGBA and
  native save/reload RGBA are exact.
- All 179 original U and 449 additional body samples are supported by the actual
  new stroke; the genuine U plus unchanged collar supports all 185 collar samples.
- Original and additional painted-source comparisons have no rejections or new
  gaps; the added painted profile has no missing samples. The complete original
  bank remains 13,145 qualified samples, with 1,630 missing reported separately.
- Original material geometry/attributes and retained shadow commands are exact;
  independent Boolean checks prove complete former-field removal, exact retained
  difference and no residual ink in the field. The stroke is open, `fill="none"`,
  crossing-free and passes source absence and the native SVG subset.

This is still **not an accepted candidate**. Original-ancestor ownership replay
needs at least 65 entries, exceeding the unchanged 64-cut ledger. Per-original-
atom class encoding fixes the separate 65-versus-64 class-count issue but not the
entry limit. The established stroke/residual plus restoration-underlay contract,
and a dominant geometric coverage classification, also exceed 64 cuts. No
namespace, cut allowance or coverage witness is altered to pass. The drawing's
pixel/source proof does not replace the missing ownership/component proof.

The next design proposed at that checkpoint was ownership of a physical source family:
retained shadow, editable stroke and declared restoration paths can render one
original source owner whose immutable raw region contains both shadow and ink.
This needs a real serialized ownership contract, complete part/dependency and
cost declarations, and synthetic controls that reject undeclared parts, missing
source support and resurrected filled outlines. Its implementation and saved-drawing
ownership proof are recorded below; complete automatic generation remains open.
Simply assigning an unsupported surface membership to avoid a cut is excluded.

The first fitted construction and independent audit retain algorithm SHA
`fb2ae34bb9f3003949f9d9582ce27df5f828d3996c1f2d72b62bc820e5cf5a1b`
under `.bench/cel-attached-full-u-raw-corner-midrail-body-replay/` and
`.bench/cel-attached-full-u-alpha-exact-independent-audit/`. After adding the
checked-in constructor, a fresh body replay exports byte-identical SVG under SHA
`629de136211f05e53cff067424d43478e24c24cdb3c6fcc030059167d881ec5f`,
with a separate full independent audit under
`.bench/cel-attached-full-u-checked-centerline-body-replay/` and
`.bench/cel-attached-full-u-checked-centerline-independent-audit/`.
The native field/opaque face domain is a saved diagnostic input; generic cold
end-to-end construction and experimental search enablement remain unproved.
Failed ownership drivers/logs are archived separately. The latest preview is
`stroke-preview-ledger-pending.png` in the independent audit directory.

A subsequent replay keeps the original measured-width search bounds explicit:
raw width 2.25412654876709, fitted width in 1.127063274383545–3.3811898231506348.
Starting from the same fresh checked-in rail/corner seed, it reaches a nine-node
stroke of width **1.1341779707238542** in 525 evaluations. Independent full native
alpha, literal locality, reload, original material/shadow geometry/attributes,
complete field removal, physical ports/raw corner, every original/added positive
body sample, the genuine collar, source absence and painted-source comparisons
all pass. The original painted bank reports 1,629 missing samples with no
rejections; all 449 added painted samples pass. This supersedes the thinner
0.930457 preview as the current visual checkpoint and retains the same unresolved
ownership gate. The source classification based on unchanged retained geometry
and full removal field is independent of this stroke-width change.

The replay and independent preview are
`.bench/cel-attached-full-u-measured-width-centerline-body-replay/` and
`.bench/cel-attached-full-u-measured-width-independent-audit/`, at the checked-in
constructor SHA above. There are no saved body/optimizer seeds, wider physical
movement limits or alpha exceptions. The saved generic field and native face
restoration domain, original ledger gate, end-to-end cold construction and
production enablement remain explicitly separate outstanding work at that checkpoint.

## Physical source-family ownership checkpoint

The serialized ownership model now explicitly declares the physical parts of one
unchanged source owner. A raw region containing both shadow and attached ink can
render as an editable stroke, its exact residual shadow and separately painted
material restorations. The declaration does not claim that the thin stroke alone
paints the entire source region, and introduces no new primary memberships or
source cuts. Legacy partitions retain version 1; a partition with physical families
uses version 2 and cannot silently serialize as a legacy partition.

Binding records the actual original owner geometry, attributes and native frame.
Validation requires a genuine open stroke, exact original shadow command prefix
and Boolean difference, no retained ink inside the complete removal field, and
complete restoration under original material paint/frame. Every part and original
material dependency has a native-reload-stable revision. A sealed component edit
must declare all parts and materials and retain the original family memberships.
Ordinary operators preserve these declarations and dependencies while permitting
independent edits; this does not enable complete-U generation in search.

The saved nine-node U of width 1.1341779707238542 now passes an independent
ownership audit as well as every earlier body/native/source check:

- All primary owners and the exact original atom ledger remain unchanged:
  **64 cuts and 80 allocated children**, before and after this interpretation.
  The earlier accepted handle route's 81-child ledger is a different candidate.
- Two sequential component seals prove original ancestor to fitted material
  parent, then that exact parent to the physical-family drawing. Partition
  metadata and the native project both reload and validate.
- Exact full native alpha, outside-field-plus-actual-body RGBA and save/reload
  RGBA hold. Original materials, retained shadow controls, complete field removal,
  exact physical ports/raw corner, real source gaps and absence checks remain valid.
- The actual stroke supports all 179 original and 449 additional U samples;
  combined with the unchanged genuine collar, it supports all 185 collar samples.
  The original painted bank still reports 1,629 missing out of 13,145 qualified
  samples, with no rejections or new gaps; the added profile has no missing samples.
- Every physical path is charged: three added paths, 158 added nodes, four added
  contours and one private gradient increase complete representation cost from
  5,696 to 5,888. Ownership grouping hides no geometry or paint cost.

Twenty-six focused controls cover saturated ledgers, original membership, native
and metadata reload, ancestor/material/part changes, private gradient clones,
equivalent numeric opacity, independent operator composition, undeclared or aliased
parts, reassigned owners, forged parent snapshots, retained filled outlines,
restoration gaps/escapes, paint order, changed alpha and interruption.
All **1,534 CEL planning tests pass in 203.53 seconds**. The affected ownership,
component and opacity subset passes all 87 tests; source/tests/scripts Ruff,
changed-file formatting and Pyrefly pass (two existing suppressions, 70 warnings).

The saved-drawing audit and preview are in
`.bench/cel-attached-full-u-physical-family-independent-audit/`, at algorithm SHA
`5e770d1a6c8272e0b9977ebf03b90409d80e2fa4e0e1941a15a78e0bee0e5f50`.
The body/field are saved diagnostic inputs; **cold complete construction and
experimental search acceptance remain unproved**. `accepted` remains false in
the report. This resolved the ownership representation at that checkpoint, not the remaining
generation/integration goal, and does not change ordinary CEL behavior.

## Fresh complete-span construction checkpoint

`span_restoration.construct` now builds the complete material floor from an
original opposing-rail span and optional original shared face. It copies the
whole removal field under the original base material and restricts the adjoining
face overlay to measured opaque parent cells. Retained shadow commands and all
original material geometry/paint/frame remain exact. Complete native alpha and
literal outside-field RGBA equality are required **before** fitting a stroke.
The intermediate owner is unpainted: it cannot bind an accepted physical family
until a genuine positive open stroke is installed and separately verified.
The constructor remains disabled in search.

Eighteen controls cover opaque/translucent and fractional frames, private
user-space gradients, complete removal and retained controls, full native alpha,
locality/reload, genuine final family binding, old-ink resurrection, changed
prefixes, escaping/open/oversized fields, unsupported materials, singular frames,
unshared faces, valid geometry with invalid native alpha, bounds and interruption.
All **1,552 CEL tests pass in 202.97 seconds**. Ruff, changed-file formatting and
Pyrefly pass (two existing suppressions, 70 warnings).

A fresh runtime orchestration enumerates the unchanged original source profiles
and derives 17 source fields and 40 distinct complete primitive alternatives.
Matching both inferred endpoints to an actual same-component genuine stroke
leaves two U alternatives. Base paint is selected through the longest complete
shared original material curves; the adjoining face is independently inferred
from original source fields/material edges. The smaller 591.156-area field is
excluded by restoration construction. The complete 598.003-area, 20-node field passes with a
108-node opaque face overlay.

No saved field, face, guide, source trace, body or optimizer vector enters this
generation. The fitted material parent and supplied owner filter remain explicit
diagnostic inputs. A fresh source/corner seed gives nine nodes and 23 fitting
parameters. A bounded local-normal fit, charging each original body obligation
once, reaches width **1.1625470830648519** within 600 evaluations. The original
measured width is 2.254244804382324; its original half-to-one-and-a-half width
bounds and two-pixel movement bounds remain unchanged. The actual existing
collar body supports 184 of its 185 observations; its genuine union with the
generated U supports all 185. Classification uses actual existing stroke support,
not a collar profile index.

Complete generation checks pass native alpha, literal field-plus-actual-body
RGBA locality, original material/shadow geometry and attributes, original atom
ownership and two component seals, source absence/gaps, native subset and reload.
All 179 original U and 448 freshly sampled additional body positions pass. A
separate independent audit also preserves the **earlier frozen 449-position
trace**, including every actual isolated body query and painted-source observation.
Its whole expected former-field subtraction is exact; original primary ownership
and all 64 cuts/80 children remain byte-identical. The original 13,145-sample
painted bank still reports 1,629 missing, with no rejections or new gaps. Complete
representation cost remains 5,888 against the parent's 5,696.

The new fresh construction, complete fit and independent preview are in
`.bench/cel-attached-cold-span-restoration-discovery/`,
`.bench/cel-attached-cold-span-complete-body/` and
`.bench/cel-attached-cold-span-complete-independent-audit/`, at algorithm SHA
`299b5dfbb3cb8c3480981b222816d39a10bc642bd9ee1d986f40de4c10c5a945`.
The independent audit's older trace/domain are frozen validation obligations,
not generation inputs. Runtime drivers/helper/dependency hashes are archived
separately from the checked-in constructors.

**The goal remains unfinished.** Fresh generation and complete validation now
pass for this supplied owner, but discovery/port/material/body orchestration is
still runtime code and is not scheduled by the checked-in experimental planner.
Both reports retain `accepted: false`. Ordinary CEL and its ordinary High route
remain unchanged; no new reference corpus or learned ranking is introduced.

## Checked-in complete body fitter checkpoint

`SpanBodyFit` now owns the successful local-normal body fit. It consumes a
complete restoration floor, a freshly constructed source centerline, its frozen
whole-guide source bank and an actual original same-component stroke. Both
ports must coincide with a genuine open contour's literal endpoints; measured
raw corners remain exact vertices. Only cubic controls, unfixed interior cubic
vertices and width enter the existing 24-parameter, 600-evaluation budget.
Original material, residual shadows and junction geometry stay outside it.

Every original profile touched by the literal removal field contributes its
**entire** frozen qualified body contract, including queries outside the crop.
The actual existing stroke body classifies joint obligations; the new stroke
alone supplies all other obligations and the additional whole-guide contract.
Shared ancestor opacity, actual paint alpha, local width, caps and joins are
preserved in native white-body queries. Background fill supplies no body support.
The source qualification bank is unchanged: raw contrast qualification is not
replaced by the separate whole-guide ink-support percentage.

The fitter returns a drawing only after complete native alpha equality, literal
field-plus-actual-body RGBA locality, exact full-context/crop agreement, all
isolated/joint body queries, original and added painted-source comparisons and
source absence pass. Physical family binding/component acceptance stay separate;
its result reports `accepted: false` until the enclosing planner accepts it.

Fifteen controls cover curved connected chains, exact corners/ports, original
shadow/material preservation, fractional/translucent frames, profile-order
independence, full source tails outside the crop, real gaps, unknown source
identities, low-contrast qualification, unrelated corruption, unsupported bars,
nonuniform frames, parameter bounds, interruption, shared group opacity and
actual paint alpha/caps/width. All **1,567 CEL planning tests pass in 218.52
seconds**; Ruff, changed-file formatting and Pyrefly pass (two existing
suppressions, 70 warnings).

Fresh runtime discovery followed by this checked-in fitter reproduces the
successful nine-node U at width **1.1625470830648519**, with 23 parameters and
600 evaluations. Its complete generation and independent validation pass every
previous gate: all 179 original U and 448 fresh additional body queries, the
independent earlier frozen **449-position** trace, all 185 joint collar queries,
exact native alpha/locality/reload, original shadow/material commands and paint,
complete former-field subtraction, original ownership and two component seals.
The original painted bank remains 13,145 qualified / 1,629 missing with no new
gaps or rejections. Original 64 cuts / 80 children and representation cost
5,696 -> 5,888 remain unchanged from the previous complete interpretation.

Evidence and the updated visible preview are in
`.bench/cel-attached-checked-span-body/` and
`.bench/cel-attached-checked-span-body-independent-audit/`, under algorithm SHA
`cb160996bcfea15081c5376d64112d2a1b3f97900195742446f62876e2b8a850`.
Drivers, generation dependencies, independent foundation and runtime logs are
archived. No saved field, face, guide, source trace, body or optimizer vector
enters generation. The existing fitted parent and supplied owner filter remain
explicit diagnostic inputs. This checkpoint reproduces the previous visual
improvement through checked-in fitting code; it does not claim a new visual gain
or automatic planner completion.

## Automatic complete-span discovery checkpoint

`AttachedSpans` now discovers the interpretation from the actual fitted parent,
its original partition and the unchanged original source bank. It enumerates
eligible original owners near genuine open-stroke endpoint pairs, complete
opposing primitive fields, actual same-component port orientations and original
materials. Complete shared native material curves rank the base paint; source
fields/shared original material edges supply adjoining faces. The newly inferred
guide is bound to actual ports before raw ink measurement and corner-preserving
construction. Original observations and gaps remain separate frozen obligations.

The stage bounds original surface/geometry/port scans, sixteen eligible owners,
32 source fields and 16,384 source points per owner, 64 primitive alternatives,
four adjoining-floor alternatives and four retained complete candidates. Only
that bounded candidate pool retains documents and additional source banks.
Original source indices record provenance; geometry/face area and path commands
rank construction. An exact restoration floor does not prove a positive stroke
or authorize publication. All constructed candidates retain `accepted: false`.

Physical restoration part IDs now derive from the actual material and restored
path commands. Equivalent reparsed faces keep the same part identities instead
of inheriting incidental parser-generated geometry IDs or source-list indices.
A dedicated repeat-construction control verifies this behavior.

Seventeen new discovery controls cover curved complete chains, rotated and
translucent frames, real same-component ports, original shared material curves,
automatic owner selection followed by genuine body fitting/family binding,
adjoining-face inference, immutable inputs/repeated construction, filled and
foreign/displaced ports, nearby unshared edges, unsupported paint, real gaps,
native-frame validation, allocation bounds and interruption. All **1,585 CEL
planning tests pass in 220.73 seconds**. Ruff, changed-file formatting and
Pyrefly pass (two existing suppressions, 70 warnings).

The sword replay supplies **no owner/profile/primitive/material IDs**. Automatic
discovery examines five eligible original owners and 32 genuine port pairs among
65 original supported materials. It derives 31 source fields and 44 primitive
alternatives across those owners, reaches two complete port matches, rejects
four native-invalid restoration alternatives and retains four complete floors.
The first constructed interpretation is the inspected complete U, with its
original Gold/Dark materials and actual existing collar stroke.

A fresh checked-in body fit reproduces the verified nine-node stroke at width
**1.1625470830648519**, within the unchanged 23-parameter/600-evaluation bounds.
Generation and independent validation preserve all 179 original U and 448 fresh
additional actual body queries, the earlier frozen **449-position** independent
trace and all 185 genuine joint collar queries. Complete former-field removal,
retained original shadow commands, original material geometry/paint, literal
ports/raw corners, native alpha/locality/reload, original ownership and two
component seals pass. The original source bank remains 13,145 qualified / 1,629
missing, with no new gaps or rejections. The original 64 cuts / 80 children and
complete representation cost 5,696 -> 5,888 are unchanged.

Evidence, drivers/dependencies/logs and the automatic-discovery preview are in
`.bench/cel-attached-automatic-span-discovery/` and
`.bench/cel-attached-automatic-span-discovery-independent-audit/`, under algorithm
SHA `6e5cf741129abd2ed5c7bde887baa0c6683c179af5c36305fe2103d028d8f7c7`.
The fitted parent and original raw source bank are the only generation inputs.
Saved fields, faces, guides, source traces, bodies and optimizer vectors remain
outside generation. The earlier trace/domain are independent validation inputs.

**The goal remains unfinished.** Complete discovery and fitting are checked in,
but experimental search does not yet schedule or accept this interpretation.
The current replay's genuine junction is unchanged and fully verified; integration
must also declare/protect that original junction dependency so later independent
geometry or paint edits cannot disconnect the new chain. This checkpoint adds
automatic construction, not another visual gain or a release claim.

## Resume here

The explicit filled-band co-planner still schedules the previously verified
handle chain. The complete U now passes automatic checked-in discovery/fitting
and independent validation. Search integration and explicit connection sealing
remain unfinished; reports retain `accepted: false`. Ordinary CEL and ordinary
High keep the experimental flag off.

1. Declare the original genuine junction stroke in the physical family's sealed
   dependencies. Preserve its actual geometry, paint/frame and literal ports
   during subsequent independent edits and save/reload. Keep original materials
   usable as unchanged restoration donors; family protection must not prevent
   constructing another independent chain from those original materials.
2. Schedule complete attached interpretations against their actual fitted parent
   in explicit experimental search. Bind families atomically, preserve existing
   families and the original namespace/cut/child limits and primary memberships,
   and declare every part/material/junction in dependency, component and complete
   representation cost accounting. Retain all body/source/gap/native/locality
   checks and use the common acceptance/frontier rules.
3. Independently validate the scheduled result, including the earlier frozen
   trace, whole former-field removal, original ownership/component lineage and
   native/metadata reload. Show its stroke preview with the actual acceptance
   status before claiming the goal achieved.

No new reference corpus, learned ranking, automatic-operation quality result or
release gate is claimed by this draft. The active goal remains unfinished.
