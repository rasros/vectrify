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

The prototype distinguishes exact alpha from premultiplied-color quantization:
`color_quantization` permits at most one premultiplied color byte outside the
selected field, with alpha unchanged exactly. This is explicitly weaker than
byte-exact RGBA locality. Its intended use depends on exact geometric subtraction
and unchanged original controls/paint. A complete-candidate check now compares the final fitted drawing with its actual
material parent, bounding changes by the removed field and old/new affected
stroke bodies. Retained shadow bounds do not authorize unrelated changes.
Global native alpha equality is mandatory. Off-field color quantization remains
reported separately from byte-exact locality and is not a release policy.

## Current evidence and limits

- Eight synthetic controls pass. Opaque, translucent and nested-group cases
  publish a two-node editable stroke, remove its old filled field, retain shadow
  commands/values, replay a saturated ancestor ledger and round-trip rendering.
  Background paint changes, changed group alpha and displaced retained shadows
  fail the complete-candidate locality check.
- The previous partial-opacity exclusion was caused by comparing a native alpha
  byte with an unrounded opacity product. `opaque_core` now measures a fully
  covered pixel through the same group-opacity stack in the native renderer.
  This bounds construction; it does not relax the final alpha equality check.
- A source-only direct replay on the previously accepted fitted sword parent
  still publishes **no attached-band candidate**. Profile 54 passes subtraction,
  seed selection, material continuation and bounded fitting. Its continuation
  complement differs at one alpha pixel; the final drawing differs from its
  material parent at 61 alpha pixels and fails the complete-parent check as well.
  A successful fit alone does not prove an acceptable replacement.
- Broad shadow interiors, internal source gaps, ambiguous chains, unsupported
  winding and bounded/interrupted work remain excluded. Source observations never
  copy the human rendering's repaired handle connection.

Ruff, formatting and the full source/test Pyrefly check pass locally. The 88
source-slice, band-plan, source-band, source-junction, cap and material-continuation
tests pass in 6.02 seconds. Draft CI was
green at `06949c5`; the changes described here require their own CI gate.

## Active goal

Make complete source-supported attached outline chains on the fitted material
candidate into clean editable strokes, removing their former filled outlines
atomically while preserving the inspected shadows and real source gaps.
Validate complete-candidate pixel locality, exact native alpha, original
ownership ledgers and save/reload. Demonstrate a visible sword improvement with
focused synthetic controls, then present a stroke preview for review. Learned
ranking and broader reference collection remain deferred.

The current direct diagnostic is in the ignored
`.bench/cel-attached-complete-locality-probe/` directory. It uses the captured
material ancestor and fitted candidate 11, receives no human target and records
the completed rejection stages. Its rejected project is a diagnostic, not an
accepted candidate or a new quality result.

## Resume here

1. Fit the attached stroke and its material restoration so the final drawing
   retains its parent's native silhouette/alpha. Keep the 61-pixel profile-54
   rejection as the concrete diagnosis; do not waive those changes.
2. Preserve exact residual shadow controls and removed-field geometry. Any
   accepted replacement must pass the complete-parent locality check, including
   real gaps, caps and source-supported junctions.
3. Replay complete ownership, original ledgers, residual controls, native body
   gaps, save/reload and the exported editable stroke inventory, then show the
   actual strokes for visual feedback.

No new reference corpus, learned ranking, automatic-operation quality result or
release gate is claimed by this draft. The active goal remains unfinished.
