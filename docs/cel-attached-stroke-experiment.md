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
and unchanged original controls/paint. The final complete-candidate locality
proof still needs work; the helper is not a release acceptance policy.

## Current evidence and limits

- Six synthetic controls pass. The opaque fixture publishes a two-node editable
  stroke, removes its old filled field, retains original shadow commands/values,
  replays a saturated ancestor ledger and round-trips the native rendering.
  Published residuals receive fresh editing ids; geometric controls remain exact.
- The partial-opacity fixture remains excluded. Its test verifies that no partial
  interpretation is published and that the original documents/ownership remain
  unchanged. This records a safe rejection, not successful translucent support.
- The pre-merge prototype sword replay completed with the original twelve
  candidates and **no accepted attached-band candidate**. One thin handle chain
  still changes an off-field alpha byte and remains rejected. Another passes the
  initial cut screen but fails later publication. Diagnose that later exclusion.
- Broad shadow interiors, internal source gaps, ambiguous chains, unsupported
  winding and bounded/interrupted work remain excluded. Source observations never
  copy the human rendering's repaired handle connection.

Ruff, formatting and the full source/test Pyrefly check pass locally. The 62
source-slice, band-plan, source-band and source-junction tests pass in 4.46 seconds.
The full suite passed on the merged parent (2,618 passed, 36 skipped); it has not
been rerun locally for this draft. Draft CI remains the separate full gate.

## Resume here

1. Diagnose the publication failure after the successful initial cut screen.
2. Prove locality on the complete final candidate relative to its parent, not
   merely on the raw subtraction and restoration complement. Decide the color
   quantization policy explicitly without relaxing native alpha or real gaps.
3. Establish partial-opacity support, or retain a documented exclusion with no
   claim that those chains are complete.
4. Replay complete ownership, original ledgers, residual controls, native body
   gaps, save/reload and the exported editable stroke inventory. Preserve the
   useful shadows and remove each replaced filled outline atomically.

No new reference corpus, learned ranking, automatic-operation quality result or
release gate is claimed by this draft. Further algorithm work is deferred.
