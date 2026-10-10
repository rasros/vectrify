# Attached outline stroke experiment

Draft [PR #316](https://github.com/rasros/vectrify/pull/316) follows merged PR #315.
Its checkpoint goal is to convert complete source-supported attached outlines
on the fitted material candidate into editable strokes, remove the former filled
outlines atomically, preserve the inspected shadows and real source gaps, and
prove native locality, alpha, original ownership and save/reload. Learned ranking
and broader reference collection remain deferred. The wider
[CEL redesign release plan](cel-redesign-plan.md) retains its open gates.

Ordinary CEL and ordinary High keep this experiment off. Complete attached
planning requires `Operators(..., filled_bands=True)` in High; editable-ink
ranking additionally requires an explicitly constructed
`Policy(..., editable_ink=EditableInk(original_guard, fitted_parent, partition))`.
The criterion freezes its original source bank and parent before generation.
A fresh frontier must score all candidates under that same policy and retain the
original detailed representation normalizer. Changing an existing frontier's
policy would mix incompatible scores.

## Independent review fixes

The October 10 review reproduced two blockers; both now have regression tests
and fixes. In the occlusion reproduction, moving a fully hidden opaque stroke
under a visible 0.04-opacity line leaves all native RGBA bytes unchanged. It now
also leaves missing support unchanged at **117**, gives **zero** score gain and
is rejected by common search. Coverage is measured per stroke through the whole
paint stack before pooling, so an overlapping stroke cannot lend it visibility.
Controls also cover same-background and white strokes, partial occlusion,
legitimate duplicate ink and shared group opacity.

Attached discovery excludes geometry with protected position, tip, corner or
junction nodes. Direct fitting checks the original and restored owner before
replacing geometry. A protected owner declines without constructing a proposal
or aborting common search; a read-only protected junction remains valid and is
preserved exactly. The original reproductions and updated reports remain in
`.bench/pr-316-independent-review/`.

The branch includes merged [PR #329](https://github.com/rasros/vectrify/pull/329)
and subsequent `main` changes through `acb7a79`. Earlier linked outlines from
[PR #325](https://github.com/rasros/vectrify/pull/325) remain supported: shared
fill/outline geometry is excluded from conversion, preserving the linked pair.
All **298 targeted tests**, Ruff and Pyrefly pass after the fixes.

## Complete construction

`AttachedSpans` discovers complete opposing source fields, genuine sibling
stroke ports and original shared material curves from the actual fitted parent.
Generation receives no selected owner/profile/material IDs, saved removal field,
body guide, human trace or fitted parameter vector. The original raw source bank
is authoritative; a correction in a human rendering cannot create source support.

Discovery supports opaque closed materials under the same parent, literal
endpoints of genuine open strokes, uniform invertible frames and private
user-space gradients. It bounds native images to 1,536² pixels, material tables
to 512 surfaces / 32,768 nodes, port pairs to 128, nearby owners to 16, distinct
source fields to 64 and returned floor alternatives to four. Unsupported cases
produce no attached interpretation. Already sealed physical families are
protected and excluded from source discovery; their original donors and
junctions remain read-only evidence.

`SpanRestoration` subtracts the whole former outline field with directed winding,
retaining the exact original shadow command prefix. Original paint/frame restore
the complete field; adjoining dark material is restricted to proved opaque
original parent cells. Every assembled floor must preserve full native alpha and
byte-exact RGBA outside the literal removal field. No embedded raster, clip or
mask supplies restoration.

`SpanBodyFit` fits the actual open stroke body with the literal junction ports and
measured raw corners fixed. At most 24 normal control/width parameters move;
controls remain within two native pixels and inferred width stays within the
existing bounds. User-fixed widths are declined. Optimization has at most 600
evaluations and a bounded crop, followed by independent full-viewport checks.
Every affected original qualified body query and the fresh complete-span body
queries must be supported by the isolated genuine stroke, under its actual width,
paint alpha, caps, joins and shared ancestor opacity. Painted source support,
real gaps, absence, crossings, exact full alpha and literal locality remain hard
gates. The fit optimizes those contracts and a small movement regularizer;
it does not reshape the inspected stroke merely to reduce raster error.

## Atomic ownership and search

`SourceFamily` seals the original owner snapshot and complete subtraction, every
physical stroke/residual/restoration part, original material dependencies, and
the original genuine junction and its two literal ports. The intermediate cut
cannot be accepted as a drawing. Original primary memberships, atom cuts and
allocated children remain unchanged. Family metadata binds native editor
identities and revisions and survives native project save/reload.

`AttachedOutlines` fits at most four automatically discovered spans, declares all
physical and material/junction dependencies, and seals the actual fitted-parent
component. Original ancestor → fitted parent and fitted parent → physical family
are two separate component proofs. It preserves existing families and retains
new additions only when their component seal is valid. Every added path, node,
contour and paint server is charged by the common representation cost.

Experimental High schedules these complete interpretations first on a fitted
material/band parent while retaining the other operator slots. Common search
still checks partition lineage, component dependencies, local score, native
subset and full checkpoint agreement. `seed_document` preserves sealed native
identities on resume; its exported SVG must exactly equal the selected scored
seed. Reimporting SVG cannot substitute for a native project handoff.

## Explicit editable-ink quality

Raster error alone can prefer a filled outline to a clean editable stroke.
`EditableInk` measures visible genuine stroke support at every originally
qualified source position, using the bank's frozen raw-positive matching windows.
Neither added profiles nor candidate-created queries increase its denominator.
A source-frame/digest check prevents pairing the criterion with a different raw
reference.

Eligible ink consists of original genuine strokes and newly sealed complete
attached removal families. New unrelated or unsealed paths receive no credit.
The criterion validates the original family owner, original geometry/paint/frame
and unchanged genuine junction. Changed or added undeclared paint intersecting a
new family's removal field excludes that new stroke from quality credit, so a
clone of the old filled outline under the stroke cannot imitate complete removal.

Criterion version **2** measures each solid stroke independently. Whitening
that stroke in the complete native drawing reveals its alpha after every later
paint and ancestor opacity; a separate probe verifies that it darkens the
non-ink backdrop without borrowing another eligible stroke's darkening. Visible
contributions are pooled only after both checks. Filled paint, occluded strokes,
white or same-background paint and double-counted group opacity cannot supply
support. Duplicate genuine ink retains credit. Opaque body checks also reject
newly completed original source gaps; complete families retain the original
painted-source preservation gate. Version 1 scores must not be mixed into a
version 2 frontier.

The experimental visual score is the unchanged raster visual term plus the
existing `Weights.detail` weight (0.04 by default) times the original qualified
query missing fraction. It introduces no sword-specific weight or cost discount.
The full representation cost and fixed original normalizer still control the
existing complexity slider. Reports separate `raster_visual` and editable-ink
terms, including criterion version, qualified count and missing count. Ordinary
policies retain their original terms and behavior.

Local evaluation recomputes this bounded full-native criterion for the actual
proposal document/partition. Checkpoint evaluation independently remeasures it
against the same frozen bank. Native document/SVG mismatches, corrupt families,
unsupported frames, memory/query limits and interruption cannot publish a partial
score. The bounds retain the common 8,192-object / 32,000-node seed limits, with
256 eligible strokes and 32,768 qualified or gap queries.

## Validation scope and remaining release work

Focused synthetic controls cover complete family removal and reload; original
source qualification; normal matching windows; transformed/translucent bodies;
filled, hidden, bright, zero-alpha, unsealed and unrelated paths; reintroduced
filled ink; real gaps; shared-group opacity; corrupt metadata; limits and
interruption. Common-search controls verify local/full score parity and a
complexity tradeoff, without an acceptance exemption for stroke proposals.

The sword diagnostic runs the actual experimental scheduler/common search from
the existing fitted native parent, restricting the proposal stream to its first
complete interpretation. It uses a 480-second offline budget and the original
20,818 cost normalizer. It is not a full ordinary-mode performance result. An
independent audit may use the earlier frozen trace and known field as validation
obligations; neither is supplied to generation. The accepted geometry and native
project must also match the actual selected frontier candidate.

This checkpoint covers a complete attached U, rather than every sword outline.
Ordinary-mode latency, wider-corpus quality and the release-plan gates remain
open. Further references and learned ranking are separate follow-up work.

## Selected sword checkpoint

Fresh automatic scheduling and common search accept and checkpoint the complete
nine-node U at width **1.1625470830648519**. The actual frontier selects it at
complexity **75 and 100** and retains the cheaper fitted parent at **0, 25 and
50**. No rejection override or stroke cost exemption is used. Search completes
in **200.04 seconds**, with one attempted/accepted/checkpointed proposal and zero
score disagreements. Local/full term agreement is within **9.56e-10**.

| Measure | Fitted parent | Selected attached stroke |
| --- | ---: | ---: |
| Original qualified source queries | 13,145 | 13,145 |
| Queries missing visible editable support | 2,236 | 2,056 |
| Visible editable support | 82.9897% | 84.3591% |
| Raw raster visual loss | 0.0390481935 | 0.0392659599 |
| Explicit experimental visual loss | 0.0458523015 | 0.0455223311 |
| Complete representation cost | 5,696 | 5,888 |

The visual improvement is the inspected U geometry now earned by actual search;
this scoring change does not claim a further geometric improvement over the
previous rejected U. Raw raster loss still increases. The new semantic criterion
values the conversion of originally qualified filled ink into visible editable
stroke support, while the complexity cost retains the cheaper alternative.

The independent selected-candidate audit proves:

- Every affected **179 original**, **448 freshly measured** and **185 joint
  collar** body query passes; the earlier frozen **449-position trace** also
  passes as an independent validation obligation.
- Boolean intersection with the retained residual is zero, and its XOR against
  the complete original-minus-field difference is zero. Original shadow commands
  remain exact, as do all other original material geometry and attributes.
- The new outline is one open `fill="none"` stroke, with literal collar ports and
  raw corners fixed, no crossings and no newly completed real source gaps.
- Complete native alpha is exact; RGBA is byte-identical outside the literal
  removed field plus actual stroke body. Native project reload is byte-identical.
- All original primary memberships and atom labels are unchanged: **64 cuts /
  80 allocated children**. Both sequential component seals and all seven
  declared physical/material/junction objects validate.
- Family/partition metadata reload validates. Reload gives exactly the same
  editable-ink query counts and full native score as the selected search result.

Evidence is in `.bench/cel-attached-reviewed-search/` and
`.bench/cel-attached-reviewed-audit/`: native projects,
SVGs, partitions, complete reports, copied runtime/audit drivers and the preview
`stroke-preview-automatic.png`. Generation and audit verify the unchanged
algorithm SHA **f4eab0e621b98f2c37c1c8054bed656b93e49860b8e8844ebffe45afd8857d07**.
Generation uses the actual fitted parent; the independent audit's older trace
and field remain validation inputs only.

All **298 targeted tests pass**, including the visibility, common-search,
protected-feature, body-fit, discovery, restoration, ownership and latest-main
integration controls. Ruff across source/tests/scripts and changed-file
formatting pass; Pyrefly reports zero errors (two existing suppressions).
Validation
commands use `PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4` and the
project virtual environment:

```sh
python -m pytest tests/refine/test_cel_plan_{editable_ink,attached_outlines,span_body_fit,paint_continuation,span_restoration,attached_spans,source_family,ink_replace}.py tests/mcp tests/refine/{test_crossings,test_cleanup}.py tests/ui/test_tidy_settings.py -q
ruff check src tests scripts
ruff format --check src/vectrify/refine/cel_plan/{editable_ink,paint_continuation,span_body_fit}.py tests/refine/test_cel_plan_{editable_ink,attached_outlines,span_body_fit}.py
pyrefly check --python-interpreter-path /home/rasmus/Workspaces/vectrify/.venv/bin/python
```

This satisfies the attached-stroke checkpoint; the broader release work listed
above remains open and this PR remains a draft.
