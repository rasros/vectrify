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

## Independent review status

The October 10 review found a blocking experimental-ranking defect. Isolated
stroke alpha is pooled before a pooled visibility test. A fully occluded opaque
stroke can therefore borrow the darkening of an overlapping 0.04-opacity visible
stroke. Moving the hidden stroke underneath that faint line leaves every native
RGBA byte unchanged but changes missing support from **117 to 0**, improves the
visual score by **0.04**, and is accepted and selected by common search.

Actual visible ink contribution must be measured after complete paint occlusion
before applying the body-support threshold. The existing hidden-stroke controls
do not cover this overlapping case. The validated sword U below remains a valid
physical conversion, but the quality criterion is not generally safe against
hidden ink and the PR remains a draft. The minimal reproduction and report are
in `.bench/pr-316-independent-review/hidden-ink-repro.py` and
`hidden-ink-report.json`.

Issue [#317](https://github.com/rasros/vectrify/issues/317) was implemented by
merged [PR #325](https://github.com/rasros/vectrify/pull/325). Its linked-outline
support and the subsequent merged changes are integrated from `main`. Shared
geometry remains excluded from attached-field conversion, which must not detach
a linked fill/outline pair. The combined branch passes 186 focused MCP, editable
ink, attached-proposal and source-family tests, Ruff and Pyrefly.

The combined-main review also reproduces a protected-feature compatibility
defect: giving the conversion owner a protected `position` node makes attached
fitting raise `DocumentError` when it replaces the owner geometry. The generator
does not catch that exception, so planning aborts instead of declining an
unsupported interpretation. Protected owners must be rejected before fitting or
their protected node identities must be preserved atomically. The reproduction
and report are `protected-feature-repro.py` and `protected-feature-report.json`
in the same review artifact directory. The previous sword candidate still passes
its complete native/source/ownership/reload audit on the integrated branch.

Merged [PR #329](https://github.com/rasros/vectrify/pull/329), the requested
integration target, is now included from `main`. Its detail subdivision,
coupled smooth tangents, bounded handles and enclosure protection integrate
without conflicts. The combined branch passes 290 focused tests (20 skipped),
Ruff and Pyrefly. The saved sword compatibility audit again proves native
alpha/locality, original ownership, source body/gaps and exact native/metadata
reload. This is compatibility validation of the existing candidate, not fresh
generation; both independent-review findings above remain open.

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

It renders actual isolated stroke alpha, the complete drawing and the drawing
with eligible strokes removed. A query receives support only where actual stroke
alpha visibly darkens the complete drawing. Filled paint, occluded strokes, white
or bright overlays, and shared-group opacity counted twice cannot supply support.
Opaque body checks also reject newly completed original source gaps; complete
families retain the original painted-source preservation gate.

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
in **191.12 seconds**, with one attempted/accepted/checkpointed proposal and zero
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

Evidence is in `.bench/cel-attached-editable-ink-search/` and
`.bench/cel-attached-editable-ink-search-independent-audit/`: native projects,
SVGs, partitions, complete reports, copied runtime/audit drivers and the preview
`stroke-preview-automatic.png`. Generation and audit verify the unchanged
algorithm SHA **7655aaa7f049f2acbf0dc7425fa242b8686a2a1899ce42240393499e2b4ba635**.
Generation uses the actual fitted parent; the independent audit's older trace
and field remain validation inputs only.

All **1,694 CEL tests pass in 236.54 seconds**, including **22 editable-ink
controls**. Ruff across source/tests/scripts and changed-file formatting pass;
Pyrefly reports zero errors (two existing suppressions, 72 warnings). Validation
commands use `PYTHONPATH=src:. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4` and the
project virtual environment:

```sh
python -m pytest tests/refine/test_cel* -q
ruff check src tests scripts
ruff format --check src/vectrify/refine/cel_plan/{editable_ink,line_fidelity,local,policy,frontier,search}.py tests/refine/test_cel_plan_editable_ink.py
pyrefly check --python-interpreter-path /home/rasmus/Workspaces/vectrify/.venv/bin/python
```

This satisfies the attached-stroke checkpoint; the broader release work listed
above remains open and this PR remains a draft.
