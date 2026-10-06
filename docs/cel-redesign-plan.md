# CEL structural vectorization plan

Build a CEL generator that produces clean, editable shapes closer to a human redraw. Introduce a complexity slider backed by a measured representation budget, combine line and color evidence in one structural planner, and refine the planned drawing automatically. Start with deterministic planning and add a small learned candidate ranker only if it improves held-out results.

This is an implementation plan. The experimental method remains separate from the existing CEL method until it passes the rollout gates below.

The recommended overhaul has three parts: CEL supplies ink and silhouette evidence; color segmentation supplies surface and shading evidence; a structural planner chooses an editable drawing from both. Automatic path optimization follows those choices. A small learned model may later rank the planner's proposals, after the deterministic version establishes a measurable baseline. The complexity slider controls the drawing's representation cost; quality controls how much time the planner spends finding it.

The implementation status and measured experiments live in [CEL redesign progress](cel-redesign-progress.md). This plan defines the intended behavior and completion gates; experimental code is not evidence that those gates have passed.

## Decisions and immediate priority

The architecture decision is to combine CEL ink/silhouette evidence with color-region surface evidence in one owned graph. Choose surfaces, ink, shared boundaries and layers together, then automatically fit their geometry and paint. Keep the current editor representation and operation preview/apply contract.

The complexity slider is part of the first product release. It controls the cost of the drawing, while quality controls search effort. A small learned ranker is conditional work after deterministic operator coverage and score calibration; it is not on the critical path.

The latest committed native sword experiment (`8679895`, recorded in the progress document) shows why the next work must address structural compaction:

| Drawing | Nodes | Contours | Error against the human render |
| --- | ---: | ---: | ---: |
| Completed human fixture | 523 | 93 | 0 |
| Legacy CEL operation baseline | 2,312 | 339 | Approximately 663.31 |
| Experimental owned-family search, 60-second budget | 22,097 | 3,901 | 439.64 |
| Proposed balanced sword gate | At most 800 | At most 140 | At most 497.39 |

The experimental row uses complexity 50, balanced quality and refinement disabled. It meets the numerical error ceiling but fails both structural targets. It is a development result, not a matched-runtime improvement over legacy CEL. Native partial-alpha safeguards currently lead to a very dense starting drawing; two family merges cannot compensate for thousands of partitions. The immediate goal is to offer compact, faithful alternatives without depending on that density for coverage. Full measurements and hashes remain in the progress document.

Complete coherent surface models and ink replacement using the bounded native evaluator. Follow them with constrained geometry fitting, budget-directed scheduling and tuning-corpus calibration. Expose the controls when their behavior is validated, then run release evaluation. The tile evaluator now has independent native/full agreement tests; its sword run retains the same published drawing and does not establish a quality improvement or complete the runtime/memory gates.

## Evidence and success criteria

The completed sword fixture merged on 6 October 2026 contains 80 paths, 93 contours, 523 nodes and six gradients. The current default CEL output contains 137 paths, 339 contours, 2,312 nodes and 24 gradients. Requesting 12 regions still produces 64 visible regions because protected shadows and features are exempt from the target. The default budget uses canvas area even though only about 10% of the sword reference is foreground.

The sword experiments used the merged fixture from commit `4e93ccf` and CEL and Tidy sources unchanged between that commit and `07b5406`. Full-resolution RGB mean squared error was measured over the reference's alpha of at least 0.5, dilated by four pixels, with both images composited over white. Error against the raster was 415.65 for the human drawing and 292.65 for default CEL. The human drawing deliberately simplifies the reference, so raster error cannot be the sole acceptance criterion.

Two joint Tidy runs selected every traced path, allowed two rounds, and had 30-second limits. Both exhausted their limits and retained the shape step. Default CEL grew from 2,312 to 2,319 nodes and did not improve its full-resolution match. A trace using 12 regions and tolerance 3 followed by Tidy reached 1,040 nodes and lowered its error against the human rendering from 733.95 to 578.07. Default CEL's error against the human rendering was 663.19.

Use these initial engineering targets for the sword at balanced complexity and quality: at most 800 nodes, at most 140 contours, and at least 25% lower foreground error against the human rendering than default CEL, which means at most 497.39 under the frozen scoring mask. These are proposed targets, not achieved results. Keep the jewel, blade tip, handle wrapping and main shade boundaries identifiable. Do not impose an exact path count, copy the fixture's geometry, or introduce sword-specific recognition rules.

Evaluate the broader illustration set before changing defaults. Photos remain a separate workload; the first release targets cel art and illustrations.

### What the prototype establishes

The current development prototype is evidence for the next steps, not a completed delivery. A native-render run at complexity 50 produced 435 nodes, 71 contours and nine gradients in about 4.75 seconds. Its human-reference MSE was 749.68, worse than legacy CEL's approximately 663.31 through the same operation/export benchmark. Reducing geometry alone therefore does not pass the sword milestone. Earlier experiments also showed that centering a continuous outer stroke on the alpha boundary creates substantial spill; the fitted stroke must account for its width and the intended outer silhouette.

The small difference between the original diagnostic's 663.19 and the operation benchmark's 663.31 comes from its rendering/export path. Freeze the operation benchmark, mask and renderer for future comparisons, and retain the conservative 497.39 target. Version source hashes as well as commits for experiments on a dirty worktree.

Prioritize cleaner boundary interpretations and local ink/feature preservation before further aggressive merging. Do not use this prototype to justify default migration. In particular, accepting `refine` or a budget setting in the operation schema does not complete that setting until it changes behavior and has validation evidence.

A later structured candidate measured 449 nodes, 87 contours and human-reference MSE 744.49. When additional merge candidates finished, the score instead selected 329 nodes and MSE 816.46. Both fail the human-match target. Time-limited searches completed different candidate sets, so these runs cannot establish a quality improvement at equal search effort. The guard, jewel and blade-tip crops show why global counts are insufficient:

| Observation | Likely mechanism to investigate | Required competing proposal |
| --- | --- | --- |
| Small shade fragments remain within broad blade facets | Segmentation retains raster variation; curve fitting preserves its bends | One coherent shade surface with straight facet boundaries and a preserved tip |
| Ink breaks around the guard and jewel | Initial line detection, region boundaries and later ink reconstruction disagree | Continuous locally supported ink, with a filled mark when stroke width varies |
| The jewel remains subdivided despite its compact outline | Whole-shape proposals depend on fragmented region families | A closed compact outline fitted from boundary evidence across adjacent shades |
| Lower-cost output scores well while human resemblance worsens | The visual score and detail penalty may reward the wrong simplification | Individually validated edits and score calibration against clean paired targets |
| Selected output changes with the completed search prefix | Candidate availability changes under the deadline | A recorded common frontier, deterministic proposal order and explicit search completion status |

These are development hypotheses, not semantic rules for recognizing swords. Test each mechanism on synthetic and tuning artwork. A learned ranker cannot choose a clean facet or continuous outline if the proposal generator never offers it.

Opacity-aware validation subsequently exposed another initialization problem:
most of the sword reference is slightly translucent. The conservative RGBA
fallback reaches human MSE 446.79, but uses 54,320 nodes and 8,217 contours. It
therefore fails the structural milestone despite passing the error ceiling.
Compact proposals must preserve native low-opacity marks and holes while
combining paint variation into coherent surfaces; treating each alpha byte as
a separate region is not viable. The measured run and limitations are recorded
in the implementation evidence. This does not change the rollout gates below.

## Product and API decisions

Introduce `generate/cel-planned` as an experimental method using the existing operation contract. Preserve the existing `generate/cel` behavior and settings while comparing the methods. After rollout, the UI can recommend the planned method while API callers retain the legacy method during migration.

The primary controls are:

| Control | Proposed contract | Effect |
| --- | --- | --- |
| Complexity | Integer 0 to 100, default 50 | Sets the cost and budget of shapes, contours, nodes and paint models |
| Quality | `fast`, `balanced`, `high`; default `balanced` | Sets search breadth and refinement effort independently of complexity |
| Automatic refinement | Boolean, default true | Refines the completed plan within the quality and operation time limits |
| Gradients | Boolean, default true | Allows a linear gradient to compete with a flat fill |
| Line width | Zero measures ink; positive overrides it | Preserves an explicit user choice of stroke weight |

Keep palette size, boundary tolerance, feature protection and budget overrides in advanced settings. An advanced override must declare which derived value it replaces; changing complexity must not silently erase it. Do not expose a second region-count control as the primary complexity setting. Do not add a learned-planner toggle to the initial UI.

Use the setting keys `complexity`, `quality`, `refine`, `gradients`, `line_width`, `palette`, `tolerance`, `protection` and `node_budget`. Zero means automatic for width, palette, tolerance and node budget. A positive tolerance is expressed in source-reference pixels and replaces the derived geometry tolerance; a positive palette replaces the color-proposal palette limit. Protection scales the finite feature penalty, without disabling mandatory coverage or topology checks. A positive node budget requests an explicit node ceiling. If no candidate satisfies it without breaking those checks, return the best safe candidate and report the unmet budget. Validate unknown keys and invalid types using the existing method-setting contract.

Label the slider **Complexity**, with **Simple** and **Detailed** endpoints. Explain that lower values keep main shapes and higher values retain more fine detail. Show paths and nodes in preview summaries and give candidate alternatives useful labels such as “Simpler” and “More detailed.” Keep raw reference error available in diagnostics without presenting it as the definition of quality.

Apply adds one generated group as one undoable edit. Stop returns the best fully validated candidate available; before any candidate exists, cancellation behaves as the job contract already specifies. Changing the reference, target scope or settings invalidates a pending preview. Initially, dragging the slider invalidates the preview and updates the label; Generate starts the new job. Reusing cached candidates for immediate slider previews is a later enhancement.

## Pipeline and intermediate representation

Use a temporary planning representation; the editor document model and project format do not need a new geometry type. The pipeline is:

`Reference → evidence → region and boundary graph → structural candidates → complexity frontier → geometry and paint fitting → exact validation → SVG proposal`

The evidence bundle contains original premultiplied RGBA, alpha confidence, analysis images at several scales, color statistics, ink confidence, centerline chains, texture confidence and boundary samples. Preserve original pixels for final scoring. Do not discard partial alpha or bake the white preview background into the model.

The graph contains regions, adjacency, shared boundary chains, junctions and proposed occlusion relationships. Each shared boundary has one canonical geometry owner and references from the regions either side. A region records its area, paint samples, texture, contrast, feature confidence and component membership. A stroke records its chain, width evidence, ink evidence and junction relationships.

A plan contains component groups, shape candidates, paint models, draw order, shared edges, representation cost and scored alternatives. It also retains evidence that explains accepted or rejected decisions. The diagnostic record includes stage timings, real path/contour/node counts, score terms and rejection reasons.

Use ordinary SVG paths, groups and supported gradients on export. Fit ellipses into ordinary path geometry for this release. Assign neutral component names unless a semantic label is actually known. Do not pack unrelated parts into one compound path merely because their colors match; merging for storage should respect component membership and editing usefulness.

## Evidence and segmentation

Reuse CEL's line detection, trapped-ball fill, color splitting, boundary chains and alpha handling as the initial evidence generators. Factor them behind a stage interface without changing the legacy method's behavior.

Estimate complexity from foreground content and boundary structure, excluding empty canvas padding. For opaque images, infer a simple background only when border connectivity and color consistency support it; retain the whole scene when they do not. Background classification affects budgeting and grouping without silently removing opaque reference pixels. Normalize feature sizes and tolerances by a defined analysis scale; map fitted geometry back to source coordinates. Keep final silhouettes and thin-feature checks at source resolution or in native-resolution crops so a downsampled analysis cannot erase them.

Generate color evidence at more than one spatial scale. Broad shading should support a region or gradient, while a narrow coherent dark mark should support a stroke. Use local color models where one global palette loses differences within a component. GPU palette fitting may accelerate this stage, but CPU processing remains available.

Replace unconditional shadow exemptions in the new planner with finite protection costs derived from contrast, size, coherence and stability across scales. High-confidence small features remain strongly protected. Texture fragments compete against the cost of retaining them. Keep alpha-derived empty space and intentional holes as topology constraints rather than ordinary merge candidates.

Separate silhouette cleanup from inner texture cleanup. A long supported edge may become a straight segment or a few smooth curves, while the tip remains a corner. Do not force symmetry globally; a symmetric interpretation is eligible only when bilateral evidence supports it and validation accepts it. Evaluate pale lines, dark fills, hatching and blurred edges as distinct ambiguity cases.

## Structural planning and layer construction

Start from an oversegmented graph and a valid CEL-derived candidate. Use bounded local search with a small beam of alternatives rather than enumerating every combination. Recompute proposals only near an accepted graph edit.

The first candidate operators are:

| Operator | Competing interpretations |
| --- | --- |
| Merge adjacent regions | One flat surface, one gradient surface, or retain the separating edge |
| Split a region | Two stable shades or one surface with texture |
| Reclassify a dark mark | Stroke, filled ink shape, shade boundary or texture |
| Join stroke chains | One continuous stroke or separate marks, using ink support and tangent continuity |
| Replace boundary geometry | Straight segment, smooth cubic chain, or retain corners |
| Fit a compact closed shape | Ellipse-like path or unrestricted contour |
| Reorganize coverage | Base fill with overlays or adjacent regions sharing boundaries |
| Change paint model | Flat fill or linear gradient |

Rank proposals by the change in validated visual error and representation cost. Do not accept a merge solely because it reduces region count, or a stroke join solely because its ends are nearby. Width transitions, junctions, corners and ink across gaps must support the resulting shape.

Use a two-stage proposal test. An inexpensive local estimate prioritizes edits; it cannot accept them. Rasterize a candidate in the affected source-resolution crop, including a margin for strokes and antialiasing, before changing the retained plan. Score the same visible context before and after the edit. Run full-canvas validation before retaining a preview or final result. Cache rejected edits by graph revision and operator parameters, and invalidate only proposals whose neighborhoods changed.

For the initial search, cap the beam at one, four and eight states for fast, balanced and high quality. Start with exact local evaluation caps of 16, 48 and 128 proposals per run, subject to the deadline; measure and revise these caps on the tuning corpus. Do not spend that allowance only on merges. Reserve proposals for stroke interpretation, boundary geometry and paint models. Beam states must share immutable evidence, while graph edits and geometry are independently owned. Deterministic ordering breaks ties by operator and stable region/boundary IDs.

### Local proposal contract

Each proposal records its operator, stable planning IDs, parent revision, affected geometry and paint, native-coordinate bounds, representation delta and evidence features. Its affected area includes the union of old and new visible bounds, stroke expansion, renderer antialiasing, score filter support and dependent layers. A gradient or order change can affect an entire shape; evaluating only its boundary is insufficient.

Render before and after with the same surrounding layers and coordinate mapping. Compute changed score contributions using the full policy's fixed denominators. Update cached feature contributions and their worst-feature aggregate; do not normalize a crop independently or average away damage to one feature. Count representation changes over the complete plan. Cheap graph estimates determine evaluation order, while exact local objective improvement determines whether an edit enters a working beam state.

Keep a separate fully validated checkpoint. Before publishing a beam state to the frontier, export and render the complete candidate and run document, coverage, topology, shared-edge and feature checks. If the local estimate disagrees with full scoring, reject the checkpoint and record the disagreement. This also tests whether crop margins or dependency tracking are incomplete.

Rejected proposals are reusable only while their dependency revisions and relevant settings remain unchanged. Cache by neighborhood revisions, operator parameters and score version, with a bounded entry count. Changes to a shared edge invalidate both adjacent regions and their ink; changes to draw order invalidate affected visibility dependencies. A candidate owns its edited geometry so rollback never mutates another beam state.

Stroke-versus-fill proposals compare local width variation, paired-edge support, junction topology and ink color. Keep tapered or strongly varying marks as filled ink shapes when constant-width strokes cannot explain them. Estimate ink and width locally rather than forcing one global style. For joins, measure tangent continuity and evidence across the entire connecting gap; reject unsupported bridges even when endpoints are close. A continuous exterior stroke is a competing interpretation only where ink supports it, including separate decisions for holes.

For boundary fitting, retain evidence-supported corners and graph junctions as fixed anchors. Compare straight runs, cubic fits and unrestricted contours between anchors. An ellipse proposal needs low fit residual, stable support at multiple scales and no supported corner that it would remove. The accepted model's constraint remains active during later refinement. These operators address the sword's straight blade facets, pointed tip and compact jewel without naming or recognizing those objects in algorithm code.

Build component groups from connectivity and boundary evidence. Establish a base fill, local shade/highlight overlays and linework in a consistent draw order. Treat partial occlusion explicitly: a base may continue underneath a covering shape instead of tracing every hidden edge. In the first release, restrict order changes to locally supported relationships and reject cycles or unexplained changes in visible coverage.

Maintain shared boundaries for truly adjacent visible fills. A base fill beneath adjacent subdivisions can close antialias seams without adding same-color strokes around every fill. Intentional overlaps are allowed; uncovered opaque interior and visible spill are not. Avoid introducing clip paths as a shortcut until the fitting and exact scoring paths support them consistently.

For a nearly uniform translucent component, let an isolated opacity group with a native core fill compete against adjacent RGBA surfaces. Normalize child fill and gradient-stop opacity by the group opacity so the base does not double translucency. Keep intentional holes out of the core and retain weaker fringes outside it. Group only connected material evidence; disconnected components and thin marks keep independent coverage. Broad variable-alpha surfaces require their own RGBA interpretation. Core thresholds are proposal parameters that must pay the full native color/alpha and feature score, rather than exemptions from it. Test project export/reload as well as the initial renderer, and include this coverage interpretation when establishing a validated detailed cost normalizer.

## Scoring and complexity

Use a normalized objective of the form:

`J = visual error + alpha and edge error + protected feature error + λ(complexity) × representation cost + geometric regularization`

Visual error combines robust color differences at native and coarser scales. Alpha error uses premultiplied RGBA and checks transparent references on contrasting backgrounds. Edge error measures supported contours and ink positions rather than every texture edge. Feature error gives annotated benchmark features and automatically detected high-confidence marks local weight. Geometric regularization discourages unsupported wiggles, abrupt tangent changes away from corners and needless width oscillation.

Representation cost charges for nodes, separate contours, primitive objects and gradients. Charge contours inside compound paths so color grouping cannot disguise complexity. Preserve useful component grouping rather than penalizing all groups indiscriminately. Calibrate weights on the tuning set and version them with the algorithm. A flat model should win when the gradient's improvement does not justify its extra parameters.

Low slider values increase the cost of detail and lower the available budget. Higher values reduce that penalty. Normalize budgets to visible content and structural evidence, not raster canvas size. The endpoints still preserve mandatory coverage and protected features; the UI reports actual counts when a requested budget cannot be met.

Cache a frontier of candidates that offer different tradeoffs between visual error and representation cost. Choose slider results from that common frontier where possible so total representation cost progresses consistently as complexity increases. Individual path and node counts can trade off; do not promise every count is independently monotonic. Test endpoints, intermediate levels, repeated runs, resized inputs and added transparent padding.

Hard validation rejects nonfinite geometry, new unintended self-crossings, broken shared junctions, lost protected holes, unsupported layer changes and visible coverage failures. Candidate retention uses the same documented score as planning, with exact rasterization for checkpoints. Raw pixel MSE is a diagnostic, not a veto against intentional simplification.

### Initial score and slider implementation

Start with representation cost `C = nodes + 4 × contours + 2 × (paths + primitive objects) + 12 × gradients`. Primitive objects are visible SVG elements such as rectangles, circles and ellipses that do not store path nodes; they still pay object and contour costs. Compact models exported as ordinary paths pay their actual path/node costs. Count every contour inside a compound path and every rendered use of reused geometry; exclude unused definitions. Report the raw counts alongside this cost. These weights are initial engineering choices to calibrate on the tuning set, not inferred human preferences.

Normalize cost by a reference-dependent structural estimate `C₀` computed once from the detailed candidate and foreground evidence, with a positive lower bound for empty or tiny images. Keep that normalizer fixed across slider levels and search states. Use `λ(c) = λ₅₀ × 2^((50 − c)/25)` as the initial detail-penalty schedule. Calibrate `λ₅₀` against the tuning corpus, freeze it with the score version, and do not tune it against held-out human drawings. Derive a soft representation budget from the same structural estimate; it increases with complexity and excludes transparent padding. Mandatory features establish a budget floor. A requested node ceiling is a separate, explicitly reported constraint.

Use `B(c) = max(B_min, C₀ × 2^((c − 100)/50))` as the initial soft-budget schedule: one quarter, one half and one detailed-reference cost at complexity 0, 50 and 100 before the mandatory-feature floor. Estimate `B_min` from the simplest validated interpretation that preserves mandatory coverage and features. This is a search target to calibrate, not permission to remove features or a promised node count. Report the target and achieved cost. Frontier selection remains governed by the common objective; a positive `node_budget` adds its separately documented feasibility rule.

| Complexity | Detail penalty relative to level 50 | Soft cost target before the feature floor |
| --- | ---: | ---: |
| 0 · Simple | 4× | 25% of `C₀` |
| 25 | 2× | Approximately 35% of `C₀` |
| 50 | 1× | 50% of `C₀` |
| 75 | 0.5× | Approximately 71% of `C₀` |
| 100 · Detailed | 0.25× | 100% of `C₀` |

Use these targets to schedule useful proposals as well as report them. When a candidate is far above budget, prioritize coherent family and layer replacements by expected cost reduction subject to local visual risk. Reserve opportunities for ink, boundary and feature corrections even when their immediate cost savings are small. Once near budget, emphasize visual improvements within the retained tradeoff frontier. Bound all priorities and retain deterministic ties; the requested budget never relaxes a hard gate. A cheapest-so-far candidate is a conservative observed floor, not proof that a lower safe cost is impossible.

Make the visual score's terms executable and separately inspectable:

- Compare premultiplied colors and alpha at native scale and two coarser scales. Use a bounded robust color loss so isolated corruption cannot dominate the entire plan. Weight coarse surface evidence more where texture confidence is high, while retaining native ink and silhouette checks.
- Measure alpha on both white and dark backgrounds; separately measure missing opaque interior, spill and intentional-hole preservation. Keep full-canvas alpha checks independent of the foreground RGB scoring mask.
- Compare bidirectional distances between high-confidence reference ink/contours and candidate ink/contours. Weight by ink confidence and distinguish an outer silhouette from an inner shade edge.
- Score automatically detected stable small features locally, so a long blade cannot dilute the loss of a jewel or wrapping mark. Human benchmark rectangles and clean SVG geometry are evaluator inputs only; generation never receives them.
- Penalize unsupported curvature changes, excessive handles and width oscillations. Exempt supported corners and intentional hatching.

Normalize each term by its fixed evidence support, not by the candidate's area or remaining features. Record term values, weights, score version and hard rejection reasons. Calibrate weights and per-case tolerances using a declared parameter grid and the tuning split, then freeze them before evaluation. This is a required delivery artifact; the current MSE diagnostic is not yet the proposed acceptance policy.

Build one nondominated frontier using visual score and representation cost. For a fixed evidence/settings key, choose candidates by `visual score + λ(c) × C/C₀`, breaking ties toward lower cost. This gives a consistent ordering of total cost as complexity increases. Prune dominated candidates only after exact validation. Keep at most 12 frontier candidates and three user-facing alternatives; retain a valid baseline even if it is dominated so interrupted runs have a known fallback. A separately generated frontier for each slider level cannot establish the promised ordering.

Generate the shared frontier from a bounded range of merge penalties and geometry tolerances, including the five slider checkpoints, then evaluate its candidates under the common score. Complexity-dependent proposal settings cannot change the scoring evidence or normalizer. With an explicit tolerance override, use that tolerance for all those candidates. Test monotonic selection against the same frozen frontier; do not compare unrelated time-limited searches and claim a monotonicity guarantee.

Refinement proposals also enter the common frontier before it is published. For a cached slider preview, reselect from that frozen frontier and expose its version. Independent fitting of each selected slider result creates different candidate sets and therefore does not support a monotonic cost promise. A later Generate run may extend the frontier; report the new version and completed search stages. Quality and time limits can change candidate availability, but do not change what complexity means.

### Score calibration and candidate diagnosis

Separate three failures before changing the score or training a model. Candidate coverage fails when no generated proposal resembles the clean target. Candidate selection fails when such a proposal exists but the production objective selects a worse one. Search scheduling fails when the useful proposal exists in a longer run but is unavailable at the matched deadline. Record the best clean-target result within each generated candidate pool as a benchmark-only oracle; it never guides production generation.

For tuning, retain clean target renders, all exactly evaluated candidates, production score terms, local feature metrics, representation counts and elapsed search effort. Compare production selection with the oracle at each complexity checkpoint. Inspect explicit conflicts such as a smooth but missing outline, lost small mark, flattened intentional gradient, or retained injected noise. Correct evidence and operator coverage before attempting to learn an ordering over inadequate candidates.

Declare the calibration grid in a versioned benchmark artifact before running it. Include color/alpha/edge/feature weights, geometric regularization, the detail penalty and soft-budget schedule. Choose parameters by aggregate clean-target quality subject to per-case feature, hole, coverage and line gates, then representation cost; do not optimize only the pooled average. The current initial weights supply the center of this grid, not a trained policy. Run each configuration against the same candidate pools for selection diagnosis, then rerun bounded search to measure the effects on proposal generation and runtime. Freeze the chosen grid result, metric tolerances and score version before held-out evaluation.

## Automatic refinement and runtime

Run a planned refinement sequence rather than invoking the entire Tidy job unchanged:

1. Simplify and fit compact geometric models before expensive gradient work.
2. Fit colors and choose flat versus linear gradient with geometry fixed.
3. Fit boundaries and stroke widths jointly within connected spatial groups.
4. Refit paint after geometry changes.
5. Remove redundant geometry and validate the final candidate.

Reuse shared-edge handling, joint fitting, stroke support, simplify and paint-fitting helpers. Introduce a score policy for the planned generator; existing manual Tidy keeps its current default acceptance behavior. Structural edits occur between fitting rounds, not inside a differentiable geometry step. Evidence-supported primitive constraints remain active during refinement.

Reserve time for cheap simplification and final validation before assigning the remainder to fitting. Schedule spatial groups by expected useful improvement and give groups minimum opportunities to run. Large compound paths must not consume the entire refinement budget. Check stop and time limits between stages, graph edits and fitting iterations. Bound graph size, beam width, raster buffers and cached candidates independently of input dimensions.

Start with planning budgets of roughly 5, 20 and 60 seconds for fast, balanced and high quality, subject to measurement on the benchmark hardware. These are tuning targets, not performance promises. An explicit operation time budget takes precedence. Both rendering and validation must fit within the scheduling policy; document any unavoidable deadline overshoot.

Keep deterministic planning and basic refinement available with the base CPU dependencies. PyTorch enables additional gradient fitting; CUDA accelerates supported work. Missing optional acceleration reduces effort rather than making generation unavailable. Audit existing GPU ownership before adding acceleration: use one gate owner for a fitting section, respect cancellation while acquiring it, and avoid nested acquisition or holding the GPU during CPU planning.

Use a bounded analysis cache keyed by reference pixels, alpha, crop, coordinate mapping, evidence settings and algorithm version. Planning/refinement entries additionally include complexity, quality, overrides, score version and any model version. Never reuse a candidate across changed reference content or scope. Cache memory limits and eviction are part of the implementation.

The shared-frontier cache omits the selected complexity, but includes quality/search limits and every override that changes candidate generation. Selection and refinement cache entries include complexity. Store the reference-to-document transform and target-group context with the proposal so cache reuse cannot place correct geometry into an outdated scope.

Assign an initial 96 MiB limit to immutable evidence and a separate 64 MiB limit to candidate/preview data. Bound analysis to a 1,536-pixel long side; use source-resolution crops for thin features and final checks. Use at least 10% of the run budget for validation, with a measured minimum needed to export and render one candidate. Do not begin another fitting round if it would consume that reserve. Check cancellation during graph loops, proposal evaluation and optimizer iterations, not only at stage boundaries. Measure uncancellable renderer/export calls and report their deadline overshoot explicitly.

Maintain `best_validated` separately from the current working state. A stopped or failed optional fitting step returns that checkpoint; a partially edited graph never becomes a preview. Disabling refinement still runs structural planning, paint selection and final validation. The CPU path provides geometric simplification, local paint fitting and bounded proposal search. Optional PyTorch fitting is an additional stage, with a single owner of the existing GPU admission gate for each accelerated section. It must not acquire that gate again inside helpers or hold it during CPU evidence/search.

### Refinement acceptance and fallback

Every refinement stage proposes an independently owned document, preserves fixed corners, junctions and accepted compact-model constraints, and uses the same objective and validation policy as planning. CPU paint fitting compares flat and gradient candidates locally, charging the gradient's actual representation cost. Geometry fitting runs on connected spatial groups, followed by a paint refit because changing coverage changes the fitted color. An explicit line-width override stays fixed. Automatic width changes require local evidence and bounded exact acceptance.

Reuse the joint fitter's injected score callback for exact checkpoints rather than its legacy raster-MSE acceptance. Preserve the manual operation's existing defaults. Until compact models can be optimized in their own parameter space, hold their constrained geometry fixed during unrestricted fitting. Follow shared-edge links after a geometry change and reject the result if the links or anchored junctions no longer agree.

Record whether refinement was disabled, completed, interrupted, skipped for lack of time, or limited by missing optional dependencies. Also record attempted and accepted edits and before/after objective values. A returned checkpoint need not improve in every run, but an accepted refinement cannot worsen the defined objective or break a hard gate. A constant `refinement_complete: false` does not implement this contract.

Produce and independently validate a conservative CEL-derived fallback before expensive search. Preserve partial opacity and intentional holes in that path, and test it in transformed target groups. If the structured initial candidate fails, try this fallback without weakening hard checks. If neither can be validated, report generation failure with its reasons and leave the document unchanged. Cancellation before a checkpoint remains cancellation. A deadline before a checkpoint must not publish unfinished geometry; report the unavailable result and measured unavoidable overshoot. Once a checkpoint exists, stop or optional-stage failure returns it.

## Small learned planning model

First ship a deterministic proposal ranker with recorded features and decisions. The learned experiment predicts which candidate action will provide useful visual and structural improvement. It does not emit SVG coordinates or bypass geometry validation.

Start with a small feature-based classifier or ranker and compare it with a compact MLP. Features include region size, color/gradient fit residuals, boundary length and contrast, texture, scale stability, stroke continuity, junctions and primitive fit quality. A graph neural network is a later experiment only if neighborhood features prove insufficient. Select the simplest model that improves held-out results within the runtime and memory budget.

Create paired training data by rendering clean structured SVGs and applying blur, JPEG artifacts, noise, resizing, edge contamination and alpha degradation. Supervise decisions against the clean geometry/render and compact representation, not only against the degraded raster. Add human-edited projects with reviewed operator labels. Do not use the sword or future human comparison set for training if they remain evaluation fixtures.

Split by original artwork/component family before generating augmentations. Hold out style families as well as degradation types. Avoid random patch splits that place near-duplicates of one drawing in training and testing. Store provenance, asset licenses and consent for any collected human projects.

Compare deterministic-only, learned ranking and full candidate evaluation. The learned model must improve human-reference quality or achieve comparable quality faster; it must not simply optimize the same noisy pixel score. Keep a deterministic fallback when no model is installed, inference fails or confidence is poor. Package a versioned local model without automatic downloads, network inference or user-image upload. A small model must meet a measured CPU inference budget and cannot change the hard validation rules.

Train only after proposal logs contain accepted and rejected alternatives from all core operators. Store feature vectors, graph/evidence versions, exact score deltas, representation deltas and rejection reasons. Hard-invalid proposals are excluded by rules before ranking. Start with a linear model and a two-hidden-layer MLP of at most 64 units per layer; treat architecture and weights as versioned experiment artifacts. Target less than 5 ms for ranking a batch of 128 proposals on benchmark CPU hardware. Compare at equal runtime and equal exact-evaluation budgets so fewer evaluations cannot disguise worse output.

Keep blur, JPEG and RGB noise in the first training augmentation set. Reserve alpha-edge degradation and combinations of degradations for evaluation, in addition to the artwork-family split. Four tuning drawings are a plumbing fixture, not enough evidence to ship a learned model. Expand licensed family coverage before training a release candidate. Proceed only if the model passes all deterministic safety/feature gates and either improves the frozen quality measures or saves at least 20% runtime at equivalent quality. Otherwise record the negative result and ship deterministic planning.

## Benchmarks and release gates

Turn the sword diagnostic into a reproducible benchmark that loads the compressed fixture through the project-file API, exports the human drawing, traces the embedded reference, and computes native-resolution metrics on frozen masks. Save SVGs, full previews and crops of the blade tip, blade edges, guard, handle and jewel. Record the source version, settings, device, time limits and stopped stages.

Extend the existing line, shadow and trace benchmarks rather than replacing them. Add a clean-vector/degraded-raster paired suite and a small curated human-redraw suite. Include alpha edges, legitimate tiny features, lettering, holes, tapered strokes, hatching, nearly straight contours, real gradients, dark surfaces and partial occlusions. Synthetic ground truth makes structural checks possible; human examples test whether the abstractions are useful.

| Area | Required evidence |
| --- | --- |
| Human match | Frozen foreground, component and feature errors against human renders; blind side-by-side review |
| Geometry | Nodes, contours, useful groups, stroke fragments, gradients, corners, self-crossings and shared junctions |
| Coverage | Silhouette/alpha overlap, opaque interior gaps and unintended visible spill |
| Linework | Existing line scores, continuity, width fit and preservation of hatching |
| Shading | Missing-shadow benchmark and broad shade/highlight preservation |
| Complexity | Stable tradeoffs at 0, 25, 50, 75 and 100; padding and resolution invariance |
| Runtime | Stage timing, peak memory, cancellation latency and CPU/CUDA behavior |
| Editing | One undo entry, target group placement, valid references, and save/export/reload equivalence |

Use the sword targets above as the first milestone. For the broader suite, freeze metric tolerances from the baseline before tuning. Aim for at least a 30% median node reduction and better blind preference than legacy CEL at matched runtime. On synthetic clean-vector cases, use an initial maximum line F1 drop of one percentage point and require preservation of annotated critical features and intentional holes. Set per-case silhouette and feature tolerances from the frozen baseline, with a documented allowance for removing injected corruption. Use local feature and silhouette checks so whole-image averages cannot hide damage. Investigate every important-feature loss or new coverage failure; do not average it away. Track raw raster MSE alongside perceptual and structural scores without requiring it to improve for a cleaned redraw.

Run cheap invariants and small synthetic cases in CI. Keep time-sensitive GPU and full-corpus comparisons as explicit benchmark jobs. Test meaningful behavior: preserved features under texture suppression, supported versus unsupported stroke joins, shared-edge changes without gaps, primitive fits retaining corners, gradient rejection on flat noise, stop returning a validated candidate, and legacy settings retaining their behavior.

The held-out evaluation set is used for release evaluation, not repeated parameter tuning. If failures inform new work, retire the affected examples into a development set and refresh held-out coverage before the next release decision.

Run the ablation sequence at matched time limits: legacy CEL; legacy CEL plus current Tidy; color-region generation; planned graph with merges only; planned graph with geometry/ink operators; planned graph with local layers; full deterministic planning with refinement; and learned ranking if eligible. Include refinement off, gradients off and each quality level. This identifies which part improves the human result instead of attributing every gain to the overhaul.

Blind review requires reviewers and additional licensed human redraws; it cannot be completed by a pixel score or an automated claim of preference. Randomize presentation order, hide method names, compare at the same scale, and ask separately about visual resemblance and ease of editing. Record sample size and uncertainty. Until that evidence exists, delivery 8 remains pending and the method remains experimental. The development sword is never a held-out test or ranker-training example.

## Code boundaries and delivery sequence

Add a focused package such as `src/vectrify/refine/cel_plan/` with modules for evidence, graph values, candidates, score policy, planning, primitive fitting, refinement and export. Keep module boundaries based on responsibilities rather than prematurely splitting every helper. Add `src/vectrify/operations/methods/cel_planned.py` for operation orchestration. The existing document model remains the exported editing representation.

Integrate through `operations/generate.py`, the method registry, the existing job/result contract, the Generate UI in `ui/static/index.html` and `app.js`, and the MCP generate tool. Extract reusable helpers from `refine/cel.py`, `joint.py`, `shared.py`, `simplify.py` and the paint-fitting code only when a new stage uses them. Do not combine the monolithic CEL and color-region SVG outputs.

The MCP integration is `src/vectrify/mcp/server.py`: extend the generate method literal and tool documentation together. The UI integration adds a method panel and `generateSettings` entry, uses the existing result choices and preview lifecycle, and updates preview counts without creating a second Apply workflow. Operation metrics expose resolved settings, counts, unmet budgets, stage timings and validation diagnostics. Candidate labels identify actual differences; do not label identical results as alternatives.

| Delivery | Scope | Completion condition |
| --- | --- | --- |
| 1 | Reproducible sword/human benchmarks, score definitions and corpus split | Baselines and masks reproduce the diagnosis; proposed gates are recorded |
| 2 | Stage interfaces, graph representation and experimental operation | Legacy tests pass; a CEL-derived candidate can complete, stop, apply and reload |
| 3 | Foreground budgeting, finite feature protection, merge/paint model search and complexity frontier | Slider controls actual representation cost; coverage and critical features survive |
| 4 | Stroke classification/joining, corner handling and compact primitive fits | Sword targets and line/feature development checks pass |
| 5 | Local layers, automatic refinement, scheduling and CPU/CUDA fallback | Exact checkpoints, runtime/memory limits and cancellation checks pass |
| 6 | UI/MCP controls, preview summaries, candidate labels and documentation | Browser/API round trips, settings invalidation and undo checks pass |
| 7 | Ablations, broad evaluation and optional learned ranker experiment | New behavior improves the frozen suite; learned model has separately measured benefit |
| 8 | Held-out evaluation and default migration | Release gates pass and remaining regressions have an explicit disposition |

Deliver each stage as a reviewable change with the required checks for its behavior. Keep conventional single-line commit messages without scopes, bodies or trailers. Start delivery 1 from current main so the compressed fixture loader is available.

The critical path is benchmark → score and budget → structural operators → refinement → release evaluation. UI work can begin when the settings contract stabilizes. Training depends on recorded candidate decisions and clean paired data. Deterministic rollout does not depend on training a model.

Complete the deliveries in order of their dependencies, with these explicit decision points:

1. Freeze a reproducible baseline and executable score before accepting structural changes by that score. Existing benchmark scaffolding and operation tests are a foundation; numerical reproduction alone does not complete corpus coverage.
2. Establish a valid fallback and stop/apply/reload behavior before adding beam search. Test transparent-empty inputs, partial alpha, opaque scenes, holes and target-group transforms.
3. Prove cost control at all five slider checkpoints using a shared frontier, with actual overrides and reported budget infeasibility. Then freeze the setting contract for UI/API work.
4. Pass the sword's numerical and local-feature gates with structural operators. If it fails, inspect the feature crops and revise the operators or evidence; do not relax the target merely to admit the prototype.
5. Demonstrate that automatic refinement improves or retains the selected plan under the same score and hard checks, and that CPU/cancellation paths meet their contracts.
6. Complete browser and MCP round trips and user documentation while corpus evaluation runs.
7. Run deterministic ablations first. Launch the learned experiment only when measured candidate-evaluation cost or ranking errors justify it.
8. Collect independent review and held-out evidence. Promote the new UI default only after the gates pass; retain the explicit legacy API method and an easy default rollback.

Each delivery records changed files, reproducible commands, settings, input/mask/source hashes, results and remaining failures. A test passing, a schema accepting a setting, or a small SVG is not by itself evidence that an entire delivery is complete.

### Next implementation changes

Continue from the experimental package in the following order. These changes complete missing behavior within the eight deliveries; they do not replace their broader release gates.

| Change | Main code boundary | Evidence required to retain it |
| --- | --- | --- |
| Freeze and expand paired tuning evidence | `scripts/cel_pairs.py`, `scripts/bench_cel_pairs.py`, `scripts/bench_data/planned_pairs.json` | The runner exists; expand artwork coverage, freeze baseline tolerances and record clean/degraded hashes, native masks, line/feature metrics and candidate scores |
| Complete bounded native evaluation | `cel_plan/local.py`, `search.py`, `families.py`, `opacity.py` | Long edits, streamed paint samples, cumulative edits, gradients and opacity groups agree with full scoring; interruption discards partial work and preserves the checkpoint |
| Compact the valid opacity-aware starting drawing | `cel_plan/evidence.py`, `export.py`, `pipeline.py` and the generate operation | Connected flat/gradient RGBA and core/layer competitors avoid byte-level partitions while preserving empty/partial-alpha/opaque inputs, thin features, holes and transformed-scope round trips |
| Extend individual exact acceptance to graph edits | `cel_plan/ownership.py`, `families.py`, `local.py`, `search.py`, `proposals.py`, `planning.py`, `frontier.py` | Owned family merges and paint/boundary/ink edits have a bounded working beam; add split, richer surface, ink replacement and order operators with native full-render agreement, independent rollback, bounded dependencies and stop within loops |
| Complete the CPU refinement path | A focused `cel_plan/refine.py`, existing simplify/shared/paint helpers | Refinement on/off changes behavior; accepted edits improve or retain the common objective; primitive anchors, explicit width and best-checkpoint semantics survive |
| Diagnose and improve facets, ink and compact outlines | `cel_plan/geometry.py`, `ink.py`, `strokes.py`, `layers.py` | Candidate-pool/oracle diagnosis, native feature crops, variable-width alternatives, supported joins and passing sword development gates |
| Calibrate scoring and complete common-frontier caching | `cel_plan/policy.py`, `frontier.py`, `model.py`, `pipeline.py` | Frozen tuning grid, cost progression on one frontier, padding/resizing checks, target/achieved budgets, bounded memory and reference/scope invalidation |
| Add optional joint fitting and spatial scheduling | `cel_plan/refine.py`, `joint.py`, `gpu.py` | CPU works without Torch; gate contention is cancellable; exact acceptance, minimum spatial opportunities and measured runtime/memory limits |
| Expose and document the stable contract | Generate UI, MCP `generate`, operation docs | Slider/quality/overrides round trips, preview invalidation, truthful counts and alternatives, one Apply/undo flow |
| Run the ablations and decide whether ranking needs ML | Benchmark tools and optional versioned ranker | Matched runtime and evaluation counts, operator coverage separated from selection error, measured benefit meeting the learned-model gate |
| Complete release evaluation | Held-out suite, review artifacts and default configuration | Fresh held-out results, independent blind review and all coverage/feature/editing gates before default migration |

The paired tuning runner is available for independent quality checks; expand and freeze its evidence while finishing bounded native evaluation. Structural compaction comes next because the current safe drawing is too dense. CPU refinement can proceed once it has compact eligible shapes, but the sword milestone still depends on useful interpretations being proposed and selected. UI integration waits for real refinement and budget behavior. Learned ranking remains a conditional branch after deterministic ablations.

### Structural compaction work packages

The native opacity-aware drawing still contains thousands of color/alpha
partitions. Owned family merges and individual paint/boundary edits now have an
exact acceptance path, but the current alternatives still fail the structural
targets. Original region ownership and atomic regrouping are implemented;
splitting original graph atoms and rebuilding their canonical edges remain open.
Complete the following work before spending effort on a learned ranker:

1. Preserve stable region membership through merges and export. Maintain an
   owned planning state linking each visible SVG surface, its source regions,
   canonical edges and covering ink. A graph edit updates both adjacent edge
   owners and records the original members; exporting must not lose that link
   by renumbering labels. Test successive merges, splits and rollback on sibling
   beam states, including opacity groups and shared gradient ownership.
2. Propose coherent region families as one surface. Compare a flat RGBA model,
   a linear color/opacity gradient and the existing subdivisions. Grow families
   using connectivity, residuals, boundary contrast and stability across scales.
   Retain supported shade breaks and corners. Pairwise merges remain useful,
   but a fixed allowance of 48 evaluations cannot remove thousands of fragments
   one pair at a time. Bound family size and model work; record the rejected
   alternatives rather than exempting dark or small partitions from all merges.
3. Reinterpret ink together with its underlying surface. Compare continuous
   strokes, variable-width filled marks and the current fragmented fills.
   An accepted replacement removes the corresponding fragments and restores
   supported surface coverage beneath them. Simply adding a stroke can improve
   pixels while increasing complexity. Validate entire connecting gaps, local
   width and junction evidence, translucent compositing, holes and draw order.
4. Evaluate long affected areas as bounded native tiles. Use identical renderer
   coordinates and complete relevant layers, plus the score halo. Accumulate
   each changed pixel contribution once with global denominators, then update
   feature maxima and topology aggregates. Verify agreement against independent
   complete renders for overlapping tiles, long gradients, opacity groups and
   strokes crossing tile boundaries. Keep the existing safe checkpoint when a
   renderer cannot meet the tile contract within the budget.
5. Diagnose proposals separately from selection. Record the clean-target oracle
   for the common candidate pool, including a cost ceiling, before calibrating
   the score. If no compact faithful interpretation exists in the pool, improve
   these operators. If it exists but loses selection, calibrate on the declared
   tuning families. Compare identical pools and measured search effort; keep
   the sword as development evidence and preserve held-out separation.

Each work package needs synthetic native-alpha cases and paired tuning evidence.
The combined milestone remains the frozen sword count/error gates plus local
ink, feature and coverage checks. Passing only the error ceiling with a dense
trace does not complete structural compaction or establish slider usefulness.

For routine development, run the focused evidence/policy/frontier/operator tests and operation apply/stop/reload checks. Use `scripts/bench_cel_planned.py` for the native sword comparison and save its SVGs, feature crops and source/mask hashes. Its current `--check` covers the numerical sword targets only; extend it with the documented local-feature, hole and coverage gates before treating that exit status as full acceptance. Run full-corpus and hardware-sensitive benchmarks separately, with the same completed proposal effort or a clearly stated matched deadline.

### Experiments that determine the next change

Run each experiment on synthetic cases and tuning artwork before using the sword as a development check. Change one mechanism at a time and retain identical scoring supports, settings and source hashes. Save every exact candidate needed for the comparison, with explicit omission status when diagnostic storage is exhausted.

| Question | Controlled comparison | Decision |
| --- | --- | --- |
| Can compact RGBA surfaces replace alpha/color fragments? | Current subdivisions versus connected flat RGBA, linear RGBA and core-with-overlays proposals | Retain models that reduce actual cost under the native objective and preserve holes, fringes and thin marks; fix missing models before changing weights |
| Does ink replacement improve both structure and continuity? | Existing fragments versus a supported stroke and variable-width filled mark, each with restored underlay | Require fragment removal, supported complete gaps/junctions and valid compositing; additive strokes alone do not establish compaction |
| Can fitting recover straight facets and compact outlines? | Unrestricted contour versus anchored straight/cubic and ellipse-like alternatives | Require corner/junction preservation and local boundary/ink evidence; retain the unrestricted interpretation when the prior is unsupported |
| Is automatic optimization useful after planning? | The same retained candidate with refinement off and on | Compare geometry, paint and width stages separately; retained checkpoints must satisfy the common objective, and record time or interruption without assuming an improvement |
| Is the score selecting the wrong drawing? | Production selection versus the clean-target oracle from the same hard-valid pool at the same cost ceiling | Calibrate selection only when a materially better alternative exists; otherwise extend proposal coverage |
| Is search missing useful candidates within the deadline? | Fixed proposal order/evaluation count versus bounded deadline runs, with operator and parent coverage logs | Improve scheduling when useful proposals are available only late; a different completed prefix cannot prove a scoring gain |
| Is a small learned ranker justified? | Deterministic ranking, linear ranking and compact MLP ranking at equal time and exact-evaluation caps | Proceed only after all core operators are represented and the held-out benefit meets the quality or 20% runtime gate |

For each retained change, record the hypothesis, actual representation savings, score deltas, local feature results, hard rejections, runtime and memory. A negative result closes the tested hypothesis only; it does not authorize weakening coverage checks. The current evidence favors improving proposal coverage and compact opacity handling before ML.

### Completion record and rollout

Maintain one checklist row for each of the eight deliveries, linked to its implementation changes, reproducible evidence and unresolved failures. Mark a delivery complete only when its stated condition is met. A PR adding an operator can finish a work package while the containing delivery remains open.

The first externally usable milestone is an experimental `cel-planned` method with working complexity/quality controls, automatic refinement, valid preview/stop/apply/reload behavior and truthful diagnostics. It can remain experimental while broader evaluation continues. Default migration requires the combined sword gates, frozen broader feature/coverage/line gates, runtime and editing checks, fresh held-out evaluation and independent blind review. Reviewer availability and licensed human redraws are external requirements; record them as pending until evidence exists.

Initial scope is cel art and illustrations using ordinary editable SVG paths and supported gradients. Keep photo-specific abstraction, general semantic recognition, unrestricted layer ordering and generative coordinate prediction as later research. They do not need to be solved to complete this release. Preserve the explicit legacy method and make UI-default rollback a configuration change with no project-format migration.

## Main risks and responses

Aggressive simplification can erase small marks; use local feature confidence, protected benchmark regions and alternatives for ambiguous cases. Straight or ellipse priors can erase intentional irregularity; require supporting evidence and retain unrestricted competitors. Incorrect layer inference can create spill or hide features; restrict early order changes and validate exact visibility. A new score can favor smooth but inaccurate shapes; freeze local shape/ink tests and review paired outputs. Runtime can balloon through candidate search; cap proposals and cache local results before adding ML. A learned ranker can overfit artwork styles; split by artwork family and keep deterministic planning available.

Do not make the slider a cosmetic wrapper around region count, enable current Tidy globally and call the redesign finished, or train a model before establishing the deterministic scoring and candidate baseline.

## Research references

[Towards Layer-wise Image Vectorization](https://arxiv.org/abs/2206.04655) supports investigating progressive component initialization and layered construction. [Layered Image Vectorization via Semantic Simplification](https://arxiv.org/abs/2406.05404) separates structural buildup from visual refinement. [Differentiable Vector Graphics Rasterization for Editing and Learning](https://people.csail.mit.edu/tzumao/diffvg/) establishes the differentiable fitting machinery. These references motivate the architecture; the proposed Vectrify pipeline and release targets require their own evaluation.
