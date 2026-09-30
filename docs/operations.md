# Operations

`vectrify.operations` is the contract every automated action implements. An
action (Generate, Improve, Simplify, Link) is carried out by a named method,
for example `improve/nodes` or `simplify/cleanup`. The editor and scripts
call methods the same way.

## Request

`OperationRequest` holds the editor snapshot the method reads, plus:

- `permissions`: which edit kinds may change (`geometry`, `paint`,
  `structure`, `transform`). Selection, locks and pins still apply on top.
- `settings`: method-specific options such as a tolerance or model.
- `budget`: optional `steps` and `seconds`.
- `reference`: the reference image, for methods that compare against it.
- `bounds`: the document-space viewport for previews.

`request.transaction(label)` opens an edit of the request's snapshot limited
to its permissions. It is pure until committed, so a method can build it on a
background thread while the user keeps editing.

## Result

A method returns an `OperationResult`: a recommended `Proposal` and optional
alternatives. Each proposal wraps an uncommitted transaction with `changed`,
free-form `metrics` (conventionally `before`/`after`) and preview images.
Leaving the drawing unchanged is always a valid outcome.

Applying commits the chosen proposal's transaction as one undoable edit. The
commit checks the snapshot revision, so a result can never overwrite edits made
while the method ran. Callers add any further dependency, such as the reference
image, through `Job.context_key`.

## Jobs

`Job` runs one method on its request. Methods marked `background` run on a
thread with progress, `stop` (keep best result so far) and a status of
`running`, `ready`, `failed`, `cancelled` or `applied`; other methods finish
before `start()` returns. A method naming a resource such as `gpu` holds the
shared gate in `RESOURCES` while it runs. The GPU gate is
`vectrify.refine.gpu.gpu_gate()`, one spawn-context semaphore per process, so
it can also be handed to worker processes; two GPU jobs never hold the device
at once.

Gradient fitting has one core, `refine.paths.fit_filled_svg`, which
`improve/path-fit` wraps with exact compositing (clipping, group opacity and
objects in front), pins and permissions. `fit_filled_svg_bounded` fits one
spatial group at a time; `scripts/bench_samvg_two_phase.py` uses it for SAMVG's
recovery fit.

The editor exposes jobs through one endpoint, `POST /api/operation`, with the
commands `start`, `check` (validate a request without running it), `status`,
`stop`, `apply` and `discard`.

## Built-in methods

| Action | Method | What it does |
| --- | --- | --- |
| generate | `samvg` | Traces SAM segments of the reference into a new group |
| generate | `colour-regions` | Traces a GPU-fitted colour palette's regions into a new group |
| generate | `llm` | Asks an LLM to draw the reference region as SVG |
| improve | `path-fit` | Gradient fitting of one selected path's nodes, handles and colour (CUDA, or the CPU for unstroked fills) |
| improve | `nodes` | Fits the selected paths to the reference by mixing the path fit, snapping and simplifying |
| improve | `llm` | Sends the drawing and an instruction to an LLM; replays its reply within scope |
| improve | `colours` | Closed-form flat fill colours for the selected objects, geometry locked |
| simplify | `cleanup` | Drops redundant vertices and merges compatible paths in the selection |
| link | `boundaries` | Matches touching edges into shared boundaries |

Generate methods use `vectrify.operations.generate`: `target_region` crops the
reference to the selected objects' painted bounds plus a 10% margin (or takes
the whole artboard for the whole drawing or a group that paints nothing), and
`generated_result` inserts reference-pixel SVG as one group with the transform
that places it over the artboard, measuring reference error before and after.
The target container is the whole drawing (`scope: "drawing"` in the editor
request) or one selected group. Element IDs in generated SVG are renamed, with
their references, so repeated generations never collide.

## Replaying edited SVG

LLM edits and Clean up geometry change exported SVG, where every element keeps
its object ID. `replay(tx, svg)` in `vectrify.operations.candidates` accepts
such a candidate only by repeating its differences as transaction commands:
deletions and insertions (structure), attribute edits, in-place node updates
for an unchanged path structure, and sibling reorders. The transaction enforces
selection, permissions, locks and pins, so a replayed edit is always one the
user could have made by hand.

Strict replay (the default) raises `CandidateRejectedError` for anything it
cannot express, such as a changed root or element type. A changed path
structure is rejected too, unless `contours=True` (used by cleanup) turns it
into `Transaction.replace_geometry`.

Lenient replay (`lenient=True`, used for LLM replies) instead leaves out every
change outside the transaction's scope or permissions and counts it in
`Replay.skipped`, and replaces the contours of a path whose structure changed
when geometry and structure are both allowed. Pass `baseline=` the original
after the same normalization the candidate went through, so rounding
introduced by that rewrite is never replayed. `mutation_scope(request)` turns
the selection (or the whole drawing's top-level objects) and the permissions
into the `MutationScope` the LLM prompt names as editable.

## Optimize nodes

`improve/nodes` needs selected paths (or groups containing them) whose geometry
no other object shares. Its settings are the steps to use (`shape`, `snap`,
`simplify`, and `detail` for Snap to add points, each of which has to fix
`detail_gain` reference pixels), Simplify's `tolerance` in reference pixels,
each path fit's `steps`, `movement` (SVG units) and `resolution`, `workers`,
the `gain` in percent a step must improve by, and the `margin` in percent of
the selection's size that the reference region extends past it; the budget's
`steps` is the most rounds.
`shape` and `snap` need a reference; without one the target is the drawing's
own render of the region (`generate.drawing_region`) and only `simplify` runs.

Every round runs each chosen step on the paths as they stand and measures the
region's mean squared difference to the target (`generate.error`). The step
that lowers it most, by at least `gain` percent, is kept; if none does, Simplify is kept
when it removed points, and otherwise the run ends. A step's result is not
eligible when any path crosses itself more than before the step
(`refine.crossings.crossings`: each contour drawn as a polyline, cubics at 8
points, every pair of non-neighbouring lines that properly cross counting once,
so a bow-tie counts 1, a looped cubic 1 and a concave outline 0); such steps
are counted under `folded` in the metrics. With more than one worker,
Snap and Simplify run in spawned processes while the path fit runs in the job's
thread, so only one fit runs at a time.

The steps are separate functions over `refine.frozen.Paths` (each selected
path's `Geometry`), which leave `refine.frozen.Frozen` nodes alone: pinned
endpoints and the nodes of linked boundary edges.

- Shape is `refine.selected.fit_selected_path`, one path at a time; paths it
  refuses are skipped and reported under `skipped` in the metrics. Every
  tenth step, before Cairo scores the candidate, the fit checks it for new
  self-crossings; the nodes at the ends of the crossing segments move halfway
  back to where they last did not cross (twice), then all the way, then the
  whole outline does, and the fit carries on from there. SAMVG's Xing
  penalty stays off: it sees only a cubic's own handles crossing, while the
  fit's folds are mostly neighbouring segments crossing at a node, and in
  single runs it did not reduce them.
- Snap is `refine.snap.snap`: points and segment middles move along the
  outline's normal to the nearest edge within a few pixels, and with `detail`
  the largest blobs of wrongly covered or missed pixels get new points on the
  segment beside them (one, two, or a three-point spike, kept when each point
  fixes enough pixels), creeping along a blob by aiming part of the way in.
- Simplify is `refine.simplify.simplify`: the point whose removal moves the
  outline least goes first, the joined cubic keeping the tangents either side
  with least-squares handle lengths, until any removal would move it more than
  the tolerance.

The result is applied with `Transaction.reshape_path`, which keeps surviving
node IDs, refuses to move or remove pinned endpoints, and leaves linked
boundary edges as they are.

## LLM methods

`generate/llm` and `improve/llm` pick the provider from `settings.provider`
(`auto` takes the first provider set up in Settings, in the order OpenAI,
Anthropic, Gemini, then `local`; see `vectrify.llm.keys`). The model and
reasoning effort are each provider's choice in Settings, falling back to the
provider default and `medium`. The `local` provider sends the OpenAI chat
request to the saved server URL with its saved model, and leaves out the
reasoning effort, which most local servers reject. `candidates` asks for
several replies, each ranked by reference error. Generate pins the model's viewBox to the region's pixel size and
rescales a reply that uses another. Improve requires an instruction, names the
editable object IDs in the prompt, and replays leniently; the prompt is a
request, the transaction is the enforcement. Both offer
`OperationRequest.source_name` as a hint about the subject; the editor fills it
with the reference image's file name, else the drawing's, and skips its
placeholder names.

## Writing a method

Implement the `Method` protocol (`action`, `name`, `background`,
`needs_reference`, `resources`, `validate`, `run`) and pass an instance to
`register`. Put built-in methods in `vectrify/operations/methods/` and import
them from that package's `__init__`.
