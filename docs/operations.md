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
objects in front), pins and permissions, and SAMVG's recovery fit uses through
`fit_filled_svg_bounded`.

The editor exposes jobs through one endpoint, `POST /api/operation`, with the
commands `start`, `status`, `stop`, `apply` and `discard`.

## Built-in methods

| Action | Method | What it does |
| --- | --- | --- |
| generate | `samvg` | Traces SAM segments of the reference into a new group |
| generate | `colour-regions` | Traces a GPU-fitted colour palette's regions into a new group |
| generate | `llm` | Asks an LLM to draw the reference region as SVG |
| improve | `path-fit` | GPU fitting of one selected path's nodes, handles and colour |
| improve | `nodes` | CPU search over the selected paths' points: move, split, remove, shift, stroke |
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

## Searching over the document

Search methods mutate exported SVG, where every element keeps its object ID.
`vectrify.operations.candidates.mutation_scope(request)` turns the selection
(or the whole drawing's top-level objects) and the permissions into a
`MutationScope`. Passed to the search workers as `WorkerContext.scope`, it
limits every mutation to those elements and their descendants, runs only the
operators whose edit kinds are allowed (colour and stroke changes need paint;
numeric, move and path nudges need geometry; reordering needs structure and
both siblings in scope).

`replay(tx, svg)` accepts a candidate only by repeating its differences as
transaction commands: deletions and insertions (structure), attribute edits,
in-place node updates for an unchanged path structure, and sibling reorders.
Strict replay (the default, used by searches) raises `CandidateRejectedError`
for a changed path structure or root; the transaction enforces selection,
permissions, locks and pins, so a search can never make an edit the user could
not have made by hand.

Lenient replay (`lenient=True`, used for LLM replies) instead leaves out every
change outside the transaction's scope or permissions and counts it in
`Replay.skipped`, and turns a changed path structure into
`Transaction.replace_geometry` when geometry and structure are both allowed.
Pass `baseline=` the original after the same normalization the candidate went
through, so rounding introduced by that rewrite is never replayed.

## Optimize nodes

`improve/nodes` needs selected paths (or groups containing them) whose geometry
no other object shares. Its settings are the moves to try (`shape`, `detail`,
`simplify`, `strokes`, `position`), a `tolerance` in percent for Simplify, and
`workers` and `resolution`; the budget's `steps` is the number of tries.
`detail` needs a reference; without one, `simplify` is required and the target
is the drawing's own render of the region (`generate.drawing_region`).

The state the search changes is each path's `Geometry` and stroke width
(`vector.nodes.Paths`); the moves are in `vector.nodes`, and workers render a
state by rewriting only those paths' `d` and `stroke-width` in the region SVG.
Every change is scored by the simple scorer in the main process. A removal is
kept while the score stays within the tolerance of the start (the budget is
shared by the run); a split must improve the score by 1%; any other move is kept
when the score does not get worse. The result is applied with
`Transaction.reshape_path`, which keeps surviving node IDs, refuses to move or
remove pinned endpoints, and leaves linked boundary edges as they are.

The editor's Optimize nodes dialog runs `improve/path-fit` instead when it can:
the GPU fit, checked beforehand with `POST /api/operation` `{command: "check"}`,
which validates a request without running it.

## LLM methods

`generate/llm` and `improve/llm` pick the provider from `settings.provider`
(`auto` takes the first provider set up in Settings, in the order OpenAI,
Anthropic, Gemini, then `local`; see `vectrify.llm.keys`), `model` (empty for
the provider default, or the local server's saved model) and `reasoning`. The
`local` provider sends the OpenAI chat request to the saved server URL, and
leaves out `reasoning`, which most local servers reject. `candidates` asks for several replies, each ranked by reference
error. Generate pins the model's viewBox to the region's pixel size and
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
