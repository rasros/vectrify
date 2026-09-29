# Operations

`vectrify.operations` is the contract every automated action implements. An
action (Generate, Improve, Simplify, Link) is carried out by a named method,
for example `improve/path-fit` or `simplify/curves`. The editor and scripts
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
`vectrify.refine.gpu.gpu_gate()`, one spawn-context semaphore per process: the
CLI's search, its front evaluator and its worker processes queue on the same
object, so an editor job and a search never hold the device at once.

Gradient fitting has one core, `refine.paths.fit_filled_svg`. The editor's
`improve/path-fit` wraps it with exact compositing (clipping, group opacity and
objects in front), pins and permissions; the CLI's random path-fit mutation
wraps it with a cheaper backdrop render of the rest of the drawing.

The editor exposes jobs through one endpoint, `POST /api/operation`, with the
commands `start`, `status`, `stop`, `apply` and `discard`.

## Built-in methods

| Action | Method | What it does |
| --- | --- | --- |
| generate | `samvg` | Traces SAM segments of the reference into a new group |
| generate | `colour-regions` | Traces a GPU-fitted colour palette's regions into a new group |
| improve | `path-fit` | GPU fitting of one selected path's nodes, handles and colour |
| improve | `nsga` | NSGA-II local search over the selected objects, ranked by reference error |
| simplify | `curves` | Refits selected contours with fewer lines and cubics |
| link | `boundaries` | Matches touching edges into shared boundaries |

Generate methods use `vectrify.operations.generate`: `target_region` crops the
reference to the focus rectangle (or takes the whole artboard), and
`generated_result` inserts reference-pixel SVG as one group with the transform
that places it over the artboard, measuring reference error before and after.
The target container is the whole drawing (`scope: "drawing"` in the editor
request) or one selected group. Element IDs in generated SVG are renamed, with
their references, so repeated generations never collide.

## Searching over the document

Search methods mutate exported SVG, where every element keeps its object ID.
`vectrify.operations.candidates.mutation_scope(request)` turns the selection
(or the whole drawing's top-level objects) and the permissions into a
`MutationScope`. Set on the SVG plugin, it limits every mutation to those
elements and their descendants, runs only the operators whose edit kinds are
allowed (colour and stroke changes need paint; numeric, move and path nudges
need geometry; reordering needs structure and both siblings in scope), and
disables crossover and random path fitting.

`replay(tx, svg)` accepts a candidate only by repeating its differences as
transaction commands: attribute edits, in-place node updates for an unchanged
path structure, and sibling reorders. Anything else (added or removed objects,
changed path structure, root changes) raises `CandidateRejectedError`, and the
transaction still enforces selection, permissions, locks and pins, so a search
can never produce an edit the user could not have made by hand.

## NSGA-II Improve

`improve/nsga` exports the drawing with its viewBox on the target region
(focus or artboard), stretched to the reference crop at the chosen resolution,
and runs `vector.search.run_search` from it with a scoped plugin, no LLM seeds
and one epoch. The budget's `steps` is the number of candidates. The final pool
is ranked by one explicit policy, pixel mean squared error against the
reference region, and the best candidates (up to `alternatives` beyond the
recommendation) are replayed as transactions. A candidate that fails replay is
skipped. If none beats the current drawing, the unchanged drawing is the
recommendation. Stop ends the search and keeps the best pool so far.

## Writing a method

Implement the `Method` protocol (`action`, `name`, `background`,
`needs_reference`, `resources`, `validate`, `run`) and pass an instance to
`register`. Put built-in methods in `vectrify/operations/methods/` and import
them from that package's `__init__`.
