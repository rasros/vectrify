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
shared lock in `RESOURCES` while it runs.

The editor exposes jobs through one endpoint, `POST /api/operation`, with the
commands `start`, `status`, `stop`, `apply` and `discard`.

## Built-in methods

| Action | Method | What it does |
| --- | --- | --- |
| generate | `samvg` | Traces SAM segments of the reference into a new group |
| improve | `path-fit` | GPU fitting of one selected path's nodes, handles and colour |
| simplify | `curves` | Refits selected contours with fewer lines and cubics |
| link | `boundaries` | Matches touching edges into shared boundaries |

Generate methods use `vectrify.operations.generate`: `target_region` crops the
reference to the focus rectangle (or takes the whole artboard), and
`generated_result` inserts reference-pixel SVG as one group with the transform
that places it over the artboard, measuring reference error before and after.
The target container is the whole drawing (`scope: "drawing"` in the editor
request) or one selected group.

## Writing a method

Implement the `Method` protocol (`action`, `name`, `background`,
`needs_reference`, `resources`, `validate`, `run`) and pass an instance to
`register`. Put built-in methods in `vectrify/operations/methods/` and import
them from that package's `__init__`.
