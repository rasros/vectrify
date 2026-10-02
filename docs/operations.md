# Operations

`vectrify.operations` is the contract every automated action implements. An
action (Generate, Improve, Simplify, Snap) is carried out by a named method,
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
spatial group at a time.

The editor exposes jobs through one endpoint, `POST /api/operation`, with the
commands `start`, `check` (validate a request without running it), `status`,
`stop`, `apply` and `discard`.

## Built-in methods

| Action | Method | What it does |
| --- | --- | --- |
| generate | `samvg` | Traces SAM segments of the reference into a new group |
| generate | `cel` | Traces cel art as flat regions bounded by its drawn lines, with the lines as strokes on top (CPU); `regions` 0 (the default) keeps one per 10,000 pixels, 50-2,000; `outline` draws one unbroken stroke round the drawing's silhouette; `fit_colours` (on by default) colours each region with the least-squares flat fill under the drawn lines, as `improve/colours` would, after a region holding two clearly separate shades is split into them; `gradients` (on by default) gives a region whose colour clearly ramps a linear gradient |
| generate | `colour-regions` | Traces a GPU-fitted colour palette's regions into a new group |
| improve | `path-fit` | Gradient fitting of one selected path's nodes, handles and colour (CUDA, or the CPU for unstroked fills) |
| improve | `nodes` | Fits the selected paths to the reference by mixing the path fit, snapping and simplifying |
| improve | `colours` | Closed-form flat fills or linear gradients for the selected objects, geometry locked |
| simplify | `cleanup` | Drops redundant vertices and merges compatible paths in the selection |
| snap | `edges` | Snaps touching edges of the selected paths together, as plain geometry |

Generate methods use `vectrify.operations.generate`: `target_region` crops the
reference to the selected objects' painted bounds plus a 10% margin (or takes
the whole artboard for the whole drawing or a group that paints nothing), and
`generated_result` inserts reference-pixel SVG as one group with the transform
that places it over the artboard, measuring reference error before and after.
The target container is the whole drawing (`scope: "drawing"` in the editor
request) or one selected group. Element IDs in generated SVG are renamed, with
their references, so repeated generations never collide.

## Fit colours and gradients (improve/colours)

Compositing is linear in an object's fill: each pixel of the region is
`dark + coverage * fill`, where rendering the object with its fill black and
then white measures `dark` and `coverage` exactly, including antialiasing,
opacity, clipping and whatever is painted in front. Objects are fitted back
to front, each against the drawing as already refitted, for `passes` rounds,
at a comparison size of `resolution` pixels on the long side.

`fill` picks what is fitted. `flat` (the default) is the least-squares colour
per channel. `linear` fits each channel as an affine field `a + b·x + c·y`
over root user space by the same weighted least squares and takes the first
singular vector of the 3×2 matrix of their slopes as the gradient's axis. The
ramp may start and stop inside the object, flat beyond its ends (the
gradient's padding): with the axis also turned up to 4° either way, its ends
are searched on a 16-step grid over the covered pixels' extent along the axis,
then twice on grids four times finer around the best pair, each candidate's
two end colours solved in closed form. Ends past the object's edges would
paint it the same as ends at its edges with the colours there, so the search
stays within it. The stop colours are clipped to 0-1. The axis is mapped from
root space through the inverse of the object's ancestry transforms and its own
`transform`, into a `userSpaceOnUse` gradient whose level lines are the fitted
ones even under skew or uneven scale. A ramp whose ends differ by less than
2/255 in every channel stays a flat fill, and instances (`use`) always get a
flat one, as their user space is their source's.

The method owns each object's gradient through `Transaction.set_fill`, so it
needs only paint permission. Objects whose fill is already a gradient can be
refitted either way: a gradient only they use is updated in place, and a
flat refit removes it. An outline painted like the fill follows it. Metrics
give the region's error `before` and `after`, `objects` changed,
`gradients` among them and the objects `considered`.

## Replaying edited SVG

Clean up changes exported SVG, where every element keeps its object ID. `replay(tx, svg)` in `vectrify.operations.candidates` accepts
such a candidate only by repeating its differences as transaction commands:
deletions and insertions (structure), attribute edits, in-place node updates
for an unchanged path structure, and sibling reorders. The transaction enforces
selection, permissions, locks and pins, so a replayed edit is always one the
user could have made by hand.

Replay raises `CandidateRejectedError` for anything it cannot express, such
as a changed root or element type. A changed path structure is rejected too,
unless `contours=True` (used by cleanup) turns it into
`Transaction.replace_geometry`.

## Tidy (improve/nodes)

`improve/nodes` needs selected paths (or groups containing them) whose geometry
no other object shares. Its settings are the steps to use (`snap` and
`simplify`, on by default, `shape`, off, and `detail` for Snap to add points,
each of which has to fix `detail_gain` reference pixels), Simplify's
`tolerance` in reference pixels, each path fit's `steps`, `movement` (SVG
units) and `resolution`, `workers` (1 by default), the `gain` in percent of the local
difference a step must fix (1 by default), `seconds`, the run's time limit
(10 by default), and the `margin` in percent of the selection's size that the
reference region extends past it; the budget's `steps` is the most rounds (4
by default).
`shape` and `snap` need a reference; without one the target is the drawing's
own render of the region (`generate.drawing_region`) and only `simplify` runs.

Every round runs each chosen step on the paths as they stand and renders the
region. A step is judged where it acted: over the pixels whose colour it
changed, widened by `BAND` (2) pixels, the share of the squared difference to
the target there that it removed has to be at least `gain`, so the bar does
not grow with the selection. Of the steps that pass, the one that lowers the
region's mean squared difference (`generate.error`) most is kept; if none
does, Simplify is kept when it removed points, and otherwise the run ends. A
step's result is not eligible when any path crosses itself more than before
the step (`refine.crossings.crossings`: each contour drawn as a polyline,
cubics at 8 points, every pair of non-neighbouring lines that properly cross
counting once, so a bow-tie counts 1, a looped cubic 1 and a concave outline
0); such steps are counted under `folded` in the metrics. With more than one
worker, Snap and Simplify run in spawned processes while the path fit runs in
the job's thread, so only one fit runs at a time.

The time limit is checked before each round, and each step of a round may
take the time left divided by one more than the number of steps, from when it
starts, the last share left for rendering and judging the results: the
path fit treats that as a stop, Snap stops its passes and Add detail's tries
there, each handing back how far it got. When the time is up the run ends
with the best result so far, as on Stop, and the metrics say `out_of_time`
along with the `seconds` it took.

The steps are separate functions over `refine.frozen.Paths` (each selected
path's `Geometry`), which leave `refine.frozen.Frozen` endpoints alone: the
pinned ones.

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
  It scores at most `SPLIT_TRIES` tries per path in one call, each by filling
  the outline in numpy at pixel centres over a window around the blob rather
  than rendering it with Cairo, and finds each blob pixel's nearest point of
  the outline with a k-d tree.
- Simplify is `refine.simplify.simplify`: the point whose removal moves the
  outline least goes first, the joined cubic keeping the tangents either side
  with least-squares handle lengths, until any removal would move it more than
  the tolerance.

The result is applied with `Transaction.reshape_path`, which keeps surviving
node IDs and refuses to move or remove pinned endpoints.

## Writing a method

Implement the `Method` protocol (`action`, `name`, `background`,
`needs_reference`, `resources`, `validate`, `run`) and pass an instance to
`register`. Put built-in methods in `vectrify/operations/methods/` and import
them from that package's `__init__`.
