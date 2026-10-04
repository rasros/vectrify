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

Applying commits the chosen proposal's transaction as one undoable edit. If
another revision happened while the method ran, commit merges independent
changes into the live document. Overlapping changes and updated locks, pins,
shared consumers, or coordinate frames reject the proposal atomically.
The current user selection is retained, including newly created objects.
Callers add any further dependency, such as the reference image, through
`Job.context_key`.

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
commands `start`, `status`, `stop`, `apply` and `discard`.

## Built-in methods

| Action | Method | What it does |
| --- | --- | --- |
| generate | `cel` | Traces cel art as flat regions bounded by its drawn lines, with the lines as strokes on top (CPU), each down the middle of its ink, in an ink it is darkened toward (judged by its darkness more than its hue, which blur and JPEG smear into the surface's), and stepping in width where the ink tapers (a blurred line's width counting the ink its blur spreads), thin lines in a grainy picture also found where they stand out of its grain along their length, with aligned ends joined across short gaps only when ink remains darker than both sides; dense irregular texture stays in smoothed colour regions, with solid outlines and directional hatching retained; flat regions within three RGB levels per channel share one compound filled path, keeping their contours and holes, and small marks of a clearly different colour, such as irises, kept as regions of their own; a reference's transparent parts (less than half opaque) are left empty, with no regions or lines over them and the regions' outlines along the transparency's edge; `regions` 0 (the default) keeps one per 10,000 pixels, 50-2,000; `outline` draws one unbroken stroke round the drawing's silhouette; `fit_colours` (on by default) colours each region with the least-squares flat fill under the drawn lines, as `improve/colours` would, after a region holding two clearly separate shades is split into them; `gradients` (on by default) gives a region whose colour clearly ramps a linear gradient |
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
no other object shares. Its settings are the steps to use (`snap`,
`simplify` and `shape`, on by default, and `detail` for Snap to add points,
each of which has to fix `detail_gain` reference pixels), Simplify's error
`budget` in percent (1 by default) and its `tolerance`, the most it may move
an outline, in reference pixels (3 by default; without a reference, where
no budget judges it, 1 unless set), each path fit's `steps` (20), `movement`
(SVG units), `resolution` (384) and `stall` (0.5: the share in percent of
its match a check, every tenth step, has to improve by for the fit to go
on), `workers` (1 by default), the `gain` in percent of the local
difference a step must fix (1 by default), the `allowance` in percent by
which the match where the run acted may be worse than at its start (1 by
default), `seconds`, the run's time limit
(10 by default), and the `margin` in percent of the selection's size that the
reference region extends past it; the budget's `steps` is the most rounds (4
by default).
`shape` and `snap` need a reference; without one the target is the drawing's
own render of the region (`generate.drawing_region`) and only `simplify` runs.
Without PyTorch `shape` is left out of the run, and refused only when it is
the one step chosen.

With `shared` (on by default) a selected path's edges that another path draws
too move together (`refine.shared`): a cel trace draws the edge between two
regions in both, the same segments run either way. Before the run, `links`
finds the maximal runs of segments a selected path has in common with another
path whose geometry is its own and unlocked (same points, in the
same frame, either direction; a contour drawn back to its start is read as a
ring). The points the runs end at, where a third region meets the two, are
frozen, so every step leaves them; after each step `follow` redraws each
neighbour's run as the selected path's outline between those points now
runs (reversed when the neighbour runs it the other way), before the result
is rendered and judged, so the judging sees no gap or overlap opening. A run
whose ends are gone is left alone. The neighbours that changed are edited and
selected too, counted under `followed` in the metrics, and their crossings
are checked as the selected paths' are. With `shared` off, neighbours stay
unchanged. The shape fit judges the selected path in that frozen surrounding
artwork, so an edge hidden behind a later path need not move; a better fit
may change another visible edge instead.

Whole-path and region selections are fitted in drawing order. The shape step
divides its remaining time among the remaining fills, so an expensive early
path cannot consume the whole selection's fitting time. After each path fit,
curves are straightened where appropriate and shared neighbours follow before
the result is rendered and judged against the reference. A path fit that no
longer improves this actual result is discarded independently, preserving
earlier improvements and giving the next fit the updated surrounding artwork.

`region` ([x, y, w, h] or a polygon [[x, y], ...] in document units) confines
a run to an area: it acts on the selected paths, or with nothing selected on
every path, that paint inside it (`HitIndex` areas), leaving out paths whose
geometry is shared or locked; their points outside the area are frozen like
pinned ones (Snap and Simplify leave them, the path fit moves only the
others, as `fit_selected_path` does for selected nodes), and the run is
judged over the area's bounds widened by `margin`. The transaction selects
the paths it found. The MCP `tidy` tool takes `region` in place of, or with,
`ids`.

Shared edges are also found between selected paths. The earlier path in drawing
order owns a shared run, the later one follows it, and both ends are held.
Only unselected followers count in `followed` and are added to the selection.

Every round runs each chosen step on the paths as they stand and renders the
region. A step is judged where it acted: over the pixels whose colour it
changed, widened by `BAND` (2) pixels, the share of the squared difference to
the target there that it removed has to be at least `gain`, so the bar does
not grow with the selection. Of the steps that pass, the one that lowers the
region's mean squared difference (`generate.error`) most is kept; if none
does, Simplify is kept when it removed points, and otherwise the run ends.
With a reference, no step is eligible whose result, against the region's
render at the start, is worse over the pixels changed since then (widened by
`BAND`) by more than `allowance` of the squared difference there: a run never
trades the match for fewer points beyond it, however many rounds Simplify
gets. Without a reference Simplify is judged against the drawing itself and
only its tolerance bounds it. A
step's result is not eligible when any path crosses itself more than before
the step (`refine.crossings.crossings`: each contour drawn as a polyline,
cubics at 8 points, every pair of non-neighbouring lines that properly cross
counting once, so a bow-tie counts 1, a looped cubic 1 and a concave outline
0); such steps are counted under `folded` in the metrics. With more than one
worker, Snap and Simplify run in spawned processes while the path fit runs in
the job's thread, so only one fit runs at a time.

Scoring reads Cairo's RGB pixels directly and reuses compiled unchanged paths
within the operation. Paint servers, clips and markers retain CairoSVG's
ordinary handling. Simplify reuses the original join costs across its budget
search and judges identical candidates once. These shortcuts keep the same
pixel error and outline tolerance checks. The native CUDA fill renderer
partitions larger crops across GPU blocks while retaining analytic cubic
coverage; parallel gradient sums can differ slightly in float32 rounding.

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
  refuses (a gradient fill, say) are skipped and reported under `skipped` in
  the metrics. Its many Cairo renders of the drawing around the path leave
  out every path whose controls' hull, widened by its stroke, misses the
  crop (`selected.pruned`; referenced paths and those under filters, markers,
  clips or masks stay), which draws the same pixels for less. Every
  tenth step, before Cairo scores the candidate, the fit checks it for new
  self-crossings; the nodes at the ends of the crossing segments move halfway
  back to where they last did not cross (twice), then all the way, then the
  whole outline does, and the fit carries on from there. The Xing
  (handle crossing) penalty stays off: it sees only a cubic's own handles crossing, while the
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
- For stroked lines (a stroke, no fill) Snap is `refine.lines.fit_lines`
  instead: the reference's ink is each pixel's cover by the line (a black
  top-hat of its brightest channel over how much darker the stroke's colour
  is than the surface), read across the line at each point and curve middle
  as far as it runs unbroken from the middle; each point moves onto the ink's
  centre (at most `SHIFT`, 1.5 px) with its handles, each curve's handles then
  bring its middle there, and an opaque line's stroke width becomes the
  median of the ink's widths along it (its open ends left out) when that
  differs by over 10%, which needs the `paint` permission. The path fit
  leaves stroked lines to it.
- Simplify is `refine.simplify.simplify`: the point whose removal moves the
  outline least goes first, the joined cubic keeping the tangents either side
  with least-squares handle lengths, until any removal would move it more than
  the tolerance. With a reference the tolerance is a cap and the error budget
  the knob: of `LADDER` (8) tolerances up to the set one, bisection finds the
  largest whose result leaves the squared difference over the pixels it
  changed (widened by `BAND`) at most `budget` worse than before; if none
  does, only the points whose removal moves nothing go.

The result is applied with `Transaction.reshape_path`, which keeps surviving
node IDs and refuses to move or remove pinned endpoints.

## Writing a method

Implement the `Method` protocol (`action`, `name`, `background`,
`needs_reference`, `resources`, `validate`, `run`) and pass an instance to
`register`. Put built-in methods in `vectrify/operations/methods/` and import
them from that package's `__init__`.
