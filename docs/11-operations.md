# Operations

For developers: the contract every automated operation implements, how the
built-in methods work, and how to write one.

## Methods

`vectrify.operations` is the contract. An action (generate, improve,
simplify, snap) is carried out by a named method, such as `improve/nodes`.
The editor and the MCP server call methods the same way.

| Action | Method | What it does |
| --- | --- | --- |
| generate | `cel` | Traces outlined art as regions with the lines as strokes on top (CPU) |
| generate | `colour-regions` | Traces a GPU-fitted palette's regions into a new group |
| improve | `nodes` | Tidy: fits selected paths to the reference, and simplifies them |
| improve | `colours` | Closed-form flat fills or linear gradients, geometry locked |
| simplify | `cleanup` | Drops redundant points and merges compatible paths |
| snap | `edges` | Snaps touching edges of the selected paths together |

## Request

`OperationRequest` holds the editor snapshot the method reads, plus:

- `permissions`: which edit kinds may change (`geometry`, `paint`,
  `structure`, `transform`). Selection, locks and pins apply on top.
- `settings`: method-specific options.
- `budget`: optional `steps` and `seconds`.
- `reference`: the reference image, for methods that compare against it.
- `bounds`: the document-space viewport for previews.

`request.transaction(label)` opens an edit of the snapshot limited to those
permissions. It is pure until committed, so a method can build it on a
background thread while the user keeps editing.

## Result

A method returns an `OperationResult`: a recommended `Proposal` and
optional alternatives. Each proposal wraps an uncommitted transaction with
`changed`, free-form `metrics` (conventionally `before` and `after`) and
preview images. Leaving the drawing unchanged is always a valid result.

Applying commits the chosen proposal as one undoable edit. If the document
changed while the method ran, independent changes merge; overlapping
changes, or changed locks, pins, shared consumers or coordinate frames,
reject it atomically. The user's selection is kept. Further dependencies,
such as the reference image, go in `Job.context_key`.

## Jobs

`Job` runs one method on its request. Methods marked `background` run on a
thread with progress, `stop` (keep the best result so far) and a status of
`running`, `ready`, `failed`, `cancelled` or `applied`; others finish
before `start()` returns. A method naming a resource such as `gpu` holds
that resource's lock in `RESOURCES` while it runs, so two GPU jobs never
share the device.

The editor exposes jobs through `POST /api/operation`, with the commands
`start`, `status`, `stop`, `apply` and `discard`.

## Generate

`vectrify.operations.generate` serves both generate methods.
`target_region` crops the reference to the selected group's painted bounds
plus a 10% margin, or takes the whole artboard. `generated_result` inserts
the traced SVG as one group placed over the artboard, measures the error
before and after, and renames element ids so repeated traces never
collide.

`generate/cel` finds lines as marks narrower and darker than the surface
on either side, judged by darkness more than hue. Regions between the
lines are filled with a shrinking ball so small gaps do not join them,
split where colour changes without a line, and merged down to the target
count, similar colours first; clear lightness steps and small marks of a
distinct colour are never merged. Each edge between two regions is traced
once and used by both. Lines are thinned to centrelines, cut where their
width changes by about a third, and drawn as round-capped strokes. Grainy
images are median-filtered first. Transparent parts of the reference stay
empty. Settings: `regions`, `outline`, `fit_colours`, `gradients`,
`tolerance` (0.75), `line_width` and `strokes`.

## Fit colours (improve/colours)

Compositing is linear in an object's fill: each pixel is
`dark + coverage * fill`, and rendering the object with its fill black and
then white measures `dark` and `coverage` exactly. Objects are fitted back
to front for `passes` rounds at `resolution` pixels on the long side.

`fill="flat"` is the per-channel least-squares colour. `fill="linear"` fits
each channel as an affine field and takes the gradient axis from the first
singular vector of their slopes, then searches the ramp's ends along it on
progressively finer grids, solving the end colours in closed form. The axis
is mapped into the object's own space as a `userSpaceOnUse` gradient. A
ramp under 2/255 in every channel stays flat, and instances always get a
flat fill. Gradients are owned through `Transaction.set_fill`, so the
method needs only paint permission.

## Replaying edited SVG

Clean up edits exported SVG. `replay(tx, svg)` in
`vectrify.operations.candidates` accepts the edited SVG only by repeating
its differences as transaction commands: deletions, insertions, attribute
edits, node updates and reorders. The transaction enforces selection,
permissions, locks and pins, so a replayed edit is one the user could have
made by hand. Anything it cannot express raises `CandidateRejectedError`;
`contours=True` allows path structure changes through
`Transaction.replace_geometry`.

## Tidy (improve/nodes)

`improve/nodes` acts on selected paths, or paths in selected groups, whose
geometry is not shared.

- Steps: `shape` (gradient fit), `snap` (edge-seeking), `detail` (add
  points) and `simplify`. With a reference the defaults are `shape`, `snap`
  and `detail`; without one, `simplify` alone, judged against the drawing's
  own render. Without PyTorch `shape` is dropped.
- Limits: `tolerance` (3 reference pixels, 1 without a reference),
  `steps` (60 per path fit), `movement` (SVG units), `gain` (1%),
  `allowance` (1%), `seconds` (10), `margin` (percent of the selection's
  size the reference region extends) and `shared` (on). The budget's
  `steps` is the number of rounds (4).
- `region` confines a run to an area; points outside it are frozen like
  pinned ones.

Each round runs every step on the paths as they stand. A step is judged
over the pixels it changed widened by `BAND` (2): it must remove at least
`gain` of the squared difference there. Of the steps that pass, the one
that lowers the region's error most is kept; if none passes, Simplify is
kept when it removed points or handles, otherwise the run ends. No step is
kept that is worse than the start by more than `allowance`, or that adds
self-crossings (`refine.crossings`, counted under `folded`). The time limit
is checked before each round; on running out the best result is kept and
`out_of_time` reported.

The steps are functions over `refine.frozen.Paths` that leave pinned
endpoints alone:

- Shape is `refine.selected.fit_selected_path`, wrapping the gradient
  fitter `refine.paths.fit_filled_svg` with exact compositing (clipping,
  group opacity, objects in front) and pins. Paths are fitted in drawing
  order, each judged in the artwork the earlier fits left. Handles are
  bounded by their segment's chord, smooth joins keep one tangent, uneven
  turns are penalized, and self-crossings found every tenth step are
  backed off. It runs on CUDA when available; mitered outlines need it.
- Snap is `refine.snap.snap`: points move along the outline's normal onto
  the nearest edge, and with `detail` the largest blobs of wrongly covered
  pixels get new points. Stroked lines use `refine.lines.fit_lines`
  instead, centring points on the ink and fitting the width.
- Simplify is `refine.simplify.simplify`: the point whose removal moves the
  outline least goes first. With a reference, bisection over `LADDER`
  tolerances finds the largest within `allowance`.
- With `shared`, `refine.shared` finds runs of segments a selected path has
  in common with a neighbour, freezes their ends, and redraws the
  neighbour's copy after every step. Neighbours that changed are counted
  under `followed`.

Scoring renders with Cairo, with opacity weighted equally to colour for
transparent references. The result is applied with
`Transaction.reshape_path`, which keeps surviving node ids.

## Writing a method

Implement the `Method` protocol (`action`, `name`, `background`,
`needs_reference`, `resources`, `settings_schema`, `prepare`, `validate`,
`run`) and pass an instance to `register`. Put built-in methods in
`vectrify/operations/methods/` and import them from its `__init__`.

Subclass `SettingsMethod` for typed settings: declare `settings_schema` as
a mapping of names to `Setting` values and `settings_label` for validation
errors. Override `normalize_settings(request, settings)` for contextual
defaults; it runs once after parsing and can tell omitted values from
explicit ones. A method without settings declares an empty schema.

`prepare` returns a `PreparedOperation` with the request and the effective
settings. `Job` prepares once, then validates and runs that same object.
`validate` and `run` accept `OperationRequest | PreparedOperation` and call
`self.prepare(input)`, which returns a prepared input unchanged.

Next: [Document model](12-document-model.md)
