# Tracing a reference

How to trace a reference image into shapes and refine the result with the
automated operations.

## The workflow

1. Load the image in the Reference panel; see
   [The reference image](02-files.md#the-reference-image).
2. Generate a first trace.
3. Select paths and Tidy them to fit the reference more closely.
4. Fit colours where the fills are off.
5. Clean up and Snap edges where paths should merge or meet.
6. Fix the rest by hand, with W to compare against the reference.

Each operation runs on the selection only, respects locks and pins, shows
a preview of the reference, before and after with the change in error, and
applies as one undoable edit. They are in the Reference panel, and the
palette; Tidy is also beside the tool name above the canvas and under
Actions → Repair. Agents run the same operations; see
[Agents](09-agents.md#operations).

## Generate

Generate traces the reference into new shapes and adds them as one group,
in front of the whole drawing or inside a selected group. With a group
selected, only the part of the reference around what it paints is traced.
Text is traced as shapes. There are two methods.

### Cel art

Cel art, the default, is for flat, outlined illustrations such as cel and
anime art. It runs on the CPU. It finds the drawn lines, fills the space
between them with flat regions, and draws the lines on top as strokes down
the middle of their ink, at their measured width. Neighbouring regions
share their edges exactly, so there are no gaps or overlaps. It handles
grainy and JPEG-compressed images, and keeps small marks of a clearly
different colour, such as irises and highlights, as regions of their own.

Its settings:

- Regions: how many regions to keep. 0, the default, keeps one per 10,000
  pixels, between 50 and 2,000. Shadows and small marks are kept on top of
  this count.
- Fit colours (on): colour each region with the flat fill that best
  matches the reference under the lines. Off, each takes the median of its
  pixels.
- Gradients (on): give a region whose colour ramps across it, such as a
  sky, a linear gradient.
- Line width: one fixed width for every line instead of measuring each.
- Trace lines as strokes (on): off, lines are traced as filled shapes.
- Continuous outer outline (off): draw one unbroken stroke round the whole
  drawing in place of its traced outer line.
- Outline tolerance: how far a traced edge may stray from the reference,
  in reference pixels.

### Colour regions

Colour regions fits a palette and traces each colour region. Use it for
anything that is not outlined art. Dark outlines can be ordinary regions,
separate linework, or linework with the regions beneath cleaned up. Merge
and clean up geometry tidies the result afterwards. It needs the `vision`
extra and an NVIDIA GPU.

## Tidy

Tidy fits the selected paths, or the paths in selected groups, to the
reference around them. Its steps:

- Fit path (on) moves points and curve handles to match the reference's
  edges. Stroked lines follow the middle of their ink, and their width can
  be fitted too. It uses the `vision` extra, on the GPU when there is one
  and otherwise on the CPU (mitered outlines need the GPU); without it,
  only the edge-seeking part runs.
- Add detail (on) adds points where a piece of the shape is missing or
  covers too much.
- Simplify (off) removes points the outline does not need, as long as the
  match gets no worse than Allowed worsening (1%) and the outline moves no
  more than Tolerance (3 reference pixels). Without a reference it is the
  only step, and the tolerance alone decides.

Tidy runs in rounds, up to 4 by default. Each round tries every enabled
step and keeps the one that improves the match most, if it fixes at least
Minimum improvement (1%) of the difference where it acted. The run ends
when no step helps, the rounds run out or the Time limit (10 s) passes. No
step may leave an outline folded over itself. Stop ends the run early and
keeps the best result so far. These and the other limits are under
Advanced settings, after Override advanced defaults.

- Move shared edges together (on) moves a neighbour's copy of an edge the
  selected path shares with it, as regions from a Cel art trace do, so no
  gap opens. The corners where three regions meet stay put.
- Only what is in view tidies just the points the window shows. With
  nothing selected, it tidies every path there, so you can zoom in on a
  patch and fix that patch alone.

Pinned points stay put, surviving points keep their identity, and Tidy
never changes colour. To reshape a stretch of outline by hand, use
[Redraw outline](05-drawing.md#redraw-outline).

## Fit colours

Fit colours finds the flat fill for each selected object that best
matches the reference, without moving any geometry. It measures each
object's exact coverage, including antialiasing, opacity, clipping and
objects in front, so the result is exact rather than sampled. More passes
help where fitted objects overlap. It needs no GPU.

Set Fill to Linear gradient, or open Fit gradient directly, to give each
shape its own two-stop gradient along the direction the reference changes
most. A shape the reference paints evenly keeps a flat colour.

## Clean up

Clean up removes duplicate and collinear points and empty or duplicate
paths, and merges compatible neighbouring paths into compound paths. It
never rounds coordinates, and keeps paths that instances or clips use.

## Snap edges

Snap edges makes touching edges of two or more closed paths meet exactly.
Set the contact distance, preview the snapped stretches in cyan, and
apply. Of each pair the front path stays as it is and the back path's
edge moves onto it, split where needed so curves keep their shape. The
result is plain geometry: nothing links the paths afterwards. The paths
must be closed, unclipped and not share geometry.

Next: [Agents](09-agents.md)
