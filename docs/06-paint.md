# Paint

How to colour objects: fills and strokes, gradients, line ends and tapers,
and sampling colours.

## Fill and stroke

The Paint section of the Object tab holds fill, stroke, width, opacity,
line ends and arrows. It shows each object's effective colour and names the
group it inherits from; editing inherited paint sets the object's own.
Mixed selections and no paint have their own indicators.

## Gradients

To give a shape a gradient, select one path or basic shape and choose
Paint → Fill type → Linear gradient. The fill controls then offer:

- horizontal, vertical and diagonal directions;
- stops, with their position, colour and opacity, added, removed or
  reversed;
- Endpoints & settings, for exact coordinates, the coordinate system, the
  gradient transform and the repeat behaviour.

Picking a flat colour makes the fill flat again. A new gradient belongs to
its shape alone. A gradient from an imported file may be shared by several
shapes; the panel says whether an edit changes one shape or all of them.

To place a gradient by hand, press Edit on canvas or G for the Gradient
tool. A square handle marks the start and a round one the end. Drag either
handle to move that end: it snaps to the shape's corners, edge middles and
centre (Alt drags freely), and Shift turns it about the other end in 15°
steps. A shared gradient's line is dashed amber, as a reminder that every
shape using it changes. Only linear fills have handles.

Fit gradient, in the Reference panel, fits a gradient to the reference for
you; see [Tracing a reference](08-tracing.md#fit-colours).

## Line ends, arrows and tapers

For open paths and lines, the Paint section also offers:

- Line ends: flat, rounded or square.
- Corner joins: miter, rounded or bevelled, with the miter limit. These
  apply to closed shapes too.
- Start arrow and End arrow, which add arrowheads that follow the line,
  its colour and width.
- Start taper and End taper. Begins from start (or end) is how far from
  that end the line starts to narrow, in drawing units, and 0 turns the
  taper off. Shrink to % is the width at the tip: 0% makes a point, 100%
  keeps the full width.

For example, an end taper beginning 20 units from the end and shrinking to
25% narrows the last 20 units to a quarter of the width. Tapers that
overlap shrink to fit, and the centreline stays editable. Exported SVG
draws arrowheads and tapers as ordinary paths, and reopening the file in
vectrify restores them as settings. Stroke to path turns a tapered line
into an ordinary filled shape.

## Sampling colours

The Eyedropper (I) takes a colour from the canvas.

1. Select the objects to paint, or nothing to set the drawing paint for
   new shapes.
2. In the tool strip choose what to sample, Artwork or Reference, and what
   to set, Fill or Stroke. The sample buttons beside the colours in the
   Object tab start the tool with that target chosen.
3. Click. The colour under the pointer becomes the fill or stroke of every
   selected object, as one undoable edit.

Before you click, the tool strip says where the sample will go, such as
"Reference → fill of 2 objects". The sample is the colour as you see it:
an average over a few screen pixels, with translucent paint blended over
white. Zoom in to sample more precisely. Locked paint is refused with the
object's name before you click.

Next: [Path operations](07-paths.md)
