# Files and the artboard

How to open, save and export drawings, and how to set up the page.

## Opening and saving

The editor opens SVG files and vectrify projects. A project keeps what SVG
cannot: object and point identities, locks, pins, the selection and the
reference image. Projects are saved as `.vectrify` files, gzip-compressed
JSON that keeps the reference's original bytes; a plain `.json` project
opens and saves as JSON.

- Save (Ctrl/⌘ S) overwrites the current file. A new drawing asks for a
  destination the first time. In a browser without file access support,
  Save downloads a copy.
- Save As (Ctrl/⌘ Shift S) always asks; the extension chooses SVG,
  `.vectrify` or JSON.
- Export SVG and Download project (in the palette) write separate copies
  and leave the current file as it is.

Only a static subset of SVG is supported. A file using anything outside it,
such as text, arcs, radial gradients, filters or patterns, is
refused with a list of what is unsupported, rather than opened with content
missing. See [the document model](12-document-model.md#import-export-and-project-files)
for the exact subset.

## Sessions and recovery

The server keeps each drawing in memory as a session. Reloading the page
reconnects to it, but stopping the server loses it, so save before you stop.
Undo history is not saved in files.

Every save also stores a recovery copy of the project in the browser, for
that tab. After a server restart the tab restores it. Restore saved, in the
top bar, picks a recovery copy from any tab. Edits made after the last save
are not in it.

## A new drawing

New (Ctrl/⌘ Alt N) starts an empty drawing and asks for the artboard's
width and height in document units (SVG pixels). It replaces the open
drawing as Open does: it asks first if there are unsaved edits, and starts
as `Untitled.svg` with no reference and a fresh undo history.

## The artboard

The artboard is the drawing's page, the root `viewBox`. The canvas frames
it, SVG export shows it, and every operation compares the drawing with the
reference over it.

Click the size in the footer (or Artboard size in the palette) to edit its
X and Y (top-left corner), width and height. Changing the artboard never
moves, scales or crops the artwork: what falls outside stays in the drawing
and in exported SVG, just off the page. Fit to artwork makes the artboard
the painted bounds of everything drawn, strokes included. Each change is
one undoable edit.

## The reference image

Load a PNG, JPEG or WebP image with Load in the Reference panel; × removes
it. A new reference is fitted to the artboard without distortion: an empty
drawing takes the image's size as its artboard, and a drawing with content
keeps its artboard and centres the image in it.

After that the reference keeps its place in the drawing. Changing the
artboard later crops it, or pads it with transparency, but never refits it.
Projects save where it lies.

Press W to cycle the view: the drawing, the reference over it, the
reference alone, and the error view. Shift W cycles backward, and the
Reference panel offers the same four views and the overlay's opacity. The
error view shows where the drawing differs from the reference: black where
they match, through red and yellow to white where they differ most.

Next: [Objects and selection](03-objects.md)
