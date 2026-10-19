# Local SVG editor

Start the editor with:

```sh
uv run vectrify
```

Or open a specific SVG and its reference image:

```sh
uv run vectrify drawing.svg --reference original.png
```

With the `desktop` extra the editor opens in a native window (pywebview: the
platform's web view, or Qt WebEngine on Linux), and the page calls the Python
backend directly; saving asks for a file with a native dialog. It opens no
port of its own: with Agents on, agents reach it on the MCP server's port.
Without the
extra, or with `--serve` (and optionally `--port`), vectrify serves the editor
on loopback and prints the address to open in a browser. The UI is included in
the Python package and needs no Node build step.
For source checkouts, `PYTHONPATH=src python -m vectrify.ui` works too.

## Layout

- **Top bar**: the name, the drawing's title and the file actions
  (**Commands**, **New…**, **Open…**, **Restore saved…**, **Save**, **Save
  As…**, **Export SVG**). It keeps to one row at any width down to a phone's:
  the badge, the Ctrl K hint and then the title make room first, and the
  actions that still do not fit move into a **⋯** menu at its end, the least
  important first (**New…** and **Restore saved…**, then **Open…**, then
  **Save As…** and **Save**), so
  **Commands** and **Export SVG** stay in view.
- **Left**: the rail keeps **Select** (`A`), **Move** (`V`), **Nodes** (`S`),
  **Redraw outline** (`R`) and **Eyedropper** (`I`) within reach. **More tools**
  holds **Draw**, **Knife** (`C`), **Region** (`X`), **Gradient** (`G`) and
  **Pan** (hold Space). Tool shortcuts still activate these directly; the
  current tool appears above the canvas and More highlights for a secondary
  tool. The menu supports arrow keys, Home/End and Escape.
  Draw has a type picker for **Path** (`D`), **Rectangle** (`M`),
  **Ellipse / circle** (`E`, Shift for a circle), **Line** (`L`) and
  **Pencil** (`P`), and remembers its last type.
  The **Reference** panel sits above the **Objects** tree. Generate, Tidy and
  Fit colours stay visible, with Fit gradient under **More matching options**.
  The panel also holds reference views and overlay opacity; its heading folds
  it away, and loading a reference opens it again.
- **Tool strip**, above the canvas: undo and redo, then only the active tool's
  controls, and on the right the selection's level and count, such as "Objects
  · 3 selected" or "Points · 5 points in 3 paths". Select shows, inside an
  entered group, a breadcrumb such as "Drawing › Trees"
  (click a part to pick at that level). Nodes has the handle buttons (None /
  One / Two), **Break**, **Join points**, **Add node**, **Straighten** and
  **Delete**, shown as icons with tooltips. Delete targets selected handles or
  points first, otherwise the selected objects. Coordinates and hole
  operations live in the **Points** inspector tab. Numbered controls show their digit, and the keys 1-9 press them. Stacking
  is under Actions → Arrange, with keys 1–4 in Select or Move (see
  Keyboard shortcuts). Draw shows a type picker beside its name; Path has
  Finish, Close shape and Cancel; Rectangle
  and Ellipse have W and H, Line has Length and Angle, and Rectangle has
  Corners; Pencil has Close shape and Smoothing; all the drawing tools end
  with the drawing paint (Fill, Stroke and Width). The Eyedropper has **Artwork** / **Reference** (what it samples),
  **Fill** / **Stroke** (what the sample sets) and a status saying where the
  next sample goes. Knife, Region, Gradient, Redraw outline, the drawing
  tools, the Eyedropper and Pan show a
  one-line hint; Region's choices are in its own panel over the canvas. The strip keeps to
  one row: when the window is too narrow for the tool's controls, the least
  important of them (Add node first, and so on) move
  into a **⋯** menu at the end of the controls, which holds them until there
  is room again; the status stays in view. A command run from the menu closes
  it.
- **Right**: three tabs keep the inspector focused. Switching to Nodes or
  Redraw outline opens **Points**; other tool switches open **Object**. Clicking
  a point, clearing points or changing the selection never changes the chosen
  tab, expands a section, or inserts controls above the object properties.
  - **Object** holds the name, role and connections, **Paint**, **Transform** and **Locks** sections,
    folded by default. Paint holds fill, stroke, width, opacity, corners,
    ends and arrows. Transform holds X, Y, W, H, proportions, rotation,
    flips and Move by. Field commands from the palette and F2 open the tab and
    section containing their field before focusing it.
  - **Points** keeps coordinates, handles, alignment, pinning and the common
    point operations in fixed positions. Coordinates are disabled and blank
    without exactly one selected point; multiple points show “Multiple”.
    **Contours & holes** and **Point editing tips** expand for secondary
    operations and guidance, including notes about shared geometry and instances.
  - **Actions** offers folded **Arrange**, **Path**, **Clipping**, **Repair**
    and **Object** groups, with labelled icons. Arrange separates stacking,
    grouping, alignment and distribution. Path separates join/split, boolean
    shapes, and conversion/extraction. Each section retains its open state
    while the selection changes and when switching tabs.

  Reference tools remain in the **Reference** panel on the left. Unavailable
  commands stay in place, dimmed, with the reason in their tooltip; pressing
  one shows that reason. The inspector, tool strip, context menu and command
  palette all use the same commands and availability checks.
- **Right-click** opens a short menu with icons and shortcuts. A selected
  object's menu holds Edit points/Enter group, Tidy, Fit colours, Copy, Paste,
  Rename and Delete. With points selected, point menus hold corner/smooth handles, Add node, Break,
  Join, Select all points and Delete points (or Delete handle). Background
  menus offer Undo/Redo, Paste, Select all and reference loading/generation.
  **More actions…** opens the searchable command palette for every advanced
  command. Unavailable quick actions stay in place and explain why in their
  tooltip. Right-clicking something unselected selects it first; the object
  tree always offers object actions, including while editing points. In Nodes,
  with an object selected but no points, the canvas offers object actions too.
- **Command palette** (Ctrl/⌘ K, or **Commands** in the top bar): type to find
  any command by name, group or a keyword (tools, paint, arrange, path,
  repair, nodes, reference tools, open, save, export, view), with its
  shortcut. ↑ / ↓ choose, Enter runs, Escape closes. A command that cannot
  run now shows why instead.
- The footer shows the artboard size, which opens the **Artboard** dialog
  (see New drawings and the artboard), the reference overlay toggle (`W`),
  with its opacity and an amber canvas border while it is visible, and zoom.
  `W` cycles the view: the drawing, the reference over it, the reference
  alone at full strength, and the reference error. `Shift+W` cycles backward.
  The Reference panel picks the same four views. Error shows matching pixels
  in black and increasing differences through red and yellow to white,
  comparing both images on white at the reference's detail (up to 2048 pixels
  on the longer side). It ignores overlay opacity and refreshes after edits;
  selection outlines and node handles stay available for editing. MCP's
  `render(region="view")` crops this same map, and `describe(objects=false)`
  reports the mode in `reference_view`.
- The footer's **Agents** toggle (also *Allow agents to edit* in the command
  palette) lets an agent edit this window's drawing ([the MCP server](mcp.md)).
  Turning it on makes the editor host the MCP server at
  `http://127.0.0.1:8770/mcp` (or the next free port; `--mcp-port` picks
  another) and opens a popover with that URL and an *Add to an app* section
  of text to copy: the `claude mcp add` command that adds it to Claude Code
  (with *Regenerate token*), the `[mcp_servers.vectrify]` table for Codex's
  `~/.codex/config.toml` (or its `codex mcp add` command), and the
  `mcpServers` entry for Claude Desktop's `claude_desktop_config.json`, each
  saying where it goes; restart the app after adding it. Clicking the
  footer button again opens or closes the popover; its checkbox turns agents
  off, which stops the server. `vectrify-mcp`'s `connect()` reaches the
  window too. While it is on, the agent's edits appear as it makes them and
  land in the undo history labelled "Agent: …", and the footer reads *Agent
  connected* with the agent's last action. The objects each edit touched
  flash briefly in the canvas; the person's selection (objects, points and
  the entered group) is never changed by the agent, except that objects it
  deletes drop out of it. Zoom stays yours too. The Undo/Redo buttons and
  shortcuts affect only your own edits; agent edits have separate history.
  MCP can undo either author's changes by exact history ID, preserving
  unrelated edits and refusing conflicts. Your pending redo is kept while
  the agent works.
  Independent simultaneous edits merge automatically, including different
  properties of the same object. A real overlap reports the conflicting field
  and the editor catches up automatically; no manual refresh is needed.
- Agents' edits reach the page as they happen: the editor pushes each agent
  call to it (server-sent events with `--serve`, a script the desktop window
  runs in the app), so an edit shows within a few tens of milliseconds, and
  a poll every 5 s covers a dropped channel. Mid-drag the page waits and
  shows the edit once the pointer is released. The MCP server's port is
  remembered (in `$XDG_STATE_HOME/vectrify/mcp-port`) and taken again next
  time, so a client added once keeps working; `--mcp-port` asks for another.
  When the port is taken the server moves to the next free one, and the
  popover warns in amber with the old URL and the new one: clients added
  with the old URL must be updated.

Dialogs remain where a preview or confirmation is needed: Generate from
reference, Tidy, Fit colours (and Fit gradient), Merge shapes (for two or more
paths), Clean up, Snap edges, Restore saved project and Keyboard shortcuts,
each titled as the command that opens it. They open from the Reference panel,
the right panel, the context menu or the palette.

Input never gets lost while an edit is running. Keys, clicks, commands, tree
clicks and drops, and canvas drags made meanwhile wait for it, then apply in
the order they were made, each once, on the drawing the edit left: a click or
drag is hit-tested again when it runs, so a drag started during an edit acts
on its result. A drag still held when the edit finishes carries on live.
Escape cancels a waiting drag at once. Viewing needs no wait: panning,
zooming, fit and the overlay toggle act immediately. Edits made during a
canvas drag also wait for the drag to finish.

## Selecting

The selection always has objects, and optionally points inside them. Points
may span several paths, say five points in three paths; point commands act on
all of them as one undoable edit.

- In an object tool (Select, Move, Knife) a click picks the outermost group
  under the pointer, as in most editors. Select only picks and box-selects;
  Move moves and resizes objects by dragging; in Knife a drag cuts.
  **Double-click** a group to enter it:
  clicks then pick within it, and the breadcrumb shows where you are. A click
  outside the entered group leaves it. Double-click a path to switch to Nodes
  on it. Picking an object in the tree enters the group it is in.
- Repeated canvas clicks at the same spot cycle through the overlapping
  objects at that level, front to back, then wrap around. Moving or zooming
  the view, editing, or selecting in the tree resets the cycle. A double-click
  acts on what was picked before it: its second click does not cycle on.
- **Shift-click** adds or removes an object; in the tree, Ctrl-click
  (Command-click on Mac) does too. Tree selection centres and zooms the canvas
  on the whole selection; selecting on the canvas reveals the row in the tree,
  clearing a search filter if needed.
- **Box select**: drag on the canvas, from empty space or from an unselected
  object, to select what lies wholly inside the box: objects at the entered
  group's level in an object tool, points of the selected paths in a point
  tool. Hidden objects are never caught. Shift toggles what the box finds. In Select, every drag box-selects,
  even when it starts on a selected object or its frame. In Move, dragging a
  selected object, or empty canvas inside the selection's frame, moves the
  selection instead.
- **Select all** (Ctrl/⌘ A, under Select in the palette and the right-click
  menu) selects every object at the editing level in an object tool: the
  children of the entered group, or the drawing's top-level objects. It
  leaves out what is not drawn where it is: definitions, gradients,
  clipping boundaries and hidden objects (select those in the tree). In a point tool it
  selects every point of the selected paths, keeping the paths selected;
  with no path selected it says so. A text field keeps its own Ctrl/⌘ A.
- **Arrow keys** nudge the selection by 1 document unit (the root SVG's
  user unit, as X, Y and Move by use), or 10 with Shift, whatever the zoom:
  selected points when present in a point tool, otherwise the selected objects. Objects
  move through the same **Move selection** as a drag, in document
  coordinates, so objects inside rotated or scaled groups move as far on
  the page as the rest, and repeated nudges add up into one `translate` on
  each object. Points move through **Move points**, each by the offset taken
  into its own path's frame through its groups' transforms, with their
  handles, as dragging moves them; nudges do not snap. A locked position
  (objects), locked geometry or a pinned point refuses and says why, and
  moves nothing. Nudges pressed while an edit runs add up into one edit,
  sent when it is done, so holding a key never leaves a backlog of stale
  moves. Nudges of one selection less than a second apart are one undo
  step, as long as nothing else changes the drawing in between: undo, an
  agent's edit or any other edit starts a new step. The arrows keep their
  meaning in text fields, dialogs, the command palette, menus and the
  object tree, and do nothing during a drag or with nothing selected.
- **Resize** in Move works like resizing a window. The selection's frame is
  the bounding box of the selected objects on the page, with small ticks at
  its corners; the tool strip gives its size. Within 6 screen pixels of an
  edge the cursor shows ↔ or ↕, and dragging resizes along that axis only; at
  a corner it shows a diagonal arrow and resizes both. Inside the frame it
  shows the move cursor (over an unselected object, which a press picks
  instead, the plain one). Shift keeps the aspect ratio (at an edge, about the
  middle of the other axis), Alt resizes from the centre, and the dragged
  edge or corner snaps to other objects' bounds and the artboard's edges
  within 8 screen pixels; hold Ctrl/⌘ to drag without snapping. The canvas
  previews the new size, the tool strip shows it as W × H, and releasing
  applies it as one undoable **Resize**; Escape cancels. Each object keeps its
  geometry: the scale is composed onto its own `transform`, in its parent
  group's frame, so an object inside a rotated or scaled group resizes on the
  page exactly as the frame shows, and a selected group takes its selected
  children along once. Strokes keep their width: the stroke widths in a
  resized object are divided by the scale's mean (√(sx·sy)), written on the
  object where it inherited one, except where paint is locked. SVG strokes
  follow their transform, so a non-uniform resize still makes a stroke a
  little wider along the stretched axis, and resizing an object in a rotated
  frame along one page axis skews it; the editor does not use
  `vector-effect`, which exporters and renderers do not all support. Locked
  position or geometry refuses the resize, and the cursor over the frame says
  so before you drag.
- **Position and size** in the right panel's Transform section type the
  selection frame. Its bounds are *geometric*: each selected object's own
  outline box (without stroke width, arrowheads, markers or clipping), mapped
  through its transforms and those of its groups into document coordinates,
  the root SVG's user units, and joined for several objects. A rotated object
  measures by its rotated box's corners. These are the frame Move shows, in
  any tool, and differ from the *painted* bounds the agent's `transform` box
  fits, which include strokes and clipping. **X** and **Y** are the frame's
  top-left corner, the anchor: W and H resize from it, so changing a size
  keeps X and Y. A committed field (Enter, or leaving it) is one undoable
  edit, and Escape puts the field back. X and Y send the same **Move
  selection** a drag does, by the difference in document units: each object
  moves in its parent group's frame by what that difference is there, so
  objects inside rotated or scaled groups move as far on the page as the rest,
  and a selected group takes its selected children along once. W and H send
  the same **Resize** as dragging the frame, about the top-left corner, with
  its stroke-width and lock policy: strokes keep their width, and a locked
  position or geometry disables the fields and says why. The proportion lock
  beside W and H scales both by the same factor; off, the other side keeps
  its size. Several objects move and scale together, so they keep their
  relative placement. A side shorter than 0.001 units has no size: a
  horizontal line's H, or a vertical line's W, shows 0 and cannot be edited,
  and the proportion lock leaves it at 0. A size must be at least 0.001,
  and an edit that would make the other, locked side smaller is refused, so
  no edit makes a dimension zero, negative or invalid. Values show to three
  decimals; committing a field unchanged does nothing. **Move by** offsets
  the selection in document units too.
- **Align and distribute**, under Arrange, move each selected object without
  resizing it, measured by the same geometric bounds as the selection frame
  and the Transform fields: each top selected object's outline box through
  its transforms, so a transformed group, or an object inside a rotated
  group, aligns by its box on the page, and a selected group moves its
  selected children with it, once. Each command is one undoable edit, **Align
  objects** or **Distribute objects**, sent as the same per-object
  translation a move makes, in each object's parent frame.
  - **Align left**, **centre** and **right** line up the left edges,
    horizontal centres or right edges; **Align top**, **middle** and
    **bottom** the top edges, vertical centres or bottom edges. **To** beside
    them chooses what they line up to, and every Align tooltip names it:
    **Selection**, the frame around the selected objects (two or more);
    **Artboard**, the document's artboard (one object is enough); or **Key
    object**, one of the selected objects, which keeps its place while the
    others line up to it.
  - The key object is the object last added to the selection on its own: a
    click, a Shift-click or a row in the object tree. **Key** beside **To**
    lists the selected objects and picks another; it is drawn on the canvas
    with its own amber frame. A box select adds several at once and so
    chooses none, and Align asks for one; deselecting the key forgets it.
    The editor never takes the key from the server's selection order, which
    has none.
  - **Distribute horizontally** or **vertically** spaces three or more
    objects along that axis, in their order by centre. The first and last
    stay where they are and those between them move: **equal gaps** puts the
    same space between neighbours' edges (a negative space, an even overlap,
    when they are wider than the room), and **equal centres** puts their
    centres the same distance apart. Only positions change, never sizes, and
    the other axis is left alone. Distribute does not use the reference,
    except that with Key object chosen the key must be one of the two outer
    objects, or Distribute refuses rather than move it.
  - An object whose position is locked, or that is inside a locked group or
    holds a locked object, refuses with its name when it would move, and
    nothing changes; a locked object that stays put, such as the key, does
    not refuse. Pins keep points in place within their path, as for a drag,
    and do not stop the whole object moving. Definitions and clipping
    boundaries have no bounds to align. Objects already in place send
    nothing, and the message says so.
- **Rotate and flip.** In Move, a round handle sits on a stem above the
  middle of the selection frame's top edge, and the **pivot**, a ringed
  cross, at the frame's centre. Drag the handle to turn the selection about
  the pivot: the preview turns each object exactly as the edit will, and the
  frame turns with it, the status line showing the angle. Shift turns in 15°
  steps; Escape cancels and restores the objects; letting go applies one
  undoable **Rotate**. Drag the pivot to turn about another point: it snaps
  to the frame's corners, edge middles and centre (Alt places it freely), and
  Escape puts it back. A placed pivot, and the one a rotation turned about,
  stays for the same objects until the drawing changes some other way, so
  several turns add up about one point; then it returns to the centre. A
  frame too small to grab the pivot apart from it keeps its pivot at the
  centre, so a small object still moves wherever it is pressed. **Rotation**
  under Transform shows the angle the selected objects are turned by on the
  page, in degrees clockwise, through their groups too; typing an angle
  turns the selection by the difference, about the pivot, the short way
  round. Objects turned differently show *Mixed*, and a value turns them all
  by that many degrees. A mirrored object is measured as flipped
  horizontally first, so Flip horizontal keeps its angle and Flip vertical
  adds 180°. **Flip horizontal** (Shift H) mirrors the selection left to
  right, and **Flip vertical** (Shift V) top to bottom, about the centre of
  its frame, each one undoable edit; they are under Transform, in the
  right-click menu and in the palette. Rotations and flips are composed onto
  each object's own transform in its parent group's frame, as resizing is,
  never by a negative resize: objects inside rotated or scaled groups turn
  and mirror about the same page point as the rest, and a selected group
  takes its selected children along once. Strokes keep their width (a turn
  or mirror changes no stroke width), and arrowheads, clipping, gradients
  and nested groups turn with their objects, as all of them live in the
  object's frame. A locked position or geometry refuses them, hides the
  handle and dims the commands, saying why; definitions and clipping
  boundaries cannot be turned.
- In a point tool, the points of every selected path show, and those of
  every path inside a selected group, however deep; box select and the point
  commands span all of them. Selected instances (`use`) have no points of
  their own: the right panel asks to Detach them. Rectangles, circles,
  ellipses and lines have none until **Convert to path** (Path) makes them
  paths; double-clicking one says so. Hovering an unselected path
  in Nodes shows its points faintly; clicking one selects that point and its
  path in place of the selection (Shift adds the path). Shift-click adds or
  removes a point; a click on one of several selected points, without
  dragging, selects it alone.
- Paths drawing one shared geometry share its points. A point is selected in
  the path it was clicked in, drawn strong there; the other selected paths
  drawing that geometry show it faintly, since editing it moves them all, and
  the right panel says how many other paths it changes. Detach a path to edit
  it alone. A point picked in two such paths is still one point: it moves,
  splits or changes handles once.
- Switching tools never discards the selection. Going from a point tool to an
  object tool keeps the objects and hides the points; going back restores the
  same points if the objects have not changed meanwhile, and clears them if
  they have.
- **Escape** first cancels what is under way (a drag, a path being drawn, a
  hole inspection, a redraw stroke). Otherwise it steps up one level: points
  to their paths, a path to the group it is in (a point tool stays, showing
  the group's points), top-level objects to nothing, and, with nothing
  selected, out of an entered group.
- A click outside the artboard clears the selection in every tool; in a point
  tool a click on empty canvas drops the points first.

**Click a curve handle** in Nodes to select it; the handle is highlighted.
**Delete** (or Backspace) retracts only that handle, keeping its anchor and the
other handles. Escape clears the handle selection first. **Straighten** (`7`),
available in the tool strip, point properties, context menu and command palette,
toggles alignment for the selected points. When enabled, it aligns their existing
handles through their anchors and keeps them aligned while either handle is
dragged, preserving the opposite handle's length. Enabling alignment preserves
both lengths and chooses the direction that minimizes the total squared movement
of the handles, so longer handles rotate less. Already aligned handles stay where
they are. With a handle selected, enabling the toggle keeps it fixed and aligns
its opposite. Turn it off to move the handles
independently. The toggle is saved per point in projects and supports undo/redo.
Deleting a handle is also undoable.

Selected geometry has a translucent blue interior and a blue contour on the
canvas; holes stay clear and stroke-only paths remain unfilled. Highlights
preserve inherited clipping, including visible instances of a selected
definition. The server keeps the selection, points included, so undo, redo
and saved projects restore it.

## What works

- Open SVGs and Vectrify project files; export SVG or download an editable project.
- Pan/zoom/fit, reference image overlay and opacity adjustment (Reference
  panel). Press `W` to cycle the view, or click its toggle in the footer.
- The object list shows definition and clipping containers in their actual
  hierarchy. Shared geometry and clipping contours have neutral symbols and
  role labels; instances name their source. The right panel explains these
  roles and links to source geometry, clipping boundaries, and their users.
- Select an open path or line to choose **Paint → Line ends**: Flat, Rounded,
  or Square. **Corner joins** offers Miter (pointed), Rounded, or Bevelled;
  the miter limit is shown for Miter joins. Corner controls also apply to
  closed shapes. **Start arrow** and **End arrow** independently add or remove
  arrowheads on open contours. These settings affect strokes, not filled
  shapes' silhouettes. Arrowheads follow the line's nodes, transforms,
  stroke colour and width, and remain part of its selection. Projects preserve
  the settings; SVG exports contain ordinary triangle paths with owner metadata
  so reopening restores editable endpoint decorations without duplicate objects.
  **Start taper** and **End taper** each have two controls. **Begins from start/end**
  sets the distance from that endpoint, in local drawing units, where narrowing
  begins; **0** disables that taper. **Shrink to %** sets the endpoint's width
  relative to the stroke width: **0%** makes a point, **50%** leaves half the
  width, and **100%** keeps full width. For example, an end taper beginning
  **20** units from the end and shrinking to **25%** narrows over the last
  20 units to a quarter of the stroke width. Existing tapers default to 0%.
  Nonzero tips use the chosen line cap at their reduced width. Each open
  contour gets its own tapers; closed contours retain their ordinary stroke.
  If the lengths overlap, they shrink proportionally to fit the contour. The
  centerline remains editable, including its nodes, and tapers follow paint and
  geometry edits. SVG exports include filled outlines and preserve the settings
  for reopening; **Stroke to path** makes the outline an ordinary filled path.
- Tree swatches show resolved fill/stroke colors, including inherited paint;
  group swatches preview their contents. The right panel shows effective hex
  colors and names the parent they come from. No-paint and mixed selections
  have distinct indicators. Editing inherited paint sets an object override.
  A gradient fill shows as its ramp, in the tree's swatch and the right
  panel's picker, with "Linear gradient" in place of a colour value; picking
  a colour or typing one makes the fill flat again and removes the object's
  own gradient once its stroke no longer uses it. To create a gradient without
  a reference, select one path or basic shape and choose **Paint → Fill type →
  Linear gradient**. New gradients are private to each shape and do not appear
  under Definitions. The Fill properties offer horizontal, vertical and
  diagonal directions, editable stop positions, colours and opacity, adding
  and removing stops, reversing colours, and a transparency preview.
  **Endpoints & settings** accepts numeric or percentage coordinates and
  controls the coordinate system, gradient transform and repeat behaviour.
  Imported/shared gradients use the same editor: select a shape using the
  gradient, including an inherited fill, or its entry under Definitions.
  Shared edits update every user and retain the existing references; the
  panel identifies whether changes apply to one shape or a shared gradient.
  Locks on the gradient, its stops and its users still apply, and edits undo
  and redo together. **Edit on canvas (G)** switches to the Gradient tool to
  drag the endpoints instead (see below).
- Fill/stroke/opacity, dragging and resizing the selected objects, their
  numeric position and size (**X**, **Y**, **W**, **H**), and numeric offsets
  (**Move by**).
- **Select (A)** only picks and box-selects objects. Dragging on a selected
  object or its frame also box-selects, without moving or resizing anything.
- **Move (V)** selects, moves and resizes objects. Click to select, then drag
  the selected object or inside its frame to move it; drag a frame edge or
  corner to resize. A horizontal or vertical line's whole frame moves it.
  Shift adds or removes objects; dragging on empty canvas outside the frame
  box-selects.
  While moving, the frame's edges and centre snap to the edges and centres of
  other objects' geometric bounds and of the artboard, within 8 screen pixels
  at any zoom, and dashed guides show what it lines up with. The selection
  never snaps to itself, to objects inside it or to the groups holding it.
  Modifiers count once the drag is under way, since a Shift- or Ctrl-press
  still adds to the selection, and they take effect as they are pressed or
  released: **Shift** keeps the move horizontal or vertical, whichever way
  the pointer has gone further, and the free axis still snaps; **Alt** or
  **Ctrl/⌘** moves freely, without snapping. These are the modifiers of
  dragging points, and Ctrl/⌘ also resizes without snapping. The whole
  selection moves by one distance on the page, which each object takes in its
  own group's frame, so objects in rotated or scaled groups land where the
  preview showed them. Escape cancels the move.
- **Nodes (S)** edits the points of every selected path. Without selected
  points or handles, Delete/Backspace deletes the selected objects, as in
  Select (A); arrow keys nudge those objects too. With points selected,
  Delete/Backspace deletes the selected points, a contour's start point included: the next
  point then starts the contour. A contour left with fewer than two points
  (three when closed) is deleted, and a path left without contours is deleted
  too, so a stray speck can be removed point by point or at once with
  Shift+Delete, which deletes the contours the points are on. A press takes
  the nearest point or handle within 10 screen pixels, so a point need not be
  hit exactly; the one under the pointer is drawn larger, with its handles.
  Drag blue handles for curves; a handle shorter than 16 screen pixels is
  drawn that far out along its direction on a dashed line, so it shows clear
  of its point, and a handle on its point counts as none. Zoom in (up to
  25600%) to reveal dense points. **None**, **One** and **Two** handles turn
  the selected points into corners, points curved on one side (again to switch
  sides), or smooth points with their handles in line. The keys 1, 2 and 3 do
  the same. An open contour's endpoints can only have one handle, even when
  its fill makes it look closed. **Two** explains this; select both endpoints
  and use **Join points** (5) to close the contour before adding two handles.
  **Pinned** in the right panel (or **Pin points** in the command
  palette) keeps them in place; **Add node** (6, also under Point operations) adds a node
  halfway along the edge leading into each selected node, preserving the
  curve. At the start of an open path it uses the outgoing edge instead.
  Each edge is split once even when both its ends are selected.
  For drawn lines the strip also has
  **Break** (4) and **Join points** (5). **Break** cuts a line at the selected
  points: an open line comes apart there, each piece ending on its own copy of
  the point (both copies stay selected, so a drag moves them together; click
  one to move it alone), and a closed contour opens at the point. A line's
  ends are free already. With the two points at a segment's ends selected,
  **Break** instead takes out the segment between them, splitting the line
  there (a closed contour opens); with more points selected, every segment
  between two of them goes, and a piece left as a lone point is dropped.
  **Join points** joins any two selected points, in one path or two, filled or not,
  into one line however far apart they are; a point that is not a free end yet
  is broken there first: ends that meet become one point, others are bridged
  by a curve leaving each end along its line. Two ends of one line close it. A
  small loop in a line comes out by selecting the points on it and pressing
  Delete, which reconnects the line past them, or by cutting it off with the
  knife. Dragging a point moves every selected point by the same offset, each
  in its own path's frame, and each takes both of its handles along, so the
  curve keeps its shape around it; dragging a handle moves only that handle.
  Hold Shift while dragging to constrain movement horizontally or vertically
  from the starting position, following the larger pointer displacement.
  Typing a single point's coordinates does the same for it. Pinned points stay
  put, and no other path moves. While dragging, the point or handle under the
  pointer snaps to the points of every visible path (the other points of the
  dragged paths included) and to the artboard's edges and corners, within 8
  screen pixels at any zoom; an orange diamond marks the target and a dashed
  line the edge. Hold Alt or Ctrl/⌘ to drag without snapping. Where a point
  started is never a target. When the selected points are all on holes, in one
  path or several, **Fill hole** removes those holes (and islands inside them)
  and **Hole to shape** moves them out into new paths, each with its path's
  paint, stacked just above it and selected afterwards, as one undoable edit.
  Islands inside a hole go along as holes of the new shape.
- **Draw path (D)**: click to add corners, drag to set mirrored Bézier handles.
  Click the first point or **Close shape** for a filled shape, or press Enter /
  **Finish** for an open stroked path. Backspace removes the last point; Escape
  or **Cancel** discards the draft. Creation is one undoable edit and selects the
  new path for paint and point editing. Middle mouse panning works while drawing.
- **Rectangle (M)**, **Ellipse (E)** and **Line (L)**: drag to draw, with a
  preview painted as the shape will be and a dashed outline; the status line
  shows its size ("Rectangle · 120 × 80", "Line · 100 at 45°") and so do the
  strip's fields. The modifiers are the same in all three: Shift makes a
  square, a circle or a line at 45° steps, and Alt (Option) draws from the
  centre, around where the drag started. Both take effect as they are pressed
  or released, mid-drag. A click places a shape at the size in the strip, from
  its top-left corner (a line from its start), or with Alt around the click:
  W and H, or a line's Length and Angle (degrees counterclockwise from
  rightwards). The fields show, and keep, the size of the last shape dragged
  out, so a click repeats it; type in them for an exact size, or set it after
  drawing with X / Y / W / H under Transform. **Corners** rounds new
  rectangles, at most to half their shorter side. Escape cancels a drag. Each
  shape is one undoable edit, selected afterwards, and the tool stays on for
  the next. Shapes are native `rect`, `circle` (an ellipse with Shift),
  `ellipse` and `line` elements: they save and export as such and stay shapes
  until **Convert to path** (Path) gives them points to edit.
- **Pencil (P)**: drag to draw a freehand path. As it is drawn the stroke is
  fitted to a few cubic curves, previewed painted as the path will be, with a
  dashed outline and its points, and the status line counts them ("Pencil · 6
  points"); the path added on release is exactly the one previewed. The pointer
  is sampled every screen pixel, the samples the browser gathers between
  events included, so a fast stroke is followed as closely as a slow one.
  **Smoothing** (1-20, by default 2) is how far, in screen pixels, the curves
  may stray from the stroke: more gives smoother curves and fewer points.
  Being in screen pixels, it fits alike at any zoom. Sharp turns stay corners;
  elsewhere the curves meet smoothly. **Close shape** (1), remembered with the
  smoothing, closes each stroke back to where it started, as a shape that
  takes the fill; it may be pressed mid-stroke. A click, or a stroke too short
  to fit, draws nothing. Escape (or a tool key) cancels a stroke under way,
  and a stroke started while an edit runs waits for it, as in the other
  tools. Each stroke is one undoable edit, selected afterwards, into the
  drawing container with the drawing paint, and the tool stays on for the
  next. Pencil only creates paths; Redraw outline is the tool that redraws a
  stretch of an existing outline.
- **Drawing paint**: the strip's **Fill**, **Stroke** and **Width** in all the
  drawing tools, remembered in this browser. Closed shapes and paths take the
  fill and the stroke (each can be off, never both); lines and open paths take
  the stroke colour, or the fill colour when the stroke is off. Width is in
  document units. Picking a colour turns that paint on.
- **Drawing container**: the drawing tools draw into the entered group (see
  Selecting), else the top level, on top of what is there. In a group that is
  moved, scaled evenly or mirrored, a shape's own numbers are converted into
  the group's frame, stroke width included, so it shows exactly where and as
  wide as it was drawn; in a rotated or skewed group it keeps the numbers as
  drawn under a transform that undoes the group's.
- Group/ungroup, stacking order, delete and detach shared geometry.
  **Group** gathers selected objects, even across groups or with other objects
  between them, at the frontmost selected branch in their common container.
  **Ungroup** distributes the group's transforms, paint, clipping and opacity
  to its children. Overlapping children may become more opaque; each operation
  is one undoable edit.
  Deleting objects that contain pinned points asks for confirmation. Cancel
  keeps them; confirming deletes them in one undoable edit. Undo restores
  their pins too. Deleting individual pinned points still requires unpinning.
  **Send backward**/**Bring forward** (Ctrl/⌘ `[` / `]`, or the ↓ / ↑ buttons
  of the Select strip) move one object a step; **Send to back**/**Bring to
  front** (add Shift, or ⇊ / ⇈) move the selection to the back or front of its
  group, keeping the selected objects' order.
- Click the arrow beside a group in the Objects tree to collapse or expand
  its children. With the group row focused, Left collapses and Right expands.
  Folding keeps the selection and survives edits; opening a different drawing
  starts with groups expanded. Search shows matching objects inside folded
  groups, and selecting on the canvas an object in a folded group opens its
  parents. Folding only changes the tree; hiding, below, changes the drawing.
- The eye at the end of each object's or group's row in the Objects tree hides
  or shows it. **Hide objects**, **Show objects** and **Show all hidden
  objects** (Object, also in the right-click menu and the palette) do the same
  for the selection or the whole drawing. Hiding is part of the drawing, not a
  view setting: a hidden object is saved as `display="none"`, so it stays
  hidden in the exported SVG, in projects and to agents, and its eye's tooltip
  says so. Each toggle is one undoable edit that keeps the selection; it
  changes no paint or opacity and deletes nothing, so showing the object
  brings it back as it was. A hidden group hides everything in it, whatever
  their own eyes say; their rows are dimmed and their eyes faint, and showing
  the group shows them again. Definitions and clipping boundaries have no
  eye. A **Paint** lock on the object or a group around it protects its
  visibility too, as it does its opacity: the eye is dimmed and says so.
- Hidden objects, and everything in a hidden group, are left out on the
  canvas: clicks, box select and Select all pass over them, moved and resized
  frames do not snap to them, and point tools show none of their points.
  Their rows stay in the tree, dimmed and in italics with the eye crossed
  out, and selecting a row there still selects them; the status bar then
  says how many of the selection are hidden. Selected hidden objects have no
  frame or outline on the canvas. The panel still edits their paint, name,
  locks and stacking, and copying, grouping and deleting work as usual, but
  what needs their place on the canvas (moving, nudging, resizing, rotating,
  flipping, X/Y/W/H, aligning and editing points) refuses until they are
  shown, naming the hidden object. Tidy and Fit colours leave hidden paths
  out: they are not drawn, so there is nothing to compare with the reference.
- Drag rows in the object tree to restack them. The tree lists objects back to
  front: a row paints over the rows above it. Dragging a selected row carries
  the whole selection, which keeps its order. A line shows where the objects
  land; where a group ends, move the pointer left or right to drop inside it
  or after it. Dropping onto the middle of a group row moves the objects into
  that group, at its front. Escape cancels. Objects moved to another group keep
  their placement and paint: the transform and paint they inherited are written
  onto them. They take on the clipping and opacity of their new group and its
  ancestors, and leave behind those effects from their old groups. Instances
  follow changes to the source object's transform and paint. Explicit locks,
  shared edges under changed transforms, definitions, clipping contours and
  moves of a group into itself remain protected. Each drop is one undoable
  edit. A drag can start while an edit is running: dropped before the edit is
  done, the objects land next to the rows
  they were dropped by, on the tree the edit leaves, and the drop is refused
  with a reason if those rows are gone or have moved to another group. A drag
  still held when the edit finishes follows the new rows under the pointer.
- Joining has a command for each kind of join, named for what it does:
  **Join points** (Nodes) joins two selected points, as above; **Join line
  ends** (Path) joins lines at their ends, below; and **Merge shapes** (Path)
  merges paths by area, through a dialog (**Merge shapes…**) for two or more.
  Ctrl/⌘ J runs the one that fits: selected points in a point tool,
  otherwise Merge shapes, or Join line ends when only lines are selected. Each
  says what to select when it cannot run.
- With a single filled path selected, **Merge shapes** unions overlapping contours
  inside that path and resolves self-intersections, removing internal seams.
  Nested hole rings stay with their surrounding regions; a region overlapping
  a hole fills only the portion it covers. The path keeps its identity, paint,
  transform, clipping and stacking. This runs directly, without a color dialog,
  and is one undoable step. Pins must be removed before contours are rewritten.
  A simple outline or disjoint contours are left unchanged.
- Merging paths combines selected paths and groups, recursively
  including paths in nested groups and counting overlapping selections only
  once, even if other objects sit between them. The result always occupies the
  frontmost selected position; intervening objects stay in their existing
  order. Fill/stroke colours and numeric paint properties are averaged by
  clipped painted area (including strokes, excluding holes), so larger shapes
  contribute more. Colour alpha is included; fully unpainted selections fall
  back to equal weights. Merge shapes opens an options dialog: keep the
  default area-weighted color mix or choose any participating path's
  fill/stroke colors (including inherited colors and fill/stroke alpha). Width
  and overall opacity stay area-weighted. Cancel leaves the document
  untouched; stacking remains fixed at frontmost. Filled overlaps use a curved
  boolean union, preventing cancellation holes and internal seams. Disjoint
  contours keep exact nodes; boolean intersections create new nodes and
  require affected pins to be removed first. Two pieces that abut, such as the
  halves of a knife cut or parts split apart, merge back into one region this
  way. Empty selected group wrappers are removed after joining; groups
  containing non-path shapes must be converted or selected more narrowly
  first. Undo restores the original geometry, paint and stacking. Transforms
  and per-path clipping are resolved into the common parent when joining
  across groups. The frontmost ancestry is split as needed so unselected
  objects retain their stacking, styling and clipping. Area weighting uses the
  original clipped painted areas. Object-bounding-box clips are resolved using
  the original geometric bounds. Joins involving stroke-only outlines retain
  the frontmost or chosen source's clipping on the result. A nonuniformly scaled
  stroke uses a scalar width based on its transform's area scale. Splitting
  groups with opacity is allowed even when it changes their compositing.
  Degenerate clip intersections use a winding-aware fallback; its curve
  tolerance is 0.01 SVG
  units (straight edges stay exact).
- **Join line ends** joins the open ends of the selected stroked lines (paths with
  no fill, or the paths in selected groups) where one line carries on from
  another: ends at most 12 screen pixels apart, so zooming out reaches wider
  gaps, and in line with each other, the line turning at most 60° across the
  gap and at each end. The nearest and straightest pairs go first, each end
  once; ends at one spot pair however sharp the corner when they are the only
  two there. A dashed outline becomes one line, and a ring of dashes closes.
  Ends that meet become one point; a gap is bridged by a curve that leaves
  each end along its line. The joined line lands in the frontmost of its paths
  and keeps that path's paint; a path left without lines is deleted, and the
  paths holding joined lines are selected. Side-by-side ends of parallel lines
  do not join. Lines join at their ends rather than by area, and filled
  paths in the selection are left alone.
- **Combine paths** (Path, right-click menu or command palette) collects
  selected paths and the paths in selected groups into one compound path.
  Every contour, curve, node and pin is retained, including touching or
  overlapping contours and open lines. The result uses the frontmost path's
  entire paint, including stroke width and fill rule, and occupies its stacking
  position. A compound path has one style: different original colours become
  that style, and overlapping contours can form holes under its fill rule.
  Transforms are resolved as needed without changing the contours' positions.
  Clipping follows the frontmost or chosen source, including its group's clips
  below the common container. It applies to the entire result; Combine keeps
  the original contours instead of cutting clipping into their geometry.
  Locks and shared-geometry guards apply. Undo restores the original paths
  and selection in one step.
- **Fill to centerline** and **Stroke to path** (Path) convert between a
  filled shape and a stroked line, each path in the selection that it applies
  to, as one edit. **Fill to centerline** turns each selected thin filled shape, such as a part of
  an outline that a trace drew as a fill, into a stroked line down its middle:
  the shape is rasterised about 12 pixels across its thickness, thinned to a
  centreline as the Cel art tracer does, and fitted with curves. Lines meeting
  at a junction run on through it the straightest way. The stroke width is the
  shape's thickness measured along the centreline, its colour and opacity the
  fill's, with round caps and joins, so the stroke covers about what the fill
  did. The path keeps its ID, place in the stack and group; its points are
  new, so pinned points are refused. A shape about as wide as it is long is
  not a line and is refused. **Stroke to path** does the reverse for stroked paths
  without a fill: each becomes the filled shape its stroke paints, with its
  width, caps and joins, in the stroke's colour.
- **Split disconnected parts** separates selected drawing paths into independent
  geometries. Holes and touching/overlapping contours stay together; original
  curves and pins are preserved. A group retains styling,
  transforms and clipping. Undo restores the compound path in one step.
  Only geometric contact or overlap connects contours; gaps are not bridged,
  regardless of stroke width, caps or joins. Cubic curves are approximated at
  0.01 document units for classification; output curves remain exact. Shared assets must be
  detached first. **Detach to editable path** converts a path instance into a
  selected independent path ready to split, retaining its paint and transforms
  in wrapper groups. Other instances keep their shared definition.
- **Convert to path** (Path) turns the selected rectangles, circles, ellipses
  and lines into paths of their outlines, so their points can be edited in
  Nodes. Each keeps its ID, place in the stack and group, paint, transform
  and clipping, and looks the same; a line's path is not filled. The whole
  selection converts as one undoable edit, or none of it does: anything else
  in the selection is refused (Detach is for instances and shared geometry),
  as is a shape another object references or a locked one, with the backend's
  reason. Detach refuses basic shapes and points to Convert to path.
- **Cut out as hole** (Path) takes two selected filled paths. It is not a
  plain Subtract of the front path from the back one: it picks which path
  cuts by containment first and stacking second. When one
  lies inside the other it becomes a hole in it; when they only partly overlap
  the front one cuts the back one. The outer path keeps its paint and
  stacking place, and the cutting path is deleted, as one undoable edit. The
  cutter's contours are added to the outer path, reversed when needed so they
  cut under its fill rule (nonzero or evenodd), and its own holes become
  islands again; transforms of either path's groups are taken into account.
  When added contours cannot reproduce the difference (a partial overlap), the
  outer outline is recomputed as a curved boolean difference, which renumbers
  its points and so is refused for pinned points.
  Locks, shared geometry and instances of the outer path are refused.
- **Union**, **Subtract**, **Intersect** and **Exclude** (Path, right-click
  menu or command palette) combine the filled areas of two or more selected
  paths. The backmost selected path, in stacking order and whatever the order
  they were selected in, is the **target**; the paths in front of it are the
  **cutters**. Each command's tooltip names them (for example "Subtract Patch
  from Shape. Shape, the backmost, keeps its paint and place"), and while the
  pointer or keyboard focus is on the command the canvas outlines the target
  solid and the cutters dashed, before anything changes.
  - **Union** keeps the area inside any of them; **Subtract** the target's
    area outside every cutter (back minus front); **Intersect** the area
    inside all of them; **Exclude** the area inside an odd number of them, so
    where two overlap opens up.
  - The target keeps its ID, name, paint (fill, stroke, opacity, markers),
    locks, group and place in the stacking order, and takes the result as its
    outline; the cutters are deleted. Its fill rule becomes nonzero, the
    winding the result is written in, and its points are new. The result is
    selected, and the edit is one undoable step.
  - Each path's area follows its own fill rule, so holes stay holes. A cutter
    inside the target subtracts as a hole; a target inside a cutter subtracts
    to nothing. Disjoint paths unite (or exclude) into one compound path of
    separate parts. Curves stay curves.
  - An edit that would leave nothing is refused and changes nothing:
    subtracting a target covered by the cutters, intersecting paths that do
    not all overlap, or excluding identical shapes. Subtract also refuses when
    no cutter overlaps the target, since only the cutters would go.
  - Transforms and clipping: the area is worked out in the paths' nearest
    common group and written in the target's own frame, so its transform and
    group stay as they are. The visible part of each path counts: its own
    clip, and the clips of groups it does not share with the target, are cut
    into its area, and the target's own clip is cut in and removed. The
    clipping of the target's group still applies to the result, so a Union
    with a path outside that group's clip shows only the part inside it.
  - Rectangles, circles, ellipses and lines are refused with a pointer to
    **Convert to path**, which keeps the conversion its own explicit edit, as
    are groups, instances, definitions and paths without a fill. As with
    Merge shapes, a geometry or structure lock on any of the paths, pinned
    points, shared geometry and paths other objects reference (instances or
    clips) are refused with the reason.
  - **Merge shapes** differs in mixing paint by area and taking groups;
    **Combine paths** keeps every contour unchanged; **Cut out as hole** picks
    the cutter by containment and keeps the outer path's own points.
- The **Knife** tool (`C`) cuts the paths it crosses along a straight line:
  drag across them and release. With nothing selected it cuts every drawing
  path the line crosses whose geometry and structure are unlocked, inside the
  entered group if one is entered; with a selection, only the selected paths
  and the paths inside selected groups. Shift snaps the line to 15° steps. A stroked line without a fill comes apart
  wherever the dragged line crosses it, each piece ending on its own copy of
  the crossing, and a closed one opens up. The pieces on the side of the knife
  with less of the cut lines become a new path just above, so cutting across
  a loop in a line takes the loop away whole to be deleted; two cuts across
  its neck do the same. Lines the knife misses stay where they are, and
  **Join line ends** (or **Join points**) joins pieces again. A filled path is cut only when the line
  runs through it from outside to outside; a line that ends inside a shape or
  misses it leaves it alone. The cut stops at the start and end of the drag,
  so cutting across one arm of a U leaves its other arm attached. Each cut
  path becomes separate paths (compound contours and holes kept), with the
  original's paint, transform, locks and stacking place; the first piece
  keeps its ID. Curves stay curves. The pieces meet exactly
  on the same seam points but stay independent: dragging a seam point moves
  only that piece, and **Merge shapes** merges them back into one. Pinned
  points keep their positions and identities on the pieces; a cut that would
  remove one is refused. Shared geometry, instances and locks are refused.
  The pieces are selected afterwards, and the cut is one undoable edit. A line that crosses
  nothing the knife can cut says so; a plain click selects.
- The **Region** tool (`X`) extracts or deletes the part of the drawing inside
  a region (see Extracting and deleting regions below). Drag a rectangle, or
  choose **Polygon** in its panel and click the corners; a click in
  Rectangle still selects.
- The **Gradient** tool (`G`) places the selected shape's linear-gradient
  fill on the canvas: a square handle at its start and a round one at its
  end, joined by a line. Drag either handle to move that end; the other stays.
  A dragged end snaps to the shape's corners, edge middles and centre within
  8 screen pixels (Alt or Ctrl/⌘ drags freely), and **Shift** turns it about
  the other end in 15° steps. The shape and every other shape using the
  gradient show the result while dragging, exactly as it will be saved, and
  the **Endpoints & settings** fields follow; typing in those fields moves the
  handles in turn. The handles work for both coordinate systems (each shape's
  bounds, or drawing coordinates) and through the shape's and its groups'
  transforms and the gradient transform. A percentage endpoint stays a
  percentage of the bounds; in drawing coordinates the end is written as a
  number. Only the moved end changes, as one undoable edit; Escape cancels
  the drag.
  The status line says whose paint moves: a **private gradient** changes only
  this shape, a **shared gradient** every shape using it, and its line is
  dashed and amber. A paint lock on the gradient, on any shape using it or on
  their groups refuses the drag, and the handles are greyed. Only one
  selected path or basic shape shows handles (not a group, an instance or the
  gradient's own entry under Definitions), and only its fill; a gradient
  stroke and radial gradients have no handles.
  The handles show only in this tool, so they never compete with Move's
  frame, rotation handle and dragging, even where an end sits on a corner.
  A press anywhere else selects and box-selects as in Select.
- The **Eyedropper** (`I`) samples a colour from the artwork or the
  reference (see Sampling colours below) and makes it the fill or the
  stroke of the selected objects, or with nothing selected the drawing
  paint new shapes and paths take.
- The **Redraw outline** tool (`R`) fixes a stretch of one path's outline in
  one gesture, like a magnetic lasso: a missing spike, a notch or a grass
  blade. Press on a path's outline (or on one of its points), draw roughly
  along the reference's edge and release on the same contour: the stroke
  redraws the path it starts on, the last unlocked path the pointer was over
  (inside the entered group, if one is), which is then selected. With paths
  selected (or a group: every path the point tools show), only those can be
  redrawn, and the selection stays as it was. Each
  end attaches to the nearest point within 6 screen pixels, else to the
  nearest place on the outline within 10; white dots show where, and the
  stretch that will be replaced is dashed. On a closed contour that is the
  shorter way round between the two ends; hold Shift while drawing for the
  longer. The stretch is replaced by curves fitted to the reference's edge
  near the stroke: the cheapest path through a band 8 screen pixels (at
  least 3 reference pixels) either side of the stroke, cheap along strong
  colour edges and following the stroke where there is none, moved onto the
  edge's peak within a pixel, fitted with short cubics that keep sharp turns
  as corners and simplified to 0.6 reference pixels. Without a reference the
  stroke itself is fitted, to about a screen pixel. Every point outside the
  stretch keeps its ID, place and handles; the ends are split exactly where
  they attach. Pinned points inside the stretch and geometry or structure
  locks are refused. The redraw is one undoable edit; a
  plain click selects, and Escape cancels a stroke.
- Geometry/paint/position/structure locks and backend-enforced constraints.
- Undo/redo; a drag is one transaction, not one undo entry per pointer move.

Keyboard shortcuts are available from the `?` button. They come in three
layers that never collide. Tool and view keys: `A`
Select, `V` Move, `S` Nodes, `D` Draw path, `M` Rectangle, `E` Ellipse, `L` Line,
`P` Pencil, `I` Eyedropper, `C` Knife, `X` Region, `G` Gradient and `R` Redraw outline switch
tools at once, even while an edit runs (cancelling a drag under way); `F` fits
the drawing, `Z` zooms to the selection, `W` cycles the view forward and `Shift+W`
cycles backward. The digits 1-9 press the active tool's numbered strip
controls (in Nodes: 1-3 handles, 4 Break, 5 Join points, 6 Add node, 7 Straighten; in Select or Move: 1-4
the stacking commands under Actions → Arrange; in Draw path: 1 Finish, 2 Close shape, 3 Cancel; in
Pencil: 1 Close shape; in the Eyedropper: 1 Artwork, 2 Reference, 3 Fill, 4 Stroke),
whether they are in the strip or its ⋯ menu. Commands use Ctrl/Command: K
opens the command palette, G groups (with Shift, ungroups), J joins what the
selection allows (Join points, Merge shapes or Join line ends), A selects
all. The arrow keys nudge the selection 1 document unit, 10 with Shift (see
Selecting). Drag with
the middle mouse button (in any tool), or hold Space and drag, to pan. Use the
scroll wheel to zoom, and press `F` to fit. Ctrl/Command-Z undoes; add Shift
to redo. Ctrl/Command-S saves the current file. Ctrl/Command-Shift-S opens
Save As to choose a destination; its extension selects SVG, `.vectrify` or
JSON project format. Ctrl/Command-C copies the selected
objects; Ctrl/Command-V pastes independent copies at the same position as one
undoable edit and selects them. Copy includes groups, their styling and
referenced shapes, and keeps the snapshot even if the originals are edited or
deleted. The object clipboard belongs to this editor session. These shortcuts
keep their normal text behavior while editing a field. Copy and Paste are also
available in the command palette, the object context menu and the right
panel's **Object** commands.

Projects preserve object/node identities, locks, pins,
selection and the reference image. New `.vectrify` files use lossless gzip
compression, keeping the embedded reference's original bytes and all drawing
data intact. Saves omit geometry left behind by deleted or replaced shapes.
Existing uncompressed JSON projects still open and save as JSON. Saving as
`.vectrify` uses compression. Older Vectrify versions cannot open compressed projects.
SVG files contain the drawing. **Save** overwrites the current file in the
desktop editor; a new drawing asks for a destination the first time. Browsers
with file access support also reuse files opened or selected through their
pickers. Other browsers download a copy. **Save As…** always opens the save
picker when file access is available. **Export SVG** and the command palette's
**Download project** write separate copies without changing the current file.
The server keeps sessions in memory; save a project before stopping it. Browser
reloads reconnect to the current session, but undo history is not stored in
project files. Saving SVGs or projects also keeps a full project recovery copy keyed to this tab's
session; after a server restart, the tab restores that saved project. Edits made
after the last save are not included in recovery. **Restore saved** also lets
you choose a saved browser recovery copy from a new tab.

### New drawings and the artboard

**New…** in the top bar (or `Ctrl/⌘ Alt N`, or **New drawing…** in the
command palette) starts an empty drawing. Its dialog asks for the artboard's
width and height in document units (SVG pixels), starting from the current
drawing's. The new drawing replaces this one exactly as **Open…** does: with
unsaved edits it asks first, and it starts as `Untitled.svg` with no file to
save to, no reference image and a fresh undo history. The tab's recovery copy
stays the last one saved until you save the new drawing.

The artboard is the drawing's page: the root's `viewBox`, which the canvas
frames, SVG export shows and every method measures the reference against.
Clicking its size in the footer (or **Artboard size…** in the palette) opens
the **Artboard** dialog with its **X** and **Y** (the top-left corner) and
**Width** and **Height**. Changing them never touches the artwork: nothing
is moved, scaled or cropped, and what falls outside the artboard stays in
the drawing and in exported SVG, only off the page. Width and height grow or
shrink the page from its top-left corner; X and Y move that corner. **Fit to
artwork** (also **Fit artboard to artwork** in the palette) makes the
artboard the painted bounds of everything drawn, strokes included. A root
`width` and `height`, if the file has them, keep their ratio to the
`viewBox`, so the drawing keeps its scale. Each change is one undoable edit,
saved with the drawing, and the canvas keeps the artwork where it was on
screen.

The reference image keeps its place in the drawing: it stays over the part
of the artwork it covered and shows only where it overlaps the new artboard.
Methods, sampling and the reference error view see it the same way, cropped,
or padded with transparency (white in comparisons) where the artboard grew
past it; undoing the artboard change puts it back over the whole page.
Projects save where the reference lies. Only loading a reference fits it to
the artboard (see Operations); changing the artboard never refits it.

## Operations

The **Reference** panel, above the objects on the left, is where the
reference lives, whichever tool is active: **Load…** (or **Replace…**) takes
a PNG, JPEG or WebP image, shown as a thumbnail with its name, and × removes
it. A new reference covers the artboard, so the image is fitted to it
without distortion: an empty drawing takes the image's size as its artboard
(one undoable edit), and a drawing with content keeps its artboard while the
image is centred on a transparent canvas of the artboard's shape. It then
keeps that place when the artboard changes (see New drawings and the
artboard). With a reference loaded the panel picks the view (**Drawing**,
**Overlay** or **Reference** alone, as `W` cycles them) and the overlay's
opacity; its heading gives the state. Below are the tools that compare the
drawing with it: **Generate…**, **Tidy…**, **Fit colours…** and **Fit
gradient…**, acting on the selection made with any tool. A tool that cannot
run is dimmed, with the reason as its tooltip. Each is also in the command
palette.

**Tidy…** is also available above the canvas, beside the active tool's name,
including when the Reference panel is hidden, and under **Repair** in the
right panel, since it simplifies paths without a reference too. Its compact dialog keeps
**Fit path**, **Add detail**, **Simplify** and **Only what is in view** visible;
expand **Advanced settings** for the tolerances, fitting controls and run
limits. The action buttons stay
visible while the settings or preview scroll.
The dialog runs one round by default, with up to 20 internal steps per
path fit. Each round tries the enabled steps and keeps the best
result; extra rounds repeat from that result. Enable
**Override automatic run limits** to edit these; maximum fitting steps is
available only with **Fit path** enabled. The **Time limit** remains editable.

Tidy is a quick clean-up of the selected paths, or the paths
inside selected groups, against the reference around them, never the whole
image. By default it fits their points and curves to the reference and
removes the points they do not need; to reshape a path, use Redraw outline
(R). Tick the steps it may use:

- **Fit path** (on by default) compares a nearby-edge proposal with a fit of
  the points and curve handles by gradient descent, keeping the better render.
  Stroked lines follow the middle of their ink, and their width may be fitted
  to the ink's width. Gradient fits use up to 20 steps by default and respect
  **Maximum movement** in local SVG units. Straight segments can gain handles
  where the reference curves; segments left straight stay lines. It uses CUDA
  when available, otherwise CPU; round strokes work on either, while miter
  outlines require CUDA. Without PyTorch it uses edge-seeking alone.
  **Add detail** is off by default. When enabled, it adds points where a piece
  of the shape is missing or covers too much. **Search beyond the path**
  controls how far past the selection the reference is read.
- **Simplify** (on by default) removes the points the outline does not need
  while the match to the reference where it acts gets no worse than the
  **Allowed worsening** (1% by default), moving the outline no more than the
  **Tolerance** in reference pixels anywhere (3 by default), and turns curves
  whose handles lie on their line within that into straight segments.
  Without a reference the tolerance alone decides, 1 by default.

Transparent references give average colour error over white and opacity
error equal weight, so opaque white areas and transparent gaps are
distinguished. Each fitted
path is judged in the current surrounding artwork before fitting the next.

Where a selected path shares an edge with a neighbour, as the regions of a
cel trace do, **Move shared edges together** (on by default) moves the
neighbour's copy of the edge with it, so no gap opens between them and
neither covers the other; the corners where three regions meet stay put.

Paths sharing one geometry asset, including body shapes used by clipping
boundaries and outlines, can also be tidied without detaching. Tidy edits
each asset once and judges the result over all affected artwork. Instances
keep their references, and clipping and linked outlines follow the edited
geometry. When the fast path fitter cannot handle these dependencies, Tidy
uses bounded fitting against the actual SVG render. Locks on any dependent
object are respected, and the preview reports the linked artwork.

**Only what is in view** tidies just the points inside the part of the
drawing the window shows: of the selected paths, or, with nothing selected
(Tidy then opens with it ticked), of every path painting there. Zoom in on a
patch of a trace to touch up that patch alone.

Each round tries every ticked step on the paths as they stand and keeps the one
that brings them closest to the reference, if it fixes at least the **Minimum
improvement** (1% by default) of the difference where it acted: over the
pixels it changed and a thin band around them, so a small fix on a large
selection counts as much as on a small one. When none helps, Simplify gets its
turn. No step is kept that leaves the match where Tidy acted worse than it
started by more than the **Allowed worsening** (1% by default; 0 never trades
the match for fewer points), and the run ends once nothing qualifies, the rounds (1 by default in the dialog) run out
or the **Time limit** (10 s by default) passes. Each round gives each step a
share of the time left, and a step that runs out hands back how far it got; a
run that runs out of time keeps the best result so far, as Stop does, and says
so. No step may leave an outline folded over itself: a result where a path
crosses itself more than it did, a twist or a curve looped round, is not kept,
and the shape fit pulls such folds back as it goes. Concave outlines are fine,
and a path that already crossed itself may keep those crossings. Without a reference only Simplify runs, keeping the paths' look. Pinned
endpoints stay fixed, and surviving points keep their identity. Colour is left
to Fit colours. The job runs in the background with progress, Stop & keep
best, and reference/before/after previews; the result lists the steps it
took. Apply is one undoable edit.

**Fit colours…** solves the flat fill colour of every selected object that
best matches the reference, with geometry locked. Each object is rendered with
its fill black and white, which measures its exact coverage (including
antialiasing, opacity, clipping and objects in front), so the best colour has
a closed form; outlines painted in the fill colour follow it. Objects are
fitted back to front; more passes help where fitted objects overlap. No GPU is
needed. Its **Fill** setting chooses a flat colour or a linear gradient. **Fit
gradient…** opens the same dialog with Linear gradient chosen: each selected
shape gets its own private two-stop gradient, along the direction the reference changes
most, that best matches it, starting and ending where the reference's ramp
does, inside the shape when it is flat beyond them. A shape the reference
paints evenly keeps a flat colour. Apply keeps it as one undoable edit.

**Clean up…** removes duplicate and collinear vertices and empty or
duplicate paths, and merges compatible neighbouring paths into compound paths,
within the selection only. Paths referenced by instances or clips are kept.
Coordinates are never rounded.

Generate, Improve, Simplify and Snap edges run through the shared operation
contract in `vectrify.operations` (see `docs/operations.md`) via
`POST /api/operation`.

**Generate from reference…** (Reference panel) traces the reference into
new shapes, with the Cel art method (the default) or Colour regions. The
result is placed over the artboard exactly where the reference is shown, as
one new group at the front of the whole drawing or of a selected group. With
a group selected, only the reference around what it already paints is traced.
Text is traced as shapes, since the editor does not support SVG text. Preview shows the reference, before and after, with
the change in reference error; Apply adds the group as one undoable edit.

The Cel art method is for flat, outlined illustrations such as cel and anime
art, and is the dialog's default. It follows their drawn lines: it finds the
lines (marks narrower than the surface either side and darker than it in
their brightest channel, so a black line still counts against a navy fill of
the same brightness, or only a little darker than a dark fill beside it; bold
outlines as dark as ink up to twice as wide; faint narrow shading, and dark
notches as dark as the surface they open into, are left to the fills, as are
dark shapes much wider than the lines, and marks about five pixels wide or
more whose middle is clearly paler than the ink, such as a fold's dark
wedge on a robe, which stay shaded regions rather than branching strokes;
in a grainy picture, where blur makes even a line's middle pale, that last
test is skipped). In a grainy or JPEG-compressed
picture the lines are found after a small median filter smooths the grain
away; since that also takes lines a pixel wide, the lines are looked for again
after a median of three pixels along each direction, which keeps them, and
those found only then are drawn over the regions without cutting them. So
are thin lines hardly darker, pixel by pixel, than the grain around them,
such as hatching or a line on a dark fill after blur and JPEG: the picture,
smoothed a little, curves up steeply across them and hardly along them, by
at least three times the curvature the picture's own grain gives, so the
threshold follows how grainy each picture is. It fills the space between the lines with a shrinking ball so a small
gap in a line does not join the regions either side, splits each region where
its colour changes with no line, and merges neighbours down to the chosen
number of **Regions** (0, the default, keeps one per 10,000 pixels of the
traced area, at least 50 and at most 2,000), those of a similar colour first,
and regions of
different colours a drawn line separates last (the same colour either side of
a line merges freely, since the line is drawn over it). Lightness counts
more than hue in how alike two colours are, and two regions of some size a
clear step apart in lightness, a shadow or highlight and the surface it lies
on, are never merged; nor is a small mark of a clearly different colour
(60 levels or more apart in RGB, from 12 pixels up), such as an iris, a
pupil, a highlight or a mouth, merged into the region around it. The shadows
and marks kept this way do not count toward Regions, so a shaded drawing can
come out with more regions than asked for (a few dozen more on the benchmark
drawings). Pieces of a colour under 12 pixels go to their neighbours. A region the merging left with two
clearly separate shades in it (few of its pixels between them) is then split
back into them, so its colour is not fitted to a grey between the two. The line pixels go to
the regions either side, so neighbours
meet at the line's middle; each edge between two regions is traced once and
used by both, so they meet exactly with no gap or overlap. Before an edge
is fitted, its pixel staircase is smoothed away between its corners, which
stay sharp, so the outlines come out smooth and with few points: each run
between corners is cut into as few curves as a polyline within twice the
tolerance needs, then simplified within the tolerance. With **Fit colours**
(on by default) each region takes the flat colour that best matches the
reference under the lines as they are drawn, solved in closed form as Fit
colours solves it, leaving out the pixels along its edge, which the
outline's antialiasing already mixes with its neighbour; off, it takes the
median of its own pixels, not the lines'. With **Gradients** (on by
default) a region whose colour ramps across it, such as a sky, takes the
linear gradient Fit gradients would give it, where that lowers its error
under the lines by at least a quarter and by 4 (in squared 0-255 levels)
a pixel; the rest stay flat. The lines are thinned to
centrelines and drawn over the regions as round-capped strokes in their ink,
each moved across its line onto the middle of its ink (thinning leaves a line
an even number of pixels wide half a pixel off it):
a thin line's antialiased middle mixes its ink with the surface, so it is
drawn in the darker ink at the width of ink it holds, not in grey; a line
holding less than a pixel of ink is drawn solid at least 0.8 px wide, so it
keeps its weight without becoming a sliver. In a grainy image, where a line
that thin may be grain, only the thin, tapering end of a line at least
1.5 px wide is drawn so; any other is drawn a pixel wide and as much
fainter (its stroke opacity). A dark
mark wider than its stroke but too small to be a filled shape, such as an
eye or an eyebrow, is filled in its ink beneath the stroke where the stroke
leaves it out, so it keeps its shape. There is
one path per line colour and width (up to eight widths a colour, each path's
lines within about 20% of its width; a little ink much wider or thinner
than the widths either side, such as bold lettering among thin lines, keeps a
path of its own rather than joining a thinner one), each at its lines'
measured width;
a line whose width changes along it by more than about a third, such as
tapered or calligraphic ink swelling and thinning to a point, is cut where
it changes into parts as short as six pixels or two line widths, so each
part gets the width it has, and the parts still meet end to end: it stays
a stroke, drawn as a few round-capped strokes stepping in width. Thinning's whiskers and the tiny loops it leaves round
a pinhole where lines meet are dropped (a short branch reaching three pixels
or more beyond the ink of the line it leaves, such as a mouth off a jaw's
outline, is a mark of its own and stays), the lines of one path run on through
the junctions where they meet the straightest way rather than stopping
there, and a gap of up to one and a half line widths between two lines that
carry on from each other is bridged, so an outline comes out as a few long
lines rather than many short ones; **Line width** fixes the width instead. Lines stay strokes
however much they taper; only with **Trace lines as strokes** off are they
filled shapes: each line's pixels at least half as dark as its middle, in
the ink most of its centreline is nearest (so an antialiased line is one
colour along its length, not broken up among the inks), traced like the
regions with its holes kept clear. **Continuous outer outline** (off by default) draws one
unbroken stroke round the whole drawing in place of its traced outer line,
so the outline has no gaps or breaks where the drawing has outer ink. The
drawing is everything the background does not cover: the background is the
commonest colour along the canvas border (when at least half the border has
it and it is plain, not textured), reaching in from the border, and a
drawing filling the canvas has no outer outline; holes inside the drawing,
even background-coloured ones such as a gap between an arm and the body, are
not part of the outer outline and keep their own traced lines. The stroke is
always a stroke, also with **Trace lines as strokes** off, in the outer
line's ink. Across the edge at each point the ink there is measured, from
just outside the drawing in to where it clears, and the stroke runs down its
middle at its width, cut into pieces where the width steps by a quarter and
grouped into a few widths, like the inner lines. Where the edge has no ink,
as along a cropped hem, the outline stops and leaves the edge open; breaks
in the ink up to six stroke widths long are bridged. A drawing with no outer
ink at all gets one closed stroke at the lines' typical width in their
darkest ink, or a 2 px black line with no lines at all. It is smoothed and
simplified like the region edges and stops where the drawing runs off the
canvas. Traced line pieces along it are dropped, inner
lines that reach it are carried on to it, so they join, and the drawing's
regions meet the background beneath its middle. **Outline
tolerance** is how far a traced edge or line may
stray from the reference, in reference pixels. It runs on the CPU and needs
no extra.

The Colour regions method fits a palette on the GPU and traces each colour
region. Dark outlines can be treated as ordinary regions, kept as separate
linework, or kept as linework with the regions beneath cleaned up, which
defines each region once and reuses it for fill and clip. **Merge and clean up
geometry** merges compatible paths and drops redundant vertices afterwards. It
needs CUDA and PyTorch (the `vision` extra).

## Limits

Point tools edit `path` elements, selected or inside a selected group; basic
shapes are converted to paths first, as they are to be clipping boundaries. Local
`use` instances can be selected, styled, moved and detached; editing a referenced source still
requires selecting every affected consumer, as enforced by the backend. Groups
with compositing or reference relationships that cannot be ungrouped without
changing appearance return a clear error. The knife cuts only along straight
lines: a filled shape only where the line crosses it from outside to outside,
a stroked line wherever the line crosses it. Join points welds two chosen
points and Join line ends the ends that continue each other within 12 screen
pixels; neither joins lines side by side. Redraw outline replaces a stretch between two places
on one contour; it cannot extend an open contour past its ends, and a spike
only a few reference pixels wide may come out thinner than drawn.

Only the documented static SVG subset is accepted. Unsupported imports are
reported instead of silently dropping content. The editor namespaces SVG IDs
inside the page so artwork cannot collide with editor controls. The local server
rejects external origins, unknown sessions and stale document revisions.

## Names and snapped edges

Select one object and edit **Name** in the right panel (or press **F2**).
Enter or leaving the field applies the name; Escape cancels. Clearing the field
restores its automatic label. Names support undo/redo and survive project saves
and SVG export/reimport, without changing object IDs or shared references.

**Snap edges…** matches touching edges of two or more selected, closed
paths: every pair of them that comes within the contact distance. Adjust the
distance, preview the snapped spans highlighted in cyan, then apply. Of each
pair the front contour is the reference: the edges of the region behind are
split where the front's points project onto them, and its matched points and
curve handles move onto the front's. Curves are subdivided to accommodate
different node spacing without flattening. The result is ordinary geometry
and one undoable edit; nothing links the paths afterwards, so editing,
moving or deleting one never changes the other, and SVG export keeps the
coincident contours as they are.

The paths must be closed and unclipped and may not share geometry. Pins and
locks are enforced. Within one run, an edge snapped for one pair is not
matched or moved again for another, so a region can meet two neighbours at
once. It matches line-to-line and cubic-to-cubic spans, rather than
rebuilding mismatched contour types. Contact distance is measured in canvas
SVG units, including paths in differently transformed groups; transforms,
paint and stacking remain intact.

Projects saved while the editor linked shared boundaries (version 2) still
open; the links are dropped and the contours they joined are kept as they
are.

### Setting and removing clips

A clip hides whatever lies outside a closed path, the boundary. The boundary
path itself stays in the drawing, painted and where it was.

1. Select the objects to clip (paths, shapes, groups or instances) and the
   closed path or paths to clip them to, for example a shading group and the
   body shape it should stay inside.
2. Run **Set clip…** (Clipping, in the right panel, the right-click menu or
   the command palette). Its dialog lists every selected object with two
   roles, **Clipped** and **Boundary**. It starts with the frontmost closed
   path as the boundary and everything else clipped; change any row, and the
   sentence under the list says what will be clipped to what. Nothing changes
   until you press **Set clip**.
3. Choose the **boundary geometry**:
   - **Linked** (the default): the clip shares the boundary path's points.
     Reshaping the path in Nodes reshapes the clip too, so a body and the
     shading clipped to it stay in step.
   - **Copied**: the clip takes a copy of the outline the boundary has now.
     Later edits to the path, or deleting it, leave the clip as it is.

   Either way the clip is placed where the boundary is now: moving, resizing
   or regrouping the boundary path later does not move the clip. Set the clip
   again to follow it.

Several boundaries clip to their combined area, and holes in a boundary (by
its fill rule) stay hidden. A boundary may be inside the group it clips. Each
clipped object gets its own clip in its own coordinates, so objects inside
rotated or scaled groups are clipped where the boundary appears on the page.
Setting a clip on an object that already has one replaces it.

Boundaries must be closed paths. A rectangle, circle, ellipse or line cannot
be one until **Convert to path** (Path) makes it a path; Set clip says so, as
it does for open paths. A group cannot be both clipped and contain another
clipped object in the same edit. A clipped object whose structure is locked
is refused; locks on the boundary do not matter, since it is only read.

**Remove clip** stops clipping the selected objects that are clipped; the
others are left alone. The boundary paths stay in the drawing. A clip that
was imported and is shared with other objects stays for them.

Setting and removing a clip are each one undoable step. Saved projects keep
the clip and its link to the boundary; exported SVGs contain an ordinary
`clipPath`, and reopening one in Vectrify keeps a linked clip linked. Agents
set and remove clips with the same operation, through the MCP `clip` tool.

### Sampling colours

The **Eyedropper** (`I`) takes a colour from what is on the canvas:

1. Select the objects to paint, or nothing to set the drawing paint (the
   Fill or Stroke in the drawing tools' strip, used by new shapes and paths).
2. Choose **Artwork** or **Reference** in the tool strip: what is sampled,
   whatever the view shows. **Fill** or **Stroke** chooses what the sample
   sets. **Sample fill colour** and **Sample stroke colour**, the buttons
   beside the inspector's colours, pick up the tool with that target.
3. Click. The colour under the pointer becomes the fill or stroke of every
   selected object as one undoable paint edit, the same edit the
   inspector's colour fields make; a toast names the colour.

Before the click, the strip says where the sample goes ("Reference → fill
of 2 objects", or "Artwork → drawing fill") and a mark at the pointer shows the disc that
is sampled and the target: a solid swatch for a fill, an outlined one for a
stroke. Nothing is sampled until the press is released, where it is
released; Escape before then cancels and changes nothing, as does a sample
that is refused or fails.

The sample is taken by the session, in document coordinates: the pointer's
place in the drawing, at any pan and zoom. It is the colour as seen, not
any object's paint, and it is the same sampling MCP's `pick` reports:

- the average over a disc about 4 screen pixels across (2 pixels in
  radius), never less than one document unit across, so zooming in samples
  more precisely; over an edge or a gradient it gives their mix;
- opaque: the artwork is rendered over white paper and the reference
  composited onto white, so translucent paint (opacity, fill-opacity,
  transparent pixels) gives its blend with what lies under it, and empty
  paper reads as white;
- applied as a flat, opaque colour; the objects' own opacity is kept.

The reference covers only the artboard: sampling it outside is refused, and
without a reference **Reference** is dimmed and says to load one. Locked
paint (a paint lock on the object or a group around it, or a lock on the
fill or stroke chosen) and clipping contours are refused with the object's
name, shown in the strip and at the pointer before any click.

### Extracting and deleting regions

**Extract region** takes the part of the paths inside a region into paths of
their own, one per path, just above it with its paint; **Delete region**
deletes that part. Both are drawn with the **Region** tool (`X`), and
running either from the Path list, the right-click menu or the command
palette picks up the tool with that operation chosen.

1. Choose **Extract** or **Delete** in the Region panel, at the top right of
   the canvas.
2. Draw the region: drag a **Rectangle**, or with **Polygon** click its
   corners and then click the first corner again or press Enter. Backspace
   takes back the last corner. A click in Rectangle selects, as in Select,
   so the objects to act on can be picked without changing tools.
3. Choose what it acts on: **The selected objects** (paths and basic shapes,
   and those inside selected groups), or **Every unlocked path or shape
   here**: those whose geometry and structure are not locked, inside the
   entered group if one is entered, as the knife does with nothing selected.
4. Choose whether to **Cut objects crossing the edge** (on by default).

Nothing changes yet. The region is outlined on the canvas, and the objects it
would change are highlighted by what happens to each: amber for contours
taken whole, orange for an object cut along the edge, red for one deleted
whole, and grey dashed for one the region reaches but leaves as it is. The
panel says the same in words, object by object, with a sentence such as
"Extracts from 2 objects, cutting 1 along the edge." or the reason the
operation is refused. The preview is the edit itself, made and dropped, so it
cannot disagree with what applying does. It follows the drawing: a change to
the selection, the options or the drawing previews again.

Press the panel's button (Extract region or Delete region), Enter, or run the
command again to apply it as one undoable edit; Escape or **Cancel** drops
the region and leaves the drawing as it was. Extracted pieces are selected
afterwards.

A contour wholly inside the region goes as it is, keeping its points. With
cutting on, a stroke crossing the edge comes apart where it crosses, and a
filled shape whose outline crosses it is split along it, as the knife splits
shapes; with cutting off, those stay whole and only contours wholly inside
are taken. A filled shape's holes stay with the shape around them, so a
region holding a hole but not its shape takes nothing. A fill merely around
the region is left alone, and a path wholly inside is left as it is by
Extract (it is a path of its own already) and deleted by Delete. A rectangle,
circle, ellipse or line the region cuts becomes the path of what is left of
it, keeping its id and paint. An instance draws shared geometry, so one in
the selection, or painting inside the region with every unlocked path
chosen, is refused: use **Detach** first. Pinned points refuse deleting,
shared geometry needs Detach, and locks refuse the edit, as elsewhere.
Agents make the same edit with the MCP `extract` and `delete(region=...)`
tools.

### Clipping boundaries

**Definitions** starts collapsed when you open a drawing. Expand it to browse
its contents; selecting a clipping contour on the canvas opens its branch
automatically.

Select a clipping boundary or one of its contours in **Objects** to see its
actual outline in blue. The outline is shown wherever that boundary is used,
including transformed objects and clips sized relative to an object's bounds.
**Show clips** in the footer (also in Commands) reveals other boundaries as
amber dashed outlines. Click an outline to select its contour, or drag a box
around it; clicks inside the boundary still select the artwork.

Use **Nodes** (`S`) to edit a selected clipping path's points. For a shared
boundary, points appear on the instance you clicked, or the first instance
when selected in the tree. Edits update every object using that boundary and
can be undone. These outlines are editor guides and do not appear in SVG
exports.
