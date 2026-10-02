# MCP server

Vectrify's MCP server lets an agent (Claude Code, Claude Desktop or any MCP
client) look at a drawing and its reference and edit the drawing with the
same commands a person has, at the same granularity. It replaces the removed
LLM features: instead of Vectrify calling a model, the agent calls Vectrify.

It runs two ways, with the same drawing tools:

- **Hosted by the editor** over Streamable HTTP, while **Agents** is on in
  a window's footer: always on that window. Added to a client once.
- **`vectrify-mcp`** over stdio, started by the client: a file headlessly,
  or a running editor's window through `connect()`.

## The editor hosts it

Turning on **Agents** (installed with `vectrify[mcp]`) starts the MCP
server inside the editor, at `http://127.0.0.1:8770/mcp`: uvicorn running
the SDK's Streamable HTTP ASGI app (`MCPServer.streamable_http_app`) in a
background thread. It works the same in the desktop app and in `--serve`.
Turning Agents off stops the server, and so does quitting.

- **A stable port.** The port it was hosted on is kept in
  `$XDG_STATE_HOME/vectrify/mcp-port` and preferred next time, so the URL a
  client was given keeps working; `vectrify --mcp-port N` asks for N
  instead. With neither it is 8770. If the port is taken it takes the next
  free one of the following 20 (and remembers that), and the popover warns,
  in amber, that the URL changed, with the old URL and the new one: clients
  added with the old URL must be updated (the footer's dot turns amber too).
  The status carries it as `agent.mcp.moved: {old, new}`.

The footer's popover shows the URL and the command, with a Copy button:

```bash
claude mcp add --transport http --scope user vectrify http://127.0.0.1:8770/mcp \
  --header "Authorization: Bearer <token>"
```

(`--scope user` makes it available in every project; leave it out for the
current one only.) The tools act on the window's session in the editor's
own process, as `connect()` does, so edits show live and land in the
window's history as "Agent: …", and the footer's *Agent connected · <last
action>* counts these clients too. There are no `open` or `connect` tools
here; everything else is as listed below.

- **Token.** One stable token, made once and kept in
  `$XDG_STATE_HOME/vectrify/agent-token` (default
  `~/.local/state/vectrify/agent-token`; 0600 in a 0700 directory), so the
  client is added once. *Regenerate token* in the popover replaces it;
  clients holding the old one are refused. Every request needs
  `Authorization: Bearer <token>`; a missing or wrong one gets 401.
- **Only this machine.** It binds 127.0.0.1. Requests whose `Host` is not
  `127.0.0.1:<port>` or `localhost:<port>` get 403, and the SDK's transport
  security (`TransportSecuritySettings`) refuses a browser `Origin` other
  than those, against DNS rebinding.
- **Only while allowed.** With Agents off nothing listens; a request that
  arrives as it turns off gets 403.
- In the code: `vectrify/mcp/hosted.py` (`HostedMCP`, the guard),
  `build_window_server` and `register_tools` in `vectrify/mcp/server.py`
  (one function registers every drawing tool against a target, for both
  servers), and `WindowTarget` in `vectrify/mcp/target.py`.

## `vectrify-mcp` over stdio

```bash
uv tool install "vectrify[mcp]"          # or [all]; pipx works too
claude mcp add vectrify -- vectrify-mcp
# without installing:
claude mcp add vectrify -- uvx --from "vectrify[mcp]" vectrify-mcp
# from a source checkout:
claude mcp add vectrify -- uv run --directory /path/to/vectrify --extra mcp vectrify-mcp
```

`vectrify-mcp drawing.svg` opens that file at start. It is a stdio server
built on the official `mcp` Python SDK (`MCPServer`); the client starts it.

### Other apps, from the popover

The popover's *Add to an app* section shows, under the Claude Code command,
what Codex and Claude Desktop need to start `vectrify-mcp` themselves, each
with a Copy button. The command is this install's own executable (beside
the editor's Python, else the one on `PATH`; `app_setup()` in
`vectrify/ui/agent.py`), and there is no token: started with no target, the
server attaches to the window with Agents on through `editor.json` (below).
The editor only shows the text; it never edits another app's config.

- **Codex** (the ChatGPT desktop app, the Codex CLI and the IDE extension
  share one config): paste into `~/.codex/config.toml`
  (`$CODEX_HOME/config.toml` when that is set), next to any other
  `[mcp_servers.*]` tables, then restart Codex:

  ```toml
  [mcp_servers.vectrify]
  command = "/path/to/.venv/bin/vectrify-mcp"
  startup_timeout_sec = 30
  ```

  `startup_timeout_sec` raises Codex's 10 s default, since the server
  imports the vision stack as it starts. Or run
  `codex mcp add vectrify -- /path/to/.venv/bin/vectrify-mcp` (the default
  timeout).
- **Claude Desktop**: merge into `mcpServers` in
  `claude_desktop_config.json` (macOS
  `~/Library/Application Support/Claude/`, Windows `%APPDATA%\Claude\`;
  unofficial Linux builds read `~/.config/Claude/`), keeping any other
  servers, then restart Claude Desktop:

  ```json
  {"mcpServers": {"vectrify": {"command": "/path/to/.venv/bin/vectrify-mcp"}}}
  ```

## Targets

The server edits one target at a time, with the same tools for both:

1. **A file, headless.** `open(path)` loads an `.svg` or `.vectrify` project
   into its own `Backend`/`Session` in the server's process. `save()` writes
   it back (a `.vectrify` path keeps locks, pins, the reference and the
   selection; any other is plain SVG), and `save(path)` writes elsewhere
   (and makes that the file `save()` writes).
2. **The running editor, live.** When the person turns on **Agents** in the
   editor's footer (or *Allow agents to edit* in the command palette), the
   stdio server can also join the session that window shows: each edit appears as it is
   made and lands in the window's undo history. With no target yet, the
   first tool call attaches to such an editor by itself; `connect()` does so
   explicitly, and `open(path)` switches to a file.

`describe()` says which target is in use.

### The live channel

`vectrify/ui/agent.py` holds both ends that live in the editor:

- `Agent` answers the agent's calls on one `Session`. Rendering, comparing
  and describing happen here, in the editor's process, so the live and the
  headless target give identical answers (the headless target holds an
  `Agent` in the MCP server's process).
- `AgentChannel` is the door. Turning **Agents** on (`/api/agent`) binds
  the window's session, starts the hosted MCP server, and writes
  `{url, token, pid, mcp}` to `$XDG_STATE_HOME/vectrify/editor.json`
  (default `~/.local/state/vectrify/editor.json`), owner-only (0600, in a
  0700 directory). The token is the stable one the hosted server uses.
  Turning it off, or quitting, removes the file and stops the server.
- `/agent/call` stays: it is what the stdio `vectrify-mcp`'s `connect()`
  (`LiveTarget`) talks to, from its own process. A client that adds the
  hosted URL never uses it. `url` in `editor.json` says where it is:
  - In `--serve` mode the editor's own server carries the channel on its
    port, under `/agent/`.
  - The desktop app (pywebview, which has no port) carries it on the hosted
    MCP server's port, under `/agent/` next to `/mcp`, behind the same guard
    and token (`Guard.agent_call` in `vectrify/mcp/hosted.py`, which runs
    each call off the event loop). One port, open only while Agents is on.
    Without the `mcp` extra in the editor's environment the desktop app has
    no agent channel at all.
- Requests: `POST /agent/call` with `{tool, args}` and
  `Authorization: Bearer <token>`, answered with `{data, images}`; each image
  is fetched as a raw PNG body from `GET /agent/image/<key>`, never as base64
  inside JSON. Requests must be addressed to `127.0.0.1` or `localhost`. Off:
  403; a missing or wrong token: 401; an edit of a drawing that changed since
  the agent looked: 409; a refusal: 400 with the editor's message.
- The page is told of each agent call as it happens. After every call
  (looks too, and Agents turning on or off) `AgentChannel` beats
  (`changed()`; `wait(beat, timeout)` waits for the next), and the page gets
  its *pulse*, `{session, epoch, revision, agent}`, the same body
  `/api/poll` answers:
  - with `--serve`, as server-sent events from `GET /api/events?session=…`
    (same-origin like the page's other requests; one `data:` line per pulse,
    a comment every 15 s when idle; `Backend.events`), which the page opens
    with an `EventSource` while Agents is on;
  - in the desktop app, by a thread of `desktop.Api` that runs
    `window.vectrifyPulse(<pulse>)` in the page with pywebview's `run_js`, so
    an agent call never waits on the page.

  When the agent changed something (or the revision moved), the page fetches
  the session's state and redraws and marks the drawing unsaved. Mid-gesture
  (a drag, or an edit of the person's own in flight) it waits and looks
  again every 120 ms, so the edit shows once the pointer is released. It
  still polls `/api/poll` every 5 s, which covers a dropped channel. Measured
  in headless Chrome with `--serve`, an SDK client's paint showed in the
  page 15 ms (median of 12; at most 48 ms) after the client sent it,
  before the client had its answer back (about 50 ms). The footer reads
  *Agents off*, *Agents allowed*, or *Agent connected · <last action>*
  (connected means a call in the last two minutes).
- The page reports what the window shows, for `view()`: the visible part of
  the drawing `[x, y, w, h]` in document units, the zoom (screen pixels per
  unit), the canvas size in pixels, the active tool, the entered group and
  the reference view and opacity. It sends it with each poll and, 250 ms
  after the view changes (zoom, pan, tool, entered group, reference view),
  in a poll of its own; `Session.set_view` keeps the latest. Nothing is
  sent while Agents is off.
- The poll's `agent.touched` lists the objects each of the agent's recent
  changes touched, as `{change, ids}` (the last 20 changes; `change` counts
  as `agent.changes` does). Once the page shows a change it outlines those
  objects in the overlay for 1.5 s, fading out. It is not a selection.

## Safety

- Every edit goes through `Session.action` or `Session.operation`, so
  selection scope, locks, pins, permissions and revision checks hold exactly
  as in the editor: an agent can do nothing a person couldn't.
- Revisions: the server remembers, for each client, the epoch and revision
  of the last answer it had (describe, render, an edit…) and sends it with
  that client's edits. If the drawing changed since, the edit is refused
  with a message telling the agent to `describe()` again, so it never
  overwrites a person's (or another agent's) concurrent edit it has not
  seen. `undo()` is checked the same way, so the agent never undoes a step
  it has not seen. A client is its Streamable HTTP session
  (`Mcp-Session-Id`); a stdio server is one client. A middleware
  (`per_client` in `vectrify/mcp/server.py`) puts the client's key in a
  context variable for the tool call; what it saw is also tied to the
  target it looked at (`identity()`: the file, the editor, or the window's
  `Agent`), so after Agents moves to another window every client must look
  again.
- The person's selection stays theirs. Every edit names its targets and
  none acts on the current selection. Inside its locked step a call selects
  its own targets (the session's checks of scope, locks and points work on
  the selection), then gives the person their selection back: objects and
  points, without those the call deleted. Giving it back is neither a
  revision nor an undo step, and the call's undo step selects the person's
  selection too, so undoing it does not select the agent's targets. The
  entered group is the page's own and is kept unless it was deleted. The
  headless file target goes through the same code.
- One undo step and one revision per call. A tool that needs several
  session commands (say `add_path` with a fill, a parent and a name,
  `properties` with paint, a name and locks, or `transform` to a box) runs them under the session lock and squashes them into
  one history entry, and `Editor.settle` then counts them as one revision:
  nobody saw the ones in between, and what was cached at them is dropped.
  `undo(steps=n)` is one revision too. If a later command is refused, the
  earlier ones are rolled back (no redo is left behind) and the refusal
  reports the new revision as seen.
- History labels: while an agent's call runs the editor puts "Agent: " before
  each new history label (`Editor.label_prefix`), so the window's undo
  history and `history()` say who made each step.
- Live editing only while the window allows it; the token keeps other local
  processes out. File writes happen in the MCP server's process, so the
  client's tool approval is the consent for each path.
- Budgets: renders are PNG, 1024 px on the long side by default and at most
  2048; `describe` pages large trees (100 objects a page, at most 500);
  `points` pages nodes (300 a page, at most 2000) and says `more`;
  `pick` lists 20 objects and 30 contours of each; `trace_reference` gives
  at most 40 shapes and about 16,000 characters of path data.
- Looking is read-only: `view()` reads the person's selection and viewport
  and never changes them; `pick` and `trace_reference` change nothing.

## Tools

Ids are object ids as `describe()` lists them; points are `[object, node]`
pairs from `points()`. Every edit requires its targets: `ids` (at least one)
for object tools and operations, `points` or `contours` for point and hole
tools; the schema refuses a call without them (`join` and `delete` take
`ids`, `points` or a region, and refuse a call with none). There is no `select` tool:
the agent has no selection that lasts between calls. An edit's answer gives
`result`, the objects and points the edit left selected for itself (the new
group, the cut pieces), and `created` and `removed`. Unset arguments take the editor's defaults. Every answer is
JSON text, followed by `Image: <name>` and the PNG for each image.

**Targets**: `open(path)`, `connect(url?, token?)`, `save(path?)` (a
`.svg` path writes plain SVG, a `.vectrify` path a project; no path saves to
the opened file), `load_reference(path?)` (PNG, JPEG or WebP, stretched over
the artboard as the editor shows it; no path removes the reference).

All coordinates are document coordinates (the root's user space) and every
region is `[x, y, w, h]`; a region that edits (`extract`, `delete`) may also
be a polygon `[[x, y], ...]`.

**Looking**
- `describe(page?, page_size?, within?, region?)`: target, artboard,
  reference (name and pixel size), the person's selection (for context; no
  tool acts on it), and objects (id, label, name, tag, parent,
  depth, paint, painted bounds `[x, y, w, h]`, transform, locks), a page at a
  time, or one group's contents. With `region`, only the objects that paint
  inside it, front to back, each path with the contours of it that do.
- `pick(x, y, radius?)`: what paints at a document point, or within
  `radius` of it, front to back: each object with its groups and, for a
  path, its contours there (`index`, `id`, `first_node`, `count`, `closed`,
  `bounds`, and whether its `stroke` or `fill` paints there). It is a hit
  test of painted coverage (`HitIndex.contours_in`): fills with their
  holes, strokes with their width, caps and joins, through transforms and
  clips; not bounding boxes. Its `colour` gives the drawing's and the
  reference's colour there (the mean over a disc of `radius`, at least half
  a unit) and their `difference` (RGB distance, 0 to 1).
- `view()`: what the person is looking at: their selection (objects and
  points) and, in the editor window, the visible `region`, the `zoom`, the
  canvas `pixels`, the active `tool`, the `entered_group`, the
  `reference_view` (drawing, overlay or reference) and how long ago the
  window reported it. Headless (or before the window has reported) it says
  there is no window and gives the whole artboard.
- `render(region?, overlay?, max_side?, grid?)`: the drawing; `region` in
  document units, or `"view"` for exactly what the window shows (its region
  at its pixel size, with the reference over it or alone as the window
  shows it); `overlay="side"` puts the reference beside it, `"over"` blends
  them, `"reference"` shows the reference alone. `grid=true` draws lines
  labelled with document coordinates.
- `compare(region?, max_side?, grid?)`: mean squared error against the
  reference (RGB in 0..1, on white), the worst four cells of a 4 x 4 grid as
  regions, and a heat map (black agrees, through red and yellow to white).
- Every image's answer has `mapping`: the `region`, the image's `pixels`,
  the `units_per_pixel`, and a sentence giving the document point of the
  top-left pixel and the formula (side by side, where the reference
  starts).
- `trace_reference(region?, colour?, dark?, tolerance?, min_area?)`: the
  reference's dark areas in the region (luminance at most `tolerance`,
  0.35 by default), or those near `colour` (RGB distance, 0.12 by
  default), traced as closed cubic path data in document coordinates
  (`samvg.mask_path` over each connected area, holes included), largest
  first, with each area's size and bounds. Areas smaller than `min_area`
  square units (default 6 pixels) are left out.
- `get_svg(ids?)`: the SVG of the drawing or of some objects.
- `points(id, region?, contours?, coords?, nodes?, page?, page_size?)`: a
  path's contours (`index`, `id`, `first_node`, `count`, `closed`,
  `bounds`, `hole`; a hole also has its `area`, and `holes_total` counts
  them) and their nodes (`id`, `i` its place in the contour, `command`,
  `values`, `pinned`), with the paths sharing the geometry (`users`) and
  its `transform` when it has one. `region` keeps the contours crossing it
  and the nodes inside it, `contours` those indices, `nodes=false` lists
  contours only. Values are document coordinates; `coords="local"` gives
  the path's own, `"both"` both. Nodes come 300 a page; `more` says how
  many are left and how to get them. A hole's contour `id` is what `holes`
  takes.

Renders of the drawing and the reference are cached per (epoch, revision,
region, size), so looking again at an unchanged drawing costs no rendering.

**History**: `history(limit?)` (undo and redo stacks, newest first: label,
author `agent` or `person`, revision), `undo(steps?)`, `redo(steps?)`.

**Objects**: `properties(ids, fill?, stroke?, stroke_width?, opacity?,
fill_opacity?, stroke_opacity?, name?, locks?)` (paint; a name, with one id
only; locks, an empty list unlocking; unlocking runs before the other
changes and locking after them, all one step), `transform(ids, dx?, dy?,
scale?, anchor?, box?)` (move and/or scale about an anchor, or fit the
painted bounds to `box`), `arrange(ids, to)` (front, back, forward,
backward) or `arrange(ids, parent, index?)` (into a group, the front unless
`index` says otherwise), `group(ids)`, `ungroup(ids)`, `join(ids? |
points?, reach?, bridge?, color_source?)` (the editor's Join: two points
join each other; stroked lines join their ends within `reach`; filled
paths merge by area; `joined` says which), `split_parts(ids)`,
`cut_hole(ids)`, `holes(contours, action?, delete_enclosed?)` (`fill` or
`shape`, of the holes `points()` marks), `convert(ids, to?)` (line, fill,
either, or `path`: an instance, a basic shape or shared geometry made an
editable path of its own, keeping its id and paint; nothing else detaches
by itself), `delete(ids? | points? | region?, contours?, cut?, detach?)`
(objects; points, or with `contours=true` the
whole contours they are on; with a region, the contours inside it, see
below), `add_path(d, fill?, stroke?, stroke_width?,
parent?, index?, name?)` (path data in document coordinates, several
subpaths allowed; without `parent` it goes into the group of the frontmost
object drawn under its centre, or its bounds, just above that object, and
`placed` says where and why; a group with a structure lock, opacity or a
clip is skipped, and with nothing under it, it goes in front at the top
level), `knife(start, end, ids?, within?)` (without `ids`
it cuts every unlocked path it crosses, as the editor's knife does with
nothing selected), `redraw_outline(id, points, long_way?, pixel?)`.

**Regions**: `extract(region, ids?, cut?, group?, detach?)` takes the
contours of the paths `ids` (or of every path or shape painting in the
region whose geometry and structure are unlocked) that lie inside the
region into a new path per path, with its attributes, just above it in its
group, and answers `extracted`: `{path, from, contours}`. With
`group=true` the new paths go into one new group (`group` in the answer)
just above the frontmost path they came from, in that path's group: pieces
from other groups move there keeping where they are drawn (the transform
and paint they inherited are written onto them, as `arrange(parent=…)`
does), all in the one undo step "Extract region into a group". A rect,
circle, ellipse or line is cut as the path of its outline (cubic arcs for
round parts, `regions.shape_geometry`): one the region cuts becomes that
path, keeping its id, paint and transform (a line's path is given
`fill="none"`); one wholly inside is left as it is (or deleted). An
instance (`use`) draws shared geometry, so a region edit that would act on
one is refused, naming it, unless `detach=true`, which first gives it a
path of its own as `convert(ids, to="path")` does. `describe(region)` and
`pick` find shapes and instances by what they paint, as they do paths
(only paths list contours). Contours wholly inside go as they are, keeping their node
ids. With `cut=true` (the default) a stroke crossing the region's edge is
cut there with the knife's machinery and its pieces go to the side they lie
on, and a filled shape whose outline crosses the edge is split along it
(Skia path ops, as the knife's fill cut; its shapes get new node ids); a
fill that merely surrounds the region is left alone, as is a shape and its
holes unless all of them are inside. A path lying wholly inside is left as
it is. `delete(region=..., ids?, cut?, detach?)` deletes them instead
(whole contours unless `cut=true`). Both are one undo step: the session
command `extract` (`Transaction.extract_region`, `document/regions.py`).

**Points**: `set_points({object: {node: values}}, coords?)` (document
coordinates unless `coords="local"`), `point_style(points, handles?,
pinned?)` (0, 1 or 2 handles; pin or unpin), `break_points(points)` (given
the two points at a segment's ends it deletes that segment, as the
editor's Break does), `split_edge(points)`, `delete(points, contours?)`,
`join(points=[a, b])`.

**Operations** (jobs): `generate(method?, settings?, group?)` (`cel`,
`colour-regions`, or `samvg` with a GPU; over the whole drawing, or into the
area of `group`), `tidy(ids, settings?, rounds?)`, `fit_colours(ids, fill?,
passes?, resolution?)` (flat or linear gradients), `snap_edges(ids,
tolerance?)`, `cleanup(ids)`. Each starts a job with the
permissions the editor's dialog would give it; `job(id, action?,
wait_seconds?, choice?)` follows it: `status` (the default) waits (at most
120 s, without holding the session) and returns the metrics and the
recommended result's previews (reference, before, after) as images;
`apply` keeps a result (or alternative `choice`) as one undo step,
`discard` drops it, `stop` stops it early.

**Guide**: the server's instructions (what a client sees up front) give a
short "for X use Y" list (what the person sees: `view`; what is under a
spot and its colours: `pick`; a path's nodes: `points(id, region)`;
coordinates of an image: its `mapping` and `grid`; the shape to match:
`trace_reference`; paint, names and locks: `properties`; joining: `join`;
a piece of a path: `extract`; deleting: `delete`; holes, jobs, saving) and
point to the `vectrify://guide` resource, which
explains the loop: look, edit, look again, undo what made it worse. Refusals
point to the right tool where one fits (an unknown id points to `pick` and
`points`; `describe` of a large drawing points to `pick` and `region`).

### Coverage

`tests/mcp/test_coverage.py` reads the commands `Session.action` and
`Session.operation` handle from the session's source, drives every tool
through an MCP client, and fails unless each command arrives from some tool
or is listed in `LEFT_OUT` / `OPERATIONS_LEFT_OUT` (`vectrify/ui/agent.py`)
with a reason. `select` has no tool of its own; it arrives as the first step
of every targeted tool. Left out:

- `open`: an agent opens a file as its own headless target; it never
  replaces the drawing in the person's window.
- `node`: the one-point drag; `set_points` sends `move_nodes`, which moves
  one point or many.
- `to_front`, `to_back`: internal names `reorder` (`arrange(to=...)`) is
  rewritten to.
- operation `check`: the dialog's probe; the agent starts the job and reads
  the refusal instead.

Not exposed either: `improve/path-fit` (no dialog in the editor uses it),
setting a gradient fill directly (the session's paint takes colours only;
`fit_colours(fill="linear")` makes gradients), and the editor's view
controls (zoom, overlay view, tools), which have no effect on the drawing:
the agent reads them with `view()` and never sets them.

## Limits

- One window per editor process takes agents at a time: turning Agents on in
  another window of the same `--serve` server moves the channel (and the
  hosted server's target) there; the agent must `describe()` again before it
  edits. The discovery file names the most recent editor to allow agents.
- Two editors both allowing agents host on two ports; a client added with
  the first reaches whichever got it first, and the second's popover says
  its URL moved.
- A request with no session (the sessionless 2026-07-28 Streamable HTTP
  protocol, which the SDK's own client speaks by default) is told apart
  only by the `clientInfo` it sends: two such clients of the same name and
  version share one record of what they saw. Clients that open a session
  (`Mcp-Session-Id`, as today's do) are kept apart.
- The hosted server needs the `mcp` extra in the editor's own environment;
  without it the popover says so, and only a `--serve` editor can still be
  reached, by `vectrify-mcp`'s `connect()`.
- The page flashes only the objects of changes it picks up; with several
  agent changes it takes at once (made mid-drag, or while it redraws) it
  flashes them together, and a deletion has nothing left to flash.
- A job (`tidy`, `fit_colours`…) works on the objects it started with; the
  person's selection meanwhile has no effect on it.
- Agent edits wait while the person is mid-drag, and show once it ends.
  Each open `--serve` page holds one event stream (a server thread) while
  Agents is on.
- Renders use the editor's export and Cairo; they match the page's SVG
  rendering closely but not pixel for pixel. `render(region="view")` shows
  the window's view of the drawing, without the page's checkerboard, the
  canvas around the artboard or the selection overlay.
- The view is as of the window's last report (sent 250 ms after it
  changes, and with each 5 s poll); a window that has not reported since
  Agents was turned on has none yet.
- Cutting a filled shape gives all of its contours new node ids. A shape's
  sizes must be plain numbers (no units or percentages) to be cut, and
  `detach=true` handles instances of paths in `defs` only, as Detach does.
  `extract(group=true)` is refused where the pieces cannot move into one
  group keeping their look (a group with opacity or a clip in between).
- `trace_reference` traces the reference at its own resolution there, at
  least 256 and at most 1024 pixels on the long side; finer detail needs a
  smaller region.
