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
background thread. If 8770 is taken it takes the next free port of the
following 20 and the popover shows which; `vectrify --mcp-port N` picks
another start. It works the same in the desktop app and in `--serve`.
Turning Agents off stops the server, and so does quitting.

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

## Targets

The server edits one target at a time, with the same tools for both:

1. **A file, headless.** `open(path)` loads an `.svg` or `.vectrify` project
   into its own `Backend`/`Session` in the server's process. `save()` writes
   it back (a `.vectrify` path keeps locks, pins, the reference and the
   selection; any other is plain SVG), `save(path)` and `export_svg(path)`
   write elsewhere.
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
  hosted URL never uses it.
  - In `--serve` mode the editor's own server carries the channel on its
    port, under `/agent/`.
  - The desktop app (pywebview, which has no port) opens a localhost port
    of its own when Agents is turned on, and closes it when turned off.
- Requests: `POST /agent/call` with `{tool, args}` and
  `Authorization: Bearer <token>`, answered with `{data, images}`; each image
  is fetched as a raw PNG body from `GET /agent/image/<key>`, never as base64
  inside JSON. Requests must be addressed to `127.0.0.1` or `localhost`. Off:
  403; a missing or wrong token: 401; an edit of a drawing that changed since
  the agent looked: 409; a refusal: 400 with the editor's message.
- The page polls `/api/poll` every 0.7 s while Agents is on. When the agent
  changed something (or the revision moved), it fetches the session's state
  and redraws, unless the person is mid-gesture, and marks the drawing
  unsaved. The footer reads *Agents off*, *Agents allowed*, or *Agent
  connected · <last action>* (connected means a call in the last two
  minutes).
- The poll's `agent.touched` lists the objects each of the agent's recent
  changes touched, as `{change, ids}` (the last 20 changes; `change` counts
  as `agent.changes` does). Once the page shows a change it outlines those
  objects in the overlay for 1.5 s, fading out. It is not a selection.

## Safety

- Every edit goes through `Session.action` or `Session.operation`, so
  selection scope, locks, pins, permissions and revision checks hold exactly
  as in the editor: an agent can do nothing a person couldn't.
- Revisions: the server remembers the epoch and revision of the last answer
  (describe, render, an edit…) and sends it with each edit. If the drawing
  changed since, the edit is refused with a message telling the agent to
  `describe()` again, so it never overwrites a person's concurrent edit it
  has not seen. `undo()` is checked the same way, so the agent never undoes a
  step it has not seen.
- The person's selection stays theirs. Every edit names its targets and
  none acts on the current selection. Inside its locked step a call selects
  its own targets (the session's checks of scope, locks and points work on
  the selection), then gives the person their selection back: objects and
  points, without those the call deleted. Giving it back is neither a
  revision nor an undo step, and the call's undo step selects the person's
  selection too, so undoing it does not select the agent's targets. The
  entered group is the page's own and is kept unless it was deleted. The
  headless file target goes through the same code.
- One undo step per call. A tool that needs several session commands (say
  `add_path` with a fill, a parent and a name, or `resize` to a box) runs
  them under the session lock and squashes them into one history entry; if a
  later command is refused, the earlier ones are rolled back (no redo is left
  behind) and the refusal reports the new revision as seen.
- History labels: while an agent's call runs the editor puts "Agent: " before
  each new history label (`Editor.label_prefix`), so the window's undo
  history and `history()` say who made each step.
- Live editing only while the window allows it; the token keeps other local
  processes out. File writes happen in the MCP server's process, so the
  client's tool approval is the consent for each path.
- Budgets: renders are PNG, 1024 px on the long side by default and at most
  2048; `describe` pages large trees (100 objects a page, at most 500).

## Tools

Ids are object ids as `describe()` lists them; points are `[object, node]`
pairs from `points()`. Every edit requires its targets: `ids` (at least one)
for object tools and operations, `points` or `contours` for point and hole
tools; the schema refuses a call without them. There is no `select` tool:
the agent has no selection that lasts between calls. An edit's answer gives
`result`, the objects and points the edit left selected for itself (the new
group, the cut pieces), and `created` and `removed`. Unset arguments take the editor's defaults. Every answer is
JSON text, followed by `Image: <name>` and the PNG for each image.

**Targets**: `open(path)`, `connect(url?, token?)`, `save(path?)`,
`export_svg(path)`, `load_reference(path)` (PNG, JPEG or WebP, stretched over
the artboard as the editor shows it), `remove_reference()`.

**Looking**
- `describe(page?, page_size?, within?)`: target, artboard, reference (name
  and pixel size), the person's selection (for context; no tool acts on
  it), and objects (id, label, name, tag, parent,
  depth, paint, painted bounds `[x, y, w, h]`, transform, locks), a page at a
  time, or one group's contents.
- `render(region?, overlay?, max_side?)`: the drawing; `region` in document
  units; `overlay="side"` puts the reference beside it, `"over"` blends them.
- `reference(region?, max_side?)`: the reference alone.
- `compare(region?, max_side?)`: mean squared error against the reference
  (RGB in 0..1, on white), the worst four cells of a 4 x 4 grid as regions,
  and a heat map (black agrees, through red and yellow to white).
- `get_svg(ids?)`: the SVG of the drawing or of some objects.
- `points(id)`: a path's geometry: contours, nodes (id, command, values,
  pinned) and the paths sharing it.
- `holes(id)`: a path's holes, with areas and bounds.

Renders of the drawing and the reference are cached per (epoch, revision,
region, size), so looking again at an unchanged drawing costs no rendering.

**History**: `history(limit?)` (undo and redo stacks, newest first: label,
author `agent` or `person`, revision), `undo(steps?)`, `redo(steps?)`.

**Objects**: `paint(ids, fill?, stroke?, stroke_width?, opacity?,
fill_opacity?, stroke_opacity?)`, `rename(id, name)`, `locks(id, locks)`,
`move(ids, dx, dy)`, `resize(ids, scale?, anchor?, box?)`,
`reorder(ids, to)` (front, back, forward, backward),
`move_into(ids, parent, index)`, `group(ids)`, `ungroup(ids)`, `join(ids,
color_source?)`, `join_ends(ids, reach?, bridge?)`, `split_parts(ids)`,
`cut_hole(ids)`, `fill_holes(contours, delete_enclosed?)`,
`holes_to_shapes(contours)`, `detach(ids)`, `convert(ids, to?)` (line, fill
or either), `delete(ids)`, `add_path(d, fill?, stroke?, stroke_width?,
parent?, index?, name?)`, `knife(start, end, ids?, within?)` (without `ids`
it cuts every unlocked path it crosses, as the editor's knife does with
nothing selected), `redraw_outline(id, points, long_way?, pixel?)`.

**Points**: `set_points({object: {node: values}})`, `handles(points,
count)`, `pin(points, pinned?)`, `break_points`, `delete_segment`,
`split_edge`, `delete_points`, `delete_contours`, `join_points(a, b)`.

**Operations** (jobs): `generate(method?, settings?, group?)` (`cel`,
`colour-regions`, or `samvg` with a GPU; over the whole drawing, or into the
area of `group`), `tidy(ids, settings?, rounds?)`, `fit_colours(ids, fill?,
passes?, resolution?)` (flat or linear gradients), `snap_edges(ids,
tolerance?)`, `cleanup(ids)`. Each starts a job with the
permissions the editor's dialog would give it; `job_status(id,
wait_seconds?)` waits (at most 120 s, without holding the session) and
returns the metrics and the recommended result's previews (reference, before,
after) as images; `apply(id, choice?)` keeps a result as one undo step,
`discard(id)` drops it, `stop(id)` stops it early.

**Guide**: the `vectrify://guide` resource and the server's instructions
explain the loop: describe, render and compare, edit, render again, undo what
made it worse.

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
- `to_front`, `to_back`: internal names `reorder` is rewritten to.
- operation `check`: the dialog's probe; the agent starts the job and reads
  the refusal instead.

Not exposed either: `improve/path-fit` (no dialog in the editor uses it),
setting a gradient fill directly (the session's paint takes colours only;
`fit_colours(fill="linear")` makes gradients), and the editor's view
controls (zoom, overlay view, tools), which have no effect on the drawing.

## Limits

- One window per editor process takes agents at a time: turning Agents on in
  another window of the same `--serve` server moves the channel (and the
  hosted server's target) there; the agent must `describe()` again before it
  edits. The discovery file names the most recent editor to allow agents.
- Two editors both allowing agents host on 8770 and 8771; a client added
  with 8770 reaches whichever got it first.
- Clients of the hosted server share one record of the revision last seen,
  so with two clients at once, one's `describe()` counts as the other's
  look too. Use one client at a time.
- The hosted server needs the `mcp` extra in the editor's own environment;
  without it the popover says so and `vectrify-mcp`'s `connect()` still
  works.
- The page flashes only the objects of changes it picks up; with several
  agent changes between two polls it flashes them together, and a deletion
  has nothing left to flash.
- A job (`tidy`, `fit_colours`…) works on the objects it started with; the
  person's selection meanwhile has no effect on it.
- The page notices agent edits by polling, so they show within about a
  second, not instantly, and not while the person is mid-drag.
- Renders use the editor's export and Cairo; they match the page's SVG
  rendering closely but not pixel for pixel.
