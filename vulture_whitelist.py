"""Names vulture cannot see used: called by name, by frameworks or by pytest."""

from vulture.whitelist_utils import Whitelist

_ = Whitelist()

# Agent tools, dispatched by name: getattr(self, f"tool_{tool}").
_.tool_settings_schema
_.tool_hello
_.tool_describe
_.tool_pick
_.tool_trace_reference
_.tool_get_svg
_.tool_points
_.tool_history
_.tool_export
_.tool_isolate_feature
_.tool_isolate_components
_.tool_protect_features
_.tool_linked_outline
_.tool_clip
_.tool_properties
_.tool_transform
_.tool_arrange
_.tool_group
_.tool_combine
_.tool_join
_.tool_split_parts
_.tool_cut_hole
_.tool_boolean
_.tool_holes
_.tool_convert
_.tool_delete
_.tool_add_path
_.tool_set_points
_.tool_point_style
_.tool_break_points
_.tool_split_edge
_.tool_extract
_.tool_knife
_.tool_redraw_outline
_.tool_set_reference
_.tool_undo
_.tool_redo
_.tool_generate
_.tool_tidy
_.tool_fit_colours
_.tool_snap_edges
_.tool_cleanup
_.tool_job

# Session commands, operations and point edits, dispatched by name.
_._operation_start
_._operation_status
_._operation_stop
_._operation_discard
_._operation_apply
_._command_select
_._command_open
_._command_rename
_._command_locks
_._command_visibility
_._command_pin
_._command_reference
_._command_artboard
_._command_history
_._command_points
_._point_node
_._point_move_nodes
_._point_split
_._point_delete_handle
_._point_handles
_._point_delete_node
_._point_break
_._point_join_two_ends
_._point_delete_contour
_._command_lines
_._command_add_path
_._command_add_shape
_._command_copy
_._command_paste
_._command_move_objects
_._command_extract
_._command_fill_holes
_._command_holes_to_shapes
_._command_cut_hole
_._command_boolean
_._command_paint
_._command_move
_._command_align
_._command_resize
_._command_rotate
_._command_flip
_._command_group
_._command_ungroup
_._command_delete
_._command_reorder
_._command_detach
_._command_convert_to_path
_._command_set_clip
_._command_remove_clip
_._command_split_disconnected
_._command_combine_paths
_._command_join_paths
_._knife
_._redraw_outline

# http.server hooks the base classes call.
_.daemon_threads
_.log_message
_.do_GET
_.close_connection
_.do_POST

# uvicorn.Server flags read by its serve loop.
_.should_exit
_.force_exit

# Generated MCP tool functions are named for FastMCP.
_.__qualname__

# MCP tools registered by decorator and called by clients.
_.load_reference

# CleanupStats TypedDict fields, filled and read as dict keys.
_.paths_before
_.paths_after
_.line_vertices_before
_.line_vertices_after
_.vertices_removed
_.paths_merged
_.duplicate_paths_removed
_.empty_paths_removed
_.skipped_reason

# pytest: collection hook and fixtures used via usefixtures/autouse.
_.collect_ignore_glob
_.snap_twists
_.low_resolution
_.fit_ready
_.cpu_only

# Mock configuration.
_.return_value
_.side_effect

# Enum members the test iterates over.
_.PermissionGrantedByUser
_.PermissionDeniedByUser
