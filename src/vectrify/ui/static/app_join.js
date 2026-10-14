import {JOIN_REACH} from './app_config.js';

// Join owns its controls; editor contains the live state shared with other controllers.
export function createJoinController(editor, {
  action,
  fillPaths,
  joinCandidates,
  linePaths,
  openJoin,
  plural,
  toast,
} = {}) {

  // Merge shapes: a filled path unions its own contours; two or more paths
  // merge by area, after choosing colours. (Join points joins two selected
  // points, and Join line ends the lines that continue each other.)
  const joinOpensDialog = () => joinCandidates().length > 1;

  function mergeShapes() {
    if (joinCandidates().length === 1 && fillPaths().length === 1) return action('join_paths', {}, 'Merging overlapping contours…');
    return openJoin();
  }

  // Every selected line joins the ends that continue it nearby.
  async function joinEnds() {
    const lines = linePaths().length;
    if (await action('join_ends', {reach: JOIN_REACH / editor.zoom}, 'Joining line ends…')) toast(`Joined the ends of ${plural(lines, 'line path')} that continue each other within ${JOIN_REACH} screen pixels; zoom out to reach wider gaps.`);
  }

  function wire() {
      return api;
  }
  const api = {joinOpensDialog, mergeShapes, joinEnds, wire};
  return api;
}
