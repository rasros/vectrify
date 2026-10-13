import {JOIN_REACH} from './app_config.js';

// Join owns its controls; editor contains the live state shared with other controllers.
export function createJoinController(editor, {
  action,
  fillPaths,
  joinCandidates,
  linePaths,
  openJoin,
  plural,
  pointPairs,
  toast,
  twoEnds,
} = {}) {

  // Join: two selected points join each other; lines join the ends that
  // continue them nearby; a filled path unions its contours; multiple filled
  // paths merge by area, after choosing colours.
  const joinOpensDialog = () => !twoEnds() && joinCandidates().length > 1 && linePaths().length < joinCandidates().length;

  function join() {
    if (twoEnds()) return action('join_two_ends', {points: pointPairs()}, 'Joining ends…');
    if (joinCandidates().length === 1 && fillPaths().length === 1) return action('join_paths', {}, 'Joining overlapping contours…');
    return joinOpensDialog() ? openJoin() : joinEnds();
  }

  // Every selected line joins the ends that continue it nearby.
  async function joinEnds() {
    const lines = linePaths().length;
    if (await action('join_ends', {reach: JOIN_REACH / editor.zoom}, 'Joining line ends…')) toast(`Joined the ends of ${plural(lines, 'line path')} that continue each other within ${JOIN_REACH} screen pixels; zoom out to reach wider gaps.`);
  }

  function wire() {
      return api;
  }
  const api = {joinOpensDialog, join, joinEnds, wire};
  return api;
}
