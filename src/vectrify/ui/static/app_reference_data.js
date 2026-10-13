

// Reference_data owns its controls; editor contains the live state shared with other controllers.
export function createReferenceDataController(editor, {
  request,
  showReference,
} = {}) {

  async function loadReference(){
    const result=await request('/api/reference');
    if(result.epoch!==editor.state.epoch||result.reference_revision<editor.state.reference_revision)return;
    editor.reference=result.reference;showReference();
  }

  function wire() {
      return api;
  }
  const api = {loadReference, wire};
  return api;
}
