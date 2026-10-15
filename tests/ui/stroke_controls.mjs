import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createStrokeController} from '../../src/vectrify/ui/static/app_stroke.js';
import {createArrowsController} from '../../src/vectrify/ui/static/app_arrows.js';
import assert from 'node:assert/strict';

const fields = new Map(), field = id => { if (!fields.has(id)) fields.set(id, {value:'', hidden:false, disabled:false}); return fields.get(id); };
const path = id => ({id, tag:'path', open_path:true, attributes:{}, inherited_locks:[]});
const items = new Map([['a', path('a')], ['b', path('b')]]), edits = [], errors = [];
const state = {selection:{objects:['a']}};
const ctx = Object.assign(createEditorState(), {$:field, state, object:id=>items.get(id), resolvedPaint:(id,key,fallback)=>items.get(id).attributes[key] || fallback, enable:(f,reason)=>{f.disabled=!!reason;}, paint:c=>edits.push(c), toast:m=>errors.push(m)});
const ctx_stroke = createStrokeController(ctx, {
  $: (...args) => ctx.$(...args),
  enable: (...args) => ctx.enable(...args),
  object: (...args) => ctx.object(...args),
  paint: (...args) => ctx.paint(...args),
  resolvedPaint: (...args) => ctx.resolvedPaint(...args),
  toast: (...args) => ctx.toast(...args),
});
ctx_stroke.wire();
ctx.renderStrokeControls = ctx_stroke.renderStrokeControls;
const render=()=>(ctx.renderStrokeControls());
render();assert.equal(field('path-end-controls').hidden,false);assert.equal(field('stroke-linecap').value,'butt');
field('stroke-linecap').onchange({target:{value:'round'}});assert.equal(edits.at(-1)['stroke-linecap'],'round');
field('end-arrow').onchange({target:{value:'arrow'}});assert.equal(edits.at(-1)['data-vectrify-arrow-end'],'arrow');
state.selection.objects=['a','b'];items.get('b').attributes['stroke-linecap']='round';render();assert.equal(field('stroke-linecap').value,'');
items.get('a').inherited_locks=['paint'];render();assert.equal(field('end-arrow').disabled,true);
items.get('a').inherited_locks=[];items.get('b').open_path=false;render();assert.equal(field('path-end-controls').hidden,true);assert.equal(field('stroke-shape-controls').hidden,false);
state.selection.objects=['a'];items.get('a').attributes['stroke-linejoin']='round';render();assert.equal(field('miter-limit-field').hidden,true);
field('stroke-miterlimit').onchange({target:{value:'0'}});assert.equal(errors.length,1);
field('stroke-miterlimit').onchange({target:{value:'7'}});assert.equal(edits.at(-1)['stroke-miterlimit'],'7');
items.get('a').resource=true;render();assert.equal(field('stroke-shape-controls').hidden,true);
// Live endpoint triangles move with nodes and the path's transform.
const attrs={'data-vectrify-arrow-start':'arrow','data-vectrify-arrow-end':'arrow',transform:'translate(10 20)'};
const element={dataset:{objectId:'a'},hasAttribute:k=>k in attrs,getAttribute:k=>attrs[k]??null};
const arrows=[0,1].map(()=>({attrs:{},getAttribute:k=>k==='data-vectrify-arrow-owner'?'a':null,setAttribute(k,v){this.attrs[k]=v;},removeAttribute(k){delete this.attrs[k];}}));
const g={subpaths:[{closed:false,nodes:[{values:[20,50]},{values:[20,50,80,80,80,80]}]}]};
const preview=Object.assign(createEditorState(), {svgElement:()=>element,drawing:{querySelectorAll:()=>arrows},geometries:new Map([['a',g]]),getComputedStyle:()=>({strokeWidth:'2'})});
const preview_arrows = createArrowsController(preview, {
  drawing: preview.drawing,
  getComputedStyle: preview.getComputedStyle,
  svgElement: (...args) => preview.svgElement(...args),
});
preview.syncArrowPreview = preview_arrows.syncArrowPreview;
(preview.syncArrowPreview(preview.svgElement()));assert.ok(arrows[1].attrs.d.startsWith('M80 80'));assert.equal(arrows[0].attrs.transform,'translate(10 20)');
g.subpaths[0].nodes[1].values=[20,50,90,30,90,30];delete attrs.transform;
(preview.syncArrowPreview(preview.svgElement()));assert.ok(arrows[1].attrs.d.startsWith('M90 30'));assert.equal(arrows[0].attrs.transform,undefined);
// Taper fields share paint edits, mixed values and property locks.
items.get('a').resource=false;items.get('a').open_path=true;state.selection.objects=['a'];render();
field('start-taper').onchange({target:{value:'12.5'}});assert.equal(edits.at(-1)['data-vectrify-taper-start'],'12.5');
field('end-taper').onchange({target:{value:'0'}});assert.equal(edits.at(-1)['data-vectrify-taper-end'],'0');
const errorCount=errors.length;field('start-taper').onchange({target:{value:'-1'}});assert.equal(errors.length,errorCount+1);
const editCount=edits.length;field('start-taper').onchange({target:{value:''}});assert.equal(edits.length,editCount);
items.get('a').attributes['data-vectrify-taper-start']='10';state.selection.objects=['a','b'];render();assert.equal(field('start-taper').value,'');
items.get('a').inherited_locks=['data-vectrify-taper-start'];render();assert.equal(field('start-taper').disabled,true);assert.equal(field('end-taper').disabled,false);
field('start-taper-width').onchange({target:{value:'50'}});assert.equal(edits.at(-1)['data-vectrify-taper-start-width'],'50');
field('end-taper-width').onchange({target:{value:'25'}});assert.equal(edits.at(-1)['data-vectrify-taper-end-width'],'25');
for (const value of ['-1','101','Infinity','bad']) {
  const count=edits.length;field('end-taper-width').onchange({target:{value}});assert.equal(edits.length,count);
}
state.selection.objects=['a'];items.get('a').attributes['data-vectrify-taper-end-width']='25';render();assert.equal(field('end-taper-width').value,'25');
state.selection.objects=['a','b'];render();assert.equal(field('end-taper-width').value,'');
items.get('a').inherited_locks=['data-vectrify-taper-end-width'];render();assert.equal(field('end-taper-width').disabled,true);assert.equal(field('start-taper-width').disabled,false);
state.selection.objects=['a'];items.get('a').inherited_locks=['paint'];render();assert.equal(field('start-taper-width').disabled,true);
const {taperPreview}=await import('../../src/vectrify/ui/static/taper.js');
const straight={subpaths:[{closed:false,nodes:[{values:[10,50]},{values:[90,50]}]}]};
const outline=taperPreview(straight,10,{start:20,end:20});
assert.ok(outline.includes('10 50'));assert.ok(outline.includes('90 50'));assert.ok(outline.includes('30 55'));assert.ok(outline.includes('70 55'));
const partial=taperPreview(straight,10,{start:20,end:30,startWidth:50,endWidth:20});
assert.ok(partial.includes('10 52.5'));assert.ok(partial.includes('90 51'));assert.ok(partial.includes('30 55'));assert.ok(partial.includes('60 55'));
const capped=taperPreview(straight,10,{start:20,end:20,startWidth:50,endWidth:20,cap:'square'});
assert.ok(capped.includes('7.5 52.5'));assert.ok(capped.includes('91 51'));
attrs['data-vectrify-taper-start']='20';
const taperedCompanion={attrs:{},getAttribute:k=>k==='data-vectrify-taper-owner'?'a':null,setAttribute(k,v){this.attrs[k]=v;},removeAttribute(k){delete this.attrs[k];}};
preview.drawing.querySelectorAll=()=>[taperedCompanion];
preview.syncArrowPreview(element);assert.ok(taperedCompanion.attrs.d.includes('90 30'));
attrs.transform='translate(5 6)';preview.syncArrowPreview(element);assert.equal(taperedCompanion.attrs.transform,'translate(5 6)');
attrs['data-vectrify-taper-start-width']='50';preview.syncArrowPreview(element);
assert.ok(taperedCompanion.attrs.d.startsWith(taperPreview(g,2,{start:'20',startWidth:'50'})+' '));
