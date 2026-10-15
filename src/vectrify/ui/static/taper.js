// Filled stroke pieces for live previews; the server unions the final outline.
export function taperPreview(geometry, width, style) {
  const pieces = [], tolerance = Math.max(1e-5, Math.min(0.02, width * 0.005));
  const add = points => {
    if (points.length < 3) return;
    const signed = points.reduce((sum,p,i) => {const q=points[(i+1)%points.length]; return sum+p[0]*q[1]-q[0]*p[1];},0);
    if (signed < 0) points.reverse();
    pieces.push(`M${points.map(p=>p.join(' ')).join(' L')} Z`);
  };
  const distance = (a,b) => Math.hypot(b[0]-a[0],b[1]-a[1]);
  const offset = (p,n,r) => [p[0]+n[0]*r,p[1]+n[1]*r];
  const arc = (p,r,angle,sweep) => {
    const step=2*Math.acos(Math.max(-1,Math.min(1,1-tolerance/Math.max(r,tolerance))));
    const count=Math.min(4096,Math.max(1,Math.ceil(Math.abs(sweep)/Math.max(step,1e-6))));
    add([p,...Array.from({length:count+1},(_,i)=>[p[0]+r*Math.cos(angle+sweep*i/count),p[1]+r*Math.sin(angle+sweep*i/count)])]);
  };
  function flatten(a,b,c,d,depth=0) {
    const chord=p=>{const dx=d[0]-a[0],dy=d[1]-a[1],length=dx*dx+dy*dy;const t=length?Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/length)):0;return distance(p,[a[0]+t*dx,a[1]+t*dy]);};
    if (depth>=24 || Math.max(chord(b),chord(c))<=tolerance) return [d];
    const mid=(p,q)=>[(p[0]+q[0])/2,(p[1]+q[1])/2];
    const ab=mid(a,b),bc=mid(b,c),cd=mid(c,d),abc=mid(ab,bc),bcd=mid(bc,cd),m=mid(abc,bcd);
    return [...flatten(a,ab,abc,m,depth+1),...flatten(m,bcd,cd,d,depth+1)];
  }
  for (const sub of geometry.subpaths) {
    let points=[sub.nodes[0].values.slice(-2)];
    for (const node of sub.nodes.slice(1)) {
      const v=node.values;
      points.push(...(v.length===6?flatten(points.at(-1),v.slice(0,2),v.slice(2,4),v.slice(4,6)):[v.slice(-2)]));
    }
    if (sub.closed && distance(points[0],points.at(-1))>0) points.push(points[0]);
    points=points.filter((p,i)=>!i || distance(p,points[i-1])>0);
    if (points.length<2) continue;
    const distances=[0];for(let i=1;i<points.length;i++) distances.push(distances.at(-1)+distance(points[i-1],points[i]));
    const length=distances.at(-1);
    let start=sub.closed?0:Number(style.start||0),end=sub.closed?0:Number(style.end||0);
    const scale=Math.max(start,end,length),sum=start/scale+end/scale;
    if (sum>length/scale) {start=length*(start/scale)/sum;end=length*(end/scale)/sum;}
    const startWidth=Number(style.startWidth||0)/100,endWidth=Number(style.endWidth||0)/100;
    const radius=d=>width/2*Math.max(0,Math.min(1,start?startWidth+(1-startWidth)*d/start:1,end?endWidth+(1-endWidth)*(length-d)/end:1));
    const samples=[[points[0],0]];
    for(let i=0;i<points.length-1;i++) {
      const low=distances[i],high=distances[i+1],a=points[i],b=points[i+1];
      for(const d of [...new Set([start,length-end])].sort((a,b)=>a-b)) if(low<d && d<high) {const t=(d-low)/(high-low);samples.push([[a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])],d]);}
      samples.push([b,high]);
    }
    points=samples.map(s=>s[0]);const radii=samples.map(s=>radius(s[1])),directions=[];
    for(let i=0;i<points.length-1;i++) {
      const a=points[i],b=points[i+1],size=distance(a,b),u=[(b[0]-a[0])/size,(b[1]-a[1])/size],n=[-u[1],u[0]];
      directions.push(u);add([offset(a,n,radii[i]),offset(b,n,radii[i+1]),offset(b,n,-radii[i+1]),offset(a,n,-radii[i])]);
    }
    function join(p,r,u,v) {
      const cross=u[0]*v[1]-u[1]*v[0],dot=u[0]*v[0]+u[1]*v[1];
      if(Math.abs(cross)<1e-12) {if(dot<0 && style.join==='round') arc(p,r,0,2*Math.PI);return;}
      const side=cross>0?-1:1,n=[-u[1]*side,u[0]*side],m=[-v[1]*side,v[0]*side],a=offset(p,n,r),b=offset(p,m,r);
      if(style.join==='round') arc(p,r,Math.atan2(n[1],n[0]),Math.atan2(cross,dot));
      else if((style.join||'miter')==='miter' && 1+dot>1e-12) {
        const tip=[p[0]+r*(n[0]+m[0])/(1+dot),p[1]+r*(n[1]+m[1])/(1+dot)];
        add(distance(p,tip)<=r*Number(style.miter||4)?[p,a,tip,b]:[p,a,b]);
      } else add([p,a,b]);
    }
    for(let i=1;i<directions.length;i++) join(points[i],radii[i],directions[i-1],directions[i]);
    if(sub.closed) join(points[0],radii[0],directions.at(-1),directions[0]);
    else for(const [i,u] of [[0,directions[0]],[points.length-1,directions.at(-1)]]) {
      if(radii[i]===0) continue;
      const p=points[i],r=radii[i],outward=i?u:[-u[0],-u[1]],n=[-u[1],u[0]];
      if(style.cap==='round') arc(p,r,Math.atan2(outward[1],outward[0])-Math.PI/2,Math.PI);
      else if(style.cap==='square') {const q=offset(p,outward,r);add([offset(p,n,r),offset(q,n,r),offset(q,n,-r),offset(p,n,-r)]);}
    }
  }
  return pieces.join(' ');
}
