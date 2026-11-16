"""Write a standalone WebGL SMPL surface replay for a Unity JSONL recording."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from production.input_recording import resolve_recording_path
from production.web_visualizer import _get_smpl_model_gzip


def recorded_unity_quaternions(rows, n_frames):
    """Use the exact Unity-local quaternions saved by the broadcaster.

    The browser mesh is already reflected into Unity's basis. Feeding it the
    SMPL matrices used by offline analysis would apply the basis conversion a
    second time and makes arms/legs appear mirrored or folded.
    """
    predictions = []
    for row in rows:
        values = np.asarray(row.get("unityLocalQuaternions", []), dtype=np.float32)
        if values.size == 24 * 4:
            predictions.append((float(row.get("t", 0.0)), values.reshape(24, 4)))
    if not predictions:
        return None
    out = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (n_frames, 24, 1))
    filled = np.zeros(n_frames, dtype=bool)
    for t, values in predictions:
        index = int(np.clip(round(t * 60), 0, n_frames - 1))
        out[index] = values
        filled[index] = True
    for index in range(1, n_frames):
        if not filled[index]:
            out[index] = out[index - 1]
    return out


def _leg_label(diag):
    legs = (diag or {}).get("deadbugLegs") or {}
    if not legs.get("enabled"):
        return ""
    parts = []
    for side, name in (("left", "左腿"), ("right", "右腿")):
        leg = legs.get(side) or {}
        phi = leg.get("phi")
        parts.append(f"{name} {leg.get('phase', '-')} · 置信 {leg.get('confidence', 0):.2f} · 施加 {leg.get('applied', 0):.2f}"
                     + ("" if phi is None else f" · 大腿 {phi:.0f}°") + f"<br><span>{leg.get('reason', '')}</span>")
    return "<br>".join(parts)


def recorded_labels(rows, n_frames):
    """Per-frame dead-bug leg status from the saved prediction diagnostics (same timeline as the poses)."""
    labels = [""] * n_frames
    for row in rows:
        if row.get("type") == "prediction":
            index = int(min(max(round(float(row.get("t", 0.0)) * 60), 0), n_frames - 1))
            labels[index] = _leg_label(row.get("diagnostics"))
    for i in range(1, n_frames):
        if not labels[i]:
            labels[i] = labels[i - 1]
    return labels


def deadbug_fixed_quaternions(recording, rows, frames, leg_mode="auto", model_id=None):
    """Re-run the recording through the current realtime chain (production.deadbug_replay)."""
    from production.deadbug_replay import per_frame_unity_quaternions, replay
    result = replay(recording, model_id=model_id, leg_mode=leg_mode)
    quats = per_frame_unity_quaternions(result)
    labels = [""] * result.n_frames
    for frame, diag in zip(result.represented_frames, result.diagnostics):
        if 0 <= frame < result.n_frames:
            labels[frame] = _leg_label(diag)
    for i in range(1, result.n_frames):
        if not labels[i]:
            labels[i] = labels[i - 1]
    return quats[:frames], labels[:frames]


def write_mesh_replay(recording: Path, output: Path, deadbug_fix: bool = False, baseline: bool = False,
                      model_id: str | None = None, precomputed=None, title_suffix: str | None = None) -> Path:
    rows = [json.loads(line) for line in recording.read_text(encoding="utf-8").splitlines() if line.strip()]
    ticks = [r for r in rows if r.get("type") == "tick"]
    frames = sum(int(r.get("ticks", 1)) for r in ticks)
    if precomputed is not None:
        q, labels = precomputed
    elif deadbug_fix or baseline:
        q, labels = deadbug_fixed_quaternions(recording, rows, frames, "off" if baseline else "auto", model_id)
    else:
        q, labels = recorded_unity_quaternions(rows, frames), recorded_labels(rows, frames)
    if q is None:
        raise ValueError("recording has no unityLocalQuaternions predictions")
    # Collapse repeated status strings to keep the page small.
    status, table = [], {}
    for text in labels:
        status.append(table.setdefault(text, len(table)))
    model = json.loads(gzip.decompress(_get_smpl_model_gzip()).decode("utf-8"))
    # Keep a manageable page while retaining the original 60 Hz timeline.
    title = recording.name + (" · 离线重放：未约束基线（网络腿部）" if baseline else
                              " · 当前处理链离线重放（死虫腿部运动证据估计）" if deadbug_fix else " · 实时输出")
    if title_suffix:
        title = recording.name + " · " + title_suffix
    data = {"title": title, "fps": 60, "model": model,
            "frames": np.round(q.reshape(len(q), -1), 5).tolist(),
            "status": status, "statusText": list(table)}
    output.write_text(_HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
    return output


_HTML = r'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SMPL surface replay</title><style>*{box-sizing:border-box}body{margin:0;background:#101318;color:#e9edf2;font:14px system-ui}header{padding:12px 16px;border-bottom:1px solid #2b313b;display:flex;gap:16px;align-items:center}h1{font-size:15px;margin:0}main{height:calc(100vh - 53px);display:grid;grid-template-columns:1fr 300px}@media(max-width:700px){main{grid-template-columns:1fr;grid-template-rows:60vh auto}aside{border-left:0;border-top:1px solid #2b313b}}canvas{width:100%;height:100%;display:block;background:#171b22;cursor:grab}.drag{cursor:grabbing}aside{border-left:1px solid #2b313b;padding:16px}button{background:#252c37;color:#e9edf2;border:1px solid #3a4553;border-radius:4px;padding:7px 12px;margin-right:5px}input{width:100%}.muted{color:#8d98a8;font-size:12px}.legs{margin-top:14px;font-size:12px;line-height:1.5}.legs span{color:#8d98a8}</style>
<header><h1 id="title"></h1><span class="muted">SMPL surface · 拖拽旋转 · 滚轮缩放</span></header><main><canvas id="c"></canvas><aside><button id="play">播放</button><button id="reset">重置视角</button><p><input id="seek" type="range" min="0" value="0"></p><div id="time" class="muted"></div><div id="legs" class="legs"></div></aside></main><script>
const D=__DATA__,M=D.model,c=document.getElementById('c'),g=c.getContext('webgl',{antialias:true}),seek=document.getElementById('seek'),time=document.getElementById('time');title.textContent=D.title+' · SMPL 网格回放';seek.max=D.frames.length-1;let i=0,run=false,yaw=-.4,pitch=.12,zoom=1,drag=null,last=0,acc=0;
function sh(t,s){let x=g.createShader(t);g.shaderSource(x,s);g.compileShader(x);return x}let cases=Array.from({length:23},(_,i)=>`if(i<${i+.5})return uB[${i}];`).join('');let p=g.createProgram();g.attachShader(p,sh(g.VERTEX_SHADER,`attribute vec3 aP,aN;attribute vec4 aI,aW;uniform mat4 uB[24],uV;varying float L;mat4 b(float i){${cases}return uB[23];}void main(){mat4 s=aW.x*b(aI.x)+aW.y*b(aI.y)+aW.z*b(aI.z)+aW.w*b(aI.w);vec3 n=normalize(mat3(s)*aN);L=.35+.65*max(dot(n,normalize(vec3(-.3,.8,.5))),0.);gl_Position=uV*s*vec4(aP,1.);}`));g.attachShader(p,sh(g.FRAGMENT_SHADER,'precision mediump float;varying float L;void main(){gl_FragColor=vec4(vec3(.36,.66,.86)*L,1.);}'));g.linkProgram(p);g.useProgram(p);function buf(a,n,k,integer){let b=g.createBuffer();g.bindBuffer(g.ARRAY_BUFFER,b);g.bufferData(g.ARRAY_BUFFER,a,g.STATIC_DRAW);let l=g.getAttribLocation(p,k);g.enableVertexAttribArray(l);g.vertexAttribPointer(l,n,integer?g.UNSIGNED_BYTE:g.FLOAT,false,0,0)}buf(new Float32Array(M.vertices),3,'aP');buf(new Float32Array(M.normals),3,'aN');buf(new Uint8Array(M.boneIds),4,'aI',1);buf(new Float32Array(M.weights),4,'aW');let ib=g.createBuffer();g.bindBuffer(g.ELEMENT_ARRAY_BUFFER,ib);g.bufferData(g.ELEMENT_ARRAY_BUFFER,new Uint16Array(M.faces),g.STATIC_DRAW);g.enable(g.DEPTH_TEST);g.enable(g.CULL_FACE);
function qm(a,b){return[a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]]}function rv(q,v){let r=qm(qm(q,[...v,0]),[-q[0],-q[1],-q[2],q[3]]);return r.slice(0,3)}function mat(q,t){let[x,y,z,w]=q,X=x+x,Y=y+y,Z=z+z,xx=x*X,xy=x*Y,xz=x*Z,yy=y*Y,yz=y*Z,zz=z*Z,wx=w*X,wy=w*Y,wz=w*Z;return[1-yy-zz,xy+wz,xz-wy,0,xy-wz,1-xx-zz,yz+wx,0,xz+wy,yz-wx,1-xx-yy,0,t[0],t[1],t[2],1]}
function draw(){let d=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*d;c.height=h*d;g.viewport(0,0,c.width,c.height);g.clearColor(.06,.07,.09,1);g.clear(g.COLOR_BUFFER_BIT|g.DEPTH_BUFFER_BIT);let q=Array.from({length:24},(_,j)=>D.frames[i].slice(j*4,j*4+4)),G=[],P=[],B=new Float32Array(384);for(let j=0;j<24;j++){let a=M.parents[j],x=M.joints.slice(j*3,j*3+3);if(a<0){G[j]=q[j];P[j]=[0,0,0]}else{G[j]=qm(G[a],q[j]);let pa=M.joints.slice(a*3,a*3+3),r=rv(G[a],x.map((v,k)=>v-pa[k]));P[j]=P[a].map((v,k)=>v+r[k])}let r=rv(G[j],x),t=P[j].map((v,k)=>v-r[k]);B.set(mat(G[j],t),j*16)}let cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),s=1.05*zoom,sx=s*h/Math.max(w,1);let V=new Float32Array([cy*sx,sp*sy*s,-.2*cp*sy,0,0,cp*s,.2*sp,0,sy*sx,-sp*cy*s,.2*cp*cy,0,0,.1,0,1]);g.uniformMatrix4fv(g.getUniformLocation(p,'uB[0]'),false,B);g.uniformMatrix4fv(g.getUniformLocation(p,'uV'),false,V);g.drawElements(g.TRIANGLES,M.faces.length,g.UNSIGNED_SHORT,0);seek.value=i;time.textContent=(i/60).toFixed(2)+' s / '+((D.frames.length-1)/60).toFixed(2)+' s';if(D.status)document.getElementById('legs').innerHTML=D.statusText[D.status[i]]||''}function tick(t){if(run){acc+=(t-last)/1000*60;while(acc>=1&&i<D.frames.length-1){i++;acc--}if(i>=D.frames.length-1)run=false;draw()}last=t;requestAnimationFrame(tick)}play.onclick=()=>{if(i>=D.frames.length-1)i=0;run=!run};reset.onclick=()=>{yaw=-.4;pitch=.12;zoom=1;draw()};seek.oninput=()=>{i=+seek.value;draw()};c.onpointerdown=e=>{drag=[e.clientX,e.clientY,yaw,pitch];c.setPointerCapture(e.pointerId);c.classList.add('drag')};c.onpointermove=e=>{if(drag){yaw=drag[2]+(e.clientX-drag[0])*.01;pitch=Math.max(-1.2,Math.min(1.2,drag[3]+(e.clientY-drag[1])*.01));draw()}};c.onpointerup=c.onpointercancel=()=>{drag=null;c.classList.remove('drag')};c.onwheel=e=>{e.preventDefault();zoom=Math.max(.55,Math.min(2.4,zoom*Math.exp(-e.deltaY*.001)));draw()};onresize=draw;draw();requestAnimationFrame(tick);
</script>'''


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--deadbug-fix", action="store_true",
                    help="re-run through the current realtime chain instead of showing the recorded output")
    ap.add_argument("--baseline", action="store_true",
                    help="re-run with the dead-bug leg estimate off (unconstrained network legs)")
    ap.add_argument("--model", default=None,
                    help="with --deadbug-fix/--baseline: model id to re-run with (default: the recorded model)")
    a = ap.parse_args()
    src = resolve_recording_path(a.recording)
    suffix = "_baseline.html" if a.baseline else "_deadbug_fix.html" if a.deadbug_fix else "_smpl.html"
    dst = a.out or src.with_name(src.stem + suffix)
    print(write_mesh_replay(src, dst, deadbug_fix=a.deadbug_fix, baseline=a.baseline, model_id=a.model))
