"""Dependency-free browser viewer for the local pose WebSocket."""
from __future__ import annotations

import threading
import gzip
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


_SMPL_MODEL_GZIP = None
_SMPL_MODEL_LOCK = threading.Lock()


def _get_smpl_model_gzip():
    """Build the browser skinning asset once from the local licensed model."""
    global _SMPL_MODEL_GZIP
    with _SMPL_MODEL_LOCK:
        if _SMPL_MODEL_GZIP is not None:
            return _SMPL_MODEL_GZIP

        import numpy as np
        from mobileposer.articulate.model import ParametricModel
        from production.paths import paths

        if not paths.smpl_file.exists():
            return None
        model = ParametricModel(str(paths.smpl_file))
        joints, vertices = model.get_zero_pose_joint_and_vertex()
        joints = joints.cpu().numpy().astype(np.float32)
        vertices = vertices.cpu().numpy().astype(np.float32)
        faces = np.asarray(model.face, dtype=np.int32)

        # The realtime quaternions have already been reflected into Unity's
        # left-handed basis. Reflect the rest mesh too and repair winding.
        joints[:, 0] *= -1
        vertices[:, 0] *= -1
        faces = faces[:, [0, 2, 1]]

        normals = np.zeros_like(vertices)
        triangle_normals = np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        )
        for corner in range(3):
            np.add.at(normals, faces[:, corner], triangle_normals)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)

        weights = model._skinning_weights.cpu().numpy().astype(np.float32)
        bone_ids = np.argpartition(weights, -4, axis=1)[:, -4:]
        skin_weights = np.take_along_axis(weights, bone_ids, axis=1)
        skin_weights /= np.maximum(skin_weights.sum(axis=1, keepdims=True), 1e-8)
        parents = [-1 if parent is None else int(parent) for parent in model.parent]
        payload = {
            "vertices": vertices.reshape(-1).round(6).tolist(),
            "normals": normals.reshape(-1).round(6).tolist(),
            "faces": faces.reshape(-1).tolist(),
            "boneIds": bone_ids.astype(np.uint8).reshape(-1).tolist(),
            "weights": skin_weights.reshape(-1).round(6).tolist(),
            "joints": joints.reshape(-1).round(6).tolist(),
            "parents": parents,
        }
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        _SMPL_MODEL_GZIP = gzip.compress(raw, compresslevel=6)
        return _SMPL_MODEL_GZIP

_HTML = r'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MotionRecover production viewer</title>
<style>*{box-sizing:border-box}body{margin:0;background:#101318;color:#e9edf2;font:14px system-ui,sans-serif;overflow:hidden}header{height:52px;padding:12px 18px;border-bottom:1px solid #2b313b;display:flex;gap:24px;align-items:center}h1{font-size:16px;margin:0}.ok{color:#8bd5a4}.bad{color:#f28b82}main{height:calc(100vh - 52px);display:grid;grid-template-columns:minmax(0,1fr) 280px}.views{position:relative;min-width:0;min-height:0}.views canvas{position:absolute;inset:0;width:100%;height:100%;display:block;background:#171b22}.views #view3d,.views #meshView{display:none;cursor:grab}.views .dragging{cursor:grabbing!important}aside{border-left:1px solid #2b313b;padding:16px;line-height:1.8;overflow:auto}dt{color:#8d98a8;font-size:12px}dd{margin:0 0 8px}button{background:#252c37;color:#e9edf2;border:1px solid #3a4553;border-radius:4px;padding:6px 9px;cursor:pointer;margin:2px}button.active{background:#315d82;border-color:#70b7ff}.hint{color:#8d98a8;font-size:12px;line-height:1.4}input,select{width:100%;background:#171b22;color:#e9edf2;border:1px solid #3a4553;padding:6px;border-radius:3px}pre{white-space:pre-wrap;font-size:11px;line-height:1.35;color:#b8c3d1}@media(max-width:700px){main{grid-template-columns:1fr}aside{position:absolute;right:0;top:52px;width:220px;background:#101318ee}}</style>
<header><h1>MotionRecover production viewer</h1><span id="status" class="bad">connecting...</span><span id="recordState"></span></header><main><div class="views"><canvas id="view"></canvas><canvas id="view3d"></canvas><canvas id="meshView"></canvas></div><aside><dl><dt>Frame</dt><dd id="frame">-</dd><dt>Layout</dt><dd id="layout">-</dd><dt>FPS</dt><dd id="fps">-</dd><dt>Inference model</dt><dd><select id="modelSelect">__MODEL_OPTIONS__</select><div id="modelStatus" class="hint">Waiting for model state</div></dd><dt>Model lag</dt><dd id="lag">-</dd><dt>Foot contact</dt><dd id="contact">-</dd><dt>Recorded frames</dt><dd id="recorded">0</dd></dl><button id="front">Front</button> <button id="side">Side</button> <button id="threeD">3D joints</button> <button id="surface">SMPL surface</button><div class="hint">3D views use the same 24 local rotations sent to Unity. Drag to orbit, wheel to zoom.</div><hr><button id="record">Start recording</button> <button id="stop">Stop</button><button id="replay">Replay</button><button id="clear">Clear</button><button id="download">Download JSONL</button></aside></main>
<script>
const c=document.getElementById('view'),x=c.getContext('2d'),c3=document.getElementById('view3d'),x3=c3.getContext('2d'),cm=document.getElementById('meshView'),gl=cm.getContext('webgl',{antialias:true}),statusEl=document.getElementById('status'),fpsEl=document.getElementById('fps'),modelSelect=document.getElementById('modelSelect'),modelStatus=document.getElementById('modelStatus');let mode='front',last=0,n=0,frame=null,posePoints=null,comparisonPoints=null,yaw=-.45,pitch=.12,zoom=1,drag=null,mesh=null,poseSocket=null,activeModel=null,pendingModel=null,pendingModelTimer=null;const par=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21],off=[[0,0,0],[-.18,-.08,0],[.18,-.08,0],[0,.22,0],[0,-.42,0],[0,-.42,0],[0,.24,0],[0,-.42,0],[0,-.42,0],[0,.22,0],[0,-.13,.08],[0,-.13,.08],[0,.20,0],[-.12,.12,0],[.12,.12,0],[0,.25,0],[-.2,0,0],[.2,0,0],[-.3,0,0],[.3,0,0],[-.24,0,0],[.24,0,0],[-.12,0,.03],[.12,0,.03]];
function qm(a,b){return[a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]]}function qr(q,v){let r=qm(qm(q,[...v,0]),[-q[0],-q[1],-q[2],q[3]]);return r.slice(0,3)}
const BODY_SPAN=2.0;// metres shown across the view; pelvis-centred, constant so the camera never refits per frame
function sizeCanvas(canvas,ctx){let w=canvas.clientWidth,h=canvas.clientHeight,d=devicePixelRatio||1;if(canvas.width!==w*d||canvas.height!==h*d){canvas.width=w*d;canvas.height=h*d}ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,w,h);return[w,h]}
function poseFromMessage(m){let q=Array.from({length:24},(_,i)=>m.joints.slice(4*i,4*i+4)),g=[],p=[];for(let i=0;i<24;i++){let a=par[i];g[i]=a<0?q[i]:qm(g[a],q[i]);p[i]=a<0?[0,0,0]:p[a].map((v,j)=>v+qr(g[a],off[i])[j])}return p}
function draw2d(p,second){let[w,h]=sizeCanvas(c,x),sets=second?[p,second]:[p],half=w/sets.length;sets.forEach((pts,si)=>{let o=pts[0],s=Math.min(half,h)*.9/BODY_SPAN,xy=v=>[si*half+half/2+(mode==='front'?v[0]-o[0]:v[2]-o[2])*s,h*.5-(v[1]-o[1])*s];x.save();x.beginPath();x.rect(si*half,0,half,h);x.clip();x.strokeStyle=si===0?'#70b7ff':'#ff8a65';x.lineWidth=5;x.lineCap='round';for(let i=1;i<24;i++){let a=xy(pts[par[i]]),b=xy(pts[i]);x.beginPath();x.moveTo(...a);x.lineTo(...b);x.stroke()}x.fillStyle=si===0?'#f2c14e':'#ffd1c2';pts.forEach((v,i)=>{let a=xy(v);x.beginPath();x.arc(...a,i?5:7,0,Math.PI*2);x.fill()});x.fillStyle=si===0?'#70b7ff':'#ff8a65';x.font='bold 14px system-ui';x.fillText(si===0?'Python heading':'Server adjusted',si*half+14,24);x.restore()});if(second){x.strokeStyle='#ffffff55';x.beginPath();x.moveTo(w/2,0);x.lineTo(w/2,h);x.stroke()}}
function viewPoint(v){let cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),a=[cy*v[0]+sy*v[2],v[1],-sy*v[0]+cy*v[2]];return[a[0],cp*a[1]-sp*a[2],sp*a[1]+cp*a[2]]}
function draw3d(){if(!posePoints)return;let[w,h]=sizeCanvas(c3,x3),v=posePoints.map(viewPoint),o=v[0],s=Math.min(w*.84,h*.9)/BODY_SPAN*zoom,project=a=>[w/2+(a[0]-o[0])*s,h*.5-(a[1]-o[1])*s,a[2]];let pp=v.map(project),fl=h*.5+.95*s,bones=Array.from({length:23},(_,k)=>k+1).sort((a,b)=>(pp[a][2]+pp[par[a]][2])-(pp[b][2]+pp[par[b]][2]));x3.fillStyle='#171b22';x3.fillRect(0,0,w,h);x3.strokeStyle='#303844';x3.lineWidth=1;for(let i=-5;i<=5;i++){x3.beginPath();x3.moveTo(w/2+i*36,fl+4);x3.lineTo(w/2+i*54,fl+.1*h);x3.stroke()}x3.lineCap='round';for(let i of bones){let a=pp[par[i]],b=pp[i],depth=Math.max(0,Math.min(1,(a[2]+b[2]+1.5)/3));x3.strokeStyle=`rgb(${80+Math.round(30*depth)},${145+Math.round(38*depth)},${205+Math.round(45*depth)})`;x3.lineWidth=Math.max(4,10+3*depth);x3.beginPath();x3.moveTo(a[0],a[1]);x3.lineTo(b[0],b[1]);x3.stroke()}let joints=pp.map((a,i)=>[...a,i]).sort((a,b)=>a[2]-b[2]);for(let a of joints){x3.fillStyle=a[3]===0?'#f2c14e':'#d8e6f3';x3.beginPath();x3.arc(a[0],a[1],a[3]===0?7:4.5,0,Math.PI*2);x3.fill()}x3.fillStyle='#8d98a8';x3.font='12px system-ui';x3.fillText('SMPL 24-joint diagnostic (not a surface mesh)',14,22)}
function shader(type,source){let s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s}
function buffer(data,size,attribute,integer=false){let b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);let loc=gl.getAttribLocation(mesh.program,attribute);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,size,integer?gl.UNSIGNED_BYTE:gl.FLOAT,false,0,0);return b}
async function loadMesh(){if(mesh)return;let button=document.getElementById('surface');button.disabled=true;button.textContent='Loading SMPL...';try{let data=await(await fetch('/api/smpl-model')).json();if(!gl)throw Error('WebGL is unavailable');let cases=Array.from({length:23},(_,i)=>`if(i<${i+.5})return uBones[${i}];`).join(''),vertex=`attribute vec3 aPosition;attribute vec3 aNormal;attribute vec4 aBone;attribute vec4 aWeight;uniform mat4 uBones[24];uniform mat4 uView;varying float vLight;mat4 bone(float i){${cases}return uBones[23];}void main(){mat4 skin=aWeight.x*bone(aBone.x)+aWeight.y*bone(aBone.y)+aWeight.z*bone(aBone.z)+aWeight.w*bone(aBone.w);vec4 world=skin*vec4(aPosition,1.0);vec3 n=normalize(mat3(skin)*aNormal);vLight=.35+.65*max(dot(n,normalize(vec3(-.3,.8,.5))),0.0);gl_Position=uView*world;}`,vs=shader(gl.VERTEX_SHADER,vertex),fs=shader(gl.FRAGMENT_SHADER,'precision mediump float;varying float vLight;void main(){gl_FragColor=vec4(vec3(.36,.66,.86)*vLight,1.0);}'),program=gl.createProgram();gl.attachShader(program,vs);gl.attachShader(program,fs);gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));mesh={program,joints:data.joints,parents:data.parents,count:data.faces.length};gl.useProgram(program);buffer(new Float32Array(data.vertices),3,'aPosition');buffer(new Float32Array(data.normals),3,'aNormal');buffer(new Uint8Array(data.boneIds),4,'aBone',true);buffer(new Float32Array(data.weights),4,'aWeight');mesh.index=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,mesh.index);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(data.faces),gl.STATIC_DRAW);gl.enable(gl.DEPTH_TEST);gl.enable(gl.CULL_FACE);button.textContent='SMPL surface'}catch(e){button.textContent='SMPL unavailable';button.title=String(e);throw e}finally{button.disabled=false}}
function quatMatrix(q,t){let[x,y,z,w]=q,x2=x+x,y2=y+y,z2=z+z,xx=x*x2,xy=x*y2,xz=x*z2,yy=y*y2,yz=y*z2,zz=z*z2,wx=w*x2,wy=w*y2,wz=w*z2;return[1-(yy+zz),xy+wz,xz-wy,0,xy-wz,1-(xx+zz),yz+wx,0,xz+wy,yz-wx,1-(xx+yy),0,t[0],t[1],t[2],1]}
function drawMesh(){if(!mesh||!frame)return;let d=devicePixelRatio||1,w=cm.clientWidth,h=cm.clientHeight;if(cm.width!==w*d||cm.height!==h*d){cm.width=w*d;cm.height=h*d}gl.viewport(0,0,cm.width,cm.height);gl.clearColor(.09,.11,.14,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);let q=Array.from({length:24},(_,i)=>frame.joints.slice(4*i,4*i+4)),g=[],p=[],bones=new Float32Array(24*16);for(let i=0;i<24;i++){let a=mesh.parents[i],j=mesh.joints.slice(i*3,i*3+3);if(a<0){g[i]=q[i];p[i]=[0,0,0]}else{g[i]=qm(g[a],q[i]);let ja=mesh.joints.slice(a*3,a*3+3),bone=j.map((v,k)=>v-ja[k]),r=qr(g[a],bone);p[i]=p[a].map((v,k)=>v+r[k])}let rj=qr(g[i],j),t=p[i].map((v,k)=>v-rj[k]);bones.set(quatMatrix(g[i],t),i*16)}let cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),s=1.05*zoom,sx=s*h/Math.max(w,1),view=new Float32Array([cy*sx,sp*sy*s,-.2*cp*sy,0,0,cp*s,.2*sp,0,sy*sx,-sp*cy*s,.2*cp*cy,0,0,.10,0,1]);gl.useProgram(mesh.program);gl.uniformMatrix4fv(gl.getUniformLocation(mesh.program,'uBones[0]'),false,bones);gl.uniformMatrix4fv(gl.getUniformLocation(mesh.program,'uView'),false,view);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,mesh.index);gl.drawElements(gl.TRIANGLES,mesh.count,gl.UNSIGNED_SHORT,0)}
function syncModel(model){if(!model||![...modelSelect.options].some(option=>option.value===model))return;activeModel=model;if(pendingModel===model){pendingModel=null;clearTimeout(pendingModelTimer);pendingModelTimer=null}if(!pendingModel){modelSelect.value=model;modelStatus.textContent='Active: '+modelSelect.selectedOptions[0].textContent}}
function draw(m){frame=m;syncModel(m.model);posePoints=poseFromMessage(m.comparison?{joints:m.comparison.python}:m);comparisonPoints=m.comparison?poseFromMessage({joints:m.comparison.server}):null;if(mode==='3d')draw3d();else if(mode==='surface')drawMesh();else draw2d(posePoints,comparisonPoints);document.getElementById('frame').textContent=m.frame;document.getElementById('layout').textContent=m.layout||'-';document.getElementById('lag').textContent=(m.modelLagFrames??'-')+' frames';document.getElementById('contact').textContent=m.footContact?m.footContact.map(v=>Number(v).toFixed(2)).join(' / '):(m.footContactEnabled?'waiting':'disabled')}
async function setView(next){if(next==='surface')await loadMesh();mode=next;c.style.display=['front','side'].includes(next)?'block':'none';c3.style.display=next==='3d'?'block':'none';cm.style.display=next==='surface'?'block':'none';for(let id of['front','side','threeD','surface'])document.getElementById(id).classList.toggle('active',(id==='threeD'?'3d':id)===next);if(frame)draw(frame)}
function bindOrbit(canvas){canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY,yaw,pitch];canvas.setPointerCapture(e.pointerId);canvas.classList.add('dragging')};canvas.onpointermove=e=>{if(!drag)return;yaw=drag[2]+(e.clientX-drag[0])*.01;pitch=Math.max(-1.2,Math.min(1.2,drag[3]+(e.clientY-drag[1])*.01));if(mode==='surface')drawMesh();else draw3d()};canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;canvas.classList.remove('dragging')};canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.55,Math.min(2.4,zoom*Math.exp(-e.deltaY*.001)));if(mode==='surface')drawMesh();else draw3d()}}
let recording=false,recorded=[],replayTimer=null;const recordState=document.getElementById('recordState'),recordedEl=document.getElementById('recorded');function setRecording(v){recording=v;recordState.textContent=v?' recording':' ';recordState.className=v?'ok':'';document.getElementById('record').disabled=v}function stopReplay(){if(replayTimer){clearInterval(replayTimer);replayTimer=null}}function connect(){let ws=new WebSocket('ws://'+location.hostname+':__WS_PORT__');poseSocket=ws;ws.onopen=()=>{statusEl.textContent='connected';statusEl.className='ok'};ws.onclose=()=>{if(poseSocket===ws)poseSocket=null;statusEl.textContent='disconnected';statusEl.className='bad';setTimeout(connect,1000)};ws.onerror=()=>ws.close();ws.onmessage=e=>{let t=performance.now();n++;if(t-last>1000){fpsEl.textContent=(n*1000/(t-last)).toFixed(1);n=0;last=t}let m=JSON.parse(e.data);if(recording)recorded.push(m);recordedEl.textContent=recorded.length;draw(m)}};modelSelect.onchange=()=>{if(!poseSocket||poseSocket.readyState!==WebSocket.OPEN){modelStatus.textContent='Not connected to the pose stream';if(activeModel)modelSelect.value=activeModel;return}pendingModel=modelSelect.value;modelStatus.textContent='Loading: '+modelSelect.selectedOptions[0].textContent;clearTimeout(pendingModelTimer);poseSocket.send(JSON.stringify({selectModel:pendingModel}));pendingModelTimer=setTimeout(()=>{pendingModel=null;if(activeModel)modelSelect.value=activeModel;modelStatus.textContent='Switch timed out; check the Python console'},30000)};document.getElementById('front').onclick=()=>setView('front');document.getElementById('side').onclick=()=>setView('side');document.getElementById('threeD').onclick=()=>setView('3d');document.getElementById('surface').onclick=()=>setView('surface').catch(e=>console.error(e));document.getElementById('record').onclick=()=>{stopReplay();recorded=[];recordedEl.textContent='0';setRecording(true)};document.getElementById('stop').onclick=()=>setRecording(false);document.getElementById('clear').onclick=()=>{stopReplay();setRecording(false);recorded=[];recordedEl.textContent='0'};document.getElementById('replay').onclick=()=>{if(!recorded.length)return;stopReplay();setRecording(false);let i=0;replayTimer=setInterval(()=>{if(i>=recorded.length){stopReplay();return}draw(recorded[i++])},1000/30)};document.getElementById('download').onclick=()=>{if(!recorded.length)return;let body=recorded.map(v=>JSON.stringify(v)).join('\n')+'\n',a=document.createElement('a');a.href=URL.createObjectURL(new Blob([body],{type:'application/jsonl'}));a.download='mobilepose-pose-'+new Date().toISOString().replace(/[:.]/g,'-')+'.jsonl';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};bindOrbit(c3);bindOrbit(cm);onresize=()=>frame&&draw(frame);setView('front');connect();
</script>'''


def _render_html(websocket_port: int, model_options):
    options = "".join(
        f'<option value="{html.escape(model_id, quote=True)}">{html.escape(label)}</option>'
        for model_id, label in model_options
    )
    return _HTML.replace("__WS_PORT__", str(websocket_port)).replace("__MODEL_OPTIONS__", options)

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if urlparse(self.path).path == "/api/smpl-model":
            body = _get_smpl_model_gzip()
            if body is None:
                self.send_error(404, "SMPL model is not installed")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Cache-Control", "public, max-age=3600")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = _render_html(self.server.ws_port, self.server.model_options).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        self.send_error(404)

    def log_message(self, *_args):
        pass

def start_web_visualizer(host: str, port: int, websocket_port: int, model_options=()):
    server = ThreadingHTTPServer((host, port), _Handler)
    server.ws_port = websocket_port
    server.model_options = tuple(model_options)
    threading.Thread(target=server.serve_forever, name="mobilepose-web", daemon=True).start()
    return server
