"""Run one Unity input recording through several models and write a side-by-side HTML viewer.

The recording keeps the raw tracker input and the calibration events, so every
model sees exactly what the live loop would have fed it: the same 60 Hz frames,
the arms-forward and start-pose calibrations re-applied at their recorded
moments, inference every 2 frames, and each output shown at the input time it
represents (its lag removed) so all models stay in sync.

Usage (from realUse/):
    python -m production.offline_compare BirdDog_RingSkinC_Recorded_20261101_175053   # all models
    python -m production.offline_compare --missing        # every recording without a viewer yet
    python -m production.offline_compare <recording> --models waist_streamposer_amass,waist_line14_amass \
        --no-calibration --anchor --out recordings/compare.html
run.py starts this automatically (low priority) after every saved Unity recording. Each run also
refreshes recordings/index.html, which lists all viewers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from mobileposer.articulate.model import ParametricModel
from production.paths import paths
from production.model_backends import (
    WAIST_MODEL_FILES, WAIST_MODEL_LABELS, apply_input_offsets, make_waist_session, resolve_waist_model,
    model_lag_frames,
)
from production.calibration import bone_orientation, identity_heading
from production.input_recording import load_input_recording, resolve_recording_path
from production.layout import ResolvedLayout
from production.pose_calibration import SMPL_PARENTS, PoseCalibrator
from production.arm_prior import straighten_reaching_arms

from production.input_recording import RECORDINGS_DIR
INFER_EVERY = 2
WARMUP_FRAMES = 12


def _lag(model_id: str) -> int:
    return model_lag_frames(model_id)


def _expand(recording, rows):
    """60 Hz frames (rows repeated by ticks) and the frame index of every calibration event."""
    accel_key = "reference_adjusted_accel_xyz" if recording.acceleration_source == "server" else "accel_xyz"
    quats, accels, row_times, row_starts = [], [], [], []
    for row, ticks in zip(recording.samples, recording.ticks):
        row_starts.append(len(quats))
        quats += [[row[label]["quat_xyzw"] for label in recording.labels]] * ticks
        accels += [[row[label][accel_key] for label in recording.labels]] * ticks
    row_times = np.asarray(recording.times)
    events = []
    for event in (r for r in rows if r.get("type") == "event"):
        if event["name"] not in ("standCalibrate", "armsForwardCalibrate", "poseCalibrate",
                                  "birdDogBaseCalibrate", "birdDogLeftReachCalibrate",
                                  "birdDogRightReachCalibrate", "birdDogStartCalibrate", "poseHoldCalibrate"):
            continue
        index = int(np.searchsorted(row_times, event["t"]))
        frame = row_starts[min(index, len(row_starts) - 1)]
        events.append((frame, event))
    return np.asarray(quats, np.float32), np.asarray(accels, np.float32), events


def calibrated_orientations(recording, quats, events, enabled=True):
    """Raw bone orientations corrected exactly as the live PoseCalibrator would, plus its messages."""
    calibrator = PoseCalibrator(recording.labels)
    calibrator.calibration_enabled = enabled
    pending = {frame: event for frame, event in events}
    out, messages = [], []
    for t, q in enumerate(quats):
        out.append(calibrator.correct(bone_orientation(q, recording.labels)).numpy())
        event = pending.get(t)
        if event is not None and enabled:
            if event["name"] == "standCalibrate":
                calibrator.request_stand()
            elif event["name"] == "armsForwardCalibrate":
                calibrator.request_arms_forward()
            elif event["name"] == "birdDogStartCalibrate":
                calibrator.start_bird_dog_sequence()
            elif event["name"] == "birdDogBaseCalibrate":
                calibrator.request_bird_dog_base()
            elif event["name"] == "birdDogLeftReachCalibrate":
                calibrator.request_bird_dog_reach(left_side=True)
            elif event["name"] == "birdDogRightReachCalibrate":
                calibrator.request_bird_dog_reach(left_side=False)
            elif event["name"] == "poseHoldCalibrate":
                calibrator.request_pose_hold(event["pose"], event.get("seconds"))
            else:
                calibrator.request(event["targets"])
        message = calibrator.advance(1)
        if message:
            messages.append((t / 60.0, message))
    return np.asarray(out, np.float32), calibrator, messages


def run_model(model_id, recording, quats, accels, ori, anchor=False):
    """Local SMPL poses (T, 24, 3, 3) indexed by the input frame they represent."""
    resolved = ResolvedLayout(recording.layout, recording.labels, dict(recording.label_to_key))
    session = make_waist_session(model_id, resolved, identity_heading(recording.labels),
                                 loaded=resolve_waist_model(model_id))
    if recording.upright is not None:
        session.prime(*recording.upright)
    lag = _lag(model_id)
    ori = np.stack([apply_input_offsets(model_id, recording.labels, torch.from_numpy(o)).numpy() for o in ori])
    identity = np.tile(np.eye(3, dtype=np.float32), (24, 1, 1))
    poses = np.repeat(identity[None], len(quats), axis=0)
    last = identity
    for t in range(len(quats)):
        feature, _, accel = session.prepare_feature(quats[t], accels[t])
        feature = feature.copy()
        feature[-ori[t].size:] = ori[t].reshape(-1)
        orientation = torch.from_numpy(ori[t])
        if hasattr(session, "push_observation"):
            session.push_observation(feature, orientation, accel)
        else:
            session.push_feature(feature)
        if t >= WARMUP_FRAMES and t % INFER_EVERY == 1:
            last = session.infer().detach().numpy().astype(np.float32)
            if anchor and t - lag >= 0:
                last = _anchor(last, ori[t - lag], recording.labels)
        if t - lag >= 0:
            poses[t - lag] = last
    return poses


def _anchor(local, measured, labels):
    from production.pose_calibration import LABEL_JOINT
    overrides = {LABEL_JOINT[label]: measured[s] for s, label in enumerate(labels) if label in LABEL_JOINT}
    out, globals_ = local.copy(), []
    for joint, parent in enumerate(SMPL_PARENTS):
        parent_rot = globals_[parent] if parent >= 0 else np.eye(3, dtype=np.float32)
        if joint in overrides:
            out[joint] = parent_rot.T @ overrides[joint]
            globals_.append(overrides[joint])
        else:
            globals_.append(parent_rot @ local[joint])
    return out


def coach_poses(recording, rows, ori, events, n_frames):
    """Coach SMPL local poses (T, 24, 3, 3) on the input timeline, turned to the user's facing, or None.

    Unity stores the coach as SMPL-convention global rotations in its world; the coach faces another
    way than the user, so one yaw (waist vs coach spine1 at the calibration moment) aligns them.
    This is the reference the user follows, not the user's ground truth."""
    from production.pose_calibration import _best_yaw, _yaw, unity_quat_to_smpl
    coach_rows = [r for r in rows if r.get("type") == "coach"]
    if not coach_rows:
        return None
    times = np.asarray(recording.times)
    starts = np.cumsum([0] + list(recording.ticks))[:-1]
    frames = [int(starts[min(int(np.searchsorted(times, r["t"])), len(starts) - 1)]) for r in coach_rows]
    globals_ = np.stack([[unity_quat_to_smpl(q) for q in r["q"]] for r in coach_rows]).astype(np.float64)
    anchor = next((f for f, e in events if e["name"] in
                   ("standCalibrate", "poseCalibrate", "birdDogBaseCalibrate")), frames[0])
    k = int(np.argmin(np.abs(np.asarray(frames) - anchor)))
    waist = recording.labels.index("waist")
    yaw = _yaw(_best_yaw([ori[min(anchor, len(ori) - 1), waist].astype(np.float64) @ globals_[k, 3].T]))
    globals_ = np.einsum("ij,tkjl->tkil", yaw, globals_)
    local = globals_.copy()
    for joint, parent in enumerate(SMPL_PARENTS):
        if parent >= 0:
            local[:, joint] = np.swapaxes(globals_[:, parent], -1, -2) @ globals_[:, joint]
    out = np.repeat(np.eye(3)[None, None], n_frames, axis=0).repeat(24, axis=1)
    index = np.clip(np.searchsorted(frames, np.arange(n_frames)), 0, len(frames) - 1)   # nearest-after sample
    out[:] = local[index]
    out[: frames[0]] = np.eye(3)        # before the first coach sample: hide (identity)
    return out.astype(np.float32)


def joint_positions(body, poses, step):
    """Pelvis-centred joint positions (T', 24, 3) at 60/step Hz."""
    sampled = torch.from_numpy(poses[::step]).float()
    _, joints = body.forward_kinematics(sampled)[:2]
    return (joints - joints[:, :1]).numpy()


def write_viewer(path: Path, title: str, fps: float, tracks: dict, messages, events, live_model=None):
    data = {
        "title": title, "fps": fps, "parents": SMPL_PARENTS, "liveModel": live_model,
        "models": [{"id": model_id, "label": WAIST_MODEL_LABELS.get(model_id, model_id),
                    "frames": np.round(positions, 3).reshape(len(positions), -1).tolist()}
                   if model_id != "coach" else
                   {"id": "coach", "label": "教练（参考动作，非真值）",
                    "frames": np.round(positions, 3).reshape(len(positions), -1).tolist()}
                   for model_id, positions in tracks.items()],
        "messages": [[round(t, 2), m] for t, m in messages],
        "events": [[round(frame / 60.0, 2), event["name"]] for frame, event in events],
    }
    html = _VIEWER.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    path.write_text(html, encoding="utf-8")


def compare_recording(path: Path, models, calibration=True, anchor=False, fps=30.0, out=None, arm_prior=True) -> Path:
    recording = load_input_recording(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    quats, accels, events = _expand(recording, rows)
    ori, _, messages = calibrated_orientations(recording, quats, events, enabled=calibration)
    for t, message in messages:
        print(f"  t={t:5.1f}s {message}", file=sys.stderr)
    body = ParametricModel(paths.smpl_file)
    step = max(1, int(round(60 / fps)))
    tracks = {}
    coach = coach_poses(recording, rows, ori, events, len(quats))
    if coach is not None:
        tracks["coach"] = joint_positions(body, coach, step)
    for model_id in models:
        print(f"running {model_id} ...", file=sys.stderr, flush=True)
        poses = run_model(model_id, recording, quats, accels, ori, anchor=anchor)
        if arm_prior:      # same post-processing as the live loop; poses[t] represents input frame t
            lw, rw = recording.labels.index("lw"), recording.labels.index("rw")
            poses = np.stack([straighten_reaching_arms(p, (ori[t, lw], ori[t, rw]))
                              for t, p in enumerate(poses)]).astype(np.float32)
        tracks[model_id] = joint_positions(body, poses, step)
    options = (("raw input" if not calibration else "calibrated") + (", anchored" if anchor else "")
               + (", straight-arm prior" if arm_prior else ""))
    out = out or path.with_name(path.stem + "_compare.html")
    write_viewer(out, f"{path.name} — {options}", 60.0 / step, tracks, messages, events,
                 live_model=rows[0].get("model"))
    return out


def write_index(directory: Path = RECORDINGS_DIR) -> Path:
    """recordings/index.html linking every comparison viewer, newest first."""
    items = []
    for viewer in sorted(directory.glob("*_compare.html"), key=lambda p: p.stat().st_mtime, reverse=True):
        recording = viewer.with_name(viewer.name[:-len("_compare.html")] + ".jsonl")
        model = ""
        if recording.exists():
            try:
                model = json.loads(recording.open(encoding="utf-8").readline()).get("model") or ""
            except (OSError, ValueError):
                pass
        items.append((viewer.name, recording.stem, model))
    rows = "".join(
        f'<li><a href="{name}">{stem}</a><span>{WAIST_MODEL_LABELS.get(model, model) or "-"}</span></li>'
        for name, stem, model in items)
    index = directory / "index.html"
    index.write_text(_INDEX.replace("__ROWS__", rows or "<li>还没有对比页面</li>"), encoding="utf-8")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", nargs="?", help="file name under realUse/recordings, or a path")
    parser.add_argument("--missing", action="store_true", help="process every recording that has no viewer yet")
    parser.add_argument("--models", default="all", help="comma-separated model ids, or 'all'")
    parser.add_argument("--no-calibration", action="store_true", help="feed raw orientations (ignore recorded calibrations)")
    parser.add_argument("--anchor", action="store_true", help="replace sensed bones with the measurement")
    parser.add_argument("--no-arm-prior", action="store_true",
                        help="skip the straight-arm prior (always skipped for side-bridge recordings)")
    parser.add_argument("--fps", type=float, default=30.0, help="viewer frame rate (divides 60)")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None, help="HTML output (default: next to the recording)")
    args = parser.parse_args()
    if not args.recording and not args.missing:
        parser.error("give a recording or --missing")
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, args.torch_threads))
    models = list(WAIST_MODEL_FILES) if args.models == "all" else [m for m in args.models.split(",") if m]
    if args.missing:
        paths_ = [p for p in sorted(RECORDINGS_DIR.glob("*.jsonl"))
                  if not p.with_name(p.stem + "_compare.html").exists()]
    else:
        paths_ = [resolve_recording_path(args.recording)]
    for path in paths_:
        try:
            arm_prior = not args.no_arm_prior and "sidebridge" not in path.name.lower()
            out = compare_recording(path, models, not args.no_calibration, args.anchor, args.fps,
                                    args.out if len(paths_) == 1 else None, arm_prior=arm_prior)
            print(f"viewer: {out}", file=sys.stderr, flush=True)
        except (ValueError, KeyError) as exc:      # e.g. legacy recordings without the needed fields
            print(f"skipped {path.name}: {exc}", file=sys.stderr)
    print(f"index: {write_index()}", file=sys.stderr)


_INDEX = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Session comparisons</title>
<style>:root{--bg:#101318;--text:#e9edf2;--muted:#8d98a8;--line:#2b313b;--link:#70b7ff}
body{margin:0;padding:16px;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}
h1{font-size:17px}ul{list-style:none;padding:0;max-width:820px}li{display:flex;justify-content:space-between;gap:12px;
padding:10px 4px;border-bottom:1px solid var(--line)}a{color:var(--link)}span{color:var(--muted)}</style></head>
<body><h1>每局游戏的全模型对比</h1><ul>__ROWS__</ul></body></html>"""


_VIEWER = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Model comparison</title>
<style>
:root{--bg:#101318;--panel:#171b22;--line:#2b313b;--text:#e9edf2;--muted:#8d98a8;--bone:#70b7ff;--joint:#d8e6f3;--accent:#f2c14e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}
header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:12px;align-items:center}
h1{font-size:15px;margin:0;font-weight:600}.muted{color:var(--muted);font-size:12px}
#grid{display:grid;gap:8px;padding:8px;grid-template-columns:repeat(auto-fill,minmax(260px,1fr))}
#toggles{display:flex;flex-wrap:wrap;gap:6px;padding:8px 16px 0}#toggles label{background:#1d232c;border:1px solid var(--line);
border-radius:12px;padding:3px 10px;font-size:12px;cursor:pointer;user-select:none}#toggles input{margin-right:5px}
.cell.hidden{display:none}.live{color:var(--accent)}
.cell{background:var(--panel);border:1px solid var(--line);border-radius:6px;position:relative}
.cell canvas{width:100%;height:340px;display:block;cursor:grab}.cell .name{position:absolute;left:10px;top:8px;font-weight:600}
footer{padding:8px 16px 16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
button,select{background:#252c37;color:var(--text);border:1px solid #3a4553;border-radius:4px;padding:6px 10px;cursor:pointer}
input[type=range]{flex:1;min-width:200px}#msg{width:100%;color:var(--muted);font-size:12px;min-height:1.4em}
</style></head><body>
<header><h1 id="title"></h1><span class="muted">拖动画面旋转视角 · 空格播放/暂停 · ←/→ 逐帧 · ★ = 这局游戏实时用的模型</span></header>
<div id="toggles"><button id="all">全部</button><button id="none">全不选</button></div>
<div id="grid"></div>
<footer><button id="play">播放</button><select id="speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select>
<input id="seek" type="range" min="0" value="0"><span id="time"></span><div id="msg"></div></footer>
<script>
const D=__DATA__;document.getElementById('title').textContent=D.title;
const N=Math.min(...D.models.map(m=>m.frames.length)),grid=document.getElementById('grid'),seek=document.getElementById('seek');
seek.max=N-1;let frame=0,playing=false,yaw=-0.5,pitch=0.15,last=0,acc=0;
const cells=D.models.map(m=>{const d=document.createElement('div');d.className='cell';d.innerHTML=`<canvas></canvas><div class="name"></div>`;
const nm=d.querySelector('.name');nm.textContent=(m.id===D.liveModel?'★ ':'')+m.label;if(m.id===D.liveModel)nm.classList.add('live');
grid.appendChild(d);const c=d.querySelector('canvas');orbit(c);return{m,c,x:c.getContext('2d'),d}});
let hidden={};try{hidden=JSON.parse(localStorage.getItem('compareHidden')||'{}')}catch(e){}
const tg=document.getElementById('toggles');
for(const cell of cells){const l=document.createElement('label'),i=document.createElement('input');i.type='checkbox';i.checked=!hidden[cell.m.id];
l.append(i,cell.m.label);tg.appendChild(l);cell.box=i;i.onchange=()=>{hidden[cell.m.id]=!i.checked;apply()}}
function apply(){for(const cell of cells)cell.d.classList.toggle('hidden',!!hidden[cell.m.id]);try{localStorage.setItem('compareHidden',JSON.stringify(hidden))}catch(e){}draw()}
document.getElementById('all').onclick=()=>{hidden={};cells.forEach(c=>c.box.checked=true);apply()};
document.getElementById('none').onclick=()=>{cells.forEach(c=>{hidden[c.m.id]=true;c.box.checked=false});apply()};
function orbit(c){let drag=null;c.onpointerdown=e=>{drag=[e.clientX,e.clientY,yaw,pitch];c.setPointerCapture(e.pointerId)};
c.onpointermove=e=>{if(!drag)return;yaw=drag[2]+(e.clientX-drag[0])*0.01;pitch=Math.max(-1.3,Math.min(1.3,drag[3]+(e.clientY-drag[1])*0.01));draw()};
c.onpointerup=c.onpointercancel=()=>drag=null}
function project(p,w,h){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
const x=cy*p[0]+sy*p[2],z=-sy*p[0]+cy*p[2],y=cp*p[1]-sp*z,d=sp*p[1]+cp*z,s=Math.min(w,h)*0.42;return[w/2+x*s,h/2-y*s,d]}
function draw(){for(const{m,c,x,d}of cells){if(d.classList.contains('hidden'))continue;const w=c.clientWidth,h=c.clientHeight,r=devicePixelRatio||1;if(c.width!==w*r||c.height!==h*r){c.width=w*r;c.height=h*r}
x.setTransform(r,0,0,r,0,0);x.clearRect(0,0,w,h);const f=m.frames[frame],pts=[];for(let i=0;i<24;i++)pts.push(project([f[3*i],f[3*i+1],f[3*i+2]],w,h));
x.lineCap='round';const order=[...Array(23).keys()].map(k=>k+1).sort((a,b)=>(pts[a][2]+pts[D.parents[a]][2])-(pts[b][2]+pts[D.parents[b]][2]));
for(const i of order){const a=pts[D.parents[i]],b=pts[i],left=[1,4,7,10,13,16,18,20,22].includes(i);x.strokeStyle=m.id==='coach'?(left?'#8bd5a4':'#c6e8a0'):(left?'#70b7ff':'#ff8a65');x.lineWidth=6;
x.beginPath();x.moveTo(a[0],a[1]);x.lineTo(b[0],b[1]);x.stroke()}for(let i=0;i<24;i++){x.fillStyle=i?'#d8e6f3':'#f2c14e';x.beginPath();x.arc(pts[i][0],pts[i][1],i?3.5:6,0,7);x.fill()}}
const t=frame/D.fps;document.getElementById('time').textContent=`${t.toFixed(1)} s / ${((N-1)/D.fps).toFixed(1)} s`;seek.value=frame;
const msg=D.messages.filter(m=>m[0]<=t).pop();document.getElementById('msg').textContent=msg?`最近一次标定 (t=${msg[0]}s): ${msg[1]}`:'尚未标定'}
function tick(now){if(playing){acc+=(now-last)/1000*parseFloat(document.getElementById('speed').value)*D.fps;while(acc>=1&&frame<N-1){frame++;acc--}if(frame>=N-1)playing=false;draw()}
last=now;document.getElementById('play').textContent=playing?'暂停':'播放';requestAnimationFrame(tick)}
document.getElementById('play').onclick=()=>{if(frame>=N-1)frame=0;playing=!playing};seek.oninput=()=>{frame=+seek.value;draw()};
addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();document.getElementById('play').click()}if(e.code==='ArrowRight'){frame=Math.min(N-1,frame+1);draw()}if(e.code==='ArrowLeft'){frame=Math.max(0,frame-1);draw()}});
addEventListener('resize',draw);apply();requestAnimationFrame(tick);
</script></body></html>"""


if __name__ == "__main__":
    main()
