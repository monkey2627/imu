"""Read-only reconstruction/ablations of recorded five-IMU deployment inputs.

Print JSON to stdout; never updates recordings, model weights or training data.
Run from runtime/python with the mobileposer environment.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from production.calibration import bone_orientation, identity_heading
from production.pose_calibration import PoseCalibrator
from production.arm_prior import straighten_reaching_arms
from production.model_backends import resolve_waist_model
from production.paths import paths

sys.path.insert(0, str(paths.workspace_root / 'local_models/model_code'))
from waist_imudino_fiveimu.downstream.data import to_imu6
from waist_imudino_fiveimu.downstream.model import encode_imu
from waist_imudino_fiveimu.model_entry import sixd_to_matrix

PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]
LABELS = ['lw','rw','ls','rs','waist']


def cadence_audit():
    """Start isolated replay bridges; measure wire cadence and real push counts.

    Wrapping methods only in the child process keeps production sources intact.
    """
    import asyncio
    import subprocess
    import websockets
    child_code = '''
from production import run
base_make=run.make_waist_session
pushed={'count':0}
def make(*args,**kwargs):
    session=base_make(*args,**kwargs)
    original=session.push_observation
    def push(*args,**kwargs):
        pushed['count']+=1
        return original(*args,**kwargs)
    session.push_observation=push
    return session
run.make_waist_session=make
original=run.PoseBroadcaster.broadcast
async def broadcast(self,*args,**kwargs):
    if len(args)>5 and args[5] is not None:
        args[5]['auditPushedCount']=pushed['count']
    return await original(self,*args,**kwargs)
run.PoseBroadcaster.broadcast=broadcast
run.main()
'''
    async def once(every):
        args=[sys.executable,'-u','-c',child_code,'--model','waist_imudino_fiveimu',
              '--infer-every',str(every),'--replay',
              'BirdDog_RingSkinC_Recorded_20261111_220149.jsonl',
              '--no-auto-compare','--no-arm-prior','--port','21209']
        process=subprocess.Popen(args,cwd=str(paths.runtime_root),stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,text=True)
        frames=[]; received=[]
        try:
            for _ in range(80):
                if process.poll() is not None:
                    raise RuntimeError('Diagnostic bridge exited early')
                try:
                    socket=await websockets.connect('ws://127.0.0.1:21209')
                    break
                except OSError:
                    await asyncio.sleep(.25)
            else:
                raise TimeoutError('Diagnostic bridge did not start')
            async with socket:
                for _ in range(100):
                    frames.append(json.loads(await asyncio.wait_for(socket.recv(),30)))
                    received.append(time.perf_counter())
            stable=frames[4:]
            mismatch=[d['auditPushedCount']-d['modelInputFrame']-1 for d in stable]
            print(json.dumps({'infer_every':every,'received':len(frames),
                'receive_dt_ms':stats(np.diff(received[4:])*1000),
                'inference_ms':stats([(d['inferenceEndMonotonicSeconds']-
                                      d['inferenceStartMonotonicSeconds'])*1000 for d in stable]),
                'pushed_minus_frame_count':{'min':min(mismatch),'max':max(mismatch)}},
                ensure_ascii=False),flush=True)
        finally:
            process.terminate()
            try:
                output=process.communicate(timeout=10)[0]
            except subprocess.TimeoutExpired:
                process.kill()
                output=process.communicate()[0]
            if not frames:
                print(output[-2500:],file=sys.stderr)
    async def both():
        await once(6)
        await once(2)
    asyncio.run(both())


def stats(x):
    x = np.asarray(x)
    return dict(zip(['p50','p90','p99','max'], map(float, np.percentile(x, [50,90,99,100]))))


def angle(a, b):
    relative = np.swapaxes(a, -1, -2) @ b
    # atan2 is more stable than acos near identity in float32.
    vee = np.stack([relative[...,2,1]-relative[...,1,2],
                    relative[...,0,2]-relative[...,2,0],
                    relative[...,1,0]-relative[...,0,1]], axis=-1) / 2
    return np.rad2deg(np.arctan2(np.linalg.norm(vee, axis=-1),
                               (np.trace(relative,axis1=-2,axis2=-1)-1)/2))


def globals_of(local):
    out = np.empty_like(local)
    for j,p in enumerate(PARENTS):
        out[:,j] = local[:,j] if p < 0 else out[:,p] @ local[:,j]
    return out


def motion_metrics(r6, waist):
    children = sixd_to_matrix(torch.from_numpy(r6.reshape(-1,23,6))).numpy()
    root = waist @ np.swapaxes(children[:,2],-1,-2)
    local = np.concatenate([root[:,None],children],axis=1)
    g = globals_of(local)
    changes = angle(local[:-1],local[1:])
    global_changes = angle(g[:-1],g[1:])
    return {'all_local_step_deg':stats(changes),'root_step_deg':stats(changes[:,0]),
            'spine1_local_step_deg':stats(changes[:,3]),
            'waist_measured_step_deg':stats(angle(waist[:-1],waist[1:])),
            'global_step_deg':{str(j):stats(global_changes[:,j]) for j in [0,1,2,3,4,5,15,18,19]},
            'root_waist_constraint_max_abs':float(np.max(np.abs(g[:,3]-waist)))},local


@torch.inference_mode()
def encode_windows(adapter, imu, starts, chunk=8, bf16=False):
    outputs=[]
    for offset in range(0,len(starts),chunk):
        batch=torch.stack([imu[s:s+480] for s in starts[offset:offset+chunk]]).to(adapter.device)
        mask=torch.ones(batch.shape[:2],dtype=torch.bool,device=adapter.device)
        with torch.autocast(device_type=adapter.device.type,dtype=torch.bfloat16,enabled=bf16):
            features=encode_imu(adapter.backbone,batch,mask,60)
        # Training caches round features to fp16, then head casts them to float32.
        if bf16: features=features.to(torch.float16)
        output=adapter.head(features,480).float().cpu().numpy()
        outputs.append(output)
    return np.concatenate(outputs)


def audit(path, adapter):
    rows=[json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]
    header=rows[0]; ticks=[r for r in rows if r.get('type')=='tick']
    times=np.array([r['t'] for r in ticks]); dt=np.diff(times)
    repeats=np.array([r['ticks'] for r in ticks],int)
    samples=[r['samples'] for r in ticks]
    q=np.array([[r[l]['quat_xyzw'] for l in LABELS] for r in samples])
    aq=np.array([[r[l]['reference_adjusted_accel_xyz'] for l in LABELS] for r in samples])
    legacy=np.array([[r[l]['accel_xyz'] for l in LABELS] for r in samples])
    ori=np.stack([bone_orientation(row,LABELS).numpy() for row in q])
    heading=identity_heading(LABELS)
    acc=np.stack([heading.apply(row).numpy() for row in aq])
    events=[{k:v for k,v in r.items() if k!='q'} for r in rows if r.get('type')=='event']
    # These game recordings have no stored calibration changes. Boundary cannot
    # reconstruct calibration completed before the recording began.
    expanded_ori=np.repeat(ori,repeats,axis=0)
    expanded_acc=np.repeat(acc,repeats,axis=0)
    o=torch.from_numpy(expanded_ori); a=torch.from_numpy(expanded_acc)
    imu=to_imu6(a,o,60)
    coach=[r for r in rows if r.get('type')=='coach']
    result={'path':str(path),'header':header,'events':events,
        'rows':dict(Counter(r.get('type') for r in rows)),
        'timing':{'tick_dt_ms':stats(dt*1000),'wall_seconds':float(times[-1]-times[0]),
                  'expanded_frames':len(o),'expanded_seconds':len(o)/60,
                  'ticks_counts':dict(Counter(map(int,repeats))),
                  'coach_dt_ms':stats(np.diff([r['t'] for r in coach])*1000)},
        'mapping':{l:sorted(set((r[l]['tracker_key'],r[l]['body_part'],r[l]['status']) for r in samples)) for l in LABELS},
        'inputs':{'quat_norm_error_max':float(np.max(abs(np.linalg.norm(q,axis=-1)-1))),
                  'null_raw_gyro_fraction':float(np.mean([r[l].get('raw_angular_velocity_xyz') is None for r in samples for l in LABELS])),
                  'linear_acc_norm_m_s2':stats(np.linalg.norm(aq,axis=-1)),
                  'legacy_vs_adjusted_norm_m_s2':stats(np.linalg.norm(legacy-aq,axis=-1)),
                  'sensor_rotation_step_deg':stats(angle(ori[:-1],ori[1:])),
                  'gyro_expanded_rad_s':stats(torch.linalg.vector_norm(imu[...,3:],dim=-1).numpy()),
                  'gyro_zero_fraction_expanded':float((torch.linalg.vector_norm(imu[...,3:],dim=-1)<1e-7).float().mean()),
                  'consecutive_identical_quat_fraction':{l:float(np.mean(np.max(abs(np.diff(q[:,i],axis=0)),axis=1)<1e-8)) for i,l in enumerate(LABELS)}}}
    # Rate mismatch in unexpanded true observations: gyro60 vs observed arrival dt.
    gyro_unique=to_imu6(torch.from_numpy(acc),torch.from_numpy(ori),60).numpy()[:-1,:,3:]
    gyro_time=gyro_unique/(60*dt[:,None,None])
    result['inputs']['gyro60_vs_arrival_dt_error_rad_s']=stats(np.linalg.norm(gyro_unique-gyro_time,axis=-1))
    result['inputs']['60dt_factor']=stats(60*dt)
    result['inputs']['derived_gyro_nonzero_fraction']=float(np.mean(np.linalg.norm(gyro_unique,axis=-1)>1e-5))
    # All variants scored at the same 6-frame grid, after full actual history.
    # Keep the original deployment baseline and compare the new lagged endpoint
    # on the same physical frames, including its one real forward-gyro sample.
    target_index=479
    targets=np.arange(485,len(o)-9,6)
    starts=targets-target_index
    rolled=encode_windows(adapter,imu,list(starts))
    raw6=rolled[:,target_index]
    m,local=motion_metrics(raw6,expanded_ori[targets,4]); result['rolling_window_last_frame']=m
    # Reproduce actual online preprocessing: compute gyro inside each 480-frame window.
    actual_imu=torch.stack([to_imu6(a[s:s+480],o[s:s+480],60) for s in starts])
    actual_predictions=[]; timings=[]
    for i in range(len(starts)):
        t=time.perf_counter()
        with torch.inference_mode():
            mask=torch.ones(1,480,dtype=torch.bool,device=adapter.device)
            f=encode_imu(adapter.backbone,actual_imu[i:i+1].to(adapter.device),mask,60)
            p=adapter.head(f,480).float().cpu().numpy()[0,target_index]
        timings.append(time.perf_counter()-t); actual_predictions.append(p)
    actual6=np.stack(actual_predictions)
    m,local_actual=motion_metrics(actual6,expanded_ori[targets,4]);result['actual_deployment_last_frame']=m
    result['actual_deployment_last_frame']['inference_ms']=stats(np.array(timings[2:])*1000)
    result['window_gyro_boundary_effect_deg']=stats(angle(local[:,1:],local_actual[:,1:]))
    # Head interpolation: final eight output frames are clamped to final token.
    result['head_last_eight_max_abs']=float(np.max(abs(rolled[:,-8:]-rolled[:,-1:])) )
    # Same physical target in a window shifted by one observation: removes real
    # motion from the comparison and exposes patch/context dependence.
    shifted=encode_windows(adapter,imu,list(starts+1))
    result['same_target_window_shift1_deg']=stats(angle(
        sixd_to_matrix(torch.from_numpy(raw6.reshape(-1,23,6))).numpy(),
        sixd_to_matrix(torch.from_numpy(shifted[:,target_index-1].reshape(-1,23,6))).numpy()))
    # Offline training-style windows, stride 240; average 6D BEFORE decoding.
    fixed_starts=list(range(0,len(o)-479,240))
    if fixed_starts[-1] != len(o)-480: fixed_starts.append(len(o)-480)
    fixed=encode_windows(adapter,imu,fixed_starts)
    sums=np.zeros((len(o),138),np.float32);counts=np.zeros((len(o),1),np.float32)
    for s,p in zip(fixed_starts,fixed): sums[s:s+480]+=p;counts[s:s+480]+=1
    fused=sums/counts
    m,_=motion_metrics(fused[targets],expanded_ori[targets,4]);result['offline_stride240_6d_mean']=m
    # Position in one unchanged window isolates head interpolation from re-patching.
    center=fixed[len(fixed)//2][np.arange(5,480,6)]
    m,_=motion_metrics(center,expanded_ori[fixed_starts[len(fixed)//2]+np.arange(5,480,6),4])
    result['one_fixed_window_6frame_grid']=m
    # Root is not learned: replacing just its source exposes whole-body amplification.
    constant_root=local_actual.copy();constant_root[:,0]=local_actual[0,0]
    gc=globals_of(constant_root);ga=globals_of(local_actual)
    result['root_policy_global_step_p99_deg']={str(j):{
        'waist_constrained':float(np.percentile(angle(ga[:-1,j],ga[1:,j]),99)),
        'constant_root_diagnostic_only':float(np.percentile(angle(gc[:-1,j],gc[1:,j]),99))} for j in [0,1,2,4,5,15,18,19]}
    # Arm prior is a separate actual-app postprocessing stage.
    post=np.stack([straighten_reaching_arms(p,(expanded_ori[t,0],expanded_ori[t,1])) for p,t in zip(local_actual,targets)])
    result['arm_prior_step_deg']=stats(angle(post[:-1],post[1:]))
    result['arm_prior_change_deg']=stats(angle(local_actual,post))
    # Time-correct resampling diagnostic: Slerp rotations and linear acc at 60Hz,
    # not filtering. Compare actual pipeline at same resampled temporal grid.
    uniform=np.arange(times[0],times[-1],1/60)
    resampled_ori=np.stack([Slerp(times,Rotation.from_matrix(ori[:,i]))(uniform).as_matrix() for i in range(5)],axis=1).astype(np.float32)
    resampled_acc=np.stack([np.stack([np.interp(uniform,times,acc[:,i,j]) for j in range(3)],axis=1) for i in range(5)],axis=1).astype(np.float32)
    ri=to_imu6(torch.from_numpy(resampled_acc),torch.from_numpy(resampled_ori),60)
    rt=np.arange(485,len(ri)-(479-target_index),6)
    rp=encode_windows(adapter,ri,list(rt-target_index))[:,target_index]
    m,_=motion_metrics(rp,resampled_ori[rt,4]);result['time_resampled_60hz']=m
    # Frozen deterministic eval and cache precision effects on identical input.
    repeated=encode_windows(adapter,imu,[int(starts[10])]*2)
    result['same_input_repeat_max_abs']=float(np.max(abs(repeated[0]-repeated[1])))
    cached_precision=encode_windows(adapter,imu,[int(starts[10])],bf16=True)
    result['float32_vs_training_cache_precision_deg']=stats(angle(
        sixd_to_matrix(torch.from_numpy(repeated[:1].reshape(1,480,23,6))).numpy(),
        sixd_to_matrix(torch.from_numpy(cached_precision.reshape(1,480,23,6))).numpy()))
    precision_rolling=encode_windows(adapter,imu,list(starts),bf16=True)[:,-1]
    m,_=motion_metrics(precision_rolling,expanded_ori[targets,4])
    result['training_cache_precision_rolling']=m
    result['target_grid']={'step_frames':6,'first_frame':int(targets[0]),'last_frame':int(targets[-1]),'count':len(targets)}
    deployed=encode_windows(adapter,imu,list(targets-471))[:,471]
    result['revised_deployment_same_target'],_=motion_metrics(deployed,expanded_ori[targets,4])
    result['revised_deployment_same_target']['lag_frames']=adapter.lag
    if '--summary' in sys.argv:
        result={'path':str(path), 'old':result['actual_deployment_last_frame'],
                'revised':result['revised_deployment_same_target'],
                'target_grid':result['target_grid']}
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    if '--cadence' in sys.argv:
        cadence_audit()
        raise SystemExit(0)
    torch.set_num_threads(8)
    adapter=resolve_waist_model('waist_imudino_fiveimu')[2]
    for name in ['BirdDog_RingSkinC_Recorded_20261111_215730.jsonl',
                 'BirdDog_RingSkinC_Recorded_20261111_220149.jsonl']:
        audit(paths.runtime_root/'recordings'/name,adapter)
