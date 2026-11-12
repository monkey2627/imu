"""Stage-by-stage jitter localisation for an IMUDINO input+prediction recording.

Read-only. Rebuilds the exact 60 Hz model input from `tick` rows, re-runs
FiveIMUAdapter on the same 481-frame windows that produced each recorded
`prediction`, and then varies one factor at a time (window position, input
channel, root solve). No smoothing is applied to any output.

    python -m analysis.imudino_jitter_stages BirdDog_RingSkinC_Recorded_20261111_235722
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from production.calibration import bone_orientation, identity_heading
from production.input_recording import resolve_recording_path
from production.model_backends import resolve_waist_model
from production.paths import paths
from production.rotmat import UNITY_SMPL_BASIS

sys.path.insert(0, str(paths.workspace_root / "local_models/model_code"))
from waist_imudino_fiveimu.downstream.data import to_imu6  # noqa: E402
from waist_imudino_fiveimu.model_entry import sixd_to_matrix  # noqa: E402

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
LABELS = ["lw", "rw", "ls", "rs", "waist"]
NAMES = {0: "pelvis", 1: "L_hip", 2: "R_hip", 3: "spine1", 4: "L_knee", 5: "R_knee", 6: "spine2",
         9: "spine3", 12: "neck", 16: "L_shoulder", 17: "R_shoulder", 18: "L_elbow", 19: "R_elbow",
         20: "L_wrist", 21: "R_wrist"}
SHOW = [0, 1, 2, 3, 4, 5, 6, 9, 16, 17, 18, 19]
W, IDX = 480, 471


# ---------------------------------------------------------------- rotation helpers
def quat_xyzw_to_mat(q):
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def angle(a, b):
    """Geodesic angle (deg) between rotation stacks of any leading shape [...,3,3]."""
    flat_a, flat_b = a.reshape(-1, 3, 3), b.reshape(-1, 3, 3)
    tr = np.einsum("nij,nij->n", flat_a, flat_b)  # trace(a^T b)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).reshape(a.shape[:-2])


def fk(local):
    g = np.empty_like(local)
    for j, p in enumerate(PARENTS):
        g[..., j, :, :] = local[..., j, :, :] if p < 0 else g[..., p, :, :] @ local[..., j, :, :]
    return g


def pct(x, qs=(50, 90, 99)):
    x = np.asarray(x).reshape(-1)
    return {f"p{q}": round(float(np.percentile(x, q)), 2) for q in qs} | {"max": round(float(x.max()), 2)}


def per_joint(diff):  # diff [N,24]
    return {NAMES[j]: pct(diff[:, j], (50, 99)) for j in SHOW}


# ---------------------------------------------------------------- recording
def load(path):
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8")]
    header = rows[0]
    ticks = [r for r in rows if r.get("type") == "tick"]
    preds = [r for r in rows if r.get("type") == "prediction"]
    accel_key = "reference_adjusted_accel_xyz" if header.get("accelerationSource") == "server" else "accel_xyz"
    quats = np.asarray([[t["samples"][l]["quat_xyzw"] for l in LABELS] for t in ticks], np.float32)
    acc = np.asarray([[t["samples"][l][accel_key] for l in LABELS] for t in ticks], np.float32)
    assert all(t.get("ticks", 1) == 1 for t in ticks), "expected one tick row per pushed model frame"
    # Tick rows are written before the prediction that consumed them -> index of each prediction.
    order, n_tick, pred_tick = [r.get("type") for r in rows], 0, []
    for kind in order:
        if kind == "tick":
            n_tick += 1
        elif kind == "prediction":
            pred_tick.append(n_tick - 1)
    pred_tick = np.asarray(pred_tick)
    model_frame = np.asarray([p["diagnostics"]["modelInputFrame"] for p in preds])
    offset = model_frame - pred_tick
    raw = np.asarray([p["rawLocalQuaternions"] for p in preds], np.float64).reshape(-1, 24, 4)
    fin = np.asarray([p["unityLocalQuaternions"] for p in preds], np.float64).reshape(-1, 24, 4)
    return header, ticks, preds, quats, acc, pred_tick, offset, raw, fin


def unity_to_smpl(mats):
    b = UNITY_SMPL_BASIS.astype(np.float64)
    return b @ mats @ b


# ---------------------------------------------------------------- model
class Runner:
    def __init__(self):
        _, _, self.adapter = resolve_waist_model("waist_imudino_fiveimu")

    def window(self, acc, ori, end):
        """Same as FiveIMUAdapter.infer for the window whose newest pushed frame is `end`.
        Returns dense local SMPL [480,24,3,3] (root left identity) for that window."""
        a = torch.from_numpy(acc[end - W:end + 1])
        o = torch.from_numpy(ori[end - W:end + 1])
        imu = to_imu6(a, o, 60)[:-1]
        return self.dense(imu)

    def dense(self, imu):
        pred = self.adapter.predict_imu(imu)[0]  # [480,138]
        ch = sixd_to_matrix(pred.reshape(W, 23, 6)).double().numpy()
        out = np.tile(np.eye(3), (W, 24, 1, 1))
        out[:, 1:] = ch
        return out

    @staticmethod
    def root(local, waist):
        local = local.copy()
        local[..., 0, :, :] = waist @ np.swapaxes(local[..., 3, :, :], -1, -2)
        return local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--dense-limit", type=int, default=1200, help="frames for the every-frame-window study")
    args = ap.parse_args()
    path = resolve_recording_path(args.recording)
    header, ticks, preds, quats, acc, pred_tick, offset, raw_q, fin_q = load(path)
    out = {"file": path.name, "model": header.get("model"), "ticks": len(ticks), "predictions": len(preds),
           "tickIndexOffsetUnique": sorted(set(offset.tolist()))[:5]}
    assert header.get("model") == "waist_imudino_fiveimu"
    ori = bone_orientation(quats, LABELS)  # identity heading/corrections recorded in this log
    # Live path: WaistInferenceSession.prepare_feature -> identity_heading.apply -> SERVER_TO_SMPL basis.
    acc = np.stack([identity_heading(LABELS).apply(x).numpy() for x in acc]).astype(np.float32)
    calib = {json.dumps(t.get("calibration")) for t in ticks}
    out["calibrationSnapshotsDistinct"] = len(calib)
    ori = ori.numpy().astype(np.float32)

    # ------------------------------------------------ stage 1: input stream
    t = np.asarray([tk["t"] for tk in ticks])
    imu_all = to_imu6(torch.from_numpy(acc), torch.from_numpy(ori), 60).numpy()
    gyro = imu_all[..., 3:]
    gyro_norm = np.linalg.norm(gyro, axis=-1)
    dgyro = np.linalg.norm(np.diff(gyro, axis=0), axis=-1)   # rad/s change per 1/60 s
    lin = np.linalg.norm(acc, axis=-1)
    dacc = np.linalg.norm(np.diff(acc, axis=0), axis=-1)
    step_ori = angle(ori[:-1], ori[1:])
    # Second difference of orientation: 0 for constant angular velocity, large for arrival-time zig-zag.
    second = np.linalg.norm(np.diff(gyro, n=1, axis=0), axis=-1) / np.maximum(gyro_norm[1:], 0.3)
    exact_repeat = (step_ori < 1e-4).mean(0)
    out["stage1_input"] = {
        "tickWriteIntervalMs": pct(np.diff(t) * 1000),
        "perSensorStepDeg": {l: pct(step_ori[:, i]) for i, l in enumerate(LABELS)},
        "perSensorExactRepeatFraction": {l: round(float(exact_repeat[i]), 4) for i, l in enumerate(LABELS)},
        "gyroNormRadS": {l: pct(gyro_norm[:, i]) for i, l in enumerate(LABELS)},
        "gyroFrameToFrameChangeRadS": {l: pct(dgyro[:, i]) for i, l in enumerate(LABELS)},
        "gyroRelativeRoughness": {l: pct(second[:, i]) for i, l in enumerate(LABELS)},
        "linearAccNormMps2": {l: pct(lin[:, i]) for i, l in enumerate(LABELS)},
        "linearAccFrameToFrameChange": {l: pct(dacc[:, i]) for i, l in enumerate(LABELS)},
        "specificForceNorm": {l: pct(np.linalg.norm(imu_all[..., :3], axis=-1)[:, i]) for i, l in enumerate(LABELS)},
    }

    # ------------------------------------------------ stage 3a: exact replay of every recorded prediction
    runner = Runner()
    valid = np.where(pred_tick >= W)[0]
    rec_raw = unity_to_smpl(quat_xyzw_to_mat(raw_q))
    rec_fin = unity_to_smpl(quat_xyzw_to_mat(fin_q))
    repro, dense_at, full = [], {}, []
    for n_k, k in enumerate(valid):
        end = pred_tick[k]
        dense = runner.window(acc, ori, end)
        dense_at[end] = dense[IDX - 16:IDX + 1]  # local index 16 == window index 471
        if n_k % 10 == 0:
            full.append(dense)
        repro.append(runner.root(dense[IDX], ori[end - W + IDX, 4].astype(np.float64)))
    repro = np.asarray(repro)
    err = angle(repro, rec_raw[valid])
    out["stage3_replayMatchesRecordedRaw"] = {"n": int(len(valid)), "angleDeg": pct(err)}
    out["rawVsUnityQuatDiffDeg"] = per_joint(angle(rec_raw, rec_fin))

    # ------------------------------------------------ stage 2/3: output continuity
    steps = np.diff(pred_tick[valid])
    rec_local_step = angle(rec_raw[valid][1:], rec_raw[valid][:-1])
    out["predFrameSteps"] = {int(s): int((steps == s).sum()) for s in np.unique(steps)}
    two = steps == 2
    out["online_localStep_2frames"] = per_joint(rec_local_step[two])

    # Within ONE window: change over 2 frames at the emitted index (pure model temporal response).
    within = []
    for end, d in dense_at.items():
        within.append(angle(d[16], d[14]))
    within = np.asarray(within)
    out["withinWindow_localStep_2frames_at471"] = per_joint(within)
    whole = np.concatenate([angle(d[2:], d[:-2]) for d in full[::2]])
    out["withinWindow_localStep_2frames_allIndices"] = per_joint(whole)
    # Patch-boundary signature: within-window 1-frame change as function of index mod 16
    mod = np.zeros(16)
    cnt = np.zeros(16)
    for d in full:
        s = angle(d[1:], d[:-1]).mean(1)  # mean over joints, step i -> i+1
        for i in range(len(s)):
            mod[(i + 1) % 16] += s[i]
            cnt[(i + 1) % 16] += 1
    out["withinWindow_meanStepByIndexMod16"] = [round(float(v), 3) for v in mod / cnt]

    # Same physical frame seen through windows shifted by 2 frames (what consecutive online outputs do).
    same_frame = []
    ends = sorted(dense_at)
    for a, b in zip(ends[:-1], ends[1:]):
        shift = b - a
        if shift == 2:
            same_frame.append(angle(dense_at[a][16], dense_at[b][14]))
    same_frame = np.asarray(same_frame)
    out["sameFrame_twoWindowsShifted2"] = per_joint(same_frame)

    # Dense per-frame windows on a contiguous segment: how much does the emitted pose of frame f
    # change with the window offset (index 471 vs 470...456)?
    lim = min(len(ori) - 1, W + args.dense_limit)
    dense_all = {}
    for end in range(W, lim + 1):
        dense_all[end] = dense_at[end] if end in dense_at else runner.window(acc, ori, end)[IDX - 16:IDX + 1]
    seq471 = np.asarray([dense_all[e][16] for e in range(W, lim + 1)])  # frame e-9
    out["denseEveryFrame_emit471_1frameStep"] = per_joint(angle(seq471[1:], seq471[:-1]))
    out["denseEveryFrame_emit471_2frameStep"] = per_joint(angle(seq471[2:], seq471[:-2]))
    for k_back in (1, 2, 16):
        diff = [angle(dense_all[e][16], dense_all[e + k_back][16 - k_back])
                for e in range(W, lim + 1 - k_back)]
        out[f"denseSameFrame_shift{k_back}"] = per_joint(np.asarray(diff))

    # Training/validation stitching on the same input (stride 240, averaged 6D overlap) for the segment.
    seg_start, seg_end = W - W + 1, lim  # frames whose full future context is in the log
    stitched = stitch(runner, acc, ori, seg_start, seg_end)
    out["stride240Stitch_2frameStep"] = per_joint(angle(stitched[2:], stitched[:-2]))

    # ------------------------------------------------ input sensitivity at the recorded windows
    sens = {}
    sample_ends = list(dense_at)[::8]
    for name in ("zeroLinearAcc", "gyroFromSmoothedOri", "accOnlyWaistRealOthersGravity"):
        diffs = []
        for end in sample_ends:
            a = acc[end - W:end + 1].copy()
            o = ori[end - W:end + 1].copy()
            imu = to_imu6(torch.from_numpy(a), torch.from_numpy(o), 60)[:-1].clone()
            if name == "zeroLinearAcc":
                imu = to_imu6(torch.zeros_like(torch.from_numpy(a)), torch.from_numpy(o), 60)[:-1]
            elif name == "gyroFromSmoothedOri":
                g = imu[..., 3:].numpy()
                k = np.array([1, 2, 3, 2, 1], np.float32) / 9  # diagnostic only
                gs = np.stack([np.apply_along_axis(lambda x: np.convolve(x, k, "same"), 0, g[:, s])
                               for s in range(5)], 1)
                imu[..., 3:] = torch.from_numpy(gs)
            else:
                a2 = np.zeros_like(a)
                a2[:, 4] = a[:, 4]
                imu = to_imu6(torch.from_numpy(a2), torch.from_numpy(o), 60)[:-1]
            d = runner.dense(imu)
            diffs.append(angle(d[IDX], dense_at[end][16]))
        sens[name] = per_joint(np.asarray(diffs))
    out["inputSensitivity_vsActual_at471"] = sens

    # Effect on online 2-frame jitter when linear acc is zeroed (shows whether acc noise drives jumps)
    z_seq = []
    for end in range(W, min(lim, W + 600) + 1):
        a = np.zeros_like(acc[end - W:end + 1])
        imu = to_imu6(torch.from_numpy(a), torch.from_numpy(ori[end - W:end + 1]), 60)[:-1]
        z_seq.append(runner.dense(imu)[IDX])
    z_seq = np.asarray(z_seq)
    out["zeroLinearAcc_denseEmit471_2frameStep"] = per_joint(angle(z_seq[2:], z_seq[:-2]))

    # ------------------------------------------------ stage 4: global amplification by root solve
    loc = rec_raw[valid]
    g = fk(loc)
    gstep = angle(g[1:][two], g[:-1][two])
    out["online_globalStep_2frames"] = per_joint(gstep)
    waist_step = angle(g[1:, 3][two], g[:-1, 3][two])
    out["online_global_spine1_equals_waist_step"] = pct(waist_step)
    root_step = angle(loc[1:, 0][two], loc[:-1, 0][two])
    out["online_root_step"] = pct(root_step)
    # Counterfactual: root = waist (spine1 local forced identity in root solve), keep everything else.
    alt = loc.copy()
    alt[:, 0] = g[:, 3]
    ga = fk(alt)
    out["counterfactual_rootEqualsWaist_globalStep"] = per_joint(angle(ga[1:][two], ga[:-1][two]))
    out["final_unity_globalStep_2frames"] = per_joint(angle(fk(rec_fin[valid])[1:][two], fk(rec_fin[valid])[:-1][two]))

    # ------------------------------------------------ stage 5: python emit timing
    d = [p["diagnostics"] for p in preds]
    rcv = np.asarray([x["receiveMonotonicSeconds"] for x in d])
    end_t = np.asarray([x["inferenceEndMonotonicSeconds"] for x in d])
    rep = np.asarray([x["modelOutputRepresentsFrame"] for x in d])
    emit_lag_frames = (end_t - end_t[0]) * 60 - (rep - rep[0])
    out["stage5_emit"] = {
        "emitIntervalMs": pct(np.diff(end_t) * 1000),
        "representedFrameStep": {int(s): int((np.diff(rep) == s).sum()) for s in np.unique(np.diff(rep))},
        "emitClockMinusFrameClockFrames_detrended": pct(np.abs(emit_lag_frames - np.median(emit_lag_frames))),
        "emitVsFrameClockDriftFramesTotal": round(float(emit_lag_frames[-1] - emit_lag_frames[0]), 2),
        "recvToInferStartMs": pct((np.asarray([x["inferenceStartMonotonicSeconds"] for x in d]) - rcv) * 1000),
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))


def stitch(runner, acc, ori, first, last):
    """Validation-style: windows at stride 240 over [first-?,last], average 6D/matrices in overlap."""
    start0 = max(0, first - 1)
    n = last - start0
    acc_s, ori_s = acc[start0:start0 + n + 1], ori[start0:start0 + n + 1]
    imu = to_imu6(torch.from_numpy(acc_s), torch.from_numpy(ori_s), 60)[:-1]
    total = np.zeros((n, 24, 3, 3))
    count = np.zeros(n)
    starts = list(range(0, max(n - W + 1, 1), 240))
    if starts[-1] != n - W:
        starts.append(n - W)
    for s in starts:
        d = runner.dense(imu[s:s + W])
        total[s:s + W] += d
        count[s:s + W] += 1
    m = total / count[:, None, None, None]
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    return r


if __name__ == "__main__":
    main()
