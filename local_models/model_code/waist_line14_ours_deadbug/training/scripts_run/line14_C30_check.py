"""Stage C, step 0 (CPU only): convention and inference-path checks BEFORE any real-data evaluation of the calibration pipeline.
A  injection convention round trip on synthetic offsets (corrected = R_meas R_est^T = R_bone Exp(-delta)).
B  real data: is R = RIM^T RIS RSB (the form every baseline receives) really the bone orientation of the optical/SMPL GT?  documented formula vs alternative products;
   per-sequence yaw search; static offset (calibration residual of the dataset itself) vs dynamic part.
C  real-data input sanity: gravity-free acceleration (median |a| should be small), matrix validity, scale vs synthetic training inputs.
D  inference path: infer() == direct model call on the same window; stitching places the right frames; plausibility on a few synthetic validation sequences.
usage (no GPU):  CUDA_VISIBLE_DEVICES= MOBILEPOSER_FPS=60 python scripts_run/line14_C30_check.py
"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import line14_realeval_lib as RL
from mobileposer.articulate import math as AM
from mobileposer.config import paths
from mobileposer.line14.error_sampler import load_sampler, draw_delta, apply_error
from mobileposer.line9.calib import identity_offsets
import mobileposer.articulate as art
torch.set_num_threads(8)
log = lambda *x: print(*x, flush=True)
dev = torch.device("cpu")
deg = torch.rad2deg
ang = lambda R: torch.acos(((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))


def rand_rot(*shape, gen):
    q = torch.randn(*shape, 4, generator=gen); q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(*shape, 3, 3)


# ---------------------------------------------------------------- A
log("== A. injection convention round trip (synthetic)")
S = load_sampler("data_local/line14_errsampler_v2/sampler.pt")
gen = torch.Generator().manual_seed(0)
worst = 0.0
for n in range(20):
    ident = f"ACCAD/Male1General_c3d" if n == 0 else f"synthetic_{n}"
    R_off = identity_offsets(ident, 5.0, 3.0, gauge=True)[0]                          # [5,3,3] stored convention: measured = R_bone @ R_off
    R_bone = rand_rot(50, 5, gen=gen)
    R_meas = R_bone @ R_off[None]
    delta = draw_delta(S, ident, n, 8)
    R_est = apply_error(R_off, delta)                                                    # = Exp(delta) @ R_off
    corrected = R_meas @ R_est.transpose(-1, -2)[None]
    injected = RL.inject(R_bone, delta)                                                  # R_bone @ Exp(-delta)
    worst = max(worst, float((corrected - injected).abs().max()))
    back = RL.log_so3((R_est.double() @ R_off.double().transpose(-1, -2)))
    assert torch.allclose(back.float(), delta, atol=2e-6)
log(f"   max |R_meas R_est^T - R_bone Exp(-delta)| over 20 wearings x 50 frames x 5 sensors = {worst:.2e}  ->", "OK" if worst < 1e-5 else "FAIL")
dlt = torch.stack([draw_delta(S, f"w{i}", 0, 8) for i in range(2000)])
log(f"   sampled delta (ns=8): mean angle {float(deg(dlt.norm(dim=-1)).mean()):.3f} deg ; injected orientation error vs oracle (frame-wise geodesic) mean {float(deg(ang(RL.inject(torch.eye(3).repeat(1, 5, 1, 1), dlt[0])[0])).mean()):.3f} deg for one draw")

# ---------------------------------------------------------------- B, C
from mobileposer.evaluate_standard import StdEvaluator, sequences, SENSORS
ev = StdEvaluator(dev)
body = art.model.ParametricModel(paths.smpl_file, device=dev)
P = "/home/duanyuhan/dyh/motion/MotionRe/PNP/data/test_datasets/"
JOINTS = [18, 19, 4, 5, 0]                                                                # forearms (elbow joints), shanks (knee joints), pelvis
names = ["lw", "rw", "ls", "rs", "pelvis"]


def yaw_mats(psis):
    c, s = torch.cos(psis), torch.sin(psis); z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([c, z, s, z, o, z, -s, z, c], -1).reshape(-1, 3, 3)


def residual(R_in, G):
    """R_in, G [T,5,3,3] (both global).  Best global yaw (2 deg grid + 0.25 refine) minimising the mean geodesic error; returns psi, per-sensor mean/median error after alignment, static offset angle."""
    sub = slice(None, None, 15)
    psis = torch.deg2rad(torch.arange(-180, 180, 2.0))
    Y = yaw_mats(psis)                                                                  # [K,3,3]
    cost = torch.stack([ang(G[sub].transpose(-1, -2)[None] @ (Y[k] @ R_in[sub])).mean() for k in range(len(Y))])
    p0 = psis[cost.argmin()]
    fine = p0 + torch.deg2rad(torch.arange(-2, 2.01, 0.25)); Yf = yaw_mats(fine)
    cf = torch.stack([ang(G[sub].transpose(-1, -2)[None] @ (Yf[k] @ R_in[sub])).mean() for k in range(len(Yf))])
    psi = fine[cf.argmin()]; Rn = yaw_mats(psi[None])[0] @ R_in
    Q = G.transpose(-1, -2) @ Rn                                                         # [T,5,3,3] bone-GT^T * IMU
    e = deg(ang(Q))
    Qm = Q.double().mean(0); U, S_, Vh = torch.linalg.svd(Qm); d = torch.det(U @ Vh)
    Qm = U @ torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1)) @ Vh
    return float(deg(psi)), e.mean(0), e.median(0).values, deg(ang(Qm)).float()


def gt_orient(pose):
    out = body.forward_kinematics(pose=pose.reshape(-1, 216), calc_mesh=False)[0].reshape(-1, 24, 3, 3)
    return out[:, JOINTS]


log("\n== B/C. real data: TC official calibration (RIM^T RIS RSB) vs SMPL GT bone orientation; DIP raw")
d = torch.load(P + "totalcapture_officalib.pt", map_location="cpu", weights_only=False)
pick = [n for n in ["s1_acting1", "s1_walking1", "s2_freestyle1", "s3_acting2", "s4_walking2", "s5_freestyle1", "s1_rom1", "s3_rom2"] if n in list(d["name"])]
variants = {"a  RIM^T RIS RSB  (documented)": lambda RIM, RIS, RSB: RIM.transpose(-1, -2) @ RIS @ RSB,
            "b  RIS RSB": lambda RIM, RIS, RSB: RIS @ RSB,
            "c  RIM RIS RSB": lambda RIM, RIS, RSB: RIM @ RIS @ RSB,
            "d  RIM^T RIS RSB^T": lambda RIM, RIS, RSB: RIM.transpose(-1, -2) @ RIS @ RSB.transpose(-1, -2),
            "e  RIM^T RIS (no RSB)": lambda RIM, RIS, RSB: RIM.transpose(-1, -2) @ RIS}
acc_stats = []
for nm, a, R, pose in sequences("totalcapture_officalib", ev.art):
    if nm not in pick:
        continue
    i = list(d["name"]).index(nm)
    RIM, RIS, RSB = d["RIM"][i], d["RIS"][i], d["RSB"][i]
    G = gt_orient(pose)                                                                  # [T,5,3,3]
    Rv = {k: f(RIM, RIS, RSB)[:, SENSORS] for k, f in variants.items()}
    assert torch.allclose(Rv["a  RIM^T RIS RSB  (documented)"], R, atol=1e-5), "variant a must equal what sequences() returns"
    log(f"-- {nm} ({len(pose)} frames)")
    for k, Rk in Rv.items():
        psi, em, emed, st = residual(Rk, G)
        log(f"   {k:34s} best yaw {psi:7.2f} deg | mean err/sensor {np.round(em.numpy(), 1)} | median {np.round(emed.numpy(), 1)} | static offset {np.round(st.numpy(), 1)}")
    an = a.norm(dim=-1)                                                                  # [T,5]  gravity-free world acceleration (m/s^2)
    acc_stats.append(an.median(0).values)
    log(f"   world acceleration: median |a| per sensor {np.round(an.median(0).values.numpy(), 2)} m/s^2 (should be ~1-3 if gravity is removed correctly; ~9.8 if not) ; max over frames {np.round(an.max(0).values.numpy(), 0)}")
    orth = float((R.transpose(-1, -2) @ R - torch.eye(3)).abs().max())
    log(f"   max |R^T R - I| = {orth:.1e}")
log("   DIP raw orientation (RIM=RSB=I), yaw-aligned residual vs GT:")
for nm, a, R, pose in list(sequences("dipimu", ev.art))[:3]:
    G = gt_orient(pose); psi, em, emed, st = residual(R, G)
    log(f"   {nm} ({len(pose)} fr) best yaw {psi:7.2f} | mean err {np.round(em.numpy(), 1)} | median {np.round(emed.numpy(), 1)} | static {np.round(st.numpy(), 1)} | median|a| {np.round(a.norm(dim=-1).median(0).values.numpy(), 2)}")

# ---------------------------------------------------------------- D
log("\n== D. inference path (CPU) on synthetic validation data")
from mobileposer.line9.data import load_identity_records
from mobileposer.line14.geom import scale_ident
store = RL.C.ChainStore(Path("data/line9_store_calib6"), Path("data/line9_store_pose6d"))
rec = load_identity_records(Path("data/pnp5_60a/pnp5"), Path("results/line9/data_audit.json"), ("validation",))["validation"]
Rmap = torch.load("data_local/line15_R_rot.pt", weights_only=False)["R"]
seen, chosen = set(), []
for r in rec:
    if r.frames >= 1500 and r.identity not in seen:
        seen.add(r.identity); chosen.append(r)
    if len(chosen) == 12:
        break
models = RL.load_model("i2s", 42, dev)
log("   loaded i2s seed 42; sequences:", len(chosen))


def prep(r):
    imu, jp = store.get(r.key); imu = imu.clone(); T = len(imu)
    ori = imu[..., 15:60].reshape(T, 5, 3, 3) @ Rmap[r.identity].transpose(-1, -2)[None]                 # orientation correction with the estimated offset
    imu = torch.cat((imu[..., :15] * scale_ident(r.identity, 0.5, 1.2), ori.reshape(T, 45)), dim=-1)    # scale hook of the training script
    return imu[:1500], jp[:1500, :, 3:]


imu0, pose6 = prep(chosen[0])
out = RL.infer(models, imu0)
# direct call on one stitched window: window starting at 126 contributes frames [157,220)
x = imu0[126:126 + 125][None]
with torch.no_grad():
    j = models[0](x, [125]); p = models[1](torch.cat((j, x), -1), [125])
dmax = float((p[0, 31:94] - out[157:220]).abs().max())
log(f"   stitching: max |direct window prediction - infer()| on frames 157..219 = {dmax:.2e} ->", "OK" if dmax < 1e-4 else "FAIL")
# window prediction invariance sanity: the same frame predicted from two different windows should be close (context effect only)
x2 = imu0[100:225][None]
with torch.no_grad():
    j2 = models[0](x2, [125]); p2 = models[1](torch.cat((j2, x2), -1), [125])
log(f"   same frames 157..219 predicted from window@100 vs window@126: mean abs diff {float((p2[0, 57:120] - p[0, 31:94]).abs().mean()):.4f} (context effect)")
errs = []
t0 = time.time()
for r in chosen:
    imu, p6 = prep(r)
    pred = RL.infer(models, imu)
    loc = RL.to_local(body, pred, dev)
    gt = AM.r6d_to_rotation_matrix(p6.reshape(-1, 6)).reshape(-1, 24, 3, 3)
    errs.append(ev(loc, gt))
E = torch.stack(errs)
log(f"   12 synthetic validation sequences (first 1500 frames) with the corrected-orientation model: SIP/Ang/Pos/Mesh = {np.round(E.mean(0).numpy(), 2)} (reference over all 49 identities, 3 seeds: SIP 11.09 / Ang 7.91 / Pos 4.69 / Mesh 5.73); {time.time() - t0:.0f}s")
