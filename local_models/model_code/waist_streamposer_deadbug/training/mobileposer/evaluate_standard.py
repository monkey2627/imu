"""STANDARD benchmark evaluation (the protocol of TransPose / PIP / PNP / GlobalPose) of our 5-IMU models.

Test sets: the preprocessed REAL-IMU sets distributed with PNP (data/test_datasets): DIP-IMU test split (19 sequences),
TotalCapture with the DIP calibration and with the official calibration (45 sequences each).  Raw measurements are
converted exactly as in PNP/test.py:  bone orientation R = RIM^T RIS RSB,  world linear acceleration a = RIM^T RIS aS + g.
The six sensors are [L_LowArm, R_LowArm, L_LowLeg, R_LowLeg, Head, Pelvis]; our 5-IMU layout uses [0,1,2,3,5]
(both lower arms, both lower legs, pelvis: the standard 6 IMUs WITHOUT the head).  Sequences are 60 fps; our models run at
30 fps (every 2nd frame), the metrics are frame-wise so the comparison is valid (jitter is not reported).

Metrics (PNP/test.py ReducedPoseEvaluator, ignored joints set to identity): SIP error (global orientation error of the
hips and shoulders), angular error (global, all joints), joint position error (cm, root-position aligned), vertex error (cm).
Mean over sequences of per-sequence means, as in PNP/test.py.

Inference modes:  full  = whole sequence (offline, bidirectional: NOT comparable with the real-time baselines)
                  win46 = 40 past + 5 future frames per output frame (MobilePoser's real-time setting)

  python -m mobileposer.evaluate_standard --name base --poser v3_base_poser --joints v3_base_joints --mode win46
  python -m mobileposer.evaluate_standard --table
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent))
sys.path.insert(0, str(_DIR.parent.parent))

PNP_DIR = Path(os.environ.get("PNP_DIR", str(Path(__file__).resolve().parents[1] / "third_party" / "PNP")))
SENSORS = [0, 1, 2, 3, 4, 5] if os.environ.get('MOBILEPOSER_NSENS') == '6' else [0, 1, 2, 3, 5]
FPS = int(os.environ.get('MOBILEPOSER_FPS', 30))
STEP = 60 // FPS
DATASETS = ['dipimu', 'totalcapture_dipcalib', 'totalcapture_officalib']
OUT = Path(os.environ.get('MOBILEPOSER_RESULTS_ROOT', 'results/standard'))

# reference numbers copied from GlobalPose (SIGGRAPH 2025) Table 1, "Local" columns: SIP / Ang / Pos / Mesh
REFERENCE = {
    'dipimu': {'TransPose': (17.06, 8.86, 6.03, 7.17), 'PIP': (15.33, 8.78, 5.12, 6.02), 'PNP': (13.71, 8.75, 4.97, 5.77),
               'DynaIP-XD': (13.75, 7.05, 4.97, 5.98), 'GlobalPose': (13.55, 8.47, 4.65, 5.41)},
    'totalcapture_dipcalib': {'TransPose': (16.60, 12.90, 6.56, 7.43), 'PIP': (12.93, 12.04, 5.61, 6.51), 'PNP': (10.89, 10.45, 4.74, 5.45),
                              'GlobalPose': (9.81, 9.99, 4.25, 4.94)},
    'totalcapture_officalib': {'TransPose': (18.12, 14.91, 7.10, 8.09), 'PIP': (14.52, 13.85, 6.22, 7.21), 'PNP': (11.35, 11.10, 4.89, 5.60),
                               'GlobalPose': (10.17, 10.16, 4.31, 4.96)},
}


def load_pnp_articulate():
    import types
    for m in ("quaternion",):                      # only needed by articulate.filter, not by the evaluator
        if m not in sys.modules:
            try:
                __import__(m)
            except ImportError:
                sys.modules[m] = types.ModuleType(m)
    spec = importlib.util.spec_from_file_location('pnp_articulate', PNP_DIR / 'articulate' / '__init__.py',
                                                  submodule_search_locations=[str(PNP_DIR / 'articulate')])
    mod = importlib.util.module_from_spec(spec)
    sys.modules['pnp_articulate'] = mod
    spec.loader.exec_module(mod)
    return mod


class StdEvaluator:
    def __init__(self, device):
        art = load_pnp_articulate()
        self.art = art
        self.fn = art.FullMotionEvaluator(str(PNP_DIR / 'models' / 'SMPL_male.pkl'), joint_mask=torch.tensor([1, 2, 16, 17]), fps=FPS, device=device)
        self.ignored = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])
        self.device = device

    def __call__(self, pose_p, pose_t):
        pose_p, pose_t = pose_p.clone().view(-1, 24, 3, 3).to(self.device), pose_t.clone().view(-1, 24, 3, 3).to(self.device)
        pose_p[:, self.ignored] = torch.eye(3, device=self.device)
        pose_t[:, self.ignored] = torch.eye(3, device=self.device)
        e = self.fn(pose_p=pose_p, pose_t=pose_t)
        return torch.stack([e[9], e[3], e[0] * 100, e[1] * 100])[:, 0].cpu()       # SIP, Ang, Pos(cm), Mesh(cm): means


def sequences(name, art):
    d = torch.load(PNP_DIR / 'data' / 'test_datasets' / f'{name}.pt', map_location='cpu')
    g = torch.tensor([0, -9.8, 0])
    for i in range(len(d['pose'])):
        RIM, RIS, RSB = d['RIM'][i], d['RIS'][i], d['RSB'][i]
        R = RIM.transpose(1, 2).matmul(RIS).matmul(RSB)                                   # [T,6,3,3]
        a = RIM.transpose(1, 2).matmul(RIS).matmul(d['aS'][i].unsqueeze(-1)).squeeze(-1) + g   # [T,6,3]
        pose = art.math.axis_angle_to_rotation_matrix(d['pose'][i].reshape(-1, 3)).view(-1, 24, 3, 3)
        yield d['name'][i], a[::STEP, SENSORS].contiguous(), R[::STEP, SENSORS].contiguous(), pose[::STEP].contiguous()


@torch.no_grad()
def online_predict(pred, acc, ori, ctx, past=None, fut=None, bs=256):
    past, fut = past or 40 * FPS // 30, fut or 5 * FPS // 30
    """window [t-past, t+fut] per frame, batched; output of frame t read at its index inside the window"""
    from mobileposer.evaluate_drift_anchorpose import prepare_imu
    T = len(acc)
    imu = prepare_imu(acc, ori)
    if pred.ctx_on:
        imu = torch.cat([imu, ctx.view(1, -1).expand(T, -1)], dim=1)
    if getattr(pred, "domain", None) is not None:
        imu = torch.cat([imu, pred.domain.view(1, -1).expand(T, -1)], dim=1)
    imu = imu.to(pred.dev)
    out = []
    for s in range(0, T, bs):
        ts = list(range(s, min(T, s + bs)))
        starts = [max(0, t - past) for t in ts]
        ends = [min(T, t + fut + 1) for t in ts]
        L = [e - b for b, e in zip(starts, ends)]
        batch = torch.zeros(len(ts), max(L), imu.shape[1], device=pred.dev)
        for k, (b, e) in enumerate(zip(starts, ends)):
            batch[k, :e - b] = imu[b:e]
        pj = pred.joints(batch, L)
        pose = pred.poser(torch.cat((pj, batch), -1), L)                                    # [B,Smax,96]
        idx = torch.tensor([t - b for t, b in zip(ts, starts)], device=pred.dev)
        sel = pose[torch.arange(len(ts), device=pred.dev), idx]                             # [B,96]
        out.append(sel)
    sel = torch.cat(out)                                                                      # [T,96]
    return pred.poser._reduced_global_to_full(sel).cpu().reshape(-1, 24, 3, 3)


def run(args):
    from mobileposer.eval_variant import Pred
    from mobileposer.body_ctx import body_context
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pred = Pred(args.poser, args.joints, dev)
    ctx = body_context(torch.zeros(10))
    ev = StdEvaluator(dev)
    res = {'name': args.name, 'mode': args.mode}
    for ds in DATASETS:
        errs = []
        if os.environ.get("MOBILEPOSER_DOMAIN") == "1":      # annotation-convention token: DIP-style labels for the DIP test set, AMASS-style for TotalCapture
            pred.domain = torch.tensor([0.0, 1.0] if ds == "dipimu" else [1.0, 0.0])
        for name, a, R, pose in sequences(ds, ev.art):
            if args.calib:                                          # self-calibration: remove the estimated sensor-to-bone offsets before anything else
                from mobileposer import calibnet
                if not hasattr(run, "cnet"):
                    run.cnet = calibnet.load(args.calib, dev)
                C = calibnet.estimate(run.cnet[0], run.cnet[1], a / 30.0, R, upto=(None if args.calib_mode == "whole" else 10 * FPS), device=dev)
                R = calibnet.correct(R, C)
            from mobileposer.frames import frame_kind, frame_slot, frame_rot
            kind = frame_kind()
            if kind != "world":
                Rf = frame_rot(R[:, frame_slot()], kind)
                RfT = Rf.transpose(-1, -2)
                a = (RfT.unsqueeze(1) @ a.unsqueeze(-1)).squeeze(-1)
                R = RfT.unsqueeze(1) @ R
            p = pred(a, R, ctx)[:len(pose)] if args.mode == "full" else online_predict(pred, a, R, ctx)
            if kind != "world":
                p = p.clone()
                p[:, 0] = Rf[:len(p)] @ p[:, 0]
            errs.append(ev(p, pose))
            print('  ', ds, name, np.round(errs[-1].numpy(), 2), flush=True)
        res[ds] = torch.stack(errs).mean(0).tolist()
        print(ds, 'SIP/Ang/Pos/Mesh', np.round(res[ds], 2), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f'{args.name}_{args.mode}.json', 'w'))


def table(names):
    print('SIP(deg) / Angular(deg) / Position(cm) / Mesh(cm); ours = 5 IMU without head, references = 6 IMU (published numbers)')
    for ds in DATASETS:
        print(f'\n{ds}')
        for k, v in REFERENCE[ds].items():
            print(f'  {k:22s}' + '  '.join(f'{x:6.2f}' for x in v) + '   (published, 6 IMU)')
        for n in names:
            f = OUT / f'{n}.json'
            if f.exists():
                print(f'  {n:22s}' + '  '.join(f'{x:6.2f}' for x in json.load(open(f))[ds]) + '   (ours, 5 IMU)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name'); ap.add_argument('--poser'); ap.add_argument('--joints')
    ap.add_argument('--mode', default='win46', choices=['full', 'win46'])
    ap.add_argument("--table", nargs="*")
    ap.add_argument("--calib", default=None, help="CalibNet checkpoint: estimate and remove the sensor-to-bone offsets (innovation 2)")
    ap.add_argument("--calib_mode", default="whole", choices=["whole", "first10"], help="estimate from the whole sequence or only the first 10 s")
    a = ap.parse_args()
    table(a.table) if a.table is not None else run(a)
