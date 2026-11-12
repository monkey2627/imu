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

Inference modes:  causal40 = 40 past + current frame, zero future frames (strictly causal, primary)
                  win46 = 40 past + 5 future frames per output frame (low-latency look-ahead comparison only)
                  full is intentionally disabled by the line14 strict-real-time protocol.

  python -m mobileposer.evaluate_standard --name causal_pnp5 --poser <poser-checkpoint> --joints <joints-checkpoint> --mode causal40
  python -m mobileposer.evaluate_standard --table
"""
import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from mobileposer.line14.layout import PNP5_LABELS, PNP5_SMPL_JOINTS, PNP5_SURFACE_VERTICES, PNP6_KEEP_INDICES

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent))
sys.path.insert(0, str(_DIR.parent.parent))

PNP_DIR = Path(os.environ.get("PNP_DIR", str(Path(__file__).resolve().parents[1] / "third_party" / "PNP")))
DATASET_DIR = Path(os.environ.get("MOTION_DATASET_DIR", str(Path(__file__).resolve().parents[3] / "dataset")))
SENSORS = list(range(6)) if os.environ.get('MOBILEPOSER_NSENS') == '6' else list(PNP6_KEEP_INDICES)
FPS = int(os.environ.get('MOBILEPOSER_FPS', 30))
STEP = 60 // FPS
DATASETS = ['dipimu', 'totalcapture_dipcalib', 'totalcapture_officalib']
OUT = Path('results/standard')


def prepare_imu(acc, ori):
    """Pack the PNP5 inputs exactly like MobilePoser training: acc/30 | R."""
    if acc.ndim != 3 or acc.shape[1:] != (len(PNP5_LABELS), 3):
        raise ValueError(f"acc must have shape [T,5,3], got {tuple(acc.shape)}")
    if ori.shape != (acc.shape[0], len(PNP5_LABELS), 3, 3):
        raise ValueError(f"ori must have shape [T,5,3,3], got {tuple(ori.shape)}")
    return torch.cat((acc.float().flatten(1) / 30.0, ori.float().flatten(1)), dim=1)


def resolve_checkpoint(value: str) -> Path:
    """Resolve an explicit Lightning checkpoint path or a named checkpoint dir."""
    candidate = Path(value).expanduser()
    if not candidate.exists():
        candidate = Path("checkpoints") / value
    if candidate.is_file() and candidate.suffix == ".ckpt":
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"checkpoint file/directory not found: {candidate}")
    files = list(candidate.rglob("*.ckpt"))
    if not files:
        raise FileNotFoundError(f"no Lightning .ckpt files under {candidate}")

    def epoch(path: Path) -> int:
        match = re.search(r"epoch=(\d+)", path.name)
        return int(match.group(1)) if match else -1

    def val_loss(path: Path) -> float:
        match = re.search(r"validation_step_loss=([0-9.]+)", path.name)
        return float(match.group(1).rstrip(".")) if match else float("inf")

    if all(not np.isfinite(val_loss(path)) for path in files):
        return max(files, key=epoch)
    return min(files, key=val_loss)


class StandardPredictor:
    """Load the standard MobilePoser pair and audit its actual architecture."""

    def __init__(self, poser_path: str, joints_path: str, device: torch.device):
        from mobileposer.models.joints import Joints
        from mobileposer.models.poser import Poser

        self.dev = device
        self.ctx_on = os.environ.get("MOBILEPOSER_CTX") == "1"
        poser_ckpt = resolve_checkpoint(poser_path)
        joints_ckpt = resolve_checkpoint(joints_path)
        self.poser = Poser.load_from_checkpoint(str(poser_ckpt), map_location=device).to(device).eval()
        self.joints = Joints.load_from_checkpoint(str(joints_ckpt), map_location=device).to(device).eval()
        self.checkpoints = {"poser": str(poser_ckpt), "joints": str(joints_ckpt)}

    def assert_strict_causal_architecture(self) -> None:
        """Reject architecture paths that can inspect beyond each target frame."""
        from mobileposer.models.rnn import RNN

        if not isinstance(self.joints.joints, RNN) or not isinstance(self.poser.pose, RNN):
            raise RuntimeError("causal40 requires RNN Joints and Poser; other temporal architectures need a separate audit")
        for label, model in (("joints", self.joints), ("poser", self.poser)):
            if getattr(model, "slowfast", False):
                raise RuntimeError(f"{label} slowfast path is not part of the audited causal40 baseline")
        # A bidirectional RNN is still causal when each call ends at the target
        # frame and only that window's final output is selected. Future
        # invariance is checked against the actual inference path before scoring.

    @torch.no_grad()
    def attach_context(self, imu: torch.Tensor, ctx: Optional[torch.Tensor]) -> torch.Tensor:
        if self.ctx_on:
            if ctx is None:
                raise ValueError("this checkpoint requires the body-context vector")
            imu = torch.cat([imu, ctx.view(1, -1).expand(len(imu), -1)], dim=1)
        if getattr(self, "domain", None) is not None:
            imu = torch.cat([imu, self.domain.view(1, -1).expand(len(imu), -1)], dim=1)
        return imu

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
    data_path = DATASET_DIR / 'processed' / 'TotalCapture' / f'{name}.pt' if name.startswith('totalcapture_') else PNP_DIR / 'data' / 'test_datasets' / f'{name}.pt'; d = torch.load(data_path, map_location='cpu')
    g = torch.tensor([0, -9.8, 0])
    for i in range(len(d['pose'])):
        RIM, RIS, RSB = d['RIM'][i], d['RIS'][i], d['RSB'][i]
        R = RIM.transpose(1, 2).matmul(RIS).matmul(RSB)                                   # [T,6,3,3]
        a = RIM.transpose(1, 2).matmul(RIS).matmul(d['aS'][i].unsqueeze(-1)).squeeze(-1) + g   # [T,6,3]
        pose = art.math.axis_angle_to_rotation_matrix(d['pose'][i].reshape(-1, 3)).view(-1, 24, 3, 3)
        yield d['name'][i], a[::STEP, SENSORS].contiguous(), R[::STEP, SENSORS].contiguous(), pose[::STEP].contiguous()


@torch.no_grad()
def online_predict(pred, acc, ori, ctx, past=None, fut=None, bs=256):
    past = 40 * FPS // 30 if past is None else int(past)
    fut = 5 * FPS // 30 if fut is None else int(fut)
    if fut < 0:
        raise ValueError("future-frame count must be non-negative")
    """Window [t-past, t+fut] per frame; causal mode selects the endpoint at t."""
    T = len(acc)
    imu = prepare_imu(acc, ori)
    imu = pred.attach_context(imu, ctx)
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


@torch.no_grad()
def check_future_invariance(pred, ctx, *, split=6, steps=12):
    """Exercise the actual causal40 inference path before evaluating a checkpoint."""
    if not (0 < split < steps):
        raise ValueError("split must be inside the synthetic sequence")
    gen = torch.Generator().manual_seed(14027)
    acc = torch.randn(steps, len(PNP5_LABELS), 3, generator=gen)
    eye = torch.eye(3).view(1, 1, 3, 3).expand(steps, len(PNP5_LABELS), 3, 3).clone()
    changed_acc, changed_ori = acc.clone(), eye.clone()
    changed_acc[split:] = torch.randn(changed_acc[split:].shape, generator=gen) * 100
    changed_ori[split:] = torch.randn(changed_ori[split:].shape, generator=gen)
    base = online_predict(pred, acc, eye, ctx, past=40, fut=0, bs=steps)
    perturbed = online_predict(pred, changed_acc, changed_ori, ctx, past=40, fut=0, bs=steps)
    error = float((base[:split] - perturbed[:split]).abs().amax())
    if error > 1e-6:
        raise RuntimeError(f"checkpoint failed future-frame invariance test: max prefix difference {error:.3g}")
    return error


def run(args):
    if tuple(SENSORS) != tuple(PNP6_KEEP_INDICES):
        raise RuntimeError("line14 standard evaluation is fixed to PNP5 sensors [0,1,2,3,5]; unset MOBILEPOSER_NSENS=6")
    from mobileposer.body_ctx import body_context
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pred = StandardPredictor(args.poser, args.joints, dev)
    if args.mode == "causal40":
        pred.assert_strict_causal_architecture()
    ctx = body_context(torch.zeros(10))
    ev = StdEvaluator(dev)
    res = {'name': args.name, 'mode': args.mode, 'checkpoints': pred.checkpoints,
           'sensor_layout': {'labels': list(PNP5_LABELS), 'smpl_joints': list(PNP5_SMPL_JOINTS),
                             'surface_vertices': list(PNP5_SURFACE_VERTICES), 'pnp6_test_indices': list(PNP6_KEEP_INDICES),
                             'sensor_count': len(SENSORS)},
           'temporal_protocol': {'model_fps': FPS,
                                 'future_frames': 0 if args.mode == 'causal40' else 5 * FPS // 30,
                                 'lookahead_ms': 0.0 if args.mode == 'causal40' else 1000.0 * (5 * FPS // 30) / FPS}}
    causality_error = None
    for ds in DATASETS:
        errs = []
        if os.environ.get("MOBILEPOSER_DOMAIN") == "1":      # annotation-convention token: DIP-style labels for the DIP test set, AMASS-style for TotalCapture
            pred.domain = torch.tensor([0.0, 1.0] if ds == "dipimu" else [1.0, 0.0])
        if args.mode == "causal40" and causality_error is None:
            causality_error = check_future_invariance(pred, ctx)
        for name, a, R, pose in sequences(ds, ev.art):
            from mobileposer.frames import frame_kind, frame_slot, frame_rot
            kind = frame_kind()
            if kind != "world":
                Rf = frame_rot(R[:, frame_slot()], kind)
                RfT = Rf.transpose(-1, -2)
                a = (RfT.unsqueeze(1) @ a.unsqueeze(-1)).squeeze(-1)
                R = RfT.unsqueeze(1) @ R
            if args.mode == "full":
                raise RuntimeError("full mode is disabled by the line14 strict-real-time protocol")
            fut = 0 if args.mode == "causal40" else None
            p = online_predict(pred, a, R, ctx, fut=fut)
            if kind != "world":
                p = p.clone()
                p[:, 0] = Rf[:len(p)] @ p[:, 0]
            errs.append(ev(p, pose))
            print('  ', ds, name, np.round(errs[-1].numpy(), 2), flush=True)
        res[ds] = torch.stack(errs).mean(0).tolist()
        print(ds, 'SIP/Ang/Pos/Mesh', np.round(res[ds], 2), flush=True)
    if causality_error is not None:
        res['future_invariance_max_abs_error'] = causality_error
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f'{args.name}_{args.mode}.json', 'w'))


def table(names):
    print('SIP(deg) / Angular(deg) / Position(cm) / Mesh(cm); ours = 5 IMU without head, references = 6 IMU (published numbers)')
    for ds in DATASETS:
        print(f'\n{ds}')
        for k, v in REFERENCE[ds].items():
            print(f'  {k:22s}' + '  '.join(f'{x:6.2f}' for x in v) + '   (published, 6 IMU)')
        for n in names:
            f = Path(n)
            if not f.is_file():
                f = OUT / (n if n.endswith('.json') else f'{n}_causal40.json')
            if f.exists():
                print(f'  {n:22s}' + '  '.join(f'{x:6.2f}' for x in json.load(open(f))[ds]) + '   (ours, 5 IMU)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name'); ap.add_argument('--poser'); ap.add_argument('--joints')
    ap.add_argument('--mode', default='causal40', choices=['causal40', 'win46'])
    ap.add_argument("--table", nargs="*")
    a = ap.parse_args()
    table(a.table) if a.table is not None else run(a)
