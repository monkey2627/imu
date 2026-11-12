"""Evaluate ONE trained variant under the primary protocol and store the numbers in results/variants/<name>.json.
The model-defining environment variables (MOBILEPOSER_ARCH / CTX / LEVER / CTX_USE / RELACC ...) must be set exactly as in
training and BEFORE python starts (they fix the input layout in config.py).

  python -m mobileposer.eval_variant --name kin_A --poser v3_kinA_poser --joints v3_kinA_joints
  python -m mobileposer.eval_variant --table base base_s2 kin kin_A ...        # print the comparison table

Tests: DIP-IMU test split (real IMU; body context = mean body), DIP + constant sensor offsets 20 deg, 68 held-out AMASS
stand->hold events (body context = true shape of the sequence), 10 own captures as given (mean-body context unless
--captures_ctx json is passed, e.g. made by body_from_image.py).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent))
sys.path.insert(0, str(_DIR.parent.parent))

from mobileposer.config import paths
from mobileposer.articulate.model import ParametricModel
from mobileposer.models.poser import Poser
from mobileposer.models.joints import Joints
from mobileposer.evaluate_canon_models import best_ckpt, GROUPS
from mobileposer.evaluate_drift_anchorpose import prepare_imu
from mobileposer.evaluate_dip_real import load_dip
from mobileposer.evaluate_causal_holds import BINS
from mobileposer.diagnose_dip_direction_vs_twist import vec_angle
from drift_eval_common import angle_between_rotmats

OUT = Path('results/variants')


class Pred:
    def __init__(self, poser, joints, dev):
        self.dev = dev
        self.ctx_on = os.environ.get('MOBILEPOSER_CTX') == '1'
        root = os.environ.get('MOBILEPOSER_CHECKPOINT_ROOT', 'checkpoints')
        self.poser = Poser.load_from_checkpoint(best_ckpt(str(Path(root) / poser)), map_location=dev).to(dev).eval()
        self.joints = Joints.load_from_checkpoint(best_ckpt(str(Path(root) / joints)), map_location=dev).to(dev).eval()

    @torch.no_grad()
    def __call__(self, acc, ori, ctx=None):
        imu = prepare_imu(acc, ori)
        if self.ctx_on:
            imu = torch.cat([imu, ctx.view(1, -1).expand(len(imu), -1)], dim=1)
        if getattr(self, "domain", None) is not None:
            imu = torch.cat([imu, self.domain.view(1, -1).expand(len(imu), -1)], dim=1)
        imu = imu.to(self.dev).unsqueeze(0)
        T = imu.shape[1]
        pj = self.joints(imu, [T])
        return self.poser._reduced_global_to_full(self.poser(torch.cat((pj, imu), -1), [T])).cpu().reshape(-1, 24, 3, 3)


class WinPred:
    """sliding-window inference at the training window length (overlap, keep the centre): for models that do not extrapolate to long sequences"""
    def __init__(self, fn, W, stride):
        self.fn, self.W, self.stride = fn, W, stride

    def __call__(self, acc, ori, ctx=None):
        T, W, st = len(acc), self.W, self.stride
        if T <= W:
            return self.fn(acc, ori, ctx)
        out = torch.zeros(T, 24, 3, 3)
        m = (W - st) // 2
        starts = list(range(0, T - W, st)) + [T - W]
        for k, s in enumerate(starts):
            p = self.fn(acc[s:s + W], ori[s:s + W], ctx)
            a = 0 if k == 0 else s + m
            b = T if k == len(starts) - 1 else s + W - m
            out[a:b] = p[a - s:b - s]
        return out


def dip_metrics(pred_fn, body, ctx, offset_deg=0.0):
    import mobileposer.articulate as art
    rows, wts = [], []
    for si, s in enumerate(load_dip(body)):
        T = len(s['pose'])
        ori = s['ori']
        if offset_deg > 0:
            g = torch.Generator().manual_seed(1000 + si)
            axis = torch.nn.functional.normalize(torch.randn(5, 3, generator=g), dim=-1)
            ang = torch.randn(5, generator=g).abs() * (offset_deg * np.pi / 180)
            ori = ori @ art.math.axis_angle_to_rotation_matrix(axis * ang.unsqueeze(-1)).view(5, 3, 3)
        G, J = body.forward_kinematics(s['pose'], torch.ones(10), tran=None, calc_mesh=False)
        G, J = G.reshape(T, 24, 3, 3), J.reshape(T, 24, 3)
        Gp, Jp = body.forward_kinematics(pred_fn(s['acc'], ori, ctx)[:T], torch.ones(10), tran=None, calc_mesh=False)
        Gp, Jp = Gp.reshape(T, 24, 3, 3), Jp.reshape(T, 24, 3)
        e = angle_between_rotmats(Gp, G).mean(0).numpy()
        da = (vec_angle(J[:, 18] - J[:, 16], Jp[:, 18] - Jp[:, 16]).mean() + vec_angle(J[:, 19] - J[:, 17], Jp[:, 19] - Jp[:, 17]).mean()) / 2
        dt = (vec_angle(J[:, 4] - J[:, 1], Jp[:, 4] - Jp[:, 1]).mean() + vec_angle(J[:, 5] - J[:, 2], Jp[:, 5] - Jp[:, 2]).mean()) / 2
        cm = (Jp - J).norm(dim=-1)[:, [1, 2, 4, 5, 16, 17, 18, 19]].mean().item() * 100
        rows.append([float(e[js].mean()) for js in GROUPS.values()] + [da.item(), dt.item(), cm]); wts.append(T)
    return np.average(rows, axis=0, weights=wts).tolist()


def run(args):
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    body = ParametricModel(paths.smpl_file)
    from mobileposer.body_ctx import body_context
    fn = Pred(args.poser, args.joints, dev)
    if args.window:
        fn = WinPred(fn, args.window, args.window // 2)
    mean_ctx = body_context(torch.zeros(10))
    res = {'name': args.name}
    res['dip'] = dip_metrics(fn, body, mean_ctx)
    res['dip_off20'] = dip_metrics(fn, body, mean_ctx, 20.0)
    blob = torch.load('data/hold_events_heldout.pt', map_location='cpu', weights_only=False)
    rows = []
    for sub, i, t0, end, T in blob['events']:
        h = blob['held'][sub]
        j = h['indices'].index(i)
        acc, ori, pose = h['acc'][j][:3000], h['ori'][j][:3000], h['pose'][j].reshape(-1, 24, 3, 3)[:3000]
        T = len(pose); end = min(end, T)
        shp = h['shape'][j] if 'shape' in h else torch.zeros(10)
        G, _ = body.forward_kinematics(pose, torch.ones(10), tran=None, calc_mesh=False)
        Gp, _ = body.forward_kinematics(fn(acc, ori, body_context(shp))[:T], torch.ones(10), tran=None, calc_mesh=False)
        terr = angle_between_rotmats(Gp.reshape(T, 24, 3, 3)[:, 1], G.reshape(T, 24, 3, 3)[:, 1]).numpy()
        rows.append([terr[t0 + int(lo * 30):min(end, t0 + int(hi * 30))].mean() if min(end, t0 + int(hi * 30)) - (t0 + int(lo * 30)) >= 10 else np.nan for lo, hi in BINS])
    res['events'] = np.nanmean(np.array(rows), axis=0).tolist()
    cctx = mean_ctx
    if args.captures_ctx:
        d = json.load(open(args.captures_ctx))
        cctx = mean_ctx.clone(); cctx[12:22] = torch.tensor(d['lengths_m'])
    caps = []
    for cap in sorted(Path('results/ours_multi_finetune/captures').iterdir(), key=lambda p: int(p.name.split('_')[0])):
        t = torch.load(cap / 'targets.pt', map_location='cpu', weights_only=False)
        gt = t['pose'].reshape(-1, 24, 3, 3)
        d = torch.load(cap / 'inputs/wrists_shanks_waist/ours1.pt', map_location='cpu', weights_only=False)
        T = min(len(d['acc'][0]), len(gt))
        Gp, _ = body.forward_kinematics(fn(d['acc'][0][:T], d['ori'][0][:T], cctx)[:T], torch.ones(10), tran=None, calc_mesh=False)
        Gq, _ = body.forward_kinematics(gt[:T], torch.ones(10), tran=None, calc_mesh=False)
        e = angle_between_rotmats(Gp.reshape(T, 24, 3, 3), Gq.reshape(T, 24, 3, 3)).mean(0).numpy()
        caps.append([float(e[js].mean()) for js in GROUPS.values()])
    res['caps'] = np.mean(caps, axis=0).tolist()
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f'{args.name}.json', 'w'))
    print(json.dumps(res))


def table(names):
    rs = [json.load(open(OUT / f'{n}.json')) for n in names if (OUT / f'{n}.json').exists()]
    f = lambda v: ' '.join(f'{x:6.1f}' for x in v)
    print('DIP clean: pelvis sensed thigh spine neck/head upperarm | ua-dir th-dir unsens-cm')
    for r in rs: print(f'  {r["name"]:14s} {f(r["dip"])}')
    print('DIP + 20 deg sensor offsets:')
    for r in rs: print(f'  {r["name"]:14s} {f(r["dip_off20"])}')
    print('AMASS held-out events, thigh error by time bin (0-4, 4-8, 8-16, 16+ s):')
    for r in rs: print(f'  {r["name"]:14s} {f(r["events"])}')
    print('own captures as given: pelvis sensed thigh spine neck/head upperarm:')
    for r in rs: print(f'  {r["name"]:14s} {f(r["caps"])}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name'); ap.add_argument('--poser'); ap.add_argument('--joints')
    ap.add_argument('--captures_ctx', default=None)
    ap.add_argument('--window', type=int, default=0, help='sliding-window inference length (0 = whole sequence)')
    ap.add_argument('--table', nargs='*')
    a = ap.parse_args()
    table(a.table) if a.table is not None else run(a)
