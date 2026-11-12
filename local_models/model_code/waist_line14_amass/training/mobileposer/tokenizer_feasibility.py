"""Feasibility check for part-wise discrete pose tokens.

Per-frame VQ tokenizer per body part on parent-relative rotations (heading-invariant). The root orientation is NOT tokenized
(the pelvis IMU measures it). Decoded parts are composed along the kinematic chain with the GT root, and the error is the
global joint-angle error of the 16 reduced joints (same quantity as the Poser output / SIP-Ang convention).
Reports an unquantized autoencoder (same architecture, no codebook) as the lower bound.

    OMP_NUM_THREADS=8 python -m mobileposer.tokenizer_feasibility --data data/pnp5_60a/pnp5 --K 512 --steps 6000
"""
import argparse, glob, math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

from mobileposer.articulate.math import r6d_to_rotation_matrix, rotation_matrix_to_r6d, axis_angle_to_rotation_matrix

from mobileposer.models.pose_tokenizer import PARENT, REDUCED, PARTS, VQ, PartAE


def load_frames(folder, stride, hold_every, seg=1):
    """Local rotation segments [N,seg,24,3,3] (start every `stride` frames) of train / held-out sequences (hold-out is by sequence)."""
    tr, te = [], []
    seq_id = 0
    for f in sorted(glob.glob(os.path.join(folder, '*.pt'))):
        d = torch.load(f)
        for pose in d['pose']:
            pose = pose.float()
            T = pose.shape[0]
            if pose.numel() == T * 72:
                pose = axis_angle_to_rotation_matrix(pose.reshape(-1, 3)).reshape(T, 24, 3, 3)
            else:
                pose = pose.reshape(T, 24, 3, 3)
            if T >= seg:
                idx = torch.arange(0, T - seg + 1, stride)[:, None] + torch.arange(seg)[None]
                (te if seq_id % hold_every == 0 else tr).append(pose[idx].clone())
            seq_id += 1
        print(f'{os.path.basename(f)}: seqs so far {seq_id}', flush=True)
    return torch.cat(tr), torch.cat(te)


def to_global(local):
    glob_ = [None] * 24
    for j in range(24):
        glob_[j] = local[:, j] if PARENT[j] < 0 else glob_[PARENT[j]] @ local[:, j]
    return torch.stack(glob_, 1)


def angle_deg(Ra, Rb):
    cos = ((Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6)) * 180 / math.pi


def part_feats(local, joints):
    return rotation_matrix_to_r6d(local[:, :, joints].reshape(-1, 3, 3)).reshape(len(local), -1)


def decode_to_local(models, local, K_on):
    """Replace tokenized joints of `local` by the decoded ones; returns new local rotations."""
    out = local.clone()
    for name, joints in PARTS.items():
        m = models[name]
        m.eval()
        with torch.no_grad():
            rec, _, _ = m(part_feats(local, joints))
        out[:, :, joints] = r6d_to_rotation_matrix(rec.reshape(-1, 6)).reshape(len(local), local.shape[1], len(joints), 3, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/pnp5_60a/pnp5')
    ap.add_argument('--K', type=int, nargs='+', default=[512])
    ap.add_argument('--nq', type=int, default=1)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--bs', type=int, default=4096)
    ap.add_argument('--stride', type=int, default=6)
    ap.add_argument('--hold_every', type=int, default=10)
    ap.add_argument('--seg', type=int, default=1, help='frames per token segment')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='results/tokenizer_feasibility.txt')
    ap.add_argument('--save', default=None, help='save the last (largest-K) tokenizer here')
    a = ap.parse_args()

    tr, te = load_frames(a.data, a.stride, a.hold_every, a.seg)
    print(f'train frames {len(tr)}  held-out frames {len(te)}', flush=True)
    tr, te = tr.to(a.device), te.to(a.device)
    te_f = te.flatten(0, 1)
    te_glob = to_global(te_f)
    lines = []

    for K in [0] + a.K:                                                          # K=0: continuous autoencoder lower bound
        models = {}
        t0 = time.time()
        for name, joints in PARTS.items():
            m = PartAE(len(joints) * a.seg, K, nq=a.nq).to(a.device)
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=1e-5)
            x_all = part_feats(tr, joints)
            for step in range(a.steps):
                m.train()
                x = x_all[torch.randint(len(x_all), (a.bs,), device=a.device)]
                rec, commit, _ = m(x)
                loss = F.mse_loss(rec, x) + 0.25 * commit
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            models[name] = m
        rec_local = decode_to_local(models, te, K)
        rec_local = rec_local.flatten(0, 1)
        rec_glob = to_global(rec_local)
        err = angle_deg(rec_glob[:, REDUCED], te_glob[:, REDUCED])               # [N*seg,16]
        loc_err = angle_deg(rec_local[:, 1:], te_f[:, 1:])                         # local rotation error
        names = {0: 'pelvis', 1: 'L_hip', 2: 'R_hip', 3: 'spine1', 4: 'L_knee', 5: 'R_knee', 6: 'spine2', 9: 'spine3',
                 12: 'neck', 13: 'L_collar', 14: 'R_collar', 15: 'head', 16: 'L_shoulder', 17: 'R_shoulder', 18: 'L_elbow', 19: 'R_elbow'}
        per = {names[j]: err[:, i].mean().item() for i, j in enumerate(REDUCED)}
        sip = (err[:, [1, 2, 12, 13]].mean().item())                              # hips + shoulders (reduced slots 1,2,12,13)
        usage = ''
        if K > 0:
            usage = ' codes used: ' + ', '.join(f'{n}={(m.vq[0].cnt > 1e-2).sum().item()}' for n, m in models.items())
        head = f'K={K if K else "continuous"} nq={a.nq if K else "-"}  global-angle err: mean16={err.mean():.2f}  SIP-joints={sip:.2f}  local mean={loc_err.mean():.2f}  ({time.time() - t0:.0f}s){usage}'
        detail = '   ' + '  '.join(f'{k}={v:.2f}' for k, v in per.items())
        print(head, flush=True); print(detail, flush=True)
        lines += [head, detail]

    if a.save and K > 0:
        torch.save({'cfg': {'K': K, 'nq': a.nq, 'dim': 64, 'hid': 512, 'T': a.seg}, 'state': {n: m.state_dict() for n, m in models.items()}}, a.save)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'a') as f:
        f.write(f'--- {time.ctime()} seg={a.seg} stride={a.stride} steps={a.steps} nq={a.nq} train={len(tr)} test={len(te)}\n' + '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
