"""End-to-end fine-tuning of the Joints -> Poser cascade.

Diagnostic (cascade_diag): the same Poser fed with ground-truth joint positions reaches SIP 8.6 / Ang 10.6 on TotalCapture (DIP calib.) instead of
13.9 / 13.2 with the Joints module's predictions - the pose error is dominated by the quality of the joint-position estimates, and the Poser is
trained on noisy GT joints (sigma 0.04 m) that are statistically unlike the structured errors of the Joints module at test time.
Here the Poser is trained on the Joints module's own (differentiable) predictions and both are optimised with the POSE loss
(rotation MSE + joint-position loss through the SMPL forward kinematics) + a Joints reconstruction term, starting from the separately trained modules.

  python -m mobileposer.train_e2e --poser pnp5_base_poser --joints pnp5_base_joints --out pnp5_e2e --epochs 15 --lr 2e-4
Writes checkpoints/<out>_poser|joints/1/<module>/epoch=*-validation_step_loss=*.ckpt (same layout as train.py).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent))
sys.path.insert(0, str(_DIR.parent.parent))

import mobileposer.articulate as art
from mobileposer.config import joint_set
from mobileposer.data import PoseDataset, pad_seq
from mobileposer.evaluate_canon_models import best_ckpt
from mobileposer.models.joints import Joints
from mobileposer.models.poser import Poser


def losses(poser, joints, batch, dev, w_joint, noise_mix):
    (imu, lens), (out, _) = batch
    imu = imu.to(dev)
    tgt_pose, tgt_joint = out['poses'].to(dev), out['joints'].to(dev)
    B, S, _ = tgt_pose.shape
    pj = joints(imu, lens)                                                       # predicted joint positions [B,S,72]
    if noise_mix > 0:                                                            # keep part of the original training signal: GT + noise
        gt = tgt_joint.reshape(B, S, -1) + 0.04 * torch.randn(B, S, 72, device=dev)
        use_gt = (torch.rand(B, 1, 1, device=dev) < noise_mix).float()
        pj_in = use_gt * gt + (1 - use_gt) * pj
    else:
        pj_in = pj
    pose_p = poser(torch.cat((pj_in, imu), dim=-1), lens)
    pose_t = tgt_pose.view(B, S, 24, 6)[:, :, joint_set.reduced].reshape(B, S, -1)
    loss_pose = torch.nn.functional.mse_loss(pose_p, pose_t) + poser.t_weight * poser.compute_jerk_loss(pose_p)
    full = poser._reduced_global_to_full(pose_p)
    jp = poser.bodymodel.forward_kinematics(pose=full.view(-1, 216))[1].view(B, S, -1)
    loss_pos = torch.nn.functional.mse_loss(jp, tgt_joint.reshape(B, S, -1))
    loss_joint = torch.nn.functional.mse_loss(pj, tgt_joint.reshape(B, S, -1)) + joints.t_weight * joints.compute_temporal_loss(pj)
    return loss_pose + loss_pos + w_joint * loss_joint, loss_pose.item(), loss_pos.item(), loss_joint.item()


def save(model, name, out, epoch, val):
    d = Path(f'checkpoints/{out}_{name}/1/{name}')
    d.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'hyper_parameters': dict(model.hparams), 'epoch': epoch,
                'pytorch-lightning_version': '2.0.0'}, d / f'epoch={epoch}-validation_step_loss={val:.4f}.ckpt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--poser', required=True); ap.add_argument('--joints', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=15); ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--batch', type=int, default=int(os.environ.get('MOBILEPOSER_BATCH', 256)))
    ap.add_argument('--w_joint', type=float, default=1.0); ap.add_argument('--noise_mix', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device('cuda:0')
    poser = Poser.load_from_checkpoint(best_ckpt(f'checkpoints/{a.poser}'), map_location='cpu').to(dev)
    joints = Joints.load_from_checkpoint(best_ckpt(f'checkpoints/{a.joints}'), map_location='cpu').to(dev)
    ds = PoseDataset('train')
    n_val = int(0.05 * len(ds))
    tr, va = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    dl = DataLoader(tr, batch_size=a.batch, collate_fn=pad_seq, shuffle=True, drop_last=True, num_workers=4)
    vl = DataLoader(va, batch_size=a.batch, collate_fn=pad_seq, num_workers=2)
    opt = torch.optim.Adam(list(poser.parameters()) + list(joints.parameters()), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=1e-6)
    best = 1e9
    for ep in range(a.epochs):
        poser.train(); joints.train()
        run = []
        for batch in dl:
            loss, *_ = losses(poser, joints, batch, dev, a.w_joint, a.noise_mix)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(poser.parameters()) + list(joints.parameters()), 1.0); opt.step()
            run.append(loss.item())
        sched.step()
        poser.eval(); joints.eval()
        vs = []
        with torch.no_grad():
            for batch in vl:
                vs.append(np.array(losses(poser, joints, batch, dev, a.w_joint, 0.0)[1:]))
        v = np.mean(vs, axis=0)
        val = float(v[0] + v[1])
        print(f'ep {ep:3d} train {np.mean(run):.5f}  val pose-mse {v[0]:.5f} position-mse {v[1]:.5f} joints-mse {v[2]:.5f}  (pose+position {val:.5f})', flush=True)
        if val < best:
            best = val
            save(poser, 'poser', a.out, ep, val)
            save(joints, 'joints', a.out, ep, val)


if __name__ == '__main__':
    main()
