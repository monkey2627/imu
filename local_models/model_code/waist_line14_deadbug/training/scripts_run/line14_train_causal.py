"""Minimal causal PivotPrint-conditioned trainer.

Starts from the audited PNP5 causal checkpoints.  It intentionally keeps the
data contract small and explicit so smoke runs can verify gradients and causal
conditioning before a longer experiment is scheduled.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from mobileposer.line9.data import load_identity_records, PairedWindowDataset, collate_paired_windows, SubsetBatchSampler
from mobileposer.line9.store import SequenceStore
from torch.utils.data import DataLoader
from mobileposer.models.joints import Joints
from mobileposer.models.poser import Poser
from mobileposer.config import joint_set
from mobileposer.line14.endpoint import CalibrationConditionedRNN
from mobileposer.line14.layout import assert_pnp5_data_root, validate_pnp5_dataset_metadata
from mobileposer.articulate import math as M
from mobileposer.line14.geom import r2_pca_basis


def best_ckpt(root: str) -> str:
    files = list(Path(root).rglob("*.ckpt"))
    if not files: raise FileNotFoundError(root)
    return str(sorted(files, key=lambda p: ("validation_step_loss=" not in p.name, p.name))[0])


def load_calibration_by_identity(device):
    rows = {}
    train_ids = sorted({r.identity for r in load_identity_records(folds=('train',), raw_root='/home/duanyuhan/dyh/motion/dataset/raw/AMASS')['train']})
    fold_of = {ident: i % 5 for i, ident in enumerate(train_ids)}
    basis = r2_pca_basis()
    for ident in train_ids:
        fold = fold_of[ident]
        p = torch.load(f'data_local/line15_cal_ids/pred_uni_s42_fold{fold}_train.pt', map_location='cpu', weights_only=False)
        sel = [i for i, x in enumerate(p['identity']) if x == ident]
        if not sel:
            raise KeyError(f'no OOF C20 estimate for {ident}')
        mu = p['mu_cm'][sel].mean(0) / 100.0
        var = torch.exp(p['logvar_cm'][sel]).mean(0) / 10000.0
        pc = ((mu - basis['mean']) @ basis['V'][:, :6])
        pc_var = (var[:, None] * basis['V'][:, :6].square()).sum(0).clamp_min(1e-8)
        R_est = p['R_est'][sel].mean(0)
        R_true = p['R_true'][sel].mean(0)
        U, _, Vh = torch.linalg.svd(R_est)
        R_est = U @ Vh
        U, _, Vh = torch.linalg.svd(R_true)
        R_true = U @ Vh
        rows[ident] = {'condition': torch.cat((pc, pc_var.sqrt())), 'R_est': R_est, 'R_true': R_true}
    allc = torch.stack([v['condition'] for v in rows.values()])
    return rows, allc.mean(0), allc.std(0).clamp_min(1e-4)


def condition_and_correct(imu, ids, device, cal_by_id, mean, std, gate_model):
    b = len(ids)
    cond = torch.stack([cal_by_id[x]['condition'] for x in ids]).to(device)
    cond = ((cond - mean) / std).clamp(-4, 4)
    corrected=imu.clone()
    for n in range(b):
        est_off = cal_by_id[ids[n]]['R_est'].to(device)
        gate = gate_model.calibration_gates(cond[n:n+1], est_off)[0]
        tangent = M.rotation_matrix_to_axis_angle(est_off)
        effective = M.axis_angle_to_rotation_matrix(tangent * gate[:, None])
        # Target IMU already contains the same identity-level mount offsets as
        # the fixed-protocol C20 calibration set (line9_store_calib6).
        ori = imu[n,:,15:].reshape(-1,5,3,3)
        corrected[n,:,15:]= (ori @ effective.transpose(-1,-2)).reshape(len(ori),45)
    return corrected, cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data/pnp5_60a/pnp5')
    ap.add_argument('--cache', default='data/cache_pnp5_60a.pt')
    ap.add_argument('--joints-ckpt', default='checkpoints/pnp5_60a_s42_joints')
    ap.add_argument('--poser-ckpt', default='checkpoints/pnp5_60a_s42_poser')
    ap.add_argument('--out', default='checkpoints/line14/causal_smoke.pt')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--max-batches', type=int, default=4)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--freeze-backbone', action='store_true')
    a = ap.parse_args()
    root = assert_pnp5_data_root(a.data_root); validate_pnp5_dataset_metadata(root)
    os.environ['MOBILEPOSER_DATA_CACHE'] = a.cache
    # Keep the audited baseline trunk architecture (bidirectional weights are
    # loaded intact); training supervises only each window endpoint, which is
    # the same past-only causal query used by standard causal40 evaluation.
    os.environ.pop('MOBILEPOSER_CAUSAL', None)
    os.environ['MOBILEPOSER_PROCESSED_DATASETS'] = str(root)
    dev = torch.device(a.device if a.device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
    torch.manual_seed(14027)
    cal_by_id, cond_mean, cond_std = load_calibration_by_identity(dev)
    j = Joints.load_from_checkpoint(best_ckpt(a.joints_ckpt), map_location=dev).to(dev).eval()
    p = Poser.load_from_checkpoint(best_ckpt(a.poser_ckpt), map_location=dev).to(dev).eval()
    if not hasattr(j, 'joints') or not hasattr(p, 'pose'):
        raise RuntimeError('expected plain PNP5 Joints/Poser RNN checkpoints')
    jw = CalibrationConditionedRNN(j.joints, 12).to(dev)
    pw = CalibrationConditionedRNN(p.pose, 12).to(dev)
    jw.train(); pw.train()
    # Freeze the baseline trunk initially; only conditioning and sensor trust
    # learn in smoke. A production run can explicitly unfreeze the trunks.
    if a.freeze_backbone:
        for net in (jw, pw):
            for name, param in net.named_parameters():
                param.requires_grad = name.startswith('film_') or name == 'sensor_gate_logits'
    opt = torch.optim.AdamW([q for q in list(jw.parameters()) + list(pw.parameters()) if q.requires_grad], lr=1e-3)
    records = load_identity_records(folds=('train',), raw_root='/home/duanyuhan/dyh/motion/dataset/raw/AMASS')['train']
    target_store = SequenceStore('data/line9_store_calib6')
    ds = PairedWindowDataset(records, window_size=250, calibration_window_size=390,
                             max_windows=(a.max_batches*a.batch if a.max_batches else None), store=target_store)
    sampler = SubsetBatchSampler(ds, a.batch, shuffle=True, drop_last=True, seed=14027)
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_paired_windows, num_workers=0)
    hist=[]
    for epoch in range(a.epochs):
        total=0.0; count=0
        for bi, b in enumerate(dl):
            if bi >= a.max_batches: break
            imu = b['target_imu'].to(dev); lens = b['target_lengths'].tolist(); ids = b['identity']
            target_j = b['target_joints'].to(dev).flatten(2)
            imu, cond = condition_and_correct(imu, ids, dev, cal_by_id, cond_mean.to(dev), cond_std.to(dev), jw)
            opt.zero_grad(set_to_none=True)
            pred_j_all = jw(imu, cond, lens)
            end = torch.as_tensor(lens, device=dev) - 1
            rows = torch.arange(len(lens), device=dev)
            pred_j = pred_j_all[rows, end]
            target_j = target_j[rows, end]
            lj = (pred_j-target_j).square().mean()
            pose_joint_input = pred_j_all + 0.04 * torch.randn_like(pred_j_all)
            pose_in = torch.cat((pose_joint_input, imu), -1)
            pred_p = pw(pose_in, cond, lens)[rows, end]
            target_pose = b['target_pose'].to(dev)
            target_p = M.rotation_matrix_to_r6d(target_pose[rows, end]).reshape(len(rows), 24, 6)[:, joint_set.reduced].reshape(len(rows), -1)
            lp = (pred_p-target_p).square().mean()
            loss = lj + lp
            loss.backward(); torch.nn.utils.clip_grad_norm_(list(jw.parameters())+list(pw.parameters()), 1.0); opt.step()
            total += float(loss); count += 1
            print(json.dumps({'epoch':epoch+1,'batch':bi+1,'loss':float(loss),'joints':float(lj),'pose':float(lp),'finite':bool(torch.isfinite(loss))}), flush=True)
        hist.append(total/max(count,1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'joints':jw.state_dict(),'poser':pw.state_dict(),'condition_dim':12,'condition_mean':cond_mean.cpu(),'condition_std':cond_std.cpu(),'history':hist,'protocol':{'future_frames':0,'sensors':['lw','rw','ls','rs','pelvis'],'calibration_geometry':'identity-matched C20 OOF pca6'}}, a.out)
    sample_id = next(iter(cal_by_id))
    sample_cond = ((cal_by_id[sample_id]['condition'] - cond_mean) / cond_std).unsqueeze(0).to(dev)
    sample_off = cal_by_id[sample_id]['R_est'].to(dev)
    gates = jw.calibration_gates(sample_cond, sample_off).detach().cpu().tolist()
    print(json.dumps({'saved':a.out,'history':hist,'sample_identity':sample_id,'sample_gates':gates}))

if __name__ == '__main__': main()
