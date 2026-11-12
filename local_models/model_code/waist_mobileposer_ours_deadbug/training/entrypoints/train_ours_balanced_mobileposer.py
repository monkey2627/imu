#!/usr/bin/env python3
"""Balanced action fine-tuning with action-held-out validation and AMASS replay."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def load_ours(root, device):
    rows = {}
    for d in sorted((root / 'captures').iterdir(), key=lambda p: int(p.name)):
        action = d.name
        input_path = d / 'inputs/wrists_shanks_waist/ours1.pt'
        target_path = d / 'targets_humanpose_calibrated_60.pt'
        inp = torch.load(input_path, map_location='cpu', weights_only=False)
        tgt = torch.load(target_path, map_location='cpu', weights_only=False)
        if inp['fps'] != 60 or tgt['fps'] != 60 or not torch.equal(inp['timestamps'], tgt['timestamps']):
            raise ValueError(f'action {action}: non-60Hz or mismatched timestamps')
        if inp['layout_labels'] != ['lw', 'rw', 'ls', 'rs', 'waist']:
            raise ValueError(f'action {action}: unexpected sensor order')
        imu = torch.cat(((inp['acc'][0] / 30).flatten(1), inp['ori'][0].flatten(1)), -1)
        rows[action] = {'imu': imu, 'pose': tgt['pose'], 'shape': tgt['shape'], 'timestamps': tgt['timestamps']}
    if sorted(rows, key=int) != [str(i) for i in range(1, 11)]:
        raise ValueError(f'Expected actions 1..10, found {list(rows)}')
    return rows


class AmassReplay:
    def __init__(self, root):
        self.files = sorted(Path(root).glob('*.pt'))
        if not self.files:
            raise FileNotFoundError(f'No AMASS surface files in {root}')
        self.loaded = {}
        self.entries = []
        for path in self.files:
            d = torch.load(path, map_location='cpu', weights_only=False)
            if d.get('layout') != 'wrists_shanks_waist' or d.get('synthesis', {}).get('fps') != 60:
                raise ValueError(f'Incompatible AMASS replay set: {path}')
            self.loaded[path] = d
            for i, pose in enumerate(d['pose']):
                if len(pose) >= 240:
                    self.entries.append((path, i))
        if not self.entries:
            raise ValueError('No AMASS sequences long enough for replay crops')
        print(f'AMASS replay: {len(self.entries)} sequences across {len(self.files)} subsets', flush=True)

    def sample(self, count, crop, body, device):
        samples = []
        for _ in range(count):
            path, i = random.choice(self.entries)
            d = self.loaded[path]
            acc = d['acc'][i].float()
            ori = d['ori'][i].float()
            # Cached AMASS layouts may put sensors before time; normalize before slicing.
            if acc.shape[0] == 6 and acc.shape[1] > 6:
                acc = acc.transpose(0, 1)
            if ori.shape[0] == 6 and ori.shape[1] > 6:
                ori = ori.transpose(0, 1)
            if acc.shape[1] < 5 or ori.shape[1] < 5:
                raise ValueError(f'Unexpected AMASS sensor layout: acc={tuple(acc.shape)}, ori={tuple(ori.shape)}')
            acc, ori = acc[:, :5], ori[:, :5]
            pose, shape = d['pose'][i].float(), d['shape'][i].float()
            n = min(crop, len(pose))
            start = random.randrange(len(pose) - n + 1)
            pose = pose[start:start+n]
            imu = torch.cat(((acc[start:start+n] / 30).flatten(1), ori[start:start+n].flatten(1)), -1)
            yield imu, pose


def forward(model, imu):
    lengths = [imu.shape[1]]
    joints = model.joints(imu, lengths)
    reduced = model.pose(torch.cat((joints, imu), dim=-1), lengths)
    pose = model._reduced_global_to_full(reduced)
    _, fk = model.bodymodel.forward_kinematics(pose)
    return joints, reduced, pose, fk


def targets(model, pose):
    from mobileposer.articulate import math
    from mobileposer.config import joint_set
    pose = pose.to(next(model.parameters()).device)
    global_pose, joint = model.bodymodel.forward_kinematics(pose)
    joint = joint.reshape(1, len(pose), 24, 3)
    reduced = math.rotation_matrix_to_r6d(global_pose[:, joint_set.reduced]).reshape(1, len(pose), -1)
    return joint, reduced, pose


def losses(model, imu, pose):
    joint_gt, rot_gt, pose_gt = targets(model, pose)
    j, r, p, fk = forward(model, imu)
    root_gt = joint_gt - joint_gt[:, :, :1]
    root_fk = fk.reshape(1, imu.shape[1], 24, 3)
    root_fk = root_fk - root_fk[:, :, :1]
    joint_loss = (j.reshape_as(joint_gt) - root_gt).square().mean()
    fk_loss = (root_fk - root_gt).square().mean()
    rot_loss = (r - rot_gt).square().mean()
    return joint_loss + rot_loss + fk_loss, p, fk.reshape(1, imu.shape[1], 24, 3), pose_gt


@torch.no_grad()
def evaluate(model, rows, actions, device):
    model.eval()
    metrics = {}
    for action in actions:
        row = rows[action]
        imu = row['imu'][None].to(device)
        _, _, pred_pose, pred_joint = forward(model, imu)
        _, gt_joint = model.bodymodel.forward_kinematics(row['pose'].to(device))
        gt_joint = gt_joint.reshape(1, -1, 24, 3)
        pred_joint = pred_joint.reshape(1, -1, 24, 3)
        pred_rel = pred_joint - pred_joint[:, :, :1]
        gt_rel = gt_joint - gt_joint[:, :, :1]
        pred_global = pred_pose[0]
        target_global = row['pose'].to(device)
        diff = pred_global.transpose(-1, -2) @ target_global
        angle = torch.acos(((diff.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
        metrics[action] = {'mpjpe_cm': float((pred_rel - gt_rel).norm(dim=-1).mean() * 100),
                           'rotation_error_deg': float(angle.mean() * 180 / np.pi)}
    model.train()
    model.joints.joints.rnn.train()
    model.pose.pose.rnn.train()
    return metrics


def macro(metrics):
    return {k: float(np.mean([v[k] for v in metrics.values()])) for k in next(iter(metrics.values()))}


def train_stage(model, ours, train_actions, val_actions, replay, args, output, name, fixed_steps=None):
    device = torch.device(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    steps = fixed_steps or args.max_steps
    best_score, best_step, stale = float('inf'), 0, 0
    history = []
    for step in range(1, steps + 1):
        model.eval(); model.joints.joints.rnn.train(); model.pose.pose.rnn.train()
        action_batch = random.sample(train_actions, k=min(args.action_batch, len(train_actions)))
        ours_losses = []
        for action in action_batch:
            row = ours[action]
            n = len(row['imu']); crop = min(args.crop, n)
            start = random.randrange(n - crop + 1)
            imu = row['imu'][start:start+crop][None].to(device)
            pose = row['pose'][start:start+crop]
            loss, _, _, _ = losses(model, imu, pose)
            ours_losses.append(loss)
        ours_loss = torch.stack(ours_losses).mean()
        replay_losses = []
        for imu_cpu, pose in replay.sample(args.replay_batch, args.crop, model.bodymodel, device):
            replay_losses.append(losses(model, imu_cpu[None].to(device), pose)[0])
        replay_loss = torch.stack(replay_losses).mean()
        loss = ours_loss + args.replay_weight * replay_loss
        if not torch.isfinite(loss):
            raise ValueError(f'{name}: non-finite loss at step {step}')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        if step % args.eval_every == 0 or step == steps:
            vm = evaluate(model, ours, val_actions, device)
            score = macro(vm)['mpjpe_cm']
            history.append({'step': step, 'train_ours_loss': float(ours_loss), 'train_amass_loss': float(replay_loss),
                            'validation': vm, 'validation_macro': macro(vm)})
            print(f'{name} step={step}/{steps} val_mpjpe={score:.3f}cm val_rot={macro(vm)["rotation_error_deg"]:.2f}deg', flush=True)
            if score < best_score:
                best_score, best_step, stale = score, step, 0
                torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, output/f'{name}_best.pth')
            else:
                stale += 1
            (output/f'{name}_history.json').write_text(json.dumps(history, indent=2))
            if fixed_steps is None and stale >= args.patience:
                break
    state = torch.load(output/f'{name}_best.pth', map_location=device, weights_only=False)
    model.load_state_dict(state)
    return model, best_step, best_score, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True); ap.add_argument('--amass', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workspace', type=Path, required=True); ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--max-steps', type=int, default=3000)
    ap.add_argument('--eval-every', type=int, default=100); ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--crop', type=int, default=240); ap.add_argument('--action-batch', type=int, default=4)
    ap.add_argument('--replay-batch', type=int, default=1); ap.add_argument('--replay-weight', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=2e-5); ap.add_argument('--special-action', default='7')
    ap.add_argument('--special-steps', type=int, default=250); ap.add_argument('--special-lr', type=float, default=5e-6)
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f'Output must be empty: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.insert(0, str(args.workspace))
    from mobileposer.config import joint_set, model_config
    from mobileposer.utils.model_utils import load_model
    from mobileposer.surface_imu import file_hash
    model_config.device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    ours = load_ours(args.data, torch.device(args.device))
    train_actions, val_actions = [str(i) for i in range(1, 9)], ['9', '10']
    if args.special_action not in train_actions:
        raise ValueError('special action must be in the training split')
    model = load_model(str(args.checkpoint)).to(args.device)
    base_hash = file_hash(args.checkpoint)
    for p in model.parameters(): p.requires_grad_(False)
    for module in (model.joints, model.pose):
        for p in module.parameters(): p.requires_grad_(True)
    replay = AmassReplay(args.amass)
    config = vars(args).copy(); config = {k: str(v) if isinstance(v, Path) else v for k,v in config.items()}
    config.update({'train_actions': train_actions, 'validation_actions': val_actions, 'baseline_sha256': base_hash,
                   'frequency_hz': 60, 'amass_replay_fraction': args.replay_batch/(args.action_batch+args.replay_batch)})
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    # Select a stopping point without touching held-out action gradients.
    model, selected_steps, _, val_history = train_stage(model, ours, train_actions, val_actions, replay, args,
                                                         args.output, 'selection')
    # Refit from the untouched AMASS baseline on all ten actions for the selected duration.
    model = load_model(str(args.checkpoint)).to(args.device)
    for p in model.parameters(): p.requires_grad_(False)
    for module in (model.joints, model.pose):
        for p in module.parameters(): p.requires_grad_(True)
    all_actions = [str(i) for i in range(1, 11)]
    model, _, _, _ = train_stage(model, ours, all_actions, val_actions, replay, args,
                                  args.output, 'all10', fixed_steps=selected_steps)
    validation_metrics = evaluate(model, ours, val_actions, torch.device(args.device))
    all_metrics = evaluate(model, ours, all_actions, torch.device(args.device))
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, args.output/'model_finetuned.pth')
    # Final action-specific pass starts from the all-action model and is kept separate.
    for p in model.parameters(): p.requires_grad_(False)
    for module in (model.joints, model.pose):
        for p in module.parameters(): p.requires_grad_(True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.special_lr, weight_decay=1e-4)
    action = ours[args.special_action]
    for step in range(args.special_steps):
        n = len(action['imu']); crop = min(args.crop, n); start = random.randrange(n-crop+1)
        imu = action['imu'][start:start+crop][None].to(args.device)
        loss, _, _, _ = losses(model, imu, action['pose'][start:start+crop])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
    specialized_metrics = evaluate(model, ours, all_actions, torch.device(args.device))
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, args.output/'model_specialized.pth')
    report = {'baseline': str(args.checkpoint), 'baseline_sha256': base_hash, 'validation_actions': val_actions,
              'training_actions': all_actions, 'selected_steps': selected_steps,
              'validation_after_all10': validation_metrics, 'validation_macro_after_all10': macro(validation_metrics),
              'all10_before_specialization': all_metrics, 'all10_macro_before_specialization': macro(all_metrics),
              'specialization_action': args.special_action, 'specialization_steps': args.special_steps,
              'all10_after_specialization': specialized_metrics, 'all10_macro_after_specialization': macro(specialized_metrics),
              'metric_note': 'SMPL pseudo-label MPJPE and local rotation error; Ours actions used for train/validation as reported; AMASS supervised replay.'}
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'selected_steps':selected_steps,'validation':macro(validation_metrics),
                      'all10_before_specialization':macro(all_metrics),
                      'all10_after_specialization':macro(specialized_metrics),'output':str(args.output)},indent=2))


if __name__ == '__main__': main()
