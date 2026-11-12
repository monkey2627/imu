#!/usr/bin/env python3
"""Balanced 10-action Line14 fine-tuning with AMASS replay."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


def normalize_sensors(value):
    value = value.float()
    if value.ndim >= 3 and value.shape[0] == 6 and value.shape[1] > 6:
        value = value.transpose(0, 1)
    if value.shape[1] < 5:
        raise ValueError(f'Expected at least five sensor slots, got {tuple(value.shape)}')
    return value[:, :5]


def prepare_target(body, pose, shape, device, chain):
    from mobileposer.articulate import math as M
    pose = pose.float().to(device)
    shape = shape.float().reshape(1, 10).to(device).expand(len(pose), -1)
    _, joints = body.forward_kinematics(pose, shape, tran=None)
    local6 = M.rotation_matrix_to_r6d(pose).reshape(1, len(pose), 24, 6)
    rotations = chain.global_targets(local6, body.parent)[0]
    return joints.reshape(len(pose), 72), rotations


def load_ours(root, body, device, chain):
    rows = {}
    for path in sorted((root / 'captures').iterdir(), key=lambda p: int(p.name)):
        inp = torch.load(path / 'inputs/wrists_shanks_waist/ours1.pt', map_location='cpu', weights_only=False)
        tgt = torch.load(path / 'targets_humanpose_calibrated_60.pt', map_location='cpu', weights_only=False)
        if inp['fps'] != 60 or tgt['fps'] != 60 or not torch.equal(inp['timestamps'], tgt['timestamps']):
            raise ValueError(f'{path.name}: input/target are not aligned 60Hz sequences')
        if inp['layout_labels'] != ['lw', 'rw', 'ls', 'rs', 'waist']:
            raise ValueError(f'{path.name}: unexpected sensor order')
        acc, ori = normalize_sensors(inp['acc'][0]), normalize_sensors(inp['ori'][0])
        pose, shape = tgt['pose'].float(), tgt['shape'].float()
        joints, rotations = prepare_target(body, pose, shape, device, chain)
        rows[path.name] = {'acc': acc, 'ori': ori, 'joints': joints.cpu(), 'rot': rotations.cpu(),
                           'timestamps': tgt['timestamps']}
    if list(sorted(rows, key=int)) != [str(i) for i in range(1, 11)]:
        raise ValueError(f'Expected actions 1..10, got {list(rows)}')
    return rows


class AmassReplay:
    def __init__(self, root):
        self.files = sorted(Path(root).glob('*.pt'))
        self.loaded, self.entries = {}, []
        for path in self.files:
            d = torch.load(path, map_location='cpu', weights_only=False)
            if d.get('layout') != 'wrists_shanks_waist' or d.get('synthesis', {}).get('fps') != 60:
                raise ValueError(f'Incompatible 60Hz AMASS replay file: {path}')
            d['acc'] = [normalize_sensors(v) for v in d['acc']]
            d['ori'] = [normalize_sensors(v) for v in d['ori']]
            self.loaded[path] = d
            self.entries.extend((path, i) for i, pose in enumerate(d['pose']) if len(pose) >= 240)
        if not self.entries: raise ValueError('No AMASS replay crops')
        print(f'AMASS replay: {len(self.entries)} sequences in {len(self.files)} subsets', flush=True)

    def sample(self, crop, body, device, chain):
        path, index = random.choice(self.entries)
        d = self.loaded[path]
        pose = d['pose'][index].float()
        n = min(crop, len(pose)); start = random.randrange(len(pose) - n + 1)
        pose = pose[start:start + n]
        joints, rotations = prepare_target(body, pose, d['shape'][index], device, chain)
        return d['acc'][index][start:start + n], d['ori'][index][start:start + n], joints.cpu(), rotations.cpu()


def predict(jnet, poser, imu):
    lengths = [imu.shape[1]]
    joints = jnet(imu, lengths)
    rotations = poser(torch.cat((joints, imu), dim=-1), lengths)
    return joints, rotations


@torch.no_grad()
def evaluate(jnet, poser, rows, actions, device):
    from mobileposer.articulate import math as M
    jnet.eval(); poser.eval()
    metrics = {}
    for action in actions:
        row = rows[action]
        imu = torch.cat(((row['acc'] / 30).flatten(1), row['ori'].flatten(1)), -1)[None].to(device)
        joints, rotations = predict(jnet, poser, imu)
        gt_j = row['joints'].to(device).reshape(1, -1, 72)
        gt_r = row['rot'].to(device)[None]
        pred_r = M.r6d_to_rotation_matrix(rotations.reshape(-1, 6))
        true_r = M.r6d_to_rotation_matrix(gt_r.reshape(-1, 6))
        delta = pred_r.transpose(-1, -2) @ true_r
        angle = torch.acos(((delta.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
        metrics[action] = {
            'mpjpe_cm': float((joints.reshape(-1, 24, 3) - gt_j.reshape(-1, 24, 3)).norm(dim=-1).mean() * 100),
            'rotation_error_deg': float(angle.mean() * 180 / np.pi),
        }
    return metrics


def macro(metrics):
    return {key: float(np.mean([row[key] for row in metrics.values()])) for key in next(iter(metrics.values()))}


def train_stage(jnet, poser, rows, train_actions, val_actions, replay, body, chain, args, output, name, fixed_steps=None):
    device = torch.device(args.device)
    optimizer = torch.optim.AdamW(list(jnet.parameters()) + list(poser.parameters()), lr=args.lr, weight_decay=.01)
    steps = fixed_steps or args.max_steps
    best, best_step, stale, history = float('inf'), 0, 0, []
    for step in range(1, steps + 1):
        jnet.train(); poser.train()
        chosen = random.sample(train_actions, min(args.action_batch, len(train_actions)))
        ours_losses = []
        for action in chosen:
            row = rows[action]; n = len(row['acc']); crop = min(args.crop, n); start = random.randrange(n - crop + 1)
            imu = torch.cat(((row['acc'][start:start + crop] / 30).flatten(1), row['ori'][start:start + crop].flatten(1)), -1)[None].to(device)
            target_j = row['joints'][start:start + crop][None].to(device)
            target_r = row['rot'][None, start:start + crop].to(device)
            pred_j, pred_r = predict(jnet, poser, imu)
            ours_losses.append((pred_j - target_j).square().mean() + (pred_r - target_r).square().mean())
        ours_loss = torch.stack(ours_losses).mean()
        replay_losses = []
        for _ in range(args.replay_batch):
            acc, ori, target_j, target_r = replay.sample(args.crop, body, device, chain)
            imu = torch.cat(((acc / 30).flatten(1), ori.flatten(1)), -1)[None].to(device)
            pred_j, pred_r = predict(jnet, poser, imu)
            replay_losses.append((pred_j - target_j[None].to(device)).square().mean() +
                                 (pred_r - target_r[None].to(device)).square().mean())
        replay_loss = torch.stack(replay_losses).mean()
        loss = ours_loss + args.replay_weight * replay_loss
        if not torch.isfinite(loss): raise ValueError(f'{name}: non-finite loss at step {step}')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(jnet.parameters()) + list(poser.parameters()), 1.0); optimizer.step()
        if step % args.eval_every == 0 or step == steps:
            vm = evaluate(jnet, poser, rows, val_actions, device)
            score = macro(vm)['mpjpe_cm']
            history.append({'step': step, 'ours_loss': float(ours_loss), 'amass_loss': float(replay_loss),
                            'validation': vm, 'validation_macro': macro(vm)})
            print(f'{name} step={step}/{steps} val_mpjpe={score:.3f}cm val_rot={macro(vm)["rotation_error_deg"]:.2f}deg', flush=True)
            if score < best:
                best, best_step, stale = score, step, 0
                torch.save({'jnet': jnet.state_dict(), 'poser': poser.state_dict()}, output / f'{name}_best.pt')
            else: stale += 1
            (output / f'{name}_history.json').write_text(json.dumps(history, indent=2))
            if fixed_steps is None and stale >= args.patience: break
    state = torch.load(output / f'{name}_best.pt', map_location=device, weights_only=False)
    jnet.load_state_dict(state['jnet']); poser.load_state_dict(state['poser'])
    return jnet, poser, best_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace', type=Path, required=True); ap.add_argument('--line14-workspace', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True); ap.add_argument('--amass', type=Path, required=True)
    ap.add_argument('--init', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0'); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-steps', type=int, default=3000); ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--patience', type=int, default=8); ap.add_argument('--crop', type=int, default=240)
    ap.add_argument('--action-batch', type=int, default=4); ap.add_argument('--replay-batch', type=int, default=1)
    ap.add_argument('--replay-weight', type=float, default=.2); ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--special-action', default='7'); ap.add_argument('--special-steps', type=int, default=250)
    ap.add_argument('--special-lr', type=float, default=5e-6)
    args = ap.parse_args()
    sys.path.insert(0, str(args.workspace)); sys.path.insert(0, str(args.line14_workspace / 'scripts_run'))
    import mobileposer
    mobileposer.__path__.append(str(args.line14_workspace / 'mobileposer'))
    from mobileposer.articulate.model import ParametricModel
    from mobileposer.config import paths
    from mobileposer.line9.models import GeometryStats, Line9Joints
    import line14_poser_chain as chain
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    baseline = torch.load(args.init, map_location='cpu', weights_only=False)
    jnet = Line9Joints('b0', geom_dim=1,
                       geometry_stats=GeometryStats(torch.zeros(1), torch.ones(1))).to(device)
    poser = chain.ChainPoser(132).to(device)
    jnet.load_state_dict(baseline['jnet']); poser.load_state_dict(baseline['poser'])
    args.output.mkdir(parents=True, exist_ok=True)
    if any(p.name != 'config.json' for p in args.output.iterdir()):
        raise ValueError(f'Output contains unexpected preexisting files: {args.output}')
    body = ParametricModel(paths.smpl_file, device=device)
    ours = load_ours(args.data, body, device, chain)
    replay = AmassReplay(args.amass)
    train_actions, val_actions = [str(i) for i in range(1, 9)], ['9', '10']
    config = {'baseline': str(args.init), 'train_actions_selection': train_actions, 'validation_actions': val_actions,
              'sampling': 'uniform actions with one crop per selected action per step', 'fps': 60,
              'amass_replay_weight': args.replay_weight}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    jnet, poser, selected_steps = train_stage(jnet, poser, ours, train_actions, val_actions, replay, body, chain, args,
                                               args.output, 'selection')
    jnet = Line9Joints('b0', geom_dim=1,
                       geometry_stats=GeometryStats(torch.zeros(1), torch.ones(1))).to(device)
    poser = chain.ChainPoser(132).to(device)
    jnet.load_state_dict(baseline['jnet']); poser.load_state_dict(baseline['poser'])
    jnet, poser, _ = train_stage(jnet, poser, ours, [str(i) for i in range(1, 11)], val_actions, replay,
                                  body, chain, args, args.output, 'all10', fixed_steps=selected_steps)
    actions = [str(i) for i in range(1, 11)]
    validation = evaluate(jnet, poser, ours, val_actions, device)
    general = evaluate(jnet, poser, ours, actions, device)
    torch.save({'jnet': jnet.state_dict(), 'poser': poser.state_dict()}, args.output / 'model_finetuned.pt')
    optimizer = torch.optim.AdamW(list(jnet.parameters()) + list(poser.parameters()), lr=args.special_lr, weight_decay=.01)
    row = ours[args.special_action]
    for _ in range(args.special_steps):
        jnet.train(); poser.train()
        n = len(row['acc']); crop = min(args.crop, n); start = random.randrange(n - crop + 1)
        imu = torch.cat(((row['acc'][start:start + crop] / 30).flatten(1), row['ori'][start:start + crop].flatten(1)), -1)[None].to(device)
        target_j, target_r = row['joints'][start:start + crop][None].to(device), row['rot'][None, start:start + crop].to(device)
        pred_j, pred_r = predict(jnet, poser, imu)
        loss = (pred_j - target_j).square().mean() + (pred_r - target_r).square().mean()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(jnet.parameters()) + list(poser.parameters()), 1.0); optimizer.step()
    specialized = evaluate(jnet, poser, ours, actions, device)
    torch.save({'jnet': jnet.state_dict(), 'poser': poser.state_dict()}, args.output / 'model_specialized.pt')
    report = {'validation_actions': val_actions, 'selected_steps': selected_steps,
              'validation_after_all10': validation, 'validation_macro_after_all10': macro(validation),
              'all10_before_specialization': general, 'all10_macro_before_specialization': macro(general),
              'specialization_action': args.special_action, 'specialization_steps': args.special_steps,
              'all10_after_specialization': specialized, 'all10_macro_after_specialization': macro(specialized),
              'metric_note': 'MPJPE and global rotation error vs calibrated Unity HumanPose SMPL pseudo-labels; not independent mocap GT.'}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
