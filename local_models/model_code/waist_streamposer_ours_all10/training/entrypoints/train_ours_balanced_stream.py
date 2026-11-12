#!/usr/bin/env python3
"""Balanced 10-action StreamPoser fine-tuning with AMASS replay."""
import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def normalize_sensors(value):
    value = value.float()
    if value.ndim >= 3 and value.shape[0] == 6 and value.shape[1] > 6:
        value = value.transpose(0, 1)
    if value.shape[1] < 5:
        raise ValueError(f'Expected at least five sensor slots, got {tuple(value.shape)}')
    return value[:, :5]


def load_ours(root, body, device):
    from mobileposer.stream.features import targets
    rows = {}
    for path in sorted((root / 'captures').iterdir(), key=lambda p: int(p.name)):
        inp = torch.load(path / 'inputs/wrists_shanks_waist/ours1.pt', map_location='cpu', weights_only=False)
        tgt = torch.load(path / 'targets_humanpose_calibrated_60.pt', map_location='cpu', weights_only=False)
        if inp['fps'] != 60 or tgt['fps'] != 60 or not torch.equal(inp['timestamps'], tgt['timestamps']):
            raise ValueError(f'{path.name}: input/target are not aligned 60Hz sequences')
        if inp['layout_labels'] != ['lw', 'rw', 'ls', 'rs', 'waist']:
            raise ValueError(f'{path.name}: unexpected sensor order')
        acc = normalize_sensors(inp['acc'][0])
        ori = normalize_sensors(inp['ori'][0])
        pose = tgt['pose'].float()
        jrel, r6d = targets(body, pose.to(device))
        rows[path.name] = {'acc': acc, 'ori': ori, 'pose': pose,
                           'jrel': jrel.cpu(), 'r6d': r6d.cpu()}
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
        if not self.entries:
            raise ValueError('No AMASS replay crops')
        print(f'AMASS replay: {len(self.entries)} sequences in {len(self.files)} subsets', flush=True)

    def sample(self, crop, body, device):
        path, index = random.choice(self.entries)
        d = self.loaded[path]
        pose = d['pose'][index].float()
        n = min(crop, len(pose)); start = random.randrange(len(pose) - n + 1)
        pose = pose[start:start + n]
        acc = d['acc'][index][start:start + n]
        ori = d['ori'][index][start:start + n]
        jrel, r6d = __import__('mobileposer.stream.features', fromlist=['targets']).targets(body, pose.to(device))
        return acc[None].to(device), ori[None].to(device), jrel[None], r6d[None], torch.ones(1, n, dtype=torch.bool, device=device)


def lagged_loss(model, acc, ori, jrel, r6d, mask, cfg, mount):
    from mobileposer.stream.data import batch_inputs
    from mobileposer.stream.features import root_relative_ori
    x, ori_seen = batch_inputs(acc, ori, cfg.gyro, mount, model.frame, model.local)
    pred_j, pred_r, _ = model(x)
    k = cfg.lag
    if k:
        pred_j, pred_r = pred_j[:, k:], pred_r[:, k:]
        jrel, r6d, mask = jrel[:, :-k], r6d[:, :-k], mask[:, k:]
    pred_r = model.compose(pred_r, root_relative_ori(ori_seen[:, :ori_seen.shape[1] - k] if k else ori_seen))
    m = mask.unsqueeze(-1).float()
    joint_loss = (((pred_j - jrel) ** 2) * m).sum() / (m.sum() * jrel.shape[-1])
    rot_loss = (((pred_r - r6d) ** 2) * m).sum() / (m.sum() * r6d.shape[-1])
    loss = rot_loss + cfg.wj * joint_loss
    if not torch.isfinite(loss):
        raise ValueError('Non-finite StreamPoser loss')
    return loss, pred_j, pred_r, jrel, r6d, mask


@torch.no_grad()
def evaluate(model, rows, actions, cfg, device):
    from mobileposer.stream.features import imu_features, root_relative_ori
    model.eval()
    result = {}
    for action in actions:
        row = rows[action]
        acc, ori = row['acc'][None].to(device), row['ori'][None].to(device)
        x, ori_seen = __import__('mobileposer.stream.data', fromlist=['batch_inputs']).batch_inputs(
            acc, ori, cfg.gyro, 0.0, model.frame, model.local)
        pj, pr, _ = model(x)
        k = cfg.lag
        if k:
            pj, pr = pj[:, k:], pr[:, k:]
            gt_j, gt_r = row['jrel'][None, :-k].to(device), row['r6d'][None, :-k].to(device)
            seen = ori_seen[:, :-k]
        else:
            gt_j, gt_r, seen = row['jrel'][None].to(device), row['r6d'][None].to(device), ori_seen
        pr = model.compose(pr, root_relative_ori(seen))
        from mobileposer.articulate import math as M
        rp = M.r6d_to_rotation_matrix(pr.reshape(-1, 6))
        rg = M.r6d_to_rotation_matrix(gt_r.reshape(-1, 6))
        delta = rp.transpose(-1, -2) @ rg
        angle = torch.acos(((delta.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
        result[action] = {
            'mpjpe_cm': float((pj - gt_j).reshape(-1, 23, 3).norm(dim=-1).mean() * 100),
            'rotation_error_deg': float(angle.mean() * 180 / np.pi),
        }
    model.train()
    return result


def macro(metrics):
    return {key: float(np.mean([row[key] for row in metrics.values()])) for key in next(iter(metrics.values()))}


def train_stage(model, rows, train_actions, val_actions, replay, cfg, args, output, name, fixed_steps=None):
    device = torch.device(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = fixed_steps or args.max_steps
    best, best_step, stale = float('inf'), 0, 0
    history = []
    for step in range(1, steps + 1):
        model.train()
        chosen = random.sample(train_actions, min(args.action_batch, len(train_actions)))
        ours_losses = []
        for action in chosen:
            row = rows[action]
            n = len(row['pose']); crop = min(args.crop, n); start = random.randrange(n - crop + 1)
            batch = (row['acc'][start:start + crop][None].to(device), row['ori'][start:start + crop][None].to(device),
                     row['jrel'][None, start:start + crop].to(device), row['r6d'][None, start:start + crop].to(device),
                     torch.ones(1, crop, dtype=torch.bool, device=device))
            ours_losses.append(lagged_loss(model, *batch, cfg, args.mount_deg)[0])
        ours_loss = torch.stack(ours_losses).mean()
        replay_losses = []
        for _ in range(args.replay_batch):
            replay_losses.append(lagged_loss(model, *replay.sample(args.crop, args.body, device), cfg, args.mount_deg)[0])
        replay_loss = torch.stack(replay_losses).mean()
        loss = ours_loss + args.replay_weight * replay_loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % args.eval_every == 0 or step == steps:
            vm = evaluate(model, rows, val_actions, cfg, device)
            score = macro(vm)['mpjpe_cm']
            history.append({'step': step, 'ours_loss': float(ours_loss), 'amass_loss': float(replay_loss),
                            'validation': vm, 'validation_macro': macro(vm)})
            print(f'{name} step={step}/{steps} val_mpjpe={score:.3f}cm val_rot={macro(vm)["rotation_error_deg"]:.2f}deg', flush=True)
            if score < best:
                best, best_step, stale = score, step, 0
                torch.save(model.state_dict(), output / f'{name}_best.pth')
            else:
                stale += 1
            (output / f'{name}_history.json').write_text(json.dumps(history, indent=2))
            if fixed_steps is None and stale >= args.patience: break
    model.load_state_dict(torch.load(output / f'{name}_best.pth', map_location=device, weights_only=False))
    return model, best_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace', type=Path, required=True); ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--amass', type=Path, required=True); ap.add_argument('--init', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--max-steps', type=int, default=3000)
    ap.add_argument('--eval-every', type=int, default=100); ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--crop', type=int, default=240); ap.add_argument('--action-batch', type=int, default=4)
    ap.add_argument('--replay-batch', type=int, default=1); ap.add_argument('--replay-weight', type=float, default=.2)
    ap.add_argument('--lr', type=float, default=2e-5); ap.add_argument('--mount-deg', type=float, default=8.)
    ap.add_argument('--special-action', default='7'); ap.add_argument('--special-steps', type=int, default=250)
    ap.add_argument('--special-lr', type=float, default=5e-6)
    args = ap.parse_args()
    sys.path.insert(0, str(args.workspace))
    from mobileposer.articulate.model import ParametricModel
    from mobileposer.config import paths
    from mobileposer.stream.model import StreamPoser
    cfg_json = json.loads((args.init / 'config.json').read_text())
    cfg = SimpleNamespace(gyro=bool(cfg_json['gyro']), lag=int(cfg_json['lag']), wj=float(cfg_json['wj']))
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    model = StreamPoser(cfg.gyro, int(cfg_json['hidden']), int(cfg_json['layers']),
                        anchor=cfg_json.get('anchor', 'none'), frame=cfg_json.get('frame', 'both'),
                        arch=cfg_json.get('arch', 'mp2'), local=bool(cfg_json.get('local_gyro', 0))).to(device)
    model.load_state_dict(torch.load(args.init / 'model.pt', map_location=device, weights_only=False))
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f'Output must be empty: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    body = ParametricModel(paths.smpl_file, device=device)
    args.body = body
    ours = load_ours(args.data, body, device)
    replay = AmassReplay(args.amass)
    train_actions, val_actions = [str(i) for i in range(1, 9)], ['9', '10']
    config = {**cfg_json, 'baseline': str(args.init), 'train_actions_selection': train_actions, 'validation_actions': val_actions,
              'sampling': 'uniform actions with one crop per selected action per step', 'fps': 60,
              'lag_frames': cfg.lag, 'amass_replay_weight': args.replay_weight, 'mount_augmentation_deg': args.mount_deg}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    model, selected_steps = train_stage(model, ours, train_actions, val_actions, replay, cfg, args, args.output, 'selection')
    model = StreamPoser(cfg.gyro, int(cfg_json['hidden']), int(cfg_json['layers']),
                        anchor=cfg_json.get('anchor', 'none'), frame=cfg_json.get('frame', 'both'),
                        arch=cfg_json.get('arch', 'mp2'), local=bool(cfg_json.get('local_gyro', 0))).to(device)
    model.load_state_dict(torch.load(args.init / 'model.pt', map_location=device, weights_only=False))
    model, _ = train_stage(model, ours, [str(i) for i in range(1, 11)], val_actions, replay, cfg, args,
                           args.output, 'all10', fixed_steps=selected_steps)
    all_actions = [str(i) for i in range(1, 11)]
    validation = evaluate(model, ours, val_actions, cfg, device)
    general = evaluate(model, ours, all_actions, cfg, device)
    torch.save(model.state_dict(), args.output / 'model_finetuned.pt')
    for parameter in model.parameters(): parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.special_lr)
    row = ours[args.special_action]
    for step in range(args.special_steps):
        n = len(row['pose']); crop = min(args.crop, n); start = random.randrange(n - crop + 1)
        batch = (row['acc'][start:start + crop][None].to(device), row['ori'][start:start + crop][None].to(device),
                 row['jrel'][None, start:start + crop].to(device), row['r6d'][None, start:start + crop].to(device),
                 torch.ones(1, crop, dtype=torch.bool, device=device))
        loss = lagged_loss(model, *batch, cfg, args.mount_deg)[0]
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    specialized = evaluate(model, ours, all_actions, cfg, device)
    torch.save(model.state_dict(), args.output / 'model_specialized.pt')
    report = {'validation_actions': val_actions, 'selected_steps': selected_steps,
              'validation_after_all10': validation, 'validation_macro_after_all10': macro(validation),
              'all10_before_specialization': general, 'all10_macro_before_specialization': macro(general),
              'specialization_action': args.special_action, 'specialization_steps': args.special_steps,
              'all10_after_specialization': specialized, 'all10_macro_after_specialization': macro(specialized),
              'metric_note': 'MPJPE and root-relative global rotation error vs calibrated Unity HumanPose SMPL pseudo-labels; not independent mocap GT.'}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
