#!/usr/bin/env python3
"""Train PosAccPoser (posacc_model.py), alongside the three waist-surface baselines (2026-11-06).

Same AMASS store, crop, lag, optimiser, schedule and seed as stream_s42_amass (run_stream_training.sh);
added:
  - the coach's Ours motions as clean synthetic IMU (build_posacc_ours_synth.py), a fixed share of every
    batch: the quadruped leg lifts / supine extensions that AMASS lacks, where only the acceleration
    tells a raised leg from a kneeling one;
  - stage-P losses on the limb sensors' positions and velocities relative to the waist sensor;
  - the stage-P state initialised from the first frame's sensor positions for a share of the crops
    (the rest: unknown start), as PNP's rnn_initialize;
  - acceleration noise measured on the desk tests: per sequence and sensor a gain ~ exp(N(0, 0.3)) and a
    bias ~ N(0, 0.1) m/s^2, plus white noise N(0, 0.15) m/s^2; and the baselines' 8 deg mount augmentation.
Existing code is only imported; output: checkpoints/<out>/ (model.pt, config.json, log.txt).

  python train_posacc.py --out posacc_s42_amass --gpu 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pnp_workspace"))
sys.path.insert(0, str(ROOT))
from mobileposer.stream.data import Store, batch_inputs        # noqa: E402
from mobileposer.stream.features import root_relative_ori      # noqa: E402
from posacc_model import PosAccPoser, sensor_targets           # noqa: E402
from mobileposer.stream.model import StreamPoser               # noqa: E402  (control only)


def noisy(acc, a):
    B = acc.shape[0]
    gain = torch.exp(torch.randn(B, 1, 5, 1, device=acc.device) * a.acc_gain_sigma)
    bias = torch.randn(B, 1, 5, 3, device=acc.device) * a.acc_bias_sigma
    return acc * gain + bias + torch.randn_like(acc) * a.acc_noise_sigma


def loss_fn(model, acc, ori, jrel, r6d, mask, a, train):
    if train:
        acc = noisy(acc, a)
    x, seen = batch_inputs(acc, ori, a.gyro, a.mount_deg if train else 0.0, model.frame, model.local)
    pv_t = sensor_targets(jrel)
    known = (torch.rand(len(x), device=x.device) < a.init_known) if train else torch.zeros(len(x), dtype=torch.bool, device=x.device)
    if isinstance(model, PosAccPoser):
        pj, pr, _ = model(x, init_pos=pv_t[:, 0, :12], known=known)
        pv = model.aux['pv']
    else:                                   # control: baseline StreamPoser, no stage P
        pj, pr, _ = model(x)
        pv = pv_t
    K = a.lag
    if K:
        pj, pr, pv = pj[:, K:], pr[:, K:], pv[:, K:]
        jrel, r6d, pv_t, mask = jrel[:, :-K], r6d[:, :-K], pv_t[:, :-K], mask[:, K:]
    pr = model.compose(pr, root_relative_ori(seen[:, :seen.shape[1] - K]))
    m = mask.unsqueeze(-1).float()
    mse = lambda p, t: (((p - t) ** 2) * m).sum() / (m.sum() * t.shape[-1])
    lr, lj = mse(pr, r6d), mse(pj, jrel)
    lp, lv = mse(pv[..., :12], pv_t[..., :12]), mse(pv[..., 12:], pv_t[..., 12:])
    return lr + a.wj * lj + a.wp * lp + a.wv * lv, (lr.item(), lj.item(), lp.item(), lv.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--amass', default=str(ROOT / 'data/stream_waist_surface_60.pt'))
    ap.add_argument('--ours_synth', default=str(ROOT / 'data/posacc_ours_synth.pt'))
    ap.add_argument('--ours_frac', type=float, default=0.25, help='share of each batch from the Ours synthetic motions')
    ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--crop', type=int, default=1200)
    ap.add_argument('--lag', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wj', type=float, default=10.0)
    ap.add_argument('--wp', type=float, default=10.0, help='sensor position loss weight (m)')
    ap.add_argument('--wv', type=float, default=1.0, help='sensor velocity loss weight (m/s)')
    ap.add_argument('--gyro', type=int, default=1)
    ap.add_argument('--mount_deg', type=float, default=8.0)
    ap.add_argument('--init_known', type=float, default=0.5, help='share of crops with a known start pose')
    ap.add_argument('--acc_gain_sigma', type=float, default=0.3)
    ap.add_argument('--acc_bias_sigma', type=float, default=0.1)
    ap.add_argument('--acc_noise_sigma', type=float, default=0.15)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--model', default='posacc', choices=['posacc', 'stream'],
                    help="'stream': control run, the baseline StreamPoser (mp2, frame both) on the same data/noise")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device(f'cuda:{a.gpu}')
    out = ROOT / 'checkpoints' / a.out
    out.mkdir(parents=True, exist_ok=False)
    json.dump({**vars(a), 'arch': 'posacc' if a.model == 'posacc' else 'mp2', 'frame': 'both', 'anchor': 'none',
               'local_gyro': 0}, open(out / 'config.json', 'w'), indent=1)
    log = open(out / 'log.txt', 'a')
    model = (PosAccPoser(bool(a.gyro), a.hidden, a.layers, frame='both') if a.model == 'posacc' else
             StreamPoser(bool(a.gyro), a.hidden, a.layers, frame='both', arch='mp2')).to(dev)
    amass, ours = Store(a.amass, dev), Store(a.ours_synth, dev)
    val = Store(a.amass.replace('.pt', '_val.pt'), dev)
    with torch.random.fork_rng(devices=[dev]):
        torch.manual_seed(0)
        val_batches = [val.sample(a.bs, a.crop) for _ in range(8)]
        ours_batches = [ours.sample(a.bs, a.crop) for _ in range(4)]
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(a.iters * 0.8)], 0.1)
    n_ours = int(round(a.bs * a.ours_frac))
    t0 = time.time()
    for it in range(1, a.iters + 1):
        model.train()
        parts = [amass.sample(a.bs - n_ours, a.crop)] + ([ours.sample(n_ours, a.crop)] if n_ours else [])
        batch = [torch.cat(p, 0) for p in zip(*parts)]
        loss, (lr_, lj_, lp_, lv_) = loss_fn(model, *batch, a, True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if it % 100 == 0:
            msg = (f'it {it} loss {loss.item():.5f} rot {lr_:.5f} joint {lj_:.6f} pos {lp_:.6f} vel {lv_:.5f} '
                   f'{time.time() - t0:.0f}s')
            if it % 1000 == 0 or it == a.iters:
                model.eval()
                with torch.no_grad():
                    va = [loss_fn(model, *vb, a, False)[0].item() for vb in val_batches]
                    vo = [loss_fn(model, *vb, a, False)[0].item() for vb in ours_batches]
                msg += f' | val amass {sum(va) / len(va):.5f} ours_synth {sum(vo) / len(vo):.5f}'
                torch.save(model.state_dict(), out / 'model.pt')
            print(msg, flush=True); log.write(msg + '\n'); log.flush()
    torch.save(model.state_dict(), out / 'model.pt')


if __name__ == '__main__':
    main()
