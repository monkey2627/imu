"""Train the streaming model on whole-sequence crops with a fixed K-frame lag (output step s = pose of frame s-K).

  pretrain:  python -m mobileposer.stream.train --out stream_k5_s42 --iters 20000
  finetune:  python -m mobileposer.stream.train --out stream_k5_s42_ftr3 --init stream_k5_s42 --dip --replay 3 --lr 5e-5 --iters 1500
Checkpoints: checkpoints_research/<out>/model.pt (+ config.json, log.txt).
"""
import argparse
import json
import time
from pathlib import Path

import torch

from mobileposer.stream.data import Store, batch_inputs
from mobileposer.stream.features import root_relative_ori, elbow_twist, REDUCED, TWIST_ARMS
from mobileposer.stream.model import StreamPoser, LEAF

OUT = Path('checkpoints_research')


def lagged_loss(model, acc, ori, jrel, r6d, mask, a, mount):
    x, ori_seen = batch_inputs(acc, ori, a.gyro, mount, model.frame, model.local)
    pj, pr, _ = model(x)
    leaf = model.aux.get('leaf') if model.arch == 'pnp3' else None
    K = a.lag
    if K:
        pj, pr, jrel, r6d, mask = pj[:, K:], pr[:, K:], jrel[:, :-K], r6d[:, :-K], mask[:, K:]
        leaf = leaf[:, K:] if leaf is not None else None
    # output step s is frame s-K: anchor it to the measured orientation of that frame (already received, causal)
    pr = model.compose(pr, root_relative_ori(ori_seen[:, :ori_seen.shape[1] - K]))
    m = mask.unsqueeze(-1).float()
    lj = (((pj - jrel) ** 2) * m).sum() / (m.sum() * jrel.shape[-1])
    lr = (((pr - r6d) ** 2) * m).sum() / (m.sum() * r6d.shape[-1])
    loss = lr + a.wj * lj
    if leaf is not None:                       # pnp3: supervise the leaf-joint stage as well
        tl = jrel.view(*jrel.shape[:2], 23, 3)[:, :, [j - 1 for j in LEAF]].reshape(leaf.shape)
        loss = loss + a.wj * (((leaf - tl) ** 2) * m).sum() / (m.sum() * leaf.shape[-1])
    return loss, lr.item(), lj.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--init', default=None)
    ap.add_argument('--amass', default='data/stream_pnp5_60a.pt')
    ap.add_argument('--dip', action='store_true', help='fine-tune on DIP s_01..s_08 with AMASS replay')
    ap.add_argument('--dip_store', default='data/stream_dip_train.pt')
    ap.add_argument('--replay', type=int, default=3)
    ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--crop', type=int, default=1200)
    ap.add_argument('--lag', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wj', type=float, default=10.0, help='joint-position loss weight (metres)')
    ap.add_argument('--gyro', type=int, default=1)
    ap.add_argument('--mount_deg', type=float, default=0.0, help='random constant sensor-to-bone offset augmentation (deg)')
    ap.add_argument('--anchor', default='none', choices=['none', 'arms', 'limbs', 'armtw'], help='predict unsensed parents relative to their measured child segment')
    ap.add_argument('--ft_only', default='all', choices=['all', 'pos', 'rot'],
                    help='diagnostic: update only the position stage(s) or only the rotation stage')
    ap.add_argument('--local_gyro', type=int, default=0, help='1: add limb angular velocity in the sensors\' own frames')
    ap.add_argument('--arch', default='mp2', choices=['mp2', 'pnp3'], help='cascade structure (see model.py)')
    ap.add_argument('--frame', default='rel', choices=['rel', 'world', 'both'], help='input representation (features.FRAMES)')
    ap.add_argument('--twist_bound_deg', type=float, default=15.0, help="'armtw': range of the learned elbow-twist residual")
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device('cuda:0')
    out = OUT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if a.init:
        cfg = json.load(open(OUT / a.init / 'config.json'))
        for k, default in (('lag', None), ('gyro', None), ('hidden', None), ('layers', None), ('anchor', 'none'), ('frame', 'rel'), ('arch', 'mp2'), ('local_gyro', 0)):
            assert getattr(a, k) == cfg.get(k, default), f'{k} differs from the initialisation checkpoint'
    json.dump(vars(a), open(out / 'config.json', 'w'), indent=1)
    log = open(out / 'log.txt', 'a')

    model = StreamPoser(bool(a.gyro), a.hidden, a.layers, anchor=a.anchor, frame=a.frame, arch=a.arch,
                        local=bool(a.local_gyro)).to(dev)
    amass = Store(a.amass, dev)
    if a.init:
        model.load_state_dict(torch.load(OUT / a.init / 'model.pt', map_location=dev))
    elif a.anchor == 'armtw':
        # anatomical constants from the AMASS (mocap) labels only: upper-arm rest axes and the mean elbow twist
        from mobileposer.articulate.model import ParametricModel
        from mobileposer.config import paths
        import mobileposer.articulate as art
        J = ParametricModel(paths.smpl_file)._J.float().view(24, 3).to(dev)
        G = art.math.r6d_to_rotation_matrix(amass.r6d[::20].reshape(-1, 6)).view(-1, len(REDUCED), 3, 3)
        for i, (j, _) in enumerate(TWIST_ARMS):
            ax = torch.nn.functional.normalize(J[j + 2] - J[j], dim=0)                 # shoulder 16/17 -> elbow 18/19
            t = elbow_twist(G[:, REDUCED.index(j)], G[:, REDUCED.index(j + 2)], ax)
            model.arm_axes[i] = ax
            model.arm_tau[i] = torch.atan2(torch.sin(t).mean(), torch.cos(t).mean())
        model.arm_bound.fill_(a.twist_bound_deg * 3.14159265 / 180)
        msg = f'armtw prior: tau = {[round(float(x) * 57.2958, 1) for x in model.arm_tau]} deg, bound {a.twist_bound_deg} deg'
        print(msg, flush=True); log.write(msg + '\n')
    val = Store(a.amass.replace('.pt', '_val.pt'), dev)
    dip = Store(a.dip_store, dev) if a.dip else None
    with torch.random.fork_rng(devices=[dev]):          # fixed validation crops, without touching the training RNG stream
        torch.manual_seed(0)
        val_batches = [val.sample(a.bs, a.crop) for _ in range(8)]
    if a.ft_only != 'all':
        rot_stage = ('ik.',) if a.arch == 'pnp3' else ('inp2.', 'rnn2.', 'out2.')
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith(rot_stage) == (a.ft_only == 'rot'))
        log.write(f'trainable: {[n for n, p in model.named_parameters() if p.requires_grad]}\n')
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(a.iters * 0.8)], 0.1)
    n_dip = a.bs // (1 + a.replay) if a.dip else 0

    t0 = time.time()
    for it in range(1, a.iters + 1):
        model.train()
        parts = [amass.sample(a.bs - n_dip, a.crop)]
        if n_dip:
            parts.append(dip.sample(n_dip, a.crop))
        batch = [torch.cat(p, 0) for p in zip(*parts)]
        loss, lr_, lj_ = lagged_loss(model, *batch, a, a.mount_deg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if it % 100 == 0:
            msg = f'it {it} loss {loss.item():.5f} rot {lr_:.5f} joint {lj_:.6f} {time.time() - t0:.0f}s'
            if it % 1000 == 0 or it == a.iters:
                model.eval()
                with torch.no_grad():
                    vl = [lagged_loss(model, *vb, a, 0.0)[0].item() for vb in val_batches]
                msg += f' | val {sum(vl) / len(vl):.5f}'
                torch.save(model.state_dict(), out / 'model.pt')
            print(msg, flush=True); log.write(msg + '\n'); log.flush()
    torch.save(model.state_dict(), out / 'model.pt')


if __name__ == '__main__':
    main()
