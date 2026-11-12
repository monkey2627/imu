"""Stage C real-motion evaluation driver (CPU/GPU selectable via --device)."""
import argparse, json, os
from pathlib import Path
import numpy as np
import torch

os.environ['MOBILEPOSER_NSENS'] = '5'
import line14_realeval_lib as RL
from mobileposer.evaluate_standard import StdEvaluator, sequences
import mobileposer.articulate as art
from mobileposer.config import paths
from mobileposer.line14.error_sampler import load_sampler, draw_delta


def uncalibrated_tc_sequences(ev):
    """Yield TC inputs with sensor-to-bone rotation RSB omitted."""
    d = torch.load('/home/duanyuhan/dyh/motion/dataset/processed/TotalCapture/totalcapture_officalib.pt', map_location='cpu')
    for i, name in enumerate(d['name']):
        RIM, RIS = d['RIM'][i], d['RIS'][i]
        R = RIM.transpose(1, 2).unsqueeze(0).matmul(RIS)
        a = RIM.transpose(1, 2).unsqueeze(0).matmul(d['aS'][i].unsqueeze(-1)).squeeze(-1)
        a = a + torch.tensor([0., -9.8, 0.])
        pose = art.math.axis_angle_to_rotation_matrix(d['pose'][i].reshape(-1, 3)).view(-1, 24, 3, 3)
        yield name, a[::2, [0, 1, 2, 3, 5]].contiguous(), R[::2, [0, 1, 2, 3, 5]].contiguous(), pose[::2].contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu'); ap.add_argument('--mc', type=int, default=8); ap.add_argument('--ns', type=int, default=8, choices=(1,4,8))
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--tc-input', choices=('official', 'uncal'), default='official')
    ap.add_argument('--out', type=Path, default=Path('results/line14/C31_real.json'))
    a = ap.parse_args(); dev = torch.device(a.device)
    sampler = load_sampler('data_local/line14_errsampler_v2/sampler.pt')
    ev = StdEvaluator(dev); body = art.model.ParametricModel(paths.smpl_file, device=dev)
    seeds = (42, 7, 3)
    models = {k: {s: RL.load_model(k, s, dev) for s in seeds} for k in ('b0','i2s','lvmean')}
    for model in models['lvmean'].values():
        model[0].input_dim = 96
    mean_r = RL.mean_r()
    picks = {'dipimu': None, 'totalcapture_officalib_uncal' if a.tc_input == 'uncal' else 'totalcapture_officalib': None}
    allres = {}
    for ds in picks:
        rows = []
        source = uncalibrated_tc_sequences(ev) if ds.endswith('_uncal') else sequences(ds, ev.art)
        for j, (name, acc, R, pose) in enumerate(source):
            if ds.startswith('totalcapture_officalib') and 'rom' in str(name).lower(): continue
            if a.smoke and rows: break
            if acc.shape[1] != 5:
                # Defensive normalization for a parent process that imported a six-IMU evaluator.
                acc, R = acc[:, [0, 1, 2, 3, -1]], R[:, [0, 1, 2, 3, -1]]
            imu0 = torch.cat([(acc / 30).flatten(1), R.flatten(1)], 1).float()
            gt = pose.to(dev)
            vals = {k: {s: [] for s in seeds} for k in models}
            for mc in range(a.mc):
                delta = draw_delta(sampler, str(name), mc, a.ns)
                for k in models:
                    x = imu0.clone()
                    if k != 'b0': x[:,15:] = RL.inject(R, delta).reshape(len(R), -1)
                    for s in seeds:
                        pred = RL.to_local(body, RL.infer(models[k][s], x, mean_r if k == 'lvmean' else None), dev)
                        vals[k][s].append(ev(pred, gt).numpy())
            rows.append({'name': str(name), 'mc_mean': {k: {str(s): np.mean(vals[k][s],0).tolist() for s in seeds} for k in models}})
            print(ds, name, flush=True)
        allres[ds] = rows
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps({'sampler': sampler['version'], 'ns': a.ns, 'mc': a.mc, 'rows': allres}, indent=1))

if __name__ == '__main__': main()
