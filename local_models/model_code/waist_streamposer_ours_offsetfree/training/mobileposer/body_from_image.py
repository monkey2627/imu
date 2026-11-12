"""Body size from ONE full-body photo, for the body-context input of the IMU pose model.

Run in the LHM conda env on the server (it contains the multi-HMR SMPL-X estimator used by TFAvatar):
    /home/duanyuhan/SoftWare/miniconda3/envs/LHM/bin/python body_from_image.py --image photo.jpg --out body.json

Writes the predicted SMPL-X betas and the 10 body-size numbers (metres) in the order used by mobileposer/body_ctx.py:
thigh, shank, upper arm, forearm, torso, neck-head, hip width, shoulder width, height, foot.
The IMU side then builds the context with body_ctx.ctx_from_joints / ctx_from_lengths (no SMPL-X dependency there).
Single-image shape estimation is only approximately metric (scale is ambiguous); pass --height_m to rescale to the
measured height of the person, which is the recommended use.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath('/home/duanyuhan/dyh/TFAvatar/LHM/engine'))
sys.path.insert(0, os.path.abspath('/home/duanyuhan/dyh/TFAvatar/LHM'))
from engine.multi_hmr.smplxEstimator import SmplxEstimator


def lengths(j):
    d = lambda a, b: float(np.linalg.norm(j[a] - j[b]))
    return [(d(1, 4) + d(2, 5)) / 2, (d(4, 7) + d(5, 8)) / 2, (d(16, 18) + d(17, 19)) / 2, (d(18, 20) + d(19, 21)) / 2,
            d(9, 0), d(15, 9), d(1, 2), d(16, 17), float(j[15, 1] - min(j[10, 1], j[11, 1])), (d(7, 10) + d(8, 11)) / 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--height_m', type=float, default=None, help='measured body height; rescales all lengths')
    ap.add_argument('--gpu', default='0')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    est = SmplxEstimator('/home/duanyuhan/dyh/TFAvatar/LHM/engine/multi_hmr/models/multiHMR/multiHMR_896_L.pt')
    human = est(args.image)
    if human is None:
        raise SystemExit('no person found in the image')
    betas = human[0]['shape'].detach().cpu().float().view(1, -1)
    bm = est.model.smpl_layer['neutral_10'].bm_x                      # smplx body model (neutral, 10 betas)
    with torch.no_grad():
        out = bm(betas=betas[:, :10].to(next(bm.parameters()).device))   # zero pose
    j = out.joints[0, :22].cpu().numpy()
    j = np.concatenate([j, j[20:22]], axis=0)                          # pad to the 24 SMPL joints (hands = wrists; not used)
    ln = lengths(j)
    if args.height_m:
        ln = [x * args.height_m / ln[8] for x in ln]
    json.dump({'betas': betas[0].tolist(), 'lengths_m': ln, 'height_rescaled_to': args.height_m}, open(args.out, 'w'), indent=2)
    print('lengths (thigh, shank, uarm, farm, torso, neck-head, hipw, shoulderw, height, foot):', np.round(ln, 3))


if __name__ == '__main__':
    main()
