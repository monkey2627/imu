"""Turn the fitted lever arms (real2sim.py placement -> configs/fitted_placement.json) into a surface-attachment configuration:
every sensor is put on the template vertex closest to  J_bone + r_fitted  (rest pose, bone frame == world frame at rest).
Writes configs/fitted_surface_imu.json (same format as pnp_surface_imu.json), usable with  no_head_layouts --attachment-config.

Run from SparseIMU/:  python -m mobileposer.make_fitted_attachments
"""
import json
from pathlib import Path

import numpy as np
import torch

from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from mobileposer.surface_imu import file_hash

JOINT = {'lw': 18, 'rw': 19, 'ls': 4, 'rs': 5, 'pelvis': 0}


def main():
    here = Path(__file__).parent
    fit = json.load(open(here / 'configs/fitted_placement.json'))
    r = torch.tensor(fit['lever_arm_m'])
    model = ParametricModel(paths.smpl_file)
    j, v = model.get_zero_pose_joint_and_vertex(None)
    faces = np.asarray(model.face)
    cfg = json.loads((here / 'configs/pnp_surface_imu.json').read_text())
    cfg['attachments'] = {}
    for k, name in enumerate(['lw', 'rw', 'ls', 'rs', 'pelvis']):
        target = j[JOINT[name]] + r[k]
        vid = int(((v - target) ** 2).sum(-1).argmin())
        f = int(np.where((faces == vid).any(1))[0][0])
        verts = list(map(int, faces[f]))
        cfg['attachments'][name] = {'joint': JOINT[name], 'face': f, 'vertices': verts, 'weights': [1.0 if x == vid else 0.0 for x in verts],
                                    'mount_rotation': np.eye(3).tolist(),
                                    'description': f'vertex {vid}: nearest to the sensor placement fitted on DIP-IMU train subjects'}
        print(f'{name:7s} fitted point {np.round(target.numpy(), 3)}  -> vertex {vid} at {np.round(v[vid].numpy(), 3)}  (dist {float((v[vid] - target).norm()) * 100:.1f} cm)')
    cfg['smpl_sha256'] = file_hash(paths.smpl_file)
    (here / 'configs/fitted_surface_imu.json').write_text(json.dumps(cfg, indent=1))
    print('wrote configs/fitted_surface_imu.json')


if __name__ == '__main__':
    main()
