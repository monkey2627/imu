"""Build the fixed PNP surface attachment configuration used by line14 and
the PNP / PIP / TransPose standard test datasets:
lower arms 1961 / 5424, lower legs 1176 / 4662, head 411, pelvis 3021.  Orientation = the global rotation of the bone joint the
sensor sits on (18/19 forearm, 4/5 shank, 15 head, 0 pelvis), mounting rotation identity.

Run from SparseIMU/:  python -m mobileposer.make_pnp_attachments
"""
import json
from pathlib import Path

import numpy as np

from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from mobileposer.surface_imu import DEFAULT_CONFIG, file_hash

VERTEX = {'lw': (1961, 18), 'rw': (5424, 19), 'ls': (1176, 4), 'rs': (4662, 5), 'head': (411, 15), 'pelvis': (3021, 0)}


def main():
    model = ParametricModel(paths.smpl_file)
    faces = np.asarray(model.face)
    cfg = json.loads(Path(DEFAULT_CONFIG).read_text())
    cfg['attachments'] = {}
    for name, (v, joint) in VERTEX.items():
        f = int(np.where((faces == v).any(1))[0][0])
        verts = list(map(int, faces[f]))
        w = [1.0 if x == v else 0.0 for x in verts]
        cfg['attachments'][name] = {'joint': joint, 'face': f, 'vertices': verts, 'weights': w,
                                    'mount_rotation': np.eye(3).tolist(), 'description': f'PNP/PIP sensor vertex {v}'}
    cfg['smpl_sha256'] = file_hash(paths.smpl_file)
    out = Path(__file__).parent / 'configs/pnp_surface_imu.json'
    out.write_text(json.dumps(cfg, indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
