"""Body context from an SMPL shape: what the SMPL template tells the model about THIS body and WHERE the sensors sit.

Layout of the context vector (N_CTX = 37, all in metres, bone frame at rest pose):
  [ 0:12]  'r'    lever arms of the four limb sensors (lw, rw, ls, rs) relative to the proximal joint of their bone
                  (elbow 18/19 for forearms, knee 4/5 for shanks): the sensor surface point minus that joint, rest pose
  [12:22]  'len'  body size: thigh, shank, upper arm, forearm, torso, neck-head, hip width, shoulder width, height, foot
  [22:37]  'pos'  template surface coordinates of the five sensors relative to the pelvis (IMUCoCo-style coordinates)
Computed for neutral pose; at rest all bone frames coincide with the world frame, so the vectors are bone-frame vectors.
The shape can come from AMASS (training), or from an image: any SMPL / SMPL-X predictor gives limb lengths, and the 'len'
group can be filled from predicted joints directly (see ctx_from_joints).
"""
import torch

from mobileposer.config import paths
from mobileposer.articulate.model import ParametricModel
from mobileposer.surface_imu import DEFAULT_CONFIG, load_attachments, attachment_points

LABELS = ['lw', 'rw', 'ls', 'rs', 'pelvis']
BONE_JOINT = [18, 19, 4, 5]
N_CTX = 37
GROUPS = {'r': slice(0, 12), 'len': slice(12, 22), 'pos': slice(22, 37)}

_state = {}
_cache = {}


def _init():
    if not _state:
        _state['model'] = ParametricModel(paths.smpl_file)
        _state['cfg'] = load_attachments(DEFAULT_CONFIG)


def lengths_from_joints(j):
    """j [24,3] rest-pose joint positions (y up) -> 10 body-size numbers"""
    d = lambda a, b: (j[a] - j[b]).norm()
    return torch.stack([(d(1, 4) + d(2, 5)) / 2, (d(4, 7) + d(5, 8)) / 2, (d(16, 18) + d(17, 19)) / 2, (d(18, 20) + d(19, 21)) / 2,
                        d(9, 0), d(15, 9), d(1, 2), d(16, 17), j[15, 1] - torch.minimum(j[10, 1], j[11, 1]), (d(7, 10) + d(8, 11)) / 2])


def body_context(shape):
    """shape: [10] SMPL betas -> [N_CTX]"""
    _init()
    key = tuple(round(float(x), 4) for x in shape.view(-1))
    if key in _cache:
        return _cache[key]
    model, cfg = _state['model'], _state['cfg']
    with torch.no_grad():
        j, v = model.get_zero_pose_joint_and_vertex(shape.view(1, 10).float())
        pts = attachment_points(v, cfg, LABELS)[0]                        # [5,3]
        r = pts[:4] - j[0, BONE_JOINT]
        ctx = torch.cat([r.reshape(-1), lengths_from_joints(j[0]), (pts - j[0, :1]).reshape(-1)])
    _cache[key] = ctx
    return ctx


def ctx_from_joints(j_rest, shape_for_r=None):
    """Context from predicted (e.g. SMPL-X image-estimator) rest-pose joints: lengths from the joints, r and pos from the
    template scaled by the ratio of predicted to template height (sensors are placed on the mean body)."""
    base = body_context(torch.zeros(10))
    ln = lengths_from_joints(j_rest)
    s = ln[8] / base[12 + 8]
    ctx = base.clone()
    ctx[GROUPS['len']] = ln
    ctx[GROUPS['r']] = base[GROUPS['r']] * s
    ctx[GROUPS['pos']] = base[GROUPS['pos']] * s
    return ctx
