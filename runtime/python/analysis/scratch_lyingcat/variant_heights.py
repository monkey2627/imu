import sys, numpy as np, torch
from analysis.scratch_lyingcat.rule_only import run
from analysis.scratch_lyingcat.contacts import model, masks
import production.mesh_video as mv
from production.rotmat import batch_smpl_matrix_to_unity_quat_xyzw
torch.set_grad_enabled(False)
WIN = {"lyingcat_20261123_044417": (21, 57), "lyingcat_20261123_044239": (9.5, 47), "lyingcat_20261120_230158": (9, 51.5)}

def heights(stem, **kw):
    t, L, _ = run(stem, **kw); a, b = WIN[stem]
    idx = np.where((t > a) & (t < b))[0][::3]
    H = np.array([[100 * mv.skin(model, batch_smpl_matrix_to_unity_quat_xyzw(L[i].astype(np.float32)), True)[m, 1].min()
                   for m in masks.values()] for i in idx])
    sp = H.max(1) - H.min(1)
    return f"handL {np.percentile(H[:,0],95):.1f} handR {np.percentile(H[:,1],95):.1f} spread p95 {np.percentile(sp,95):.1f} max {sp.max():.1f}"
