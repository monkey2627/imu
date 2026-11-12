"""Where does the orientation-offset estimation error live?  Eval-only on the assembled estimates (data_local/line15_R_<mode>.pt: R, R_true per identity).
Error rotation e = log(R_est^T R_true) (right-multiplicative, expressed in the bone frame in which R_off is defined; zero pose = T-pose so the bone long axis is
x for the forearms, y for the shanks and the pelvis).  Reports per sensor the RMS angle of e (deg) and its components along the long axis (twist) and perpendicular (swing)."""
import sys
import numpy as np, torch
from mobileposer.articulate import math as M

path = sys.argv[1] if len(sys.argv) > 1 else "data_local/line15_R_rot.pt"
d = torch.load(path, weights_only=False)
R, Rt, split = d["R"], d["R_true"], d["split"]
names = ["lw", "rw", "ls", "rs", "pelvis"]; long_axis = [0, 0, 1, 1, 1]
for sp in ("validation", "train"):
    ids = [i for i in R if split[i] == sp]
    E = torch.stack([M.rotation_matrix_to_axis_angle((R[i].transpose(-1, -2) @ Rt[i]).reshape(-1, 3, 3)).reshape(5, 3) for i in ids])   # [N,5,3] rad
    print(f"== {sp} ({len(ids)} identities), error of the identity-level estimate")
    for s in range(5):
        e = torch.rad2deg(E[:, s]); tw = e[:, long_axis[s]]; sw = torch.cat([e[:, a:a + 1] for a in range(3) if a != long_axis[s]], 1)
        print(f"  {names[s]:6s} total RMS {float(e.norm(dim=-1).pow(2).mean().sqrt()):.2f} deg | twist (about the bone axis) RMS {float(tw.pow(2).mean().sqrt()):.2f} | swing RMS {float(sw.pow(2).mean().sqrt()):.2f} "
              f"| per-axis RMS xyz {[round(float(e[:, a].pow(2).mean().sqrt()), 2) for a in range(3)]}")
