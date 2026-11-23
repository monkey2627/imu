"""Q2/Q5: is the cat/cow signal in the waist sensor?  World-frame decomposition of the waist rotation."""
import sys, numpy as np
from scipy.spatial.transform import Rotation as R
from production.arm_prior import _globals
from production.bridge_contact import _rest_offsets
from production.arm_prior import PARENTS
OFF = _rest_offsets()

def pos(g):
    p = np.zeros((24, 3))
    for j in range(1, 24): p[j] = p[PARENTS[j]] + g[PARENTS[j]] @ OFF[j]
    return p

def elev(v): return np.degrees(np.arcsin(np.clip(v[..., 1] / np.linalg.norm(v, axis=-1), -1, 1)))

def main(stems):
  for stem in stems:
      d = np.load(f"analysis/scratch_lyingcat/cap_{stem}_prod.npz")
      t, tt = d["t"], d["tick_t"]
      W = list(d["labels"]).index("waist")
      after = t > 9.0
      # headward from the network raw output (mean over the exercise)
      G = np.array([_globals(l.astype(np.float64)) for l in d["raw"]])
      P = np.array([pos(g) for g in G])
      hw = (P[after, 12] - P[after, 0]).mean(0); hw[1] = 0; hw /= np.linalg.norm(hw)
      lat = np.cross([0, 1, 0], hw)      # body-left
      up = np.array([0, 1, 0.])
      print(f"== {stem}: headward {np.round(hw,2)}  (azimuth {np.degrees(np.arctan2(hw[0], hw[2])):.0f} deg)")
      for name, Rs, tm in (("uncal", d["uncal"][:, W], tt), ("calibrated", d["measured"][:, W], t)):
          m = tm > 9.0
          Rm = Rs[m].astype(np.float64)
          ref = R.from_matrix(Rm).mean().as_matrix()
          rv = R.from_matrix(Rm @ ref.T).as_rotvec()          # world-frame delta from the mean
          comp = np.degrees(np.c_[rv @ lat, rv @ hw, rv @ up])
          # PCA dominant axis
          u, s, vt = np.linalg.svd(rv - rv.mean(0), full_matrices=False)
          ax = vt[0]; ax = ax if ax @ lat > 0 else -ax
          ang_lat = np.degrees(np.arccos(abs(ax @ lat)))
          std = comp.std(0); p2p = np.percentile(comp, 95, axis=0) - np.percentile(comp, 5, axis=0)
          # sensor local axes in world: which local axis is the waist "spine" (Y) and its elevation
          yax = Rm[:, :, 1]; zax = Rm[:, :, 2]
          print(f"  {name:10s} p5-p95 range (deg)  pitch(lat) {p2p[0]:5.1f}  roll(fwd) {p2p[1]:5.1f}  yaw(up) {p2p[2]:5.1f}"
                f" | PCA axis vs lateral {ang_lat:4.0f} deg, var share {s[0]**2/(s**2).sum():.2f}"
                f" | mean local Y {np.round(yax.mean(0),2)} elevY p5-95 {np.percentile(elev(yax),5):.0f}..{np.percentile(elev(yax),95):.0f}"
                f" mean local Z {np.round(zax.mean(0),2)}")
      print("  headings waist:", np.round(R.from_matrix(d['head'][-1][W]).as_euler('YXZ', degrees=True), 1),
            " corrections waist:", np.round(R.from_matrix(d['corr'][-1][W]).as_euler('YXZ', degrees=True), 1))
      for k in range(5):
          print(f"    label {d['labels'][k]} heading YXZ {np.round(R.from_matrix(d['head'][-1][k]).as_euler('YXZ', degrees=True),1)} corr {np.round(R.from_matrix(d['corr'][-1][k]).as_rotvec(degrees=True),1)}")

if __name__ == '__main__':
    main(sys.argv[1:])
