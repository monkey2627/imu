"""Identity-level geometry inputs for line 14 (validation-only synthetic experiments).

Kinds (all computed from the SMPL zero pose = T pose, PNP surface attachments, SMPL global axes):
  r1      12  pelvis-IMU -> {lw, rw, ls, rs} IMU vectors
  height   1  vertical extent of the zero-pose mesh
  arms     6  the two wrist vectors of r1
  legs     6  the two shank vectors of r1
  r2      24  sensor -> joint vectors J[j] - p_k for (lw: elbow, shoulder; rw: elbow, shoulder; ls: knee, hip; rs: knee, hip)
  r2m     24  r2 with the installation offset:  J[j] - (p_k + l_k)   (what the sensor motion physically constrains)
  est     24  *estimated* r2 (E8): mean of the calibration-session estimate, read from ``est_file`` (identity -> sessions)
  est_sig 48  est plus the estimator's per-component sigma-hat (same file)
  r2scale  1  mean norm of the 8 R2 vectors of the true shape (a body-scale scalar)
  est_scale 1 the same scalar computed from an estimated R2 session
  r1m     12  r1 plus the per-identity mount (lever-arm) offsets of the data store:  v_k = r1_k + (l_k - l_pelvis),
              l = mobileposer.line9.calib.identity_offsets(identity, 0, lever_cm)[1]  (zero pose: bone frame == global axes)

Perturbations (applied to the *raw* vectors, before the model's training-identity normalisation):
  none      identity
  indep     each 3-vector v_k <- v_k * (1 + sigma * n_k),   n_k ~ N(0,1) independent
  global    all vectors <- v * (1 + sigma * n),             one n shared (a pure scale error)
  add       v <- v + sigma_cm/100 * n,   n ~ N(0, I) per component (isotropic additive error, sigma in centimetres)
  shuffle   replace by the geometry of a different (deterministic partner) identity
"""
from __future__ import annotations

import hashlib

import torch

from mobileposer.line14.r1 import R1Table
from mobileposer.line9.data import _beta_key

KINDS = {"hrestp3": 4, "hrestp6": 7, "hestp3": 4, "hestp6": 7, "estp1s": 2, "estp3s": 6, "estp1g": 1, "estp3g": 3, "pca1": 1, "pca2": 2, "pca3": 3, "pca4": 4, "pca6": 6, "estp1": 1, "estp2": 2, "estp3": 3, "estp4": 4, "estp6": 6, "r1": 12, "height": 1, "arms": 6, "legs": 6, "r1m": 12, "r2": 24, "r2m": 24, "est": 24, "est_sig": 48, "r2scale": 1, "est_scale": 1}
# R2: sensor -> unsensed-joint vectors in the zero pose (sensor frame == global axes there): (sensor index, joint id)
R2_PAIRS = ((0, 18), (0, 16), (1, 19), (1, 17), (2, 4), (2, 1), (3, 5), (3, 2))


_PCA = {}


def scale_ident(identity, lo, hi):
    """Deterministic per-identity global body scale in [lo, hi] (wide-population experiments, E19+)."""
    u = int.from_bytes(hashlib.sha256(f"{identity}|scale14".encode()).digest()[:8], "little") / 2 ** 64
    return lo + (hi - lo) * u


def r2_pca_basis(path="data_local/line14_r2_pca.pt"):
    """Mean and orthonormal principal directions of the true R2 (24-d, metres) over the *training* identities (cached)."""
    if "b" in _PCA:
        return _PCA["b"]
    import os
    if os.path.isfile(path):
        _PCA["b"] = torch.load(path, weights_only=False); return _PCA["b"]
    from mobileposer.line9.data import load_identity_records
    rec = load_identity_records(folds=("train",))
    betas = {}
    for r in rec["train"]:
        betas.setdefault(r.identity, torch.tensor(r.beta, dtype=torch.float32).reshape(-1)[:10])
    table = R1Table()
    X = []
    for b in betas.values():
        pts, joints = table.points_joints(b)
        X.append(torch.stack([joints[j] - pts[k] for k, j in R2_PAIRS]).reshape(24))
    X = torch.stack(X); mean = X.mean(0)
    U, S, Vh = torch.linalg.svd(X - mean, full_matrices=False)
    _PCA["b"] = {"mean": mean, "V": Vh.T.contiguous(), "S": S / (len(X) - 1) ** 0.5}
    torch.save(_PCA["b"], path)
    return _PCA["b"]


class GeomProvider:
    def __init__(self, kind: str = "r1", lever_cm: float = 3.0, est_file: str | None = None, scale_range=None):
        if kind not in KINDS:
            raise ValueError(f"unknown geometry kind {kind!r}")
        self.kind = kind
        self.lever_cm = float(lever_cm)
        self.scale_range = scale_range              # (lo, hi) or None; applied to *oracle* geometry only (estimates are already scaled)
        self.dim = KINDS[kind]
        self._parts = None
        if kind.startswith("hestp") or kind.startswith("hrestp"):   # composite: [true height (scaled in wide populations) | estimated PivotPrint PCs (hrestp: with the height-explained part removed)]
            k = kind.split("estp")[1]
            self._resid = kind.startswith("hrestp"); self._W = None; self._k = k; self._lever = lever_cm; self._sr = scale_range
            self._parts = (GeomProvider("height", lever_cm, None, scale_range), GeomProvider("estp" + k, lever_cm, est_file, None))
            self.est = None; self.table = None; self._cache = {}; self.partner = {}; self.quality = None
            return
        import re
        m = re.match(r"estp(\d)([sg]?)$", kind)
        self.pca_k = int(kind[3:]) if kind.startswith("pca") else (int(m.group(1)) if m else 0)
        self.est_flag = m.group(2) if m else ""         # "s": append sigma-hat in PC space, "g": shrinkage gate with sigma-hat
        self.quality = None                              # None | "good" | "bad": restrict estimate sessions (E13)
        self.table = R1Table()
        self.est = torch.load(est_file, weights_only=False) if kind in ("est", "est_sig", "est_scale") or kind.startswith("estp") else None
        self._cache = {}
        self.partner = {}

    def _resid_fit(self):
        """Per-PC regression of the TRUE pca-k coordinates on [1, h, h^2] over the training identities (wide population: with the identity scale)."""
        if self._W is not None:
            return
        from mobileposer.line9.data import load_identity_records
        ph, pz = GeomProvider("height", self._lever, None, self._sr), GeomProvider("pca" + self._k, self._lever, None, self._sr)
        seen, H, Z = set(), [], []
        for r in load_identity_records(folds=("train",))["train"]:
            if r.identity in seen:
                continue
            seen.add(r.identity); beta = torch.tensor(r.beta, dtype=torch.float32).reshape(-1)[:10]
            H.append(ph.base(beta, r.identity)); Z.append(pz.base(beta, r.identity))
        H, Z = torch.stack(H), torch.stack(Z)
        self._hm, self._hs = H.mean(), H.std().clamp_min(1e-6)
        self._W = torch.linalg.lstsq(self._phi(H), Z).solution

    def _phi(self, h):
        t = (h.reshape(-1, 1) - self._hm) / self._hs if hasattr(self, "_hm") else h.reshape(-1, 1)
        return torch.cat((torch.ones_like(t), t, t * t), dim=-1)

    def _fuse(self, h, z):
        if not self._resid:
            return torch.cat((h, z), dim=-1)
        self._resid_fit()
        return torch.cat((h, z - self._phi(h) @ self._W), dim=-1)

    # ---- base (clean) geometry ------------------------------------------------------
    def base(self, beta, identity=None) -> torch.Tensor:
        if self._parts is not None:
            return self._fuse(self._parts[0].base(beta, identity).reshape(1, -1), self._parts[1].base(beta, identity).reshape(1, -1)).reshape(-1)
        v = self._base(beta, identity)
        if self.scale_range is not None and self.est is None and not self.kind.startswith("pca"):
            v = v * scale_ident(identity, *self.scale_range)      # pca kinds are scaled before projection (inside _base)
        return v

    def _base(self, beta, identity=None) -> torch.Tensor:
        if self.est is not None and self.kind.startswith("estp"):
            e = self.est[str(identity)]
            return torch.stack([self._estp(e, s) for s in range(e["mu"].shape[0])]).mean(0)
        if self.est is not None:      # statistics geometry: mean over the identity's sessions
            e = self.est[str(identity)]
            if self.kind == "est_scale": return e["mu"].reshape(-1, 8, 3).norm(dim=-1).mean(-1).mean().reshape(1)
            return torch.cat((e["mu"].mean(0), e["sigma"].mean(0))) if self.kind == "est_sig" else e["mu"].mean(0)
        beta = torch.as_tensor(beta, dtype=torch.float32).reshape(-1)[:10]
        key = _beta_key(beta) + (f"|{identity}" if self.kind in ("r1m", "r2m") else "")
        if key not in self._cache:
            if self.kind == "height":
                with torch.no_grad():
                    _, verts = self.table.body.get_zero_pose_joint_and_vertex(beta)
                y = verts.reshape(-1, 3)[:, 1]
                v = (y.max() - y.min()).reshape(1).float()
            elif self.kind.startswith("pca"):
                b = r2_pca_basis()
                pts, joints = self.table.points_joints(beta)
                r2 = torch.stack([joints[j] - pts[k] for k, j in R2_PAIRS]).reshape(24).float()
                if self.scale_range is not None:
                    r2 = r2 * scale_ident(identity, *self.scale_range)
                v = ((r2 - b["mean"]) @ b["V"][:, : self.pca_k]).float()
            elif self.kind == "r2scale":
                pts, joints = self.table.points_joints(beta)
                v = torch.stack([joints[j] - pts[k] for k, j in R2_PAIRS]).norm(dim=-1).mean().reshape(1).float()
            elif self.kind in ("r2", "r2m"):
                pts, joints = self.table.points_joints(beta)
                if self.kind == "r2m":
                    from mobileposer.line9.calib import identity_offsets
                    pts = pts + identity_offsets(str(identity), 0.0, self.lever_cm)[1]
                v = torch.stack([joints[j] - pts[k] for k, j in R2_PAIRS]).reshape(24).float()
            elif self.kind == "r1m":
                from mobileposer.line9.calib import identity_offsets
                r1 = self.table.get(beta)
                lever = identity_offsets(str(identity), 0.0, self.lever_cm)[1]      # [5,3] metres, label order lw,rw,ls,rs,pelvis
                v = (r1.reshape(4, 3) + (lever[:4] - lever[4:5])).reshape(12).float()
            else:
                r1 = self.table.get(beta)
                v = {"r1": r1, "arms": r1[:6], "legs": r1[6:]}[self.kind]
            self._cache[key] = v.clone()
        return self._cache[key]

    def _estp(self, e, s):
        """PC-space estimate of session s: z (k), optionally with sigma-hat (s) or shrinkage-gated (g)."""
        b = r2_pca_basis(); k = self.pca_k
        z = (e["mu"][s] - b["mean"]) @ b["V"][:, :k]
        if not self.est_flag:
            return z.float()
        sig = ((e["sigma"][s] ** 2) @ (b["V"][:, :k] ** 2)).sqrt()
        if self.est_flag == "s":
            return torch.cat((z, sig)).float()
        tau = b["S"][:k]
        return (z * tau ** 2 / (tau ** 2 + sig ** 2)).float()

    def _session(self, e, ident, fixed_seed):
        S = e["mu"].shape[0]
        cand = list(range(S))
        if self.quality is not None and "q" in e:
            want = 0 if self.quality == "good" else 1
            cand = [i for i in cand if int(e["q"][i]) == want] or cand
        if fixed_seed is None:
            return cand[int(torch.randint(len(cand), (1,)))]
        return cand[int.from_bytes(hashlib.sha256(f"{ident}|{fixed_seed}|sess".encode()).digest()[:4], "little") % len(cand)]

    def set_pool(self, identity_beta: dict) -> None:
        """Deterministic derangement used by the ``shuffle`` perturbation."""
        ids = sorted(identity_beta)
        n = len(ids)
        shift = max(n // 2, 1)
        self.partner = {ids[i]: (ids[(i + shift) % n], identity_beta[ids[(i + shift) % n]]) for i in range(n)}

    # ---- perturbation --------------------------------------------------------------
    def _perturb(self, v, mode, sigma, gen):
        if mode in ("none", None) or sigma == 0:
            return v
        if mode == "add":
            return v + (sigma / 100.0) * torch.randn(v.shape, generator=gen)
        k = max(self.dim // 3, 1) if self.kind != "height" else 1
        vv = v.reshape(k, -1)
        if mode == "indep":
            s = 1 + sigma * torch.randn(k, 1, generator=gen)
        elif mode == "global":
            s = 1 + sigma * torch.randn(1, 1, generator=gen).expand(k, 1)
        else:
            raise ValueError(f"unknown perturbation {mode!r}")
        return (vv * s).reshape(-1)

    @staticmethod
    def _gen(identity, seed, mode, sigma):
        h = hashlib.sha256(f"{identity}|{seed}|{mode}|{sigma}".encode()).digest()
        g = torch.Generator()
        g.manual_seed(int.from_bytes(h[:8], "little") % (2**63))
        return g

    def batch(self, betas, identities=None, mode="none", sigma=0.0, fixed_seed=None):
        if self._parts is not None:
            return self._fuse(self._parts[0].batch(betas, identities, mode, sigma, fixed_seed), self._parts[1].batch(betas, identities, mode, sigma, fixed_seed))
        """[B, dim] raw geometry. ``fixed_seed`` makes the noise a deterministic function of the identity
        (calibration-estimate-like); ``None`` draws fresh noise every call (training augmentation)."""
        rows = []
        for i, beta in enumerate(betas.detach().cpu()):
            ident = identities[i] if identities is not None else None
            if mode == "shuffle":
                ident, beta = self.partner[ident]
            v = self.base(beta, ident)
            if self.est is not None:
                e = self.est[str(ident)]
                s = self._session(e, ident, fixed_seed)
                if self.kind.startswith("estp"):
                    v = self._estp(e, s)
                elif self.kind == "est_scale": v = e["mu"][s].reshape(8, 3).norm(dim=-1).mean().reshape(1)
                else: v = torch.cat((e["mu"][s], e["sigma"][s])) if self.kind == "est_sig" else e["mu"][s]
            if mode in ("indep", "global", "add"):
                gen = self._gen(ident, fixed_seed, mode, sigma) if fixed_seed is not None else None
                v = self._perturb(v, mode, sigma, gen)
            rows.append(v)
        return torch.stack(rows)
