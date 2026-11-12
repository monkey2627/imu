import torch
from mobileposer.line14.geom import GeomProvider, r2_pca_basis

b = r2_pca_basis()
S = b["S"]
print("principal SDs (cm):", (S[:8] * 100).round(decimals=2).tolist())
print("explained variance ratio (first 8):", ((S ** 2) / (S ** 2).sum())[:8].round(decimals=3).tolist())
for k in (1, 3):
    p = GeomProvider(f"pca{k}")
    print(f"pca{k} dim", p.dim, "value for beta=0:", p.base(torch.zeros(10)).tolist())
