"""E6 diagnosis: split the R1m estimation error into the beta part (R1(beta)) and the installation (lever) part.
usage: python scripts_run/line14_E6_diag.py CKPT [DATA_DIR]"""
import sys
from pathlib import Path
import importlib.util
import numpy as np, torch
from mobileposer.line14.geom import GeomProvider

spec = importlib.util.spec_from_file_location("est", Path("scripts_run/line14_estimate_r1.py")); est = importlib.util.module_from_spec(spec); spec.loader.exec_module(est)
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
data = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data_local/line14_cal_dev10")
va = torch.load(data / "val.pt", weights_only=False)
prov = GeomProvider("r1")
r1b = torch.stack([prov.table.get(b) for b in va["beta"]]) * 100.0                      # beta part, cm [N,12]
lev = torch.stack([((l[:4] - l[4:5]).reshape(12)) for l in va["lever"]]) * 100.0         # lever part, cm [N,12]
x = est.make_inputs({"imu": va["imu"].float()}, ck["feat"]); x = (x - ck["fm"]) / ck["fs"]
m = est.R1Encoder(x.shape[-1]); m.load_state_dict(ck["model"]); m.eval()
with torch.no_grad():
    o = torch.cat([m(x[i:i + 64], va["phase_id"][i:i + 64].long()) for i in range(0, len(x), 64)])
mu = o[:, :12] * ck["ys"] + ck["ym"]
vn = lambda d: d.reshape(-1, 4, 3).norm(dim=-1).mean(0).numpy().round(2)
print("val sessions", len(x), "identities", len(set(va["identity"])))
print("lever part magnitude per vector (cm):", vn(lev))
print("beta part population spread: SD of R1(beta) around its mean (cm, vector norm):", vn(r1b - r1b.mean(0)))
print("estimate vs FULL target     (cm):", vn(mu - (r1b + lev)), " mean", float(vn(mu - (r1b + lev)).mean()))
print("estimate vs BETA-ONLY target(cm):", vn(mu - r1b), " mean", float(vn(mu - r1b).mean()))
print("train-mean baseline vs BETA-ONLY (cm):", vn(ck["ym"][None] - r1b), " mean", float(vn(ck["ym"][None] - r1b).mean()))
