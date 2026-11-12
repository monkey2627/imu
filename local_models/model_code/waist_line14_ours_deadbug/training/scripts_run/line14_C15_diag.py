"""Why are real and shuffled personal codes identical in C15?  Loads a saved C15 'code' checkpoint and reports
 (1) spread of z across validation sessions / identities vs its mean,  (2) norms of the z-FiLM weights,  (3) the output change when z is swapped between identities,
 (4) a check that CodeProvider really returns different codes for 'real' and 'shuffle'."""
import argparse, sys
from pathlib import Path
import numpy as np, torch
from torch import nn
import line14_e2e_code as E
import line14_poser_chain as C
from mobileposer.line14.geom import GeomProvider
from mobileposer.line9.models import GeometryStats
from line14_estimate_r1 import R1Encoder

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="checkpoints/line14/C15/code_z8_s42.pt"); ap.add_argument("--seed", type=int, default=42); a = ap.parse_args()
dev = torch.device("cuda:0")
S_tr, fm, fs = E.load_sessions("train"); S_va, _, _ = E.load_sessions("val", fm, fs)
st = torch.load(a.ckpt, map_location=dev, weights_only=False)
enc = R1Encoder(S_va["x"].shape[-1], dim=8).to(dev); enc.load_state_dict(st["enc"]); enc.eval()
ckj = torch.load(f"checkpoints/line14/E18/estp6_oracle_film_s{a.seed}.pt", map_location="cpu", weights_only=False)
jnet = E.FilmJointsZ(6, GeometryStats(ckj["geometry_mean"], ckj["geometry_std"]), 8).to(dev); jnet.load_state_dict(st["jnet"]); jnet.eval()
poser = E.PoserZ(8, 6).to(dev); poser.load_state_dict(st["poser"]); poser.eval()
print("best epoch", st["epoch"])
with torch.no_grad():
    z = torch.cat([enc(S_va["x"][i:i + 64].to(dev), S_va["ph"][i:i + 64].to(dev), S_va["mk"][i:i + 64].to(dev))[:, :8] for i in range(0, len(S_va["x"]), 64)]).cpu()
ids = sorted(S_va["by_id"]); idm = torch.stack([z[S_va["by_id"][i]].mean(0) for i in ids])
print("z mean (abs) per dim:", z.mean(0).abs().numpy().round(3))
print("z std across ALL sessions per dim:", z.std(0).numpy().round(4))
print("z std of identity means per dim:", idm.std(0).numpy().round(4), " | mean within-identity std:", np.mean([z[S_va['by_id'][i]].std(0).numpy() for i in ids], 0).round(4))
print("FiLM-z weight norms: joints in/out", float(jnet.film_in_z.weight.norm()), float(jnet.film_out_z.weight.norm()), " poser", float(poser.film.weight.norm()))
# effect of swapping z between two identities on identical inputs
b = None
rec = E.load_identity_records(Path("data/pnp5_60a/pnp5"), Path("results/line9/data_audit.json"), ("validation",))
store = C.ChainStore(Path("data_local/line14_store_mount3"), Path("data/line9_store_pose6d"))
val = E.PairedWindowDataset(rec["validation"], 125, store=store, file_cache_size=2, sequence_cache_size=64)
vd, _ = C.loader(val, 64, a.seed, False); b = next(iter(vd))
base = GeomProvider("estp6", 3.0, "data_local/line14_est_r2_b90.pt")
prov = E.CodeProvider(base, S_va, enc, dev, 8, "real"); prov.partner = {ids[i]: ids[(i + len(ids) // 2) % len(ids)] for i in range(len(ids))}
imu, jt = C.get_inputs(b, dev); L = b["target_lengths"].tolist()
with torch.no_grad():
    outs = {}
    for mode in ("real", "shuffle", "zero"):
        prov.mode = mode; gz = prov.batch(b["beta"], b["identity"], "none", 0.0, fixed_seed=a.seed)
        j = jnet(imu, L, gz); p = poser(torch.cat((j, imu), -1), L, gz); outs[mode] = (gz[:, 6:].cpu(), j.cpu(), p.cpu())
    print("batch identities:", len(set(b["identity"])), " z real vs shuffle: mean |dz| =", float((outs["real"][0] - outs["shuffle"][0]).abs().mean()), " (|z| mean", float(outs["real"][0].abs().mean()), ")")
    print("Joints output change real->shuffle (cm, mean abs):", float((outs["real"][1] - outs["shuffle"][1]).abs().mean()) * 100, " real->zero:", float((outs["real"][1] - outs["zero"][1]).abs().mean()) * 100)
    print("Poser output change real->shuffle (6D, mean abs):", float((outs["real"][2] - outs["shuffle"][2]).abs().mean()), " real->zero:", float((outs["real"][2] - outs["zero"][2]).abs().mean()))
