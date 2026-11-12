"""Compare trained variants against line-9 B0 (same data/labels/seeds): paired identity bootstrap, 3 seeds.
usage: python scripts_run/line14_variants_summarize.py TAG NAME [NAME ...]   (files results/line14/TAG/NAME_s{42,7,3}.json)"""
import json, sys, numpy as np
from mobileposer.line9.evaluation import oracle_control_differences, JOINT_GROUPS
S9 = "/home/duanyuhan/dyh/motion/SparseIMU/results/line9/shape"
seeds = (42, 7, 3)
tag, names = sys.argv[1], sys.argv[2:]
import os
ref = os.environ.get("B0_REF")   # e.g. E5:r1_b0  -> results/line14/E5/r1_b0_s{seed}.json ; default = line-9 shape B0
b0 = {s: json.load(open(f"results/line14/{ref.split(':')[0]}/{ref.split(':')[1]}_s{s}.json" if ref else f"{S9}/b0_s{s}.json")) for s in seeds}
def pooled(d):
    reps = [d[s]["metrics"] for s in seeds]; ids = reps[0]["per_identity"]
    pi = {i: {"metrics": {g: {"mean": float(np.mean([r["per_identity"][i]["metrics"][g]["mean"] for r in reps]))} for g in JOINT_GROUPS if g in reps[0]["per_identity"][i]["metrics"]}} for i in ids}
    return {"unit": "cm", "per_identity": pi}
B0p = pooled(b0)
print("B0 (line9)  ", " ".join("%s:%.3f" % (g, np.mean([b0[s]["metrics"]["aggregate"][g]["mean"] for s in seeds])) for g in JOINT_GROUPS))
for n in names:
    d = {s: json.load(open(f"results/line14/{tag}/{n}_s{s}.json")) for s in seeds}
    for v in d.values(): assert v["status"] == "formal_validation_only" and not v["protocol"]["test_accessed"]
    print("%-22s" % n, " ".join("%s:%.3f" % (g, np.mean([d[s]["metrics"]["aggregate"][g]["mean"] for s in seeds])) for g in JOINT_GROUPS), " best_epochs", [d[s]["best_epoch"] for s in seeds])
    r = oracle_control_differences({"b0": B0p, "b1": B0p, "oracle": pooled(d)})["oracle-b0"]
    print("    vs B0:", " ".join("%s:%+.3f[%+.3f,%+.3f]" % (g, r["metrics"][g]["mean_difference"], r["metrics"][g]["ci95"][0], r["metrics"][g]["ci95"][1]) for g in JOINT_GROUPS),
          " improved %d/%d" % (int((np.array([x["all"] for x in r["per_identity"].values()]) < 0).sum()), len(r["per_identity"])))
