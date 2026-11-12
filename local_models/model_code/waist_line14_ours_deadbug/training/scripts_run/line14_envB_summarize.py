"""Summarize line-14 environment-B measurement (validation-only).
B0 comes from line9 shape/ (same code, data, labels, seeds; reproducibility of seed 42 is checked against the line14 rerun)."""
import json, numpy as np
from mobileposer.line9.evaluation import oracle_control_differences, JOINT_GROUPS
S9 = "/home/duanyuhan/dyh/motion/SparseIMU/results/line9/shape"
seeds = (42, 7, 3)
L = lambda p: json.load(open(p))
b0 = {s: L(f"{S9}/b0_s{s}.json") for s in seeds}
or33 = {s: L(f"{S9}/oracle_s{s}.json") for s in seeds}
b1 = {s: L(f"results/line14/envB/b1_s{s}.json") for s in seeds}
or12 = {s: L(f"results/line14/envB/oracle_s{s}.json") for s in seeds}
b0r = L("results/line14/envB/b0_s42.json")
for d in (b0, or33, b1, or12):
    for v in d.values(): assert v["status"] == "formal_validation_only" and not v["protocol"]["test_accessed"]
dig = {(v["protocol"]["target_window_digest_train"], v["protocol"]["target_window_digest_validation"]) for d in (b0, or33, b1, or12) for v in d.values()}
print("digests identical across all runs:", len(dig) == 1)
a = lambda r: r["metrics"]["aggregate"]["all"]["mean"]
print("repro b0 s42: line9=%.4f line14=%.4f best_epoch %s vs %s" % (a(b0[42]), a(b0r), b0[42]["best_epoch"], b0r["best_epoch"]))
def pooled(d):
    reps = [d[s]["metrics"] for s in seeds]; ids = reps[0]["per_identity"]
    pi = {i: {"metrics": {g: {"mean": float(np.mean([r["per_identity"][i]["metrics"][g]["mean"] for r in reps]))} for g in JOINT_GROUPS if g in reps[0]["per_identity"][i]["metrics"]}} for i in ids}
    return {"unit": "cm", "per_identity": pi}
print("\n3-seed mean (identity-mean, cm)")
for name, d in (("B0 (line9)", b0), ("B1-12 (line14)", b1), ("Oracle-33 (line9)", or33), ("Oracle-R1 (line14)", or12)):
    print("%-20s" % name, " ".join("%s:%.3f" % (g, np.mean([d[s]["metrics"]["aggregate"][g]["mean"] for s in seeds])) for g in JOINT_GROUPS))
print("best epochs R1:", {s: or12[s]["best_epoch"] for s in seeds}, "B1:", {s: b1[s]["best_epoch"] for s in seeds})
out = {}
def show(tag, dd):
    for c, r in dd.items():
        print(tag, c, " ".join("%s:%+.3f[%+.3f,%+.3f]" % (g, r["metrics"][g]["mean_difference"], r["metrics"][g]["ci95"][0], r["metrics"][g]["ci95"][1]) for g in JOINT_GROUPS))
        v = np.array([x["all"] for x in r["per_identity"].values()]); print("    identities improved:", int((v < 0).sum()), "/", len(v))
out["R1_vs_B0_B1"] = oracle_control_differences({"b0": pooled(b0), "b1": pooled(b1), "oracle": pooled(or12)})
show("[R1 vs B0/B1]", out["R1_vs_B0_B1"])
out["R1_vs_Oracle33"] = oracle_control_differences({"b0": pooled(or33), "b1": pooled(or33), "oracle": pooled(or12)})
print("(next: key oracle_minus_b0 here means R1 minus Oracle-33)")
show("[R1 vs Oracle33]", out["R1_vs_Oracle33"])
out["per_seed"] = {s: oracle_control_differences({"b0": b0[s]["metrics"], "b1": b1[s]["metrics"], "oracle": or12[s]["metrics"]}) for s in seeds}
for s in seeds: show("[seed %d R1 vs B0/B1]" % s, out["per_seed"][s])
json.dump(out, open("results/line14/envB_summary_3seed.json", "w"), indent=1)
