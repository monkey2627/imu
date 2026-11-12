"""Summarize E3 test-time geometry perturbation of trained Oracle-R1 checkpoints (3 seeds), with gain-retention."""
import json, sys, numpy as np
from mobileposer.line9.evaluation import JOINT_GROUPS
S9 = "/home/duanyuhan/dyh/motion/SparseIMU/results/line9/shape"
seeds = (42, 7, 3)
D, PFX = sys.argv[1], sys.argv[2]   # e.g. E3test r1_oracle   |   E3train_test r1_oracle_n5i
R = {s: json.load(open(f"results/line14/{D}/{PFX}_s{s}.json"))["results"] for s in seeds}
B0 = {s: json.load(open(f"{S9}/b0_s{s}.json"))["metrics"]["aggregate"] for s in seeds}
names = list(R[42].keys())
g = lambda m, grp: float(np.mean(m[grp]["mean"] if isinstance(m[grp], dict) else m[grp]))
print("3-seed mean, identity-mean cm (B0 line9 / each perturbation)")
b0m = {grp: np.mean([B0[s][grp]["mean"] for s in seeds]) for grp in JOINT_GROUPS}
print("%-14s" % "B0", " ".join("%s:%.3f" % (grp, b0m[grp]) for grp in JOINT_GROUPS))
clean = {grp: np.mean([R[s]["clean"]["aggregate"][grp]["mean"] for s in seeds]) for grp in JOINT_GROUPS}
for n in names:
    m = {grp: np.mean([R[s][n]["aggregate"][grp]["mean"] for s in seeds]) for grp in JOINT_GROUPS}
    ret = (b0m["all"] - m["all"]) / (b0m["all"] - clean["all"])
    print("%-14s" % n, " ".join("%s:%.3f" % (grp, m[grp]) for grp in JOINT_GROUPS), " gain-retention(all)=%.2f" % ret)
