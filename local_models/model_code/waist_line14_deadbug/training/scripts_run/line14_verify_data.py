import torch, collections
P = "/home/duanyuhan/dyh/motion/MotionRe/PNP/data/test_datasets/"
def desc(n):
    d = torch.load(P + n + ".pt", map_location="cpu", weights_only=False)
    names = d["name"]
    print(f"== {n}: {len(names)} sequences; first names {list(names)[:6]}")
    I = torch.eye(3)
    rim = torch.stack([x for x in d["RIM"]]); rsb = torch.stack([x for x in d["RSB"]])
    print(f"   RIM identity for all? {bool(torch.allclose(rim, I.expand_as(rim), atol=1e-6))} ; RSB identity for all? {bool(torch.allclose(rsb, I.expand_as(rsb), atol=1e-6))}")
    print(f"   RSB deviation from identity (deg, mean over seq/sensors): {float(torch.rad2deg(torch.acos(((rsb.transpose(-1,-2) @ I).diagonal(dim1=-2, dim2=-1).sum(-1) - 1).div(2).clamp(-1, 1))).mean()):.2f}; distinct RSB sets: {len({tuple(x.flatten().round(decimals=4).tolist()) for x in d['RSB']})}")
    print(f"   lengths (frames) min/median/max: {min(len(x) for x in d['pose'])} / {sorted(len(x) for x in d['pose'])[len(names)//2]} / {max(len(x) for x in d['pose'])}")
    return d
d1 = desc("dipimu"); print("   all names:", list(d1["name"]))
d2 = desc("totalcapture_officalib")
nm = list(d2["name"]); print("   all names:", nm)
rom = [n for n in nm if "rom" in str(n).lower()]; print(f"   ROM sequences: {len(rom)} -> {rom}")
bysub = collections.defaultdict(list)
for n in nm: bysub[str(n).split('_')[0]].append(str(n).split('_', 1)[1])
for s, v in sorted(bysub.items()): print("   ", s, len(v), v)
nonrom = [n for n in nm if "rom" not in str(n).lower()]; print(f"   non-ROM sequences: {len(nonrom)}")
d3 = desc("totalcapture_dipcalib")
print("   dipcalib lengths == officalib lengths in order?", [len(x) for x in d3["pose"]] == [len(x) for x in d2["pose"]])
# frame counts of official vs dipcalib per index (mapping evidence)
import itertools
m = [(i, nm[i], len(d2['pose'][i]), len(d3['pose'][i])) for i in range(len(nm))]
print("   first 10 (idx, official name, official frames, dipcalib frames):", m[:10])
print("   number of indices where frame counts differ:", sum(1 for _, _, a, b in m if a != b))
