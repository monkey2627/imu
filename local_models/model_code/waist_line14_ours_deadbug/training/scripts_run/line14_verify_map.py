import torch
P = "/home/duanyuhan/dyh/motion/MotionRe/PNP/data/test_datasets/"
o = torch.load(P + "totalcapture_officalib.pt", map_location="cpu", weights_only=False)
d = torch.load(P + "totalcapture_dipcalib.pt", map_location="cpu", weights_only=False)
print("pose shapes official/dipcalib:", tuple(o["pose"][0].shape), tuple(d["pose"][0].shape))
same, diff = [], []
for i, n in enumerate(o["name"]):
    a = o["pose"][i].reshape(len(o["pose"][i]), -1); b = d["pose"][i].reshape(len(d["pose"][i]), -1)
    if len(a) == len(b) and torch.allclose(a, b, atol=1e-5):
        same.append(i)
    else:
        diff.append((i, n, len(a), len(b)))
print(f"index-aligned identical pose sequences: {len(same)}/45")
print("not identical (idx, official name, official frames, dipcalib frames):", diff)
# for the non-identical ones: does ANY dipcalib sequence have the same pose?
for i, n, la, lb in diff:
    a = o["pose"][i].reshape(la, -1)
    hit = [j for j in range(45) if len(d["pose"][j]) == la and torch.allclose(d["pose"][j].reshape(la, -1), a, atol=1e-5)]
    print(f"   {n}: dipcalib sequences with identical pose: {hit}")
# does the IMU content differ (calibration differs) for the identical-pose ones?
i = same[0]
print("example", o["name"][i], "IMU mean |aS diff|:", float((o["aS"][i] - d["aS"][i]).abs().mean()), "| RIS diff:", float((o["RIS"][i] - d["RIS"][i]).abs().mean()))
