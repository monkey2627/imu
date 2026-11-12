import torch
P = "/home/duanyuhan/dyh/motion/MotionRe/PNP/data/test_datasets/"
o = torch.load(P + "totalcapture_officalib.pt", map_location="cpu", weights_only=False)
d = torch.load(P + "totalcapture_dipcalib.pt", map_location="cpu", weights_only=False)
for i in (13, 18, 20, 32, 40, 42):
    a = o["pose"][i].reshape(len(o["pose"][i]), -1); b = d["pose"][i].reshape(len(d["pose"][i]), -1); n = min(len(a), len(b))
    print(o["name"][i], "frames", len(a), len(b), "| max|pose diff| over the common prefix:", float((a[:n] - b[:n]).abs().max()), "| over common suffix:", float((a[-n:] - b[-n:]).abs().max()))
# is the 'dip-calibrated' IMU the same physical measurement (different calibration)?  compare raw sensor-frame acceleration aS
i = 0
print("aS identical (raw sensor-frame accel) for s1_acting1:", bool(torch.allclose(o["aS"][i], d["aS"][i])), "| wS identical:", bool(torch.allclose(o["wS"][i], d["wS"][i])), "| mS identical:", bool(torch.allclose(o["mS"][i], d["mS"][i])))
print("aS shapes", tuple(o["aS"][i].shape), tuple(d["aS"][i].shape), "| mean |aS|", float(o["aS"][i].abs().mean()), float(d["aS"][i].abs().mean()))
