import torch
t = torch.load("data_local/line14_cal_dyn13_dev10/train.pt", weights_only=False)
imu = t["imu"].float()
bad = ~torch.isfinite(imu).reshape(len(imu), -1).all(1)
print("samples with non-finite:", int(bad.sum()), "of", len(imu))
i = int(bad.nonzero()[0]); x = imu[i]
nf = ~torch.isfinite(x)
print("sample", i, "nonfinite frames", int(nf.any(1).sum()), "first frames", nf.any(1).nonzero().flatten()[:10].tolist(), "channels", nf.any(0).nonzero().flatten()[:20].tolist())
print("acc channels nonfinite:", int(nf[:, :15].sum()), "ori nonfinite:", int(nf[:, 15:].sum()))
