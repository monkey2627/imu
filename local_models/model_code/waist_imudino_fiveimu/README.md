# 本地五 IMU 生产接入

在线入口：`model_entry.py` 的 `load()` / `FiveIMUAdapter`，由 `production.run --model waist_imudino_fiveimu` 调用。
生产端多收一帧以计算真实前向 gyro；输出帧 = 所有以 index ≤ 471 覆盖该帧的已算窗口的 6D 平均，`modelLagFrames` 由 `MOTION_RECOVER_IMUDINO_LAG` 设定（默认 15，设 9 即旧的单窗口 index 471 读出）；Unity 默认再用至少 4 帧显示缓冲做按模型帧时间的 SLERP。两者都计入实时延迟，不对模型姿态做低通平滑。
完整启动、输入输出、根节点约定和验证见 [工作区部署说明](../../../docs/IMUDINO_REALUSE.md)。
流式版本（`stream_entry.py`、`stream/`，模型 `waist_imudino_stream_wristankle`，手腕/脚腕佩戴）的数据合成、增广与训练路径见 [IMUDINO_STREAM_WRISTANKLE.md](../../../docs/IMUDINO_STREAM_WRISTANKLE.md)。
本目录的 `downstream/` 保留服务器数据处理、网络、训练和验证入口；线上不会调用训练、缓存或视频渲染。
本地推理兼容调整仅在 `imudino.py`，新增 `torch_blocks.py` 提供已核对的 eval-only 算子；不修改服务器代码。
下面保留上游 README，描述的是基础编码器，不是完整五 IMU 下游推理接口。

<div align="center">

# IMUDINO: Self-distilled Foundation Model for Inertial Sensing

**Zhoutong Ye**<sup>1‡</sup> ·
**Chengwen Zhang**<sup>1*</sup> ·
**Zixuan Wang**<sup>1*</sup> ·
**Xufeng Jian**<sup>1*</sup> ·
**Zhe He**<sup>1</sup> ·
**Qingyang Wan**<sup>1</sup> ·
**Xutong Wang**<sup>1</sup> ·
**Jiaqi Liu**<sup>1</sup> ·
**Mingze Sun**<sup>1</sup> ·
**Yu Mei**<sup>1</sup> ·
**Chun Yu**<sup>1†</sup> ·
**Yuanchun Shi**<sup>1</sup>

<sup>1</sup> Tsinghua University&nbsp;&nbsp;·&nbsp;&nbsp;<sup>†</sup> Corresponding Author&nbsp;&nbsp;·&nbsp;&nbsp;<sup>‡</sup> Project Leader&nbsp;&nbsp;·&nbsp;&nbsp;<sup>*</sup> Equal Contribution

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/Octopus1/IMUDINO)
[![arXiv](https://img.shields.io/badge/arXiv-2611.07437-b31b1b)](http://arxiv.org/abs/2611.07437)

</div>

Official PyTorch release of IMUDINO, a general purpose foundation model for inertial sensing.
See the [paper](http://arxiv.org/abs/2611.07437) for details.

## Method overview

![Overview of the IMUDINO training objective](assets/method.png)

*Overview of the IMUDINO training objective (Figure 1 in the paper): semantics-preserving motion
transformations produce global and local views, which are distilled from an EMA teacher to a
student with DINO, iBOT++, and KoLeo objectives.*

## Results

### Human activity recognition

Main results on seven HAR datasets (Table 1 in the paper). Bold marks the best result and
<u>underline</u> the second best. (\*) Implemented with IMUDINO data augmentation.

| Type | Method | MoSense | HAPT | HHAR | DSADS | WEAR | OPPO. | Shoaib | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Supervised Baselines | DeepConvLSTM | 93.39 | 82.00 | 91.90 | 83.25 | 75.43 | 40.88 | 97.13 | 80.57 |
| | A&D | 95.79 | 81.95 | 94.17 | 86.12 | 77.10 | 46.29 | 97.79 | 82.74 |
| | IF-ConvT. | 95.81 | 85.35 | 95.01 | 88.82 | N/A | 50.29 | 98.28 | 85.59 |
| | TSF | 97.34 | 85.89 | <u>95.81</u> | 89.70 | N/A | 52.44 | 99.35 | 86.76 |
| Self-supervised HAR FMs | SimCLR (\*) | 96.08 | 80.56 | 87.56 | 91.56 | 88.48 | 43.50 | 99.29 | 83.86 |
| | Yuan et al. | 93.41 | 64.66 | 80.41 | 83.37 | 83.10 | 18.28 | 97.30 | 74.36 |
| | RelCon | 81.93 | 55.75 | 70.57 | 81.04 | 81.55 | 19.80 | 97.32 | 69.64 |
| | HAR-FM | 84.93 | 71.41 | 76.47 | 84.93 | 80.01 | 20.24 | 97.67 | 73.20 |
| General-purpose TSFMs | TimesFM 3.0 | 85.64 | 63.30 | 82.97 | 75.09 | 78.92 | 33.50 | 93.61 | 73.29 |
| | Chronos-2 | 83.95 | 57.04 | 72.60 | 73.04 | 62.93 | 24.65 | 90.71 | 66.36 |
| IMUDINO (Ours) | IMUDINO-6M | <u>97.91</u> | <u>87.56</u> | 95.33 | <u>93.24</u> | <u>91.28</u> | <u>53.54</u> | <u>99.58</u> | <u>88.35</u> |
| | **IMUDINO-85M** | **98.54** | **89.02** | **96.00** | **94.13** | **92.34** | **58.40** | **99.68** | **89.73** |

### Inertial motion capture

MoCap results on AnDy, UNIPD, CIP, and Virginia Natural Motion (Table 2 in the paper): SIP error
(°), global angular error (°), and joint position error (cm), averaged per sequence. Bold marks
the best result and <u>underline</u> the second best. (†) Backbone initialized from the pretrained
IMUDINO checkpoint and finetuned.

| Model | AnDy SIP↓ | AnDy Ang↓ | AnDy Pos↓ | UNIPD SIP↓ | UNIPD Ang↓ | UNIPD Pos↓ | CIP SIP↓ | CIP Ang↓ | CIP Pos↓ | Nat. Mot. SIP↓ | Nat. Mot. Ang↓ | Nat. Mot. Pos↓ | Avg SIP↓ | Avg Ang↓ | Avg Pos↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DIP | 11.39 | 5.73 | 4.34 | 12.02 | 5.47 | 4.22 | 19.13 | 8.61 | 6.86 | 33.43 | 11.91 | 13.33 | 18.99 | 7.93 | 7.19 |
| TransPose | 12.15 | 6.29 | 4.91 | 15.11 | 6.05 | 4.82 | 20.06 | 8.75 | 6.86 | 30.62 | 11.26 | 11.99 | 19.49 | 8.09 | 7.15 |
| TIP | 10.11 | 4.55 | 3.56 | 9.85 | 4.06 | 2.78 | 13.05 | 5.67 | 4.30 | 22.06 | 7.90 | 7.92 | 13.77 | 5.55 | 4.64 |
| PIP | <u>9.49</u> | 4.09 | **3.29** | <u>8.90</u> | 3.59 | 2.66 | 12.68 | 5.52 | 4.12 | 19.62 | 7.49 | 6.93 | 12.67 | 5.17 | 4.25 |
| DynaIP | **8.93** | **3.45** | <u>3.41</u> | **7.29** | **2.77** | **2.21** | **11.42** | **4.54** | **3.69** | **15.78** | <u>7.18</u> | **5.83** | **10.86** | **4.49** | **3.79** |
| IMUDINO-85M + MLP | 11.64 | 4.24 | 3.99 | 11.66 | 3.95 | 3.29 | 16.25 | 5.87 | 5.13 | 19.03 | 7.30 | 6.44 | 14.65 | 5.34 | 4.71 |
| IMUDINO-85M + Transformer | 10.83 | 3.88 | 3.50 | 10.82 | 3.69 | 2.92 | 14.04 | 5.25 | 4.39 | 19.54 | 7.28 | 6.61 | 13.81 | 5.02 | 4.36 |
| IMUDINO-85M + MLP (†) | 10.15 | <u>3.72</u> | 3.43 | 9.45 | <u>3.38</u> | <u>2.50</u> | <u>12.19</u> | <u>4.76</u> | <u>3.95</u> | <u>17.35</u> | **6.65** | <u>5.92</u> | <u>12.28</u> | <u>4.63</u> | <u>3.95</u> |

Qualitative comparison of inertial human pose estimation on a natural motion sequence
(Virginia Natural Motion; Figure 2 in the paper):

![Qualitative inertial HPE comparison](assets/fig-mocap-virginia.png)

*From left to right: ground truth, DynaIP, and IMUDINO-85M with an MLP probe, a transformer
probe, and an MLP probe with finetuned backbone. Numbers below each mesh denote the per-frame
mean root-aligned global rotation error in degrees.*

### Robot odometry

Results on the 2026 TartanIMU Challenge (Table 3 in the paper), evaluated with the IMUDINO
backbone frozen under the platform-agnostic setting. Bold marks the best result and
<u>underline</u> the second best. (\*) Platform-agnostic. (†) Platform expert.

| Model | Car AVE↓ | Car ATE20↓ | Dog AVE↓ | Dog ATE20↓ | Drone AVE↓ | Drone ATE20↓ | Human AVE↓ | Human ATE20↓ | Avg AVE↓ | Avg ATE20↓ | Avg Score↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TartanIMU (\*) | 0.852 | 2.273 | 0.425 | 1.901 | 1.374 | 2.772 | 0.235 | 1.877 | 0.721 | 2.206 | 0.872 |
| TartanIMU (†) | <u>0.199</u> | 0.999 | 0.129 | <u>0.644</u> | 0.966 | 2.006 | <u>0.107</u> | **0.814** | 0.350 | 1.116 | 0.429 |
| IMUDINO-85M + MLP (\*) | 0.200 | <u>0.998</u> | <u>0.123</u> | 0.707 | <u>0.543</u> | <u>1.304</u> | 0.132 | 1.039 | <u>0.249</u> | <u>1.012</u> | <u>0.333</u> |
| IMUDINO-85M + Transformer (\*) | **0.141** | **0.760** | **0.092** | **0.501** | **0.499** | **1.285** | **0.105** | <u>0.872</u> | **0.209** | **0.854** | **0.280** |

Visualized trajectories on the TartanIMU Challenge (Figure 3 in the paper):

![TartanIMU Challenge trajectory overview](assets/trajectory-overview.png)

*Black dashed lines are ground truth; blue is the official TartanIMU expert per platform, and
green/orange are IMUDINO with transformer / MLP probes. ▲ marks the start of each trajectory.
IMUDINO+Transformer traces the ground truth most closely.*

## Download the checkpoint

The weights are hosted on Hugging Face: [Octopus1/IMUDINO](https://huggingface.co/Octopus1/IMUDINO).
Download `imudino.pt` into this repository's root directory:

```bash
# huggingface-cli (pip install "huggingface_hub[cli]")
huggingface-cli download Octopus1/IMUDINO imudino.pt --local-dir .

# or plain download
wget https://huggingface.co/Octopus1/IMUDINO/resolve/main/imudino.pt
```

The example scripts expect `imudino.pt` next to them by default.

The backbone uses 16-sample patches, 12 transformer layers, 24 attention heads,
768-dimensional features, and 4 register tokens. It embeds all six sensor channels
jointly and uses standard GELU MLPs. Frequency-aware positional encoding always
uses the supplied sampling rates and a fixed 100 Hz reference. The factory retains
its original name, `imuformer_huge_patch16`, for compatibility.
The implementation lives in `imudino.py`; constructing `IMUFormer()` directly
uses the same settings as the factory.

Install PyTorch and timm in a Python 3.10+ environment:

```bash
pip install -r requirements.txt
python example_forward.py
```

Use `--device cuda` for a CUDA device or `--checkpoint /path/to/imudino.pt` for a
different checkpoint location. The example loads weights strictly, switches to
evaluation mode, and runs inference on a synthetic batch with padding. Tested
with Python 3.12, PyTorch 2.15.0, and timm 1.0.30.

To use the model in your own code:

```python
import torch
from model_factory import imuformer_huge_patch16

model = imuformer_huge_patch16()
model.load_state_dict(
    torch.load("imudino.pt", map_location="cpu", weights_only=True), strict=True
)
model.eval()

with torch.inference_mode():
    features = model({
        "imu_data": torch.randn(2, 1024, 6),
        "freqs": torch.tensor([100.0, 200.0]),
        "masks": None,
    })
```

Input `imu_data` has shape `[B, L, 6]`, with accelerometer XYZ followed by gyroscope
XYZ. `L` must be a positive multiple of 16. `freqs` contains one positive sampling
frequency in Hz per sequence. Use floating-point inputs on the same device as the
model.

The optional `masks` tensor has shape `[B, L]`: `True`/`1` means valid and
`False`/`0` means padding. A patch is valid only when all 16 samples are valid.
Omit `masks` or pass `None` for unpadded input. Padded patches remain in the output;
exclude them when pooling or using patch features downstream.

The output contains only the final transformer layer's tokens after layer normalization:

| Key | Shape | Description |
| --- | --- | --- |
| `cls_token` | `[B, 768]` | Sequence representation |
| `reg_tokens` | `[B, 4, 768]` | Register representations |
| `patch_tokens` | `[B, L / 16, 768]` | Temporal patch representations |

For the example batch, these shapes are `(2, 768)`, `(2, 4, 768)`, and `(2, 64, 768)`.
