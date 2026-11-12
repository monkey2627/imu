# motionUse

五 IMU 实时人体姿态项目：

```text
SlimeVR -> SolarXR -> selected model backend -> SMPL 24 joints
        -> local WebSocket -> Unity SMPL / Humanoid Avatar
```

当前重点不是继续调整 Unity，而是定位自然站立时实时 IMU 输入与训练数据之间的固定旋转偏差。
完整文档按阅读顺序收录在 [docs/README.md](docs/README.md)。当前状态、已验证结论和下一步任务见
[项目交接](docs/11_交接说明_20261110.md)。

## 核心目录

| 目录 | 职责 |
| --- | --- |
| `production/` | 独立生产链：SolarXR 输入、标定、模型调度、录制回放、Unity/Web 输出 |
| `mobileposer/` | MobilePose、StreamPoser、Line14、PosAcc 等模型算法，不承载生产主流程 |
| `solarxr_protocol/` | Python SolarXR 协议生成代码 |
| `scripts/` | 本地环境和受限资产安装脚本 |

## 独立程序与同步边界

SlimeVR 和 Unity 是独立工程，分别位于工作区同级的 `../slimevr/SlimeVR-Server/` 和 `../unity/SpineRecovery_V2/`。它们与 Python 生产链没有自动同步关系，但都以当前 `MotionRecoverWorkspace` 内的副本为本地权威源。

## 快速启动

```powershell
conda activate mobileposer
python -m production.run `
  --model waist_mobileposer_amass `
  --web
```

The production runtime registers 16 selectable model IDs in `production/model_backends.py`: MobilePose, StreamPoser, and Line14 variants trained on AMASS, Deadbug, and recorded Ours data, plus offset-free and experimental PosAcc variants. In the combined workspace their checkpoints are under `../local_models/checkpoints/<model_id>/`; matching independent model packages are under `../local_models/model_code/`. The default is `waist_mobileposer_amass`. The Deadbug variants were fine-tuned and evaluated on the same capture's FBX-to-SMPL pseudo-labels; treat them as sequence overfits, not held-out results. The legacy PNP checkpoints `pnp5_60a_s42_ftr3` and `pnp5_60a_sc_ftr3` remain available under `checkpoints/` and require `HIP` rather than `WAIST`.

模型与传感器位置已同步到 SparseIMU 现行基线（60 fps，PNP 位置：前臂近腕背侧、小腿膝下前内侧、骨盆骶骨；绑定 `LEFT/RIGHT_LOWER_ARM`、`LEFT/RIGHT_LOWER_LEG`、`HIP`）。权重放在 `checkpoints/<模型名>/`（`base_model.pth` + `manifest.json`），见 [docs/07](docs/07_模型与传感器位置同步_20261019.md)。

网页地址：`http://127.0.0.1:8765/`

Unity 场景：`../unity/SpineRecovery_V2/Assets/Scenes/AvatarScene.unity`

## 本地资产

SMPL 模型、数据集、checkpoint、录制和构建依赖不会上传 Git。取得合法 SMPL v1.0.0 下载后运行：

```powershell
.\scripts\install_smpl_assets.ps1
```

SMPL 资源须遵守其研究用途许可证。
