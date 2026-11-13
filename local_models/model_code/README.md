# 17 个模型的独立代码目录

本目录按 checkpoint ID 拆成 17 个同级模型。每个目录都自带运行、网络、训练和数据处理代码，不引用其他模型目录，也没有公共网络或公共训练目录。

新增 [`waist_imudino_fiveimu`](./waist_imudino_fiveimu/)：冻结 IMUDINO 编码器 + 五 IMU 下游姿态头，AMASS 18 子集及 BirdDog 训练。权重为 `../checkpoints/waist_imudino_fiveimu/{best.pt,imudino.pt}`；预处理/训练/验证在自身 `downstream/`，在线入口为 `model_entry.py`。详见 [部署说明](../../docs/IMUDINO_REALUSE.md)。

同一目录还承载三个流式模型：`waist_imudino_stream_wristankle`、`waist_imudino_stream_wristankle_limbmix`（2026-11-12）与 `waist_imudino_stream_deadbug_limbmix`（2026-11-13）：
在线入口 `stream_entry.py`，训练/特征/增广/评估代码在 `stream/`，与窗口版共用 backbone 和 `downstream/` 预处理，因此不另建代码目录；
权重在 `../checkpoints/<模型ID>/{best.pt,imudino.pt}`。详见 [IMUDINO_STREAM_WRISTANKLE.md](../../docs/IMUDINO_STREAM_WRISTANKLE.md) 第 7.1 节。

下述 MobilePoser 目录约定适用于原 16 个模型。

```text
model_code/
├─ waist_mobileposer_amass/
│  ├─ model_entry.py
│  ├─ runtime/mobileposer/
│  ├─ training/mobileposer/
│  ├─ training/entrypoints/
│  ├─ data_processing/
│  └─ README.md
├─ ...（其余 15 个模型结构相同）
└─ README.md
```

`runtime/` 和 `training/` 是每个模型自己的源码副本。不同模型里存在相同文件是有意的：这里优先保证模型可独立移交，不以共享网络/训练代码减少重复。数据处理脚本也放进各模型目录；它们可以在将来抽成唯一允许的公共层，但当前副本不产生跨模型依赖。

权重统一位于 `../checkpoints/<模型 ID>/`。入口 `<模型 ID>/model_entry.py` 会把本模型的 `runtime/` 放入搜索路径，并显式从统一 checkpoint 根目录加载对应权重。训练大数据不复制到本地，权威数据仍位于服务器；本地保留完整代码和 16 份 checkpoint，可独立执行推理和代码审计。

## 模型索引

| 模型 | 架构 | 权重文件 | 训练阶段 |
|---|---|---|---|
| [`waist_mobileposer_amass`](./waist_mobileposer_amass/) | MobilePose | `../checkpoints/waist_mobileposer_amass/base_model.pth` | AMASS 基线：18 个 AMASS 子集生成五 IMU 表面输入，训练四阶段网络后合并。 |
| [`waist_mobileposer_deadbug`](./waist_mobileposer_deadbug/) | MobilePose | `../checkpoints/waist_mobileposer_deadbug/model_finetuned.pth` | 从 AMASS MobilePose 基线出发，以 Deadbug 序列和 FBX/SMPL 伪标签进行个体化微调。 |
| [`waist_mobileposer_ours_all10`](./waist_mobileposer_ours_all10/) | MobilePose | `../checkpoints/waist_mobileposer_ours_all10/model_finetuned.pth` | Ours 动作 1-10 平衡训练并混入 AMASS replay；动作 1-8 训练，9-10 验证。 |
| [`waist_mobileposer_ours_deadbug`](./waist_mobileposer_ours_deadbug/) | MobilePose | `../checkpoints/waist_mobileposer_ours_deadbug/model_specialized.pth` | 在 Ours all10 权重上对动作 7 做 specialized 微调。 |
| [`waist_mobileposer_ours_offsetfree`](./waist_mobileposer_ours_offsetfree/) | MobilePose | `../checkpoints/waist_mobileposer_ours_offsetfree/model_finetuned.pth` | 先移除传感器与标签的平均旋转偏差，再加入 mount/heading 扰动微调。 |
| [`waist_streamposer_amass`](./waist_streamposer_amass/) | StreamPoser | `../checkpoints/waist_streamposer_amass/model.pt` | AMASS 五 IMU 序列训练；使用流式窗口、mount 扰动与时序 lag。 |
| [`waist_streamposer_deadbug`](./waist_streamposer_deadbug/) | StreamPoser | `../checkpoints/waist_streamposer_deadbug/model.pt` | 从 StreamPoser AMASS 权重出发，以 Deadbug 序列进行个体化微调。 |
| [`waist_streamposer_ours_all10`](./waist_streamposer_ours_all10/) | StreamPoser | `../checkpoints/waist_streamposer_ours_all10/model_finetuned.pt` | Ours 动作 1-10 平衡训练并混入 AMASS replay，使用时序关节/旋转损失。 |
| [`waist_streamposer_ours_deadbug`](./waist_streamposer_ours_deadbug/) | StreamPoser | `../checkpoints/waist_streamposer_ours_deadbug/model_specialized.pt` | 在 Ours all10 权重上对动作 7 做 specialized 微调。 |
| [`waist_streamposer_ours_offsetfree`](./waist_streamposer_ours_offsetfree/) | StreamPoser | `../checkpoints/waist_streamposer_ours_offsetfree/model_finetuned.pt` | offset-free 数据加分部位 mount/heading 扰动训练，用于减小佩戴偏差。 |
| [`waist_streamposer_posacc`](./waist_streamposer_posacc/) | StreamPoser+PosAcc | `../checkpoints/waist_streamposer_posacc/model.pt` | AMASS 与 Ours 合成位置/加速度监督的实验模型；当前结论是不建议作为部署默认模型。 |
| [`waist_line14_amass`](./waist_line14_amass/) | Line14 | `../checkpoints/waist_line14_amass/ori-none_geom-none_s42.pt` | AMASS Line14 基线，使用关节链和旋转链目标训练。 |
| [`waist_line14_deadbug`](./waist_line14_deadbug/) | Line14 | `../checkpoints/waist_line14_deadbug/model.pt` | 从 Line14 AMASS 权重出发，以 Deadbug 序列做个体化微调。 |
| [`waist_line14_ours_all10`](./waist_line14_ours_all10/) | Line14 | `../checkpoints/waist_line14_ours_all10/model_finetuned.pt` | Ours 动作 1-10 平衡训练并混入 AMASS replay，优化关节链和旋转链。 |
| [`waist_line14_ours_deadbug`](./waist_line14_ours_deadbug/) | Line14 | `../checkpoints/waist_line14_ours_deadbug/model_specialized.pt` | 在 Ours all10 权重上对动作 7 做 specialized 微调。 |
| [`waist_line14_ours_offsetfree`](./waist_line14_ours_offsetfree/) | Line14 | `../checkpoints/waist_line14_ours_offsetfree/model_finetuned.pt` | offset-free 数据加 mount/heading 扰动训练，减小传感器佩戴偏差。 |

## 目录约定

- `model_entry.py`：本模型唯一加载入口。
- `runtime/mobileposer/`：本模型的网络定义、特征构造和 checkpoint 加载依赖。
- `training/mobileposer/`：该模型训练依赖的完整 Python 包副本。
- `training/entrypoints/`：直接对应此 checkpoint 训练阶段的准备、训练、导出或评估脚本。
- `data_processing/`：通用预处理、Ours 重定向、合成和校验脚本的本模型副本。
- `training/SERVER_TRAINING_NOTES.md`：服务器训练路径、数据和复现说明。

服务器仍是训练与数据的权威端。本目录是可交接的本地代码/权重快照，不会自动把本地改动上传服务器；重新训练前应先明确要以哪个模型目录为修改源，再同步该目录所需改动。
