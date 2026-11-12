# line14 固定 IMU 几何

line14 唯一训练布局固定为 PNP 六 IMU 标准位置中去掉头部后的五个传感器：

| 槽位 | 标签 | 身体位置 | SMPL joint | PNP 表面顶点 |
|---:|---|---|---:|---:|
| 0 | `lw` | 左前臂 | 18 | 1961 |
| 1 | `rw` | 右前臂 | 19 | 5424 |
| 2 | `ls` | 左小腿 | 4 | 1176 |
| 3 | `rs` | 右小腿 | 5 | 4662 |
| 4 | `pelvis` | 骨盆 | 0 | 3021 |

标准测试包按 PNP 六 IMU 原顺序取索引 `[0, 1, 2, 3, 5]`。唯一允许的 line14 训练数据根目录是已存在的 `data/pnp5_60a/pnp5`（60 Hz）。训练入口会检查路径、样本元数据、合成频率和表面顶点，避免腰/spine1 或其他替代位置混入主线。

不要重新生成或覆盖已有训练数据，也不要将处理后的数据放入 `raw`。标准合成附件配置是 `mobileposer/configs/pnp_surface_imu.json`。`no_head_layouts.py` 仅保留 PNP 标准位置的合成代码；其他实验布局不再属于 line14。

此前的 wrists/shanks/waist、thighs、upper-arms、feet 等多布局数据、权重、输出以及专用训练/评测入口已清理。它们的历史结果只能作为旧实验记录，不能作为 line14 训练配置、标准 baseline 或 PNP 对齐结果。
