# 手腕/脚腕 + 腰部 IMU 合成：挂点与代码（服务器原件的本地只读镜像）

这些文件逐字节复制自服务器 `/home/duanyuhan/dyh/motion/lines/imudino_wristankle_20261112/`，是流式 IMUDINO 两个模型
（`waist_imudino_stream_wristankle`、`..._limbmix`）训练数据**实际使用**的版本。本地其它模型包里的同名
`surface_imu.py` / `no_head_layouts.py` 是更早的副本，哈希不同，不要拿来对照这次的数据。服务器是训练侧权威；改动应在服务器做，再同步到这里。

| 文件 | 服务器位置 | SHA256 | 作用 |
|---|---|---|---|
| `wristankle_surface_imu.json` | `synth_ws/mobileposer/configs/` | `0e9da789…e1c338` | **新挂点配置**（手腕、脚腕、腰部） |
| `no_head_surface_imu.json` | 同上 | `f18a2bb9…7744f2` | 旧挂点配置（前臂 75%、小腿 88%），仅供对照；腰部从这里逐字复制 |
| `surface_imu.py` | `synth_ws/mobileposer/` | `d6584ab8…8a5753` | 读取/校验挂点，`surface_kinematics` 计算挂点表面轨迹与骨骼标定方向 |
| `no_head_layouts.py` | 同上 | `ec30e50d…6c3c38` | AMASS/BirdDog 读取、60 Hz 重采样、`_synthesize`、`_syn_acc` 加速度、写 `.pt` |
| `run_synth.sh` | 实验根目录 | `d739eb72…379d92` | 全量合成驱动（AMASS 18 子集 + BirdDog，5 进程并行） |
| `validate_surface_data.py` | 实验根目录 | `7123e2c6…7ab2a6` | 合成结果校验 |

生成新挂点配置的脚本：`../downstream/make_wristankle_config.py`；BirdDogMix 增广合成：`../stream/synth_birddog_limbmix.py`。

## 新挂点（`wristankle_surface_imu.json`）

| 标签 | 方向取自 | 位置 | SMPL 三角面 / 顶点 | 重心权重 | 中性姿态位置 (m) |
|---|---|---|---|---|---|
| lw 左手腕 | 左前臂 joint 18 | 肘→腕 95%，外侧 | face 2786，[2244, 1969, 2208] | 0.573 / 0.137 / 0.290 | (0.709, 0.497, −0.077) |
| rw 右手腕 | 右前臂 joint 19 | 同上 | face 9675，[5669, 5430, 5705] | 0.445 / 0.114 / 0.441 | (−0.705, 0.495, −0.078) |
| ls 左脚腕 | 左小腿 joint 4 | 膝→踝 95%，外侧 | face 6028，[3208, 3327, 3207] | 0.252 / 0.232 / 0.515 | (0.127, −0.874, −0.045) |
| rs 右脚腕 | 右小腿 joint 5 | 同上 | face 12915，[6607, 6728, 6608] | 0.695 / 0.198 / 0.107 | (−0.128, −0.873, −0.051) |
| waist 腰部 | spine1 joint 3 | 腰前正中、spine1 高度（未变） | face 6840，[1335, 3504, 1767] | 0.026 / 0.731 / 0.243 | (0.000, 0.124, 0.144) |

坐标为 SMPL 中性体型、零姿态、Y 向上；左右为解剖左右。每个挂点还带 `mount_rotation`（由表面法线确定的安装旋转），
因方向采用"骨骼标定"（bone_calibrated），合成后安装旋转被抵消，模型看到的方向即对应骨骼的全局旋转。

## 合成流程（代码位置）

1. 挂点生成：`make_wristankle_config.py` 沿解剖方向的射线与中性网格求交（与 `surface_imu.make_config` 同法），冻结三角面与重心权重。
2. 位置：`surface_imu.attachment_points` —— 三个蒙皮顶点按重心权重加权（随 SMPL 线性蒙皮运动；95% 处约 1/3 权重来自手掌/脚掌关节）。
3. 方向：`surface_imu.surface_kinematics` —— 对应骨骼全局旋转（安装旋转乘后再标定抵消）。
4. 加速度：`no_head_layouts._syn_acc` —— 挂点轨迹二阶差分，`smooth_n=4`，世界系线加速度（不含重力），单位 m/s²。
5. 帧率：`no_head_layouts._load_amass_dataset` —— 重采样到 `MOBILEPOSER_FPS`。**必须设为 60**，默认 30 会把加速度尺度算错。
6. 输出字段：`pose`、`shape`、`tran`、`joint`、`acc [T,6,3]`、`ori [T,6,3,3]`、`contact`、`synthesis`（含挂点全文与 SHA）；第 6 槽为骨盆兼容槽，模型只用前 5 路。
