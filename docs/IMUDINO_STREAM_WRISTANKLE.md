# IMUDINO 流式五 IMU 模型（手腕/脚腕佩戴）：训练路径与数据处理

更新：2026-11-12。本文记录两个并行训练的模型：

| 代号 | 运行目录（服务器） | 训练数据 | 目的 |
|---|---|---|---|
| **原版** `stream_wristankle` | `runs/stream_wristankle` | AMASS 18 子集 + BirdDog | 消除窗口版的跨窗跳变，逐帧平滑、1 帧延迟 |
| **增广版** `stream_wristankle_limbmix` | `runs/stream_wristankle_limbmix` | 上述 + BirdDogMix（肢体重组增广） | 在原版基础上，去掉 "伸手 ⇒ 对侧抬腿" 的数据先验 |

两者网络结构、在线调度、超参数相同，只有训练数据和 BirdDog 采样比例不同。
服务器实验根目录（下文简写 `$N`）：`/home/duanyuhan/dyh/motion/lines/imudino_wristankle_20261112`。
旧窗口版 IMUDINO（`waist_imudino_fiveimu`）及其诊断见 `IMUDINO_JITTER_DIAGNOSIS.md`，未被改动。

## 1. 为什么重做下游

- 窗口版 head 在单个 480 帧窗口内平滑，但同一物理帧在错开 1–3 帧的两个窗口中的预测差 p99 14–18°；
  在线每次滑窗只读末端，就把这种窗口放置方差变成了逐帧跳变（录制 20261111_235722 可复现，见诊断文档）。
- 冻结 backbone 是双向注意力 + 16 帧 patch，窗口依赖是其固有性质，只能在下游解决：
  改成**带状态、逐帧更新**的 head，并让训练时的输入调度与在线完全相同。
- 同时传感器佩戴位置改为手腕、脚腕（见第 2 节），需要重新合成训练数据。

## 2. 传感器挂点（数据合成用）

配置：`$N/synth_ws/mobileposer/configs/wristankle_surface_imu.json`，SHA256 `0e9da789a72715b2cc165bb65fc0ac285b5a8334ff1b8f6f3e25fc9adce1c338`。
生成脚本：`local_models/model_code/waist_imudino_fiveimu/downstream/make_wristankle_config.py`（服务器副本 `$N/make_wristankle_config.py`）。

| 标签 | 骨骼（方向来源） | 位置 | 与旧配置相比 |
|---|---|---|---|
| lw / rw | 前臂 SMPL 18 / 19 | 肘→腕 95% 处，外侧表面（手表带位置） | 旧为 75%，向远端移 5.3 / 5.4 cm |
| ls / rs | 小腿 SMPL 4 / 5 | 膝→踝 95% 处，外侧表面（脚踝绑带位置） | 旧为 88%，向远端移 3.0 cm |
| waist | spine1 SMPL 3 | 腰前正中，spine1 高度 | 与旧配置逐字节相同 |

- 构造方法与原 `mobileposer.surface_imu.make_config` 相同：沿解剖方向的射线与中性 SMPL 网格求交，冻结三角面与重心坐标；
  安装旋转由表面法线确定，方向仍为"骨骼标定"（bone_calibrated）。
- 已知近似：95% 处顶点约 1/3 蒙皮权重落在手掌/脚掌关节（旧位置约 5–20%），合成位置会随腕/踝屈伸偏移约 1 cm；
  方向仍完全取自前臂/小腿骨骼。
- 旧模型（窗口版 IMUDINO、MobilePoser 等）使用 `no_head_surface_imu.json`（SHA `f18a2bb9…`，前臂 75%/小腿 88%）。
  新佩戴位置下它们的加速度杠杆臂不匹配。

## 3. 数据合成

脚本：`$N/run_synth.sh`；工作区 `$N/synth_ws`（从旧实验复制的 `mobileposer` 代码，未改动；`data/raw/AMASS/` 下为各子集与 BirdDog 的软链接）。

```bash
env MOBILEPOSER_FPS=60 MOBILEPOSER_RESAMPLE_ANY_FPS=1 PYTHONPATH=. \
  python -m mobileposer.no_head_layouts --layout wrists_shanks_waist --output-root $N/data/processed \
    --source amass --imu-mode surface --attachment-config <wristankle_surface_imu.json> --datasets ...
```

- 数据源：AMASS 论文对齐的 18 个子集（与窗口版相同）+ BirdDog（`/home/duanyuhan/dyh/motion/flowhmr/output/training_data/BirdDog`，67 个视频、355 段伪标签）。
- 60 Hz；非 60 Hz 的 AMASS 用 SLERP/线性插值重采样。加速度为挂点表面轨迹的世界系线加速度（`smooth_n=4` 有限差分），方向为骨骼标定旋转。
- 输出 `$N/data/processed/wrists_shanks_waist/*.pt`（19 个文件，约 995 万帧，15 GB）；`validate_surface_data.py` 19/19 通过，日志 `$N/logs/synth/validation.log`。
- **必须设置 `MOBILEPOSER_FPS=60`**：合成代码默认 30 fps，会把加速度尺度算错（BirdDogMix 首次合成就漏了，被第 5 节的兼容性检查拦下后重做）。

### 3.1 BirdDogMix：肢体重组增广（仅增广版使用）

脚本：`local_models/model_code/waist_imudino_fiveimu/stream/synth_birddog_limbmix.py`。

动机：实测录制 20261112_015543 中用户全程跪姿只伸手，两侧小腿传感器完全静止（倾角恒定、竖直加速度 ±0.3 m/s²），
模型仍让对侧大腿抬起 40–85°。原因是 BirdDog 伪标签里伸手几乎总伴随对侧抬腿（P(抬腿|伸手)=0.79），
模型学到了这个先验。小腿在真实抬腿时有约 25° 倾角变化和 ±2–5 m/s² 加速度，证据存在，但模型未被迫使用它。

做法：只用**训练折**的 325 段 BirdDog，每段生成 3 段：

| 类型 | 构成 | 段数 |
|---|---|---:|
| cross | A 段的根、躯干、手臂 + 另一视频 B 段的腿（髋/膝/踝/脚局部旋转，随机时间偏移） | 325 |
| armonly | A 段，腿冻结在 A 自身最接近跪姿的帧（两大腿最接近竖直，平均 ≤35° 才生成） | 270 |
| legonly | A 段，手臂冻结在 A 自身最接近撑地的帧（两前臂最接近竖直，同上阈值） | 246 |

- 根姿态和平移始终来自 A；IMU 用与其他数据完全相同的 `_synthesize` 与挂点重新合成。
- 共 841 段、43.5 万帧；合成元数据与 BirdDog 完全一致，加速度幅度 p99 5.28 vs 4.95 m/s²。
- 全部标记 `train_only`，group 为 `BirdDogMix:<类型>:<源组>|<伙伴组>`；验证集仍只用未改动的 BirdDog 验证折。
- 解耦核验：P(对侧抬腿 | 伸手) 原始 0.79 → armonly 0.05、cross 0.60（≈ 抬腿基础比例 0.57）。

运行（mobileposer 环境，需同时能导入合成代码和 IMUDINO 的 `downstream.data`）：

```bash
cd $N/synth_ws && MOBILEPOSER_FPS=60 PYTHONPATH=$N/synth_ws:$N/imudino_stream \
  python $N/imudino_stream/stream/synth_birddog_limbmix.py \
    --birddog $N/data/processed/wrists_shanks_waist/BirdDog.pt \
    --raw /home/duanyuhan/dyh/motion/flowhmr/output/training_data/BirdDog \
    --config mobileposer/configs/wristankle_surface_imu.json \
    --out $N/data/processed/wrists_shanks_waist/BirdDogMix.pt
```

## 4. 流式契约（训练与在线共用）

代码：`stream/common.py`（本地 `local_models/model_code/waist_imudino_fiveimu/stream/`，服务器 `$N/imudino_stream/stream/`）。

- 每路 IMU 输入：`downstream.data.to_imu6`，即机体系比力（线加速度减重力，m/s²）+ 前向差分 gyro（rad/s），与窗口版训练相同。
- **慢通道（token）**：对每个 g ≡ 15 (mod 16) 的帧，取以 g 结尾的 480 帧窗口（不足时左侧用 mask），
  五路分别经冻结 IMUDINO backbone 编码，取**最后一个 patch** 的特征拼成 3840 维 token。
  token g 需要 g+1 帧来算 gyro，再留 1 帧给计算，因此**从 g+2 帧起可用**。
- **快通道（每帧）**：五路 imu6（加速度 /9.81、gyro /4）、四肢在腰部坐标系中的旋转（4×9）、腰部坐标系中的重力方向（3）、
  token 相位与有效标志（2）。全部与世界 yaw 无关。
- 输出：23 个关节（SMPL 1–23）的局部 6D；根节点由腰部测量求解 `R_root = R_waist · R_spine1_localᵀ`（与窗口版相同）。
- 精度：backbone 训练特征与在线推理**统一 float32**，token 以 fp32 存储（bf16 时同输入不同 batch 的 token 相对差 0.4%，会在输出放大成约 1°）。

## 5. 特征提取

脚本：`stream/prepare_stream.py`；驱动：`$N/run_stream.sh`（6 卡并行，每个数据集约 2–60 s）。

```bash
cd $N/imudino_stream && python -m stream.prepare_stream \
  --source-root $N/data/processed/wrists_shanks_waist --datasets <...> \
  --out $N/data/features/stream_wristankle_v1 --attachment-sha256 0e9da789... --device cuda:0
```

- 校验：五路顺序、surface 模式、60 fps、挂点 SHA 必须一致，否则拒绝（这就是拦下 30 fps BirdDogMix 的检查）。
- 每段序列存：imu6 [T,5,6]、方向 [T,5,3,3]、6D 目标 [T,138]、token [K,3840]、group、是否验证折；短于 64 帧的序列跳过。
- 划分：AMASS 按序列、BirdDog 按源视频 SHA，使用与窗口版相同的 `group_fold`（10% 验证）。
- 输出 `$N/data/features/stream_wristankle_v1/<数据集>.stream.pt`，共 16 GB。

## 6. 网络与训练

网络 `stream/model.py::StreamHead`：token → LayerNorm/Dropout/Linear 256；每帧特征 → Linear 128；拼接后 2 层 GRU(512) → Linear 138。因果，无未来帧。

训练 `stream/train_stream.py`：

- 每个样本是从序列中随机位置裁出的 360 帧片段；GRU 从零状态开始（与在线启动一致），前 60 帧不计损失。
- 损失 = 6D MSE + 1.0 × 相邻帧 6D 差分与真值差分的 MSE（匹配运动速度，不是压制运动）。
- AdamW lr 1e-3、weight decay 1e-4、OneCycle，batch 64，60000 步；每 2000 步验证一次，保存 `latest.pt` / `best.pt` / `metrics.jsonl`。
- 验证指标：平均局部关节角误差（AMASS、BirdDog 分开报告），以及预测与真值的相邻 2 帧变化 p99（用于确认输出不比真实动作更抖）。
- 选 best 的分数为 AMASS 与 BirdDog 平均角误差的均值。

| | 原版 | 增广版 |
|---|---|---|
| 驱动脚本 | `$N/run_stream.sh` | `$N/run_limbmix.sh` |
| GPU | 0 | 1 |
| 数据集 | AMASS 18 + BirdDog | AMASS 18 + BirdDog + BirdDogMix |
| `--bird-fraction` | 0.15（BirdDog 实际只占 2% 帧） | 0.10（BirdDog 与 BirdDogMix 合计，Mix 约占 70%） |
| 训练 / 验证序列 | 12790 / 1445 | 13631 / 1445（验证集相同） |
| 训练帧数 | 889 万 | 933 万 |
| 日志 | `$N/logs/stream/train.log` | `$N/logs/limbmix_train.log` |
| 完成标记 | `$N/logs/stream/COMPLETE` | `$N/logs/limbmix_COMPLETE` |

原版命令（增广版另加 `BirdDogMix` 与 `--bird-fraction 0.10`，输出改为 `runs/stream_wristankle_limbmix`）：

```bash
cd $N/imudino_stream && CUDA_VISIBLE_DEVICES=0 python -m stream.train_stream \
  --cache-root $N/data/features/stream_wristankle_v1 --output $N/runs/stream_wristankle \
  --steps 60000 --batch 64 --val-every 2000
```

截至撰写时的验证（平均局部角误差；预测/真值 2 帧变化 p99）：

| 版本 | 步数 | AMASS | BirdDog |
|---|---:|---|---|
| 原版 | 22000 | 6.25°；9.04 / 10.21° | 6.96°；2.84 / 3.18° |
| 原版（**完成**，best 在 48000，分数 6.421） | 60000 | 5.95°；9.15 / 10.21° | 6.90°；2.97 / 3.18° |
| 增广版 | 2000 | 8.75°；7.90 / 10.21° | 8.42°；2.74 / 3.18° |
| 增广版（**完成**，best 在 58000，分数 6.132） | 60000 | 5.99°；9.18 / 10.21° | 6.28°；2.86 / 3.18° |

## 6.1 留出集评估（两版最终 best）

脚本 `stream/eval_stream.py`（mobileposer 环境，`PYTHONPATH=$N/synth_ws:$N/imudino_stream`），结果 `$N/logs/eval_heldout.json`。
验证折：AMASS 1413 条序列（按序列分组）、BirdDog 30 个源视频（按视频分组）；BirdDogMix 只用于训练、不参与评估。
每条序列 GRU 从零状态一次因果解码，前 60 帧不计；位置用中性体型、根对齐。
"真值根"用真实骨盆；"部署根"按在线规则 `R_root = R_waist · R_spine1_predᵀ`（合成腰部传感器无噪声，故偏乐观）。

| 指标 | AMASS 原版 | AMASS 增广版 | BirdDog 原版 | BirdDog 增广版 |
|---|---:|---:|---:|---:|
| 局部角误差 (°) | 5.90 | 5.85 | 6.73 | 6.49 |
| Ang，真值根 (°) | 8.37 | 8.28 | 9.14 | 8.56 |
| SIP，真值根 (°) | 7.67 | 7.57 | 9.14 | 8.56 |
| 位置，真值根 (cm) | 3.37 | 3.33 | 4.66 | 4.34 |
| Ang，部署根 (°) | 7.58 | 7.48 | 7.57 | 7.01 |
| SIP，部署根 (°) | 7.46 | 7.38 | 8.76 | 8.28 |
| 位置，部署根 (cm) | 2.98 | 2.94 | 3.37 | 3.23 |
| Jitter (m/s³；真值 283 / 124) | 681 | 682 | 211 | 224 |

增广版精度全面略优，jitter 相同；两者 jitter 约为真值 2.4×（AMASS）/1.7×（BirdDog），即仍有小幅逐帧波动（2 帧变化 p99 低于真值，无大跳变）。
这是合成 IMU 留出集，不同于文献 DIP/TotalCapture 真实 IMU 协议，不可直接比较。

## 7. 在线部署

- 模型 ID `waist_imudino_stream_wristankle`，代码 `local_models/model_code/waist_imudino_fiveimu/stream_entry.py::StreamAdapter`，
  已在 `production/model_backends.py` 注册（kind `imudino_stream`，lag 1 帧），Unity 模型列表已加"IMUDINO Stream / wrist+ankle"，并沿用 IMUDINO 的显示缓冲。
- 两个模型 ID：`waist_imudino_stream_wristankle`（原版，`best.pt` = step 48000；旧 step 10000 备份为 `best_step10000.pt`）、
  `waist_imudino_stream_wristankle_limbmix`（增广版，step 58000）。各自目录 `local_models/checkpoints/<ID>/` 下需有 `best.pt` 与 `imudino.pt`。
  Unity 列表名分别为 "IMUDINO Stream / wrist+ankle" 与 "IMUDINO Stream / wrist+ankle limb-mix"。
- 启动：`python -m production.run --model waist_imudino_stream_wristankle_limbmix`（或原版 ID；在 `runtime/python` 下）。
- `push()` 只存样本；`infer()` 按帧顺序补齐所有未处理帧，严格执行第 4 节调度；历史缓冲 480+604 帧，保证 `infer` 落后时 token 窗口不被截短。
- 一致性：同一序列按 1/2/3 帧不规则节拍逐帧推入，与训练时整段前向的输出最大差 0.056°。
- 本机 RTX 4090：`infer` 中位 3.7 ms，p99 约 18 ms（每 16 帧一次 backbone）。

## 8. 验证与分析脚本（`runtime/python/analysis/`，均只读）

| 脚本 | 用途 |
|---|---|
| `imudino_jitter_stages.py` | 窗口版逐阶段抖动定位（输入、单窗、跨窗、根、时序） |
| `imudino_readout_sweep.py` | 窗口版不同读出方式的平滑度/延迟权衡 |
| `imudino_adapter_replay_check.py` | 窗口版按真实节拍重放；lag 9 与录制逐位对照 |
| `imudino_stream_replay_check.py` | 流式模型经生产 session 重放：耗时与 2 帧连续性 |
| `thigh_pitch_compare.py` | 大腿/小腿/前臂相对竖直的角度：实测输出 vs 流式模型重放 |
| `reach_thigh_episodes.py` | 每次伸手时对侧大腿抬起与小腿传感器证据；`--replay <ckpt>` 用任意 checkpoint 离线重跑 |

回放时加速度必须经 `identity_heading.apply` 转到 SMPL 基（diag(-1,1,-1)），否则与线上输入不一致。

增广版验收（录制 20261112_015543 只动手、015655 手脚一起）：只动手时对侧大腿保持在约 25° 以内；手脚一起时仍能抬到约 70° 以上。

中期对比（`reach_thigh_episodes --replay`，同一录制、同一推理节拍；对侧大腿相对竖直角度的每次伸手中位数，括号为最大值；剔除开头 0.5 s 启动预热）：

| 录制 | 原版 step 36000（best 6.453） | 增广版 step 20000（best 6.489） |
|---|---|---|
| 015543 只动手，伸左手 ×4 | 20 / 42 / 46 / 42°（25–59） | 10 / 12 / 25 / 31°（30–65） |
| 015543 只动手，伸右手 ×5 | 75 / 54 / 44 / 76°（67–83） | 21 / 20 / 19 / 67°（21–77） |
| 015655 手脚一起，抬腿 | 68–81° | 79–91° |

原版最终 checkpoint（step 48000）同一对照：只动手时伸左手 29 / 48 / 53 / 47°、伸右手 76 / 64 / 56 / 79°，手脚一起 68–82°——
与中途版本相同，说明多训练不能消除该先验。

增广版在 9 次只动手中有 7 次基本保持跪姿，真实抬腿反而更明显，验证误差不变；仍有 1 次（56 s 处）中位 67°，以及若干次瞬时 55–65°。
增广版最终 checkpoint（step 58000）：只动手时伸右手 24 / 19 / 20 / 36°、伸左手 31 / 37 / 31 / 59°；手脚一起 75–99°。
相对原版最终：伸右手的先验抬腿基本消除，伸左手部分改善（且不如 step 20000），真实抬腿更明显（伸左手时右大腿中位约 98°，可能偏大）；
BirdDog 验证误差 6.90 → 6.27°，AMASS 不变；平滑度与耗时同原版（局部 2 帧 p99 2.54°，infer 中位 3.6 ms）。
单条 60 s 录制、8 次伸手，checkpoint 间波动明显，需更多带动作说明的实测录制作为验收集。

注意 015543/015655 是新佩戴位置的实测录制，但无真值，以上是行为对照而非精度。

## 9. 已知限制与未验证项

- 大腿无传感器，只能由小腿方向、加速度和先验推断；增广版降低先验后，真实抬腿时反应可能变弱，需实测。
- BirdDog 是视频伪标签，不是动捕真值；BirdDogMix 的拼接姿态不保证物理合理（膝盖可能离地或穿地），只用于 IMU 合成。
- 训练只用合成 IMU；SlimeVR 实际噪声、绑带滑动、软组织未建模。
- 旧位置的录制（2026-11-12 之前）只能用于看平滑度，不能评估精度。
