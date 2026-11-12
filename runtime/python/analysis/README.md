# analysis/：2026-11 的离线分析与实验

## IMUDINO 真实部署抖动诊断

`python -m analysis.imudino_deployment_audit`：只读重建 2026-11-11 两份鸟狗录制，输出映射、采样、gyro、跨窗预测、根节点、重采样和离线重叠融合对照的 JSON。
`python -m analysis.imudino_deployment_audit --cadence`：在临时 21209 端口启动已有录制的 Python 回放，测量输出间隔、推理耗时及真实 push 数与帧计数偏差，完成后终止子进程。
诊断结论与生产修正记录见工作区 `docs/IMUDINO_JITTER_DIAGNOSIS.md`。分析命令只读；生产输入和 Unity 显示路径已经按报告末尾修正，不改录制、权重或训练数据。

在 `realUse/` 目录下运行：`python -m analysis.<子目录>.<脚本名> [参数]`。录制文件名指 `recordings/` 下的文件，可以不带 `.jsonl`。
这里的脚本都只读实时代码（`production/`）和录制，不会修改它们。模型训练代码在工作区 `local_models/model_code/<model_id>/training/`，大型训练数据和输出保留在服务器（见 `docs/11_交接说明_20261110.md`）。

## arms/：前平举时大臂偏外

结论：主要原因是前臂航向误差（20–35°，校准可以修正）；剩下的偏外来自模型先验，已用 `production/arm_prior.py` 做后处理。

| 脚本 | 内容 | 示例 |
|---|---|---|
| `arms_forward_probe.py` | 合成测试：直臂前平举与"侧举屈肘"的前臂朝向完全相同，看各模型选哪一种 | `python -m analysis.arms.arms_forward_probe` |
| `arms_forward_real.py` | 真实录制中，双臂平举校准那段的输入和模型输出 | `... arms_forward_real <录制>` |
| `arms_input_timeline.py` | 前臂输入在腰部朝向坐标系下的逐帧方位、俯仰、扭转，以及模型预测的大臂 | `... arms_input_timeline unity_20261103_002526` |
| `arms_calib_ablation.py` | 同一段录制：不校准、只校准航向、两姿势校准三种条件下的结果对比 | `... arms_calib_ablation unity_20261103_002526` |
| `arm_prior_eval.py` | 大臂后处理在站立和鸟狗录制上的效果 | `... arm_prior_eval` |

## calibration/：校准

| 脚本 | 内容 |
|---|---|
| `bird_dog_calib_timeline.py` | 鸟狗校准录制的逐帧传感器方向，以及校准事件 |
| `bird_dog_axis_check.py` | 用抬臂时的旋转轴独立估计前臂航向，检验伸展解算的结果 |
| `bird_dog_calib_synthetic.py` | 合成数据对比鸟狗新旧解算器（旧版见 `pose_calibration_before_reach_heading.py`） |
| `replay_calibration.py` | 用当前 `pose_calibration.py` 回放录制里的校准事件，打印解算结果 |
| `pose_hold_synthetic.py` | 死虫、臀桥、侧桥躺姿校准的合成验证（`docs/09`） |
| `coach_pose_geometry.py` | 教练动作中各骨骼的俯仰与相对躯干的方向，作为躺姿校准设计的依据 |
| `pose_calibration_before_reach_heading.py` | 2026-11-03 改动前的 `pose_calibration.py` 备份，只供上面的对比使用 |

## legs/：鸟狗伸腿能否测出（只有小腿传感器时）

结论：腿停住时，伸腿和跪地的小腿朝向几乎相同，任何模型都区分不开；需要大腿 IMU 或气压计。详见 `reports/leg_observability_report.html`。

| 脚本 | 内容 |
|---|---|
| `bird_dog_leg_eval.py` | 各模型在鸟狗校准录制中抬起那条腿的预测 |
| `bird_dog_reach_eval.py` | 校准前后伸展时大臂的预测 |
| `leg_length_test.py` | 腿长假设检验：缩放、清零加速度对预测的影响 |
| `leg_lift_event.py` | 教练抬腿时，用户对应那条小腿是否在动（"动没动"信号） |
| `leg_motion_features.py` | 抬腿、原地动小腿、踢腿的转动量和加速度特征 |
| `leg_state_margin.py` | 教练数据中伸直和弯曲两种状态下，大腿与小腿的指向差 |
| `leg_template_signal.py`、`leg_template_test.py` | 个人模板（跪地、伸腿）能否区分 |
| `shank_activity.py` | 小腿逐 0.5 秒的活动量和方向（用于负例录制） |
| `thigh_vs_shank_knn.py` | 近邻测试：只用小腿或只用大腿，能把腿部姿势确定到什么程度 |
| `export_user_birddog.py` | 把鸟狗校准录制导出为 `data/*.npz`，供服务器实验使用 |

## acceleration/：SlimeVR 加速度能否积分出位移

结论：追踪器在桌面上平推、抬高都能积分出大致正确的位移；戴在身上做手臂动作时不对，原因没有查清（用户决定停止测试）。

| 脚本 | 内容 |
|---|---|
| `bench_acc_test.py` | 桌面测试：自动切分动作并积分位移（`unity_20261106_194609`、`unity_20261106_200542`） |
| `acc_frame_test.py`、`acc_3d_disp.py`、`acc_bias_fit.py` | 手臂录制中的坐标系、三轴位移、零偏检查 |
| `acc_vs_kinematic.py`、`acc_frame_fix.py` | 实测加速度与由朝向推出的"应有"加速度对比，以及固定旋转修正 |
| `acc_gravity_lag.py` | 假设：去重力时朝向有延迟（不成立） |
| `acc_rotation_timing.py` | 假设：加速度与朝向之间有时间差（不成立） |
| `acc_sensor_z_offset.py` | 假设：加速度轴绕传感器 z 轴偏转（只能解释一部分） |
| `shank_height_from_acc.py`、`shank_lift_zupt.py` | 用小腿加速度积分抬腿高度（失败） |

## reports/ 与 data/

- `reports/leg_observability_report.html`：给老师的报告，用浏览器打开。
- `data/user_birddog_215533.npz`、`data/user_negative_225546.npz`：导出的用户输入（已校准，SMPL 坐标系），服务器的 `artifacts/` 下有同名副本。
