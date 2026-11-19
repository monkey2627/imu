# Unity 一键启动与动作起始姿态

## 正常使用顺序

1. 启动 SlimeVR Server，确认五个传感器在线并完成必要的 Full Reset／安装方向校准。
2. 打开对应动作的 Unity 场景并进入 Play。
3. Unity 自动启动 `production.run`、连接 WebSocket，并开始当前场景的动作校准。
4. 不需要先自然站立。直接按 Unity 提示摆好该动作的规定起始姿态并保持。
5. 校准完成时，Python 用刚采集的稳定起始姿态重建模型上下文，再开始正式动作。

死虫的 `tabletop` 不再只作为模型历史帧：校准完成后还会启用显式起始姿态约束。它用实测小腿方向固定可观测部分，并将不可观测的大腿约束到竖直，因此模型即使给出站立偏置的直腿，画面首帧仍会呈现约 90° 屈膝的桌面式。手腕 IMU只能直接观测前臂，因此约束还会把不可观测的上臂对齐到实测前臂，让肩、肘、腕形成双臂伸直朝天的完整链条。检测到相应肢体开始动作后，各条手臂和腿分别平滑解除约束。

默认映射如下：

| Unity 动作 | Python action | 默认起始姿态 |
| --- | --- | --- |
| 死虫 | `deadbug` | `tabletop`（仰卧桌面腿） |
| 臀桥 | `hipthrust` | `supine`（仰卧屈膝） |
| 侧桥 | `sidebridge_left` | `side_rest`（左侧卧准备） |
| 卷腹 | `crunch` | `supine`（仰卧准备） |
| 卧位猫式 | `lyingcat` | `quadruped`（四点跪姿） |

`MobilePoseCalibrationController.startPoseId` 留空时使用上表默认值；如果场景里序列化了该组件，也可以填入该动作支持的其他起始姿态 ID。

## 自动启动配置

Unity 按以下顺序寻找 Python：

1. `MobilePosePythonLauncher.pythonExecutable`；
2. 环境变量 `MOTION_RECOVER_PYTHON`；
3. 本机项目环境 `E:/dyh/conda_envs/mobileposer/python.exe`；
4. 当前 `CONDA_PREFIX`；
5. 系统 `PATH` 中的 `python`。

如果端口 `21200` 已有 Python bridge 监听，Unity 会直接连接，不会重复启动。临时禁用自动启动可设置环境变量 `MOTION_RECOVER_DISABLE_AUTOSTART=1`。

Python 命令行现在默认跳过自然站立。只有诊断旧流程时才显式添加 `--upright-calibration`。

> SlimeVR 的 Full Reset／安装方向校准解决的是传感器佩戴坐标，不等同于模型的自然站立历史帧。本改动取消的是后者；前者在传感器重新佩戴、漂移或朝向错误时仍需执行。
