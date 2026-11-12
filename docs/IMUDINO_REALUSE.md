# IMUDINO 五 IMU 接入（2026-11-11）

后续真实录制诊断已发现采样计数、窗口边界和在线末帧不连续问题，详见 [抖动诊断](IMUDINO_JITTER_DIAGNOSIS.md)。
以下命令描述当前可运行接入，不代表已通过连续性或穿戴稳定性验收。

模型 ID：`waist_imudino_fiveimu`。服务器训练源为 `10.132.25.165:/home/duanyuhan/dyh/motion/IMUDINO-Release`。
本次只读取服务器代码、checkpoint 和说明；未运行训练、数据处理或渲染入口，也未修改服务器数据或训练产物。

## 实际启动

在本地启动工作区内的 SlimeVR-Server，按原流程完成姿态/安装校准。五个设备分别绑定
`LEFT_LOWER_ARM`、`RIGHT_LOWER_ARM`、`LEFT_LOWER_LEG`、`RIGHT_LOWER_LEG`、`WAIST`，不要用 HIP 替代 WAIST。

```powershell
conda activate mobileposer
cd E:\dyh\MotionRecover\realuse\MotionRecoverWorkspace\runtime\python
python -m production.run --config configs/imudino_fiveimu.yaml --no-auto-compare
```

Unity 使用原有 `ws://127.0.0.1:21200` 连接；在模型菜单选择 `IMUDINO / Five IMU (AMASS + BirdDog)`。
Unity 会发送当前菜单选择，因此也需确认菜单没有选回其他模型。既有 Unity 可执行文件需重新构建才有新增菜单项。
只使用 Python 时也可直接运行 `python -m production.run --model waist_imudino_fiveimu --infer-every 6 --web`。
浏览器选择列表也已注册新模型。

默认设备为有 CUDA 时 `cuda:0`，否则 CPU。可在启动前用 PowerShell 显式设置：

```powershell
$env:MOTION_RECOVER_IMUDINO_DEVICE = 'cuda:0'  # 或 cpu
```

不需要为此更改本地 PyTorch 或安装 timm/torchvision。默认配置每 6 个输入帧推理一次，即目标 10 次/秒；
实际速度受硬件限制。`--infer-every 2` 可请求 30 次/秒，但必须先确认本机推理耗时。

使用已有真实录制输入检查整个 Python 广播链路，不需要佩戴设备：

```powershell
python -m production.run --config configs/imudino_fiveimu.yaml --replay BirdDog_RingSkinC_Recorded_20261101_145430.jsonl --no-auto-compare --web
```

## 文件与加载

- `local_models/checkpoints/waist_imudino_fiveimu/best.pt`：原始最佳下游 checkpoint 的完整部署副本。
- 同目录 `imudino.pt`：冻结的完整 IMUDINO 编码器，缺少它无法推理。
- `local_models/model_code/waist_imudino_fiveimu/`：独立代码目录，包含编码器、`downstream/` 原始模型/预处理/训练/验证代码，以及 `model_entry.py` 在线适配器。
- `model_entry.load()`：独立 Python 包加载接口；生产桥通过 `production/model_backends.py` 调用同一个适配器。
- `manifest.json`：checkpoint 来源、SHA256 和部署契约。

`best.pt` 字段为 `head, optimizer, scheduler, epoch, best_score, datasets, seed, sensor_order, target, window_frames`。
当前 epoch=30，seed=42，best_score=0.009917778526140675。推理只严格加载 `head`；
编码器从 `imudino.pt` 严格加载完整 state_dict。优化器和调度器不参与推理。
姿态头输入宽度 3840，隐藏维度 512，输出维度 138；并非直接把 `best.pt` 当完整网络。

独立接口示例（先将 `local_models/model_code` 加入 Python 模块搜索路径）：

```python
from waist_imudino_fiveimu.model_entry import load
adapter = load(device="cuda:0")
# orientation: [5,3,3] 校准后的 sensor/bone-to-world；acceleration: [5,3] 线加速度 m/s²
adapter.prime(None, orientation, acceleration)
adapter.push(None, orientation, acceleration)
local_rotations = adapter.infer()  # CPU float32 [24,3,3]，含由腰部约束求出的根节点
```

## 输入、输出与时序

固定传感器顺序为 `lw,rw,ls,rs,waist`。采样频率 60 Hz，窗口 480 帧（8 秒），patch 大小 16。
生产桥继续使用既有 SlimeVR reference-adjusted 四元数、线加速度、坐标变换和安装校准。
适配器使用独立的 orientation/acceleration 参数，忽略其他模型使用的 `/30` 特征，因此不重复缩放。

预处理直接调用此模型的 `downstream/data.py::to_imu6`：
世界线加速度减去 Y-up 重力 `(0,-9.81,0)`，再经旋转转置变成各校准传感器坐标系的比力；
角速度由连续方向矩阵计算，单位 rad/s。无额外均值/方差标准化。
在线路径在窗口末端额外读取一帧真实传感器样本，用于计算最后一个 gyro；不会把末帧角速度静默复制一遍。

输入编码器前张量 `[1,480,5,6]`，通道为加速度 XYZ、角速度 XYZ，
内部五路展开为 `[5,480,6]`；mask `[1,480]` 全有效，freqs 为五个 60。
复用 `downstream/model.py::encode_imu`，得到 `[1,30,3840]`，经 `FiveIMUHead` 得到 `[1,480,138]`。
6D 表示为矩阵前两列依次拼接，按验证实现的 Gram–Schmidt 转成旋转矩阵，
对应标准 SMPL 顺序中 joints 1..23 的 **局部**旋转，不是全局旋转。

具体顺序为：1 左髋、2 右髋、3 spine1、4 左膝、5 右膝、6 spine2、7 左踝、8 右踝、9 spine3、
10 左足、11 右足、12 颈、13 左锁骨、14 右锁骨、15 头、16 左肩、17 右肩、
18 左肘、19 右肘、20 左腕、21 右腕、22 左手、23 右手。

在线采用固定 60 Hz 重采样后的最近 480 帧，并额外保留 1 个真实前视样本计算连续 gyro；输出帧索引 471，`modelLagFrames=9`。
启动历史不足时左侧重复最早输入；之后由真实样本逐步替换，完整真实历史需要 8 秒，前视样本再增加约 1 帧延迟。
热切换模型会保留最多 480 帧近期输入。录制按实际进入模型的均匀帧保存；Unity 对该模型按代表帧号做显示插值。
这不是原验证脚本的 240 帧滑窗重叠平均：编码器是窗口内双向注意力，在线末帧没有未来观测，
不应将离线验证指标当作实时末帧精度。编码器推理计算本身仍会产生实际耗时。

## 根节点与全局坐标

训练不预测 joint 0 或根平移。现有 WAIST 定义测量 SMPL spine1（joint 3），其父关节为 pelvis（joint 0）。
在线使用同一帧校准后的腰部方向与网络预测的 joint 3 局部旋转求
`R_root = R_waist @ R_local_joint3.T`，因此 FK 中 joint 3 的全局方向等于腰部测量。
23 个预测局部旋转均保留，不额外把腰部当作 pelvis。
这是腰部约束下的根姿态估计，会受到腰部校准和预测脊柱角度误差影响，并非网络独立预测根姿态。

输出仍是原协议的 24 个局部旋转四元数；不新增或假造根平移。
人物世界位置与可选贴地继续由 Unity 原有角色/grounding 逻辑处理。
既有 `--anchor-sensed`、手臂先验和 Unity 姿态选项仍可作为后处理；
如需观察网络未经手臂先验修改的输出，传 `--no-arm-prior` 并在 Unity 关闭对应选项。
部署过程不访问验证伪标签、特征缓存或视频。

## 代码兼容边界

`downstream/` 保留服务器原始代码，可审查 `prepare.py` 的特征提取、`train.py` 的训练方式、
`render_validation.py` 的验证定义；这些入口不是线上依赖，也不会被启动。
训练为冻结编码器 + MLP 姿态头，18 个 AMASS 子集及 BirdDog，BirdDog loss weight=20，30 epochs。
完整数据和特征仍在服务器，部署只携带代码与两个权重文件。

本地编码器相对服务器仅增加 Python 3.9 注解兼容、PyTorch 2.1 编译状态检查兼容，
以及缺少 timm 时的 `torch_blocks.py` 推理 fallback。
fallback 使用同名参数、LayerNorm eps=1e-6、GELU 和 PyTorch scaled-dot-product attention；
仅支持 eval，训练仍使用服务器官方 timm 实现。
服务器用原 timm 与 fallback 分别严格加载相同完整编码器，对 480 帧五 IMU、含 padding mask 的输入比较：
编码特征和完整姿态头输出最大绝对差均为 **0**。

其余本地加载、录制回放和实际耗时的结果见同目录 `IMUDINO_VALIDATION.md`。

## 本次改动文件

- 新增 `local_models/model_code/waist_imudino_fiveimu/` 独立代码包，及 `local_models/checkpoints/waist_imudino_fiveimu/` 双权重与 manifest。
- `runtime/python/production/model_backends.py`：注册模型、校验输入顺序、加载独立模型包、声明 lag。
- `runtime/python/production/run.py`：新模型固定采样、前视 gyro、9 帧 lag 与真实输出记录。
- `runtime/python/production/offline_compare.py`：新模型按 9 帧 lag 比较已有实时录制。
- `runtime/python/configs/imudino_fiveimu.yaml`：启动配置。
- `runtime/unity/SpineRecovery_V2/Assets/Scripts/MobilePoseWebSocketSource.cs`：增加模型菜单项，沿用原有广播协议。
- 工作区 `README.md`、`local_models/README.md`、`local_models/model_code/README.md`、`sync/source-of-truth.md`：增加第 17 个模型与服务器来源说明。
- `docs/IMUDINO_REALUSE.md`、`docs/IMUDINO_VALIDATION.md`：部署与验证记录。
