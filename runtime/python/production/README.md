# Production runtime

这是独立于具体模型架构的 SlimeVR → Python → Unity 生产链。

启动：

```powershell
conda activate mobileposer
python -m production.run --model waist_mobileposer_amass --web
```

也可以使用 `python -m production`。

## 边界

- `run.py`：生产调度入口。
- `solarxr_client.py`、`layout.py`：SlimeVR/SolarXR 输入与传感器布局。
- `calibration.py`、`pose_calibration.py`：生产标定。
- `stream_out.py`、`web_visualizer.py`：Unity/Web 输出。
- `recording.py`、`input_recording.py`：录制与回放输入。
- `model_backends.py`：唯一的多模型架构适配边界。
- `inference.py`：旧 PNP/MobilePose checkpoint 的推理后端。
- `runtime_contract.py`（上级目录）：生产与模型共同遵守的中立输入输出契约。

生产核心不位于 `mobileposer` 包内。MobilePose、StreamPoser、Line14 和 PosAcc
只作为并列模型后端接入；增加或替换模型时，应修改后端适配层，不把模型专用逻辑放回
SolarXR、标定或 Unity 输出模块。
