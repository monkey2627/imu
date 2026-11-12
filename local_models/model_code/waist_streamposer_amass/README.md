# waist_streamposer_amass

## 定位

- 架构：StreamPoser
- checkpoint：`../../checkpoints/waist_streamposer_amass/model.pt`
- 训练方式：AMASS 五 IMU 序列训练；使用流式窗口、mount 扰动与时序 lag。
- 加载入口：`model_entry.py`

## 完整代码位置

- 生产输入、标定与输出：中央 `../../../runtime/python/production/`（不复制进模型包）
- 网络与 checkpoint 调用：runtime/mobileposer/stream/model.py (`StreamPoser`)；runtime/mobileposer/stream/features.py；model_backend.py
- 训练依赖包：`training/mobileposer/`
- 本模型训练入口：`training/entrypoints/`
- 训练入口文件：`run_stream_training.sh`、`train_stream_mountaug.py`
- 数据处理：`data_processing/run_preprocess.sh`、`validate_surface_data.py`、`server_prepare_ours_processed.py`、`retarget_ours_humanpose_processed.py`、`synth_ours_replay.py`

## 独立性

本目录内包含运行和训练所需的私有源码副本，不从其他 `waist_*` 模型目录导入网络或训练代码。源码中同时保留某些基础架构模块，是因为 checkpoint 加载适配器在导入时需要它们；这些文件也只属于本模型自己的副本，不构成模型之间的共享依赖。

完整训练数据保留在服务器，本地不复制大数据。服务器路径和复现环境见 `training/SERVER_TRAINING_NOTES.md`。
