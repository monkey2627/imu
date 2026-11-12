# waist_streamposer_ours_offsetfree

## 定位

- 架构：StreamPoser
- checkpoint：`../../checkpoints/waist_streamposer_ours_offsetfree/model_finetuned.pt`
- 训练方式：offset-free 数据加分部位 mount/heading 扰动训练，用于减小佩戴偏差。
- 加载入口：`model_entry.py`

## 完整代码位置

- 生产输入、标定与输出：中央 `../../../runtime/python/production/`（不复制进模型包）
- 网络与 checkpoint 调用：runtime/mobileposer/stream/model.py (`StreamPoser`)；runtime/mobileposer/stream/features.py；model_backend.py
- 训练依赖包：`training/mobileposer/`
- 本模型训练入口：`training/entrypoints/`
- 训练入口文件：`make_ours_offsetfree.py`、`train_stream_mountaug.py`
- 数据处理：`data_processing/run_preprocess.sh`、`validate_surface_data.py`、`server_prepare_ours_processed.py`、`retarget_ours_humanpose_processed.py`、`synth_ours_replay.py`

## 独立性

本目录内包含运行和训练所需的私有源码副本，不从其他 `waist_*` 模型目录导入网络或训练代码。源码中同时保留某些基础架构模块，是因为 checkpoint 加载适配器在导入时需要它们；这些文件也只属于本模型自己的副本，不构成模型之间的共享依赖。

完整训练数据保留在服务器，本地不复制大数据。服务器路径和复现环境见 `training/SERVER_TRAINING_NOTES.md`。
