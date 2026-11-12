# waist_mobileposer_deadbug

## 定位

- 架构：MobilePose
- checkpoint：`../../checkpoints/waist_mobileposer_deadbug/model_finetuned.pth`
- 训练方式：从 AMASS MobilePose 基线出发，以 Deadbug 序列和 FBX/SMPL 伪标签进行个体化微调。
- 加载入口：`model_entry.py`

## 完整代码位置

- 生产输入、标定与输出：中央 `../../../runtime/python/production/`（不复制进模型包）
- 网络与 checkpoint 调用：runtime/mobileposer/mobile_export.py (`MobilePose`)；model_backend.py
- 训练依赖包：`training/mobileposer/`
- 本模型训练入口：`training/entrypoints/`
- 训练入口文件：`prepare_deadbug.py`、`finetune_mobileposer_deadbug.py`、`run_deadbug_personalization.sh`、`audit_deadbug_input.py`、`fit_deadbug_humanpose_smpl.py`、`make_deadbug_targets_fbx.py`
- 数据处理：`data_processing/run_preprocess.sh`、`validate_surface_data.py`、`server_prepare_ours_processed.py`、`retarget_ours_humanpose_processed.py`、`synth_ours_replay.py`

## 独立性

本目录内包含运行和训练所需的私有源码副本，不从其他 `waist_*` 模型目录导入网络或训练代码。源码中同时保留某些基础架构模块，是因为 checkpoint 加载适配器在导入时需要它们；这些文件也只属于本模型自己的副本，不构成模型之间的共享依赖。

完整训练数据保留在服务器，本地不复制大数据。服务器路径和复现环境见 `training/SERVER_TRAINING_NOTES.md`。
