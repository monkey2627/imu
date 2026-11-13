# 本地实时模型总目录

这里集中保存当前 Python 实时链路可选择的 20 个模型（原 16 个 waist-surface 模型，加五 IMU IMUDINO 窗口版与三个流式版）。

新增 `waist_imudino_fiveimu` 的启动方式、双 checkpoint 依赖与根节点处理见 [IMUDINO 接入说明](../docs/IMUDINO_REALUSE.md)。其独立目录包含 `imudino.py`、`downstream/` 与 `model_entry.py`，不套用下述旧模型的 MobilePoser 目录结构。

2026-11-12 起有手腕/脚腕佩戴的流式 IMUDINO 模型；2026-11-13 新增 DeadBug 专用增广权重 `waist_imudino_stream_deadbug_limbmix`。
三个流式模型**没有单独的代码目录**：代码加在同一个 `model_code/waist_imudino_fiveimu/`（`stream_entry.py`、`stream/`）中、与窗口版共用 backbone；
权重分别放在各自的 `checkpoints/<model_id>/`（各含 `best.pt` + `imudino.pt`）。
文件布局、训练路径与数据处理见 [IMUDINO_STREAM_WRISTANKLE.md](../docs/IMUDINO_STREAM_WRISTANKLE.md) 第 7.1 节。

- `checkpoints/<model_id>/`：该模型实际加载的 checkpoint，以及所需配置或说明文件。
- `model_code/<model_id>/`：一个可独立查阅和移交的模型包，包含加载入口、运行/网络源码、训练源码与入口以及数据处理代码。
- `model_code/README.md`：17 个模型的总索引和目录约定。

## 单个模型包

原 16 个 `model_code/<model_id>/` 采用以下结构：

```text
<model_id>/
├─ model_entry.py
├─ runtime/mobileposer/
├─ training/mobileposer/
├─ training/entrypoints/
├─ data_processing/
└─ README.md
```

模型目录之间不导入网络或训练代码。即使不同模型使用同一种架构，它们也各自保留完整源码副本；重复是为了保证交接后能单独修改某个模型而不影响其他模型。数据处理脚本当前同样复制在各模型内，未来只有这一层允许按需要抽成公共组件。

## 与生产实时程序的关系

实际 SlimeVR → Python → Unity 程序从独立的 `../runtime/python/production/` 启动，并通过 `--model <model_id>` 选择 checkpoint。模型算法通过 `production/model_backends.py` 接入，生产输入、标定和输出不属于 MobilePoser。这里的 17 个模型目录是与模型包一一对应的代码档案和独立开发起点；对某个模型完成修改和验证后，再明确同步到生产后端，不会自动同步。IMUDINO 的生产适配直接从其独立目录加载。

完整训练数据保存在服务器，服务器训练代码和数据是训练侧权威。本地包含全部 20 个模型的权重（17 个代码目录，三个流式 IMUDINO 与窗口版共用代码目录）和可独立审计的代码副本；每个 IMUDINO 权重目录都同时需要 head（`best.pt`）与 backbone（`imudino.pt`）两个文件。不复制完整训练数据。
