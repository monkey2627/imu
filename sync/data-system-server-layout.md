# Data Generation System

服务器统一源码入口：`/home/duanyuhan/dyh/motion/data_system/`

该目录是“视频 -> SMPL/AMASS 训练数据”系统的源码权威位置。FlowHMR、DuoMo、GVHMR 与 MobilePoser 相关源代码实际归档在此；旧源码入口改为兼容链接。视频、预计算特征、checkpoint、生成结果和 Conda 环境均留在原有位置，通过资源链接或原有绝对路径访问，不在本次移动。

## 目录

- `core/flowhmr/`：主服务、FlowHMR 推理、任务调度、标注/编辑、导出与测试源码。`assets/`、`checkpoints/`、`ckpts/`、`output/` 链接回旧资源位置。
- `models/duomo/`：默认视频动作重建源码。`data/` 和 `results/` 链接回旧位置。
- `models/gvhmr/`：工作台中的对比方法源码。`inputs/`、`outputs/`、`.venv/` 链接回旧位置。
- `integrations/mobileposer/`：SMPL 拟合后的 AMASS 数据消费与 IMU 合成代码；大型训练数据和 checkpoint 不在此复制。
- `data/videos/`：输入视频。
- `data/training_data/`：已导出的 AMASS 训练集。
- `runtime/jobs/`：Web 任务记录、缓存与每任务产物。
- `runtime/output/`：FlowHMR 的完整输出目录和日志。
- `entrypoints/app.py`：Web 服务入口。
- `DATA_GENERATION_SYSTEM.md`：完整架构、格式、运行、GPU 和维护说明；链接到主仓库中的权威文档。

## 启动

```bash
cd /home/duanyuhan/dyh/motion/data_system/core/flowhmr
DATAGEN_AVOID_GPUS=6,7 setsid -f nohup \
  /home/duanyuhan/SoftWare/miniconda3/envs/flowhmr/bin/python app.py \
  --ckpt checkpoints/flowhmr_latest/flowhmr_latest.ckpt \
  --host 0.0.0.0 --port 8080 --gpus 5,1,2 \
  >> output/web_server.log 2>&1 < /dev/null
```

启动、回填、编辑和训练集导出的细节以 `DATA_GENERATION_SYSTEM.md` 为准。

## 边界

- 不要在 `data_system/` 内再次复制链接目标。
- 不要把 `runtime/jobs/` 当源码；它包含任务状态和中间产物。
- 不要移动或删除 `data/videos/`、`data/training_data/`、checkpoint 或任务目录。
- FlowHMR、DuoMo、GVHMR 和 MobilePoser 使用不同 Conda 环境；统一目录不改变环境边界。
- GPU 6、7 属于其他用户，保持在 `DATAGEN_AVOID_GPUS` 中。

## 旧路径兼容

FlowHMR、DuoMo、GVHMR 与 MobilePoser 原路径中的源码目录和入口文件链接到本目录中的源码；原路径中的资源目录仍保留。已有服务器环境和配置中的旧资源路径因此继续有效。后续开发请从本目录修改源码。

## 大型资源的后续归档建议（本次未移动）

- 视频：`/home/duanyuhan/dyh/motion/videos/` -> `data/videos/`
- FlowHMR 输出、任务与训练集：旧 `flowhmr/output/` -> `runtime/output/`、`runtime/jobs/`、`data/training_data/`
- FlowHMR checkpoint 与 body assets：旧 `flowhmr/checkpoints/`、`ckpts/`、`assets/` -> `assets/flowhmr/`
- DuoMo 权重、人体模型与预计算特征：旧 `duomo/data/` -> `assets/duomo/`（可再分 `checkpoints/`、`body_models/`、`features/`）
- GVHMR 推理权重：旧 `gvhmr/inputs/checkpoints/` -> `assets/gvhmr/checkpoints/`
- Conda 环境保持在 `/home/duanyuhan/SoftWare/miniconda3/envs/`，不要搬进源码目录。

搬动这些资源前需停用任务并更新配置/链接；本次保留其旧位置，未修改内容。
