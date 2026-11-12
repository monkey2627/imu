# 新工作区说明

本目录是后续工作的统一入口。`runtime/` 保存实际生产链，`local_models/model_code/` 下的 16 个模型目录分别保存各自完整的运行、训练和数据处理代码；不存在工作区级共享训练目录。旧目录中的文档仍保留作为历史记录，涉及旧路径时以根目录 `README.md` 和 `sync/source-of-truth.md` 为准。

训练数据不在本地工作区内。训练时使用服务器：

```text
10.132.25.165:/home/duanyuhan/dyh/motion/lines/realuse_waist_surface_20261030/data/
```

训练输出默认留在服务器的 `checkpoints/`、`logs/` 和 `artifacts/`。只有经过确认的模型和结果才复制到本地 `local_models/` 或 `sync/`。

本地运行代码的入口位于 `runtime/python/`；SlimeVR 和 Unity 位于 `runtime/slimevr/` 与 `runtime/unity/`。不要将服务器训练工作区误认为服务器上的 Python Git checkout `/home/duanyuhan/dyh/motion/realUse/`。
