# DeadBug、BirdDog / Ring Skin C

> 当前入口为 `BirdDog_RingSkinC_Recorded.unity` 和 `DeadBug_RingSkinC_Recorded.unity`。两者已接入右侧教练窗口，全部难度只保留伸展、收回；鸟狗采用对侧手脚配对。Build Settings 仅保留鸟狗、死虫和鼓盘臀桥三个场景。下文为最初视觉接入记录，旧场景名称和当时的场景数量不代表当前配置；最新验证见 `output/two-phase-haptics/`。

打开 `Assets/Scenes/DeadBug_RingSkinC.unity` 或 `Assets/Scenes/BirdDog_RingSkinC.unity`，按各自原流程选择难度并开始训练。
两者最初分别复制自死虫式和鸟狗式旧场景。现已删除 Classic 和旧测试场景，并添加三个 `_Recorded` 新动画副本，六个正式游戏的名称与构建顺序见 `Assets/Scenes/README.md`。三个原动画场景的文件内容和 GUID 保持不变。下文记录最初的视觉接入；新录制动画的手脚朝向、分段和验证见项目根目录 `README.md`。

## 视觉

- 使用桌面 RingSkinC 包的原始 PNG：星空、月亮、山峦、灯笼、四象限环、中心星球、音符、拖尾和命中光圈。
- 左手／右手／左脚／右脚分别对应 melody／harmony／bass／drums。
- 新视觉由 `RhythmSkinCVisuals` 显示；`RhythmSkinCNoteVisual` 只更新音符透明度和拖尾方向。
- 原始训练控件、教练预览窗、开始与完成菜单继续使用原组件；没有使用演示 HUD 的静态分数。
- 使用独立的 sRGB 合成 RenderTexture，避免更改整个项目的线性色彩空间。画面按 844:390 设计比例适配屏幕。
- 为眼镜视场预留空间，`RhythmModeRoot > RhythmSkinCVisuals > View Scale` 设为 `0.8`，环、音符与命中特效缩小到原来的 80%。教练预览和原有屏幕界面也以屏幕中心缩小、内收，保持原层级和按钮绑定。背景独立按屏幕比例等比铺满（cover），消除外围黑边，不跟随前景缩小。可在 0.5–1 之间微调前景；不改变判定位置或音符运动参数。
- 修正包内组装工具的纹理导入设置：明确指定 Texture2D，避免当前项目的默认 Cubemap 设置导致 Sprite 丢失。

## 保持原样的部分

动作动画、随机扇区路径、生成时机、音符根节点大小、判定半径、入圈与停留时长、命中阈值、计分、音乐、VMC 输入、触觉反馈及训练流程都保留。
`RingDemoChart` 没有装入新游戏场景。因此游戏中的出音符节奏和随机角度由实际训练驱动，不与 6.8 秒参考视频逐帧同步。
旧到达提示圈被隐藏，但参与原移动计算的半径仍保持不变。

## 验证

- 死虫式原场景 440 个序列化对象、108 个原脚本组件全部保留，其中 105 个脚本组件内容一致，另外 3 个只有视觉字段变化。
- 鸟狗式原场景 431 个序列化对象、107 个原脚本组件全部保留，其中 104 个脚本组件内容一致，另外 3 个只有视觉字段变化。鸟狗式四个抬升动作仍使用原有的对侧手脚配对。
- Unity 2022.3.62f3c1 编译与运行验证：四方向原音符生命周期、四次命中事件、原时间窗后的 MISS、停止后清理、菜单隐藏新视觉。
- 无追踪器的验证通过现有教程判定覆盖接口模拟 HIT/MISS，没有将测试覆盖保存到场景。实际追踪器输入和硬件震动需连接设备后体验。
- 本次截图与检查记录分别位于 `output/ringskinc/deadbug/`、`output/ringskinc/birddog/`。`game-interface.png` 包含原界面控件，`game-notes.png` 和 `game-hits.png` 展示视觉层。
- 背景覆盖检查包含 844:390、16:9、4:3、9:16、32:9 五种比例，并检查背景调整不会改变环的缩放。
- 旧文件 SHA-256 校验与鸟狗式组件差异记录位于 `output/scene-organization/`。
- 编辑器复查入口：`Tools > Ring Skin C > Verify BirdDog in Play Mode` 或 `Verify DeadBug in Play Mode`。仅对运行态模拟 HIT/MISS，结束后退出 Play，不保存测试覆盖。
- 批处理复查入口分别为 `-executeMethod RingSkinCVerification.Run` 和 `-executeMethod RingSkinCVerification.RunBirdDog`（不传 `-quit`，检查完成后自动退出）。

## 编辑

调整新场景 `RhythmModeRoot` 的 `RhythmSkinCVisuals` 引用，或 `RingLayout` 下的 `RingSkinC_GameVisuals` 实例。
`Tools > Ring Skin C > Create DeadBug Scene` / `Create BirdDog Scene` 为历史首次生成工具，依赖已删除的 Classic 源场景；当前直接编辑已有 RingSkinC 场景。目标场景存在时构建器不会覆盖。
包内原来的 `Build Prefab + Demo Scene` 菜单仅生成独立素材演示，不用于游戏逻辑。
