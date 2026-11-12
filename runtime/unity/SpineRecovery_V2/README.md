# SpineRecovery_V2

## 当前游戏

Build Settings 注册以下五个录制动画场景。原有三个游戏保留，新增两个臀桥评分变体。

| 动作 | 场景 | 视觉与阶段 |
| --- | --- | --- |
| 鸟狗 | `Assets/Scenes/BirdDog_RingSkinC_Recorded.unity` | RingSkinC；伸展、收回 |
| 死虫 | `Assets/Scenes/DeadBug_RingSkinC_Recorded.unity` | RingSkinC；伸展、收回 |
| 臀桥 | `Assets/Scenes/HipThrust_RingBridge_Recorded.unity` | 敲鼓；抬起、保持、放下 |
| 臀桥 · 新评分 | `Assets/Scenes/HipThrust_RingBridge_NewScoring.unity` | 同一敲鼓视觉与原节奏，换用交付包 A/B 保持评分 |
| 臀桥 · 保持30秒 | `Assets/Scenes/HipThrust_RingBridge_Hold30.unity` | 同一敲鼓视觉；最高点静止30秒后放下，精确处方次数 |

场景均保留起始菜单、校准、难度、倒计时、暂停、计分、结束和动作串流程。右侧教练窗口持续显示实际动画，并与游戏阶段、进度和暂停同步。鸟狗与死虫的全部难度不再插入保持等待，仍在伸展到达时进行原有评分检查。臀桥鼓盘与教练共用对应模式的三阶段时间轴。新评分规则和处方接入方法见 [HIPTHRUST_NEW_RULES.md](HIPTHRUST_NEW_RULES.md)。

## 录制动画与手脚方向

资源位于 `Assets/Movements/Recorded20261018/`，全部为 30 fps。完整 FBX 未修改；`*.coachrecording` 是原始 `.motion.json` 改扩展名后的同一内容，来源和校验值见 `*.source.json`。游戏分段及肢体配置见 `*.phases.json`。

| 动作 | 桌面来源 | 使用的原始帧范围 |
| --- | --- | --- |
| 鸟狗 | 腰椎第一次记录 / 8_鸟狗 / 1 / 8_v1.fbx | 90–1053 |
| 死虫 | 腰椎第一次记录 / 7_死虫 / 1 / 7_v1.fbx | 498–1041 |
| 臀桥 | 第二次膝关节记录 / 5_臀桥 / 1 / 5_v1.fbx | 180–954 |

每份录制选取四次完整往返，原节奏模式的简单、标准、进阶分别播放 1、2、3 轮；30 秒臀桥变体按处方精确次数采样并冻结最高点。自定义导入器直接从原始 HumanPose 生成完整、准备、四组往返及结束动画，保留采样帧和共享边界，避免 FBX 二次重定向改变腕、掌、踝、足的朝向。

鸟狗原始录制为同侧手脚抬起，因此仅在导入生成动画时通过 `mirrorArmChannels` 交换左右肩、臂、腕和手指的配对肌肉通道。腿部、躯干、原始文件和动作节奏保持不变。实际动画与判定均为第 1/3 次右腿配左手、第 2/4 次左腿配右手；不是仅修改界面标签。臀桥整体绕竖直轴旋转 -90° 以显示侧面，不改变骨骼之间的相对旋转。

## HIT 触觉反馈

鸟狗、死虫在命中后向对应前臂或小腿发送一次震动。臀桥只订阅鼓盘实际 HIT 事件，并只发送到 `Abdomen`（腰部/髋部设备绑定）；旧音符事件不会再触发四肢震动。MISS、待机和保持奖励不触发震动。请求沿用 `TrackerHaptics` 的 `/motor?duration_ms=100`。

## 验证

- `Tools > Ring Skin C > Verify Migrated Coach Windows`：实际开始按钮、倒计时、两阶段、全部难度零保持、暂停恢复、对侧配对、预览手脚姿态、完成和返回菜单。
- `Tools > Ring Skin C > Verify HIT-only Tracker Requests`：通过本机 HTTP 接收端验证真实请求路由、命中/未命中、臀桥腰部独占、重复启停和待机静默，不连接真实设备。
- `Tools > Recorded Coach > Verify Recorded Bone Orientation`：逐帧检查三个动作的 2,283 帧；鸟狗与解剖配对后的原始手臂通道比较，腿部保持原始朝向。
- `Tools > Taiko Self > Verify Recorded Visual Skin`：臀桥菜单、三阶段视觉与教练同步、暂停、完成和难度流程。
- `Tools > Ring Bridge > Verify New Scoring and 30 Second Scenes`：新规则边界、骨骼静止30秒、精确剂量、实际动作串组间转换。报告见 `output/hipthrust-new-rules/`。

两阶段、朝向和震动验证报告及截图位于 `output/two-phase-haptics/`。设备佩戴和实际震感仍需硬件联调。`Tools > Ring Skin C > Organize Game Scenes` 可重新登记当前五个构建场景。素材接入历史见 `Assets/RingSkinC/DEADBUG_INTEGRATION.md`、`Assets/RingBridge/INTEGRATION.md`、`Assets/TaikoSelf/INTEGRATION.md`。
