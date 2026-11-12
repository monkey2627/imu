# 正式游戏场景

| 游戏 | 场景文件 |
| --- | --- |
| 鸟狗式 | `BirdDog_RingSkinC_Recorded.unity` |
| 死虫式 | `DeadBug_RingSkinC_Recorded.unity` |
| 臀桥 · 新评分 | `HipThrust_RingBridge_NewScoring.unity` |
| 臀桥 · 保持 30 秒 | `HipThrust_RingBridge_Hold30.unity` |
| 左侧桥 · 动态 | `SideBridge_RingBridge_Dynamic.unity` |
| 左侧桥 · 保持 30 秒 | `SideBridge_RingBridge_Hold30.unity` |

按场景名称加载，不依赖固定 Build Index。可选旧版 `HipThrust_RingBridge_Recorded.unity` 仅在文件保留时注册；整理工具不会恢复已经删除的场景。

全部使用新录制动画及右侧教练窗口。鸟狗和死虫使用 RingSkinC 视觉，全部难度仅有伸展、收回；臀桥使用敲鼓素材，保留抬起、保持、放下。教练与视觉跟随同一播放进度，暂停时同步停止。

直接打开场景并 Play，按校准、选择难度、开始训练流程操作。`hipthrust_newscore`、`hipthrust_hold30` 分别选择臀桥新评分和 30 秒变体。鸟狗左腿配右手、右腿配左手；桥式 HIT 仅腰部震动。

新评分场景保持原 4/8/12 次节奏。30 秒场景在最高点静止保持，次数来自处方，默认附带标注为演示的 1 组 × 4 次处方；实际接入方法及评分规则见项目根目录 `HIPTHRUST_NEW_RULES.md`。

两个左侧桥场景使用原始 HumanPose 录制及侧桥评分，沿用臀桥视觉和保持进度光条。动作键为 `sidebridge`、`sidebridge_hold30`，都要求明确的 `reps`；默认演示处方为 1 组 × 3 次。动态版保留录制的抬起、短保持、下降与低位调整间隔；保持版在最高点冻结 30 秒。原始腕踝旋转不改，重复循环用 0.5 秒过渡衔接。导入、分段、评分调整、处方标识和设备验证范围见 `Assets/Movements/RecordedSideBridge/README.txt`。

原始录制来源、骨骼处理及验证入口见项目根目录 `README.md`。素材文件夹中的集成说明记录历史接入过程，当前场景列表以此文件和 Build Settings 为准。
