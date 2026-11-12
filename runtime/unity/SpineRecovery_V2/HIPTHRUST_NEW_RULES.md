# 两个臀桥新评分场景

原有三个场景、录制动画、材质、动作时间轴均保留。新场景都复制自当前敲鼓视觉的 `HipThrust_RingBridge_Recorded`，保留起始界面、倒计时、教练演示、暂停、结束界面、躺姿校准和 HIT 腰部震动。

| 场景 | 行为 |
| --- | --- |
| `Assets/Scenes/HipThrust_RingBridge_NewScoring.unity` | 只替换计分。保持原始录制节奏及简单/标准/进阶的 4/8/12 次。 |
| `Assets/Scenes/HipThrust_RingBridge_Hold30.unity` | 录制动作抬到实际最高帧，教练和预览骨骼静止 30 秒，再继续放下。按处方精确执行次数，不向 4 的倍数取整。 |

两个场景已追加至 Build Settings。Play 后仍需在起始界面校准、点击开始。真实评分需要已校准的人体骨骼和实时 VMC 数据；跳过校准只用于流程预览，不产生模拟患者得分。

## 新评分来源与规则

来源：桌面 `臀桥训练场景交付包/臀桥训练场景交付包` 的说明文档及 `HipThrustPractice_Minimal.unitypackage` 内 `HipThrustHoldScorer.cs`。包 SHA-256：`0da1b2928d6e3554a04109268c3cd8e9cf736abae3c3182a627ec15fb2ff4d94`。

移植的是文档指定的 **连续 A/B 保持计分**；没有引入包里的自主练习场景、启动器、旧的分组综合分、骨骼约束器或 VMC 滤波设置。实际实现为 `HipThrustHoldRule`、`HipThrustHoldGeometry` 和 `HipThrustRuleScoring`。

骨盆到胸部/双足支撑面的垂直距离除以身体跨度，映射教练 A/B 端点。质量采用髋角 45%、大腿/小腿方向 30%、膝角 15%、左右对称 10%。端点来自当前实际录制动作，不依赖交付包中另一个 Animator 的状态名。评分不会改写手脚或其他骨骼的旋转。

| 难度 | 开始抬起 | 进入 B | 退出 B | 严格返回 A | 进入质量 | 有效质量 | 起评分 | 有效保持至满分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 简单 | .08 | .20 | .14 | .10 | 35 | 30 | 85 | 3 秒 |
| 标准 | .15 | .40 | .28 | .15 | 50 | 45 | 80 | 5 秒 |
| 进阶 | .20 | .55 | .40 | .18 | 65 | 60 | 75 | 7 秒 |

进入 B 连续确认 .30 秒，退出 B .12 秒，回到 A .08 秒；放下 .20 秒后支持文档中的宽松返回条件。B 内质量不足时停止累加，B 外实时分为 0，保留上轮最高分（HUD 的 LAST）。数据缺失或超过 .5 秒无新帧时不累加；暂停冻结训练和计分。

30 秒是第二个场景的**教练动作保持时间**，没有把交付包中的满分时间改为 30 秒。玩家 A/B 识别和得分依然独立于教练时间轴。界面“完成次数”指教练示范的完整抬起—30 秒保持—放下次数；结果中的“识别往返”是实测识别次数，不能把示范完成当成患者动作合格。

## 处方接入

第二个场景附带明确标注的演示处方 `Assets/RingBridge/HipThrustHold30_PreviewPrescription.json`（1 组 × 4 次）。难度按钮只影响评分阈值，不改变处方次数。真实接入通过主线程调用，失败返回错误且保留原有效剂量；训练中/倒计时中不允许换处方。

### 已获取的医生 D01 V2 处方

```csharp
var dose = FindObjectOfType<HipThrustPrescriptionDose>();
bool ok = dose.SetPrescriptionJson(prescriptionJson, itemId: 123, out string error);
```

读取 `prescriptionContractVersion=2` 和 `items[]` 中指定 `itemId` 的 `actionId="5"`、`sets`、`reps`。多个臀桥项目必须明确指定 itemId；其他动作不能误当作臀桥。组数限制 1–20，次数限制 1–100。单独场景按 `sets × reps` 完整播放，菜单明确显示组数和次数；处方未提供组间休息，本场景不自行增加。`holdSeconds` 不覆盖该变体固定的 30 秒。此入口消费已获取的处方，不新增网络请求或资源下载。

### 现有动作串执行器

新增两个显式动作键，旧 `hipthrust` 仍指向原场景：

```json
{
  "actions": [
    { "action": "hipthrust_newscore", "difficulty": "simple", "sets": 1 },
    { "action": "hipthrust_hold30", "difficulty": "standard", "sets": 2, "reps": 3 }
  ]
}
```

调用 `GameWorkoutSession.Instance.SetPlanJson(json, out error)`，然后 `StartPlan(out error)`。执行器每组传入精确 reps，组数由执行器控制，不重复相乘；缺少合法 reps 会拒绝运行。已有手机分支的通用 `hipthrust` 导出不会自动选成 30 秒变体，调用方选择新变体时应显式设置上述动作键。

## 验证

`Tools → Ring Bridge → Verify New Scoring and 30 Second Scenes`：覆盖三个难度的数值阈值与确认时间、低质量/断流、几何平移旋转和等比缩放、精确次数、所有峰值帧、30 秒计时、手脚等骨骼静止、教练/预览/鼓面同步、开始/暂停/恢复/结束，以及实际执行器 2 组 × 3 次。

报告和截图位于 `output/hipthrust-new-rules/`。验证使用合成输入和实际 Unity 场景动画，没有使用真人或实体 IMU 来验证动作识别准确率。
