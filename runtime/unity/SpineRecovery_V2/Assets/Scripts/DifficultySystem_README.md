# 难度选择系统使用说明

## 概述
这个系统允许用户在练习开始前通过眼动交互选择稳定性判断和评分的难度等级（宽松、中等、严格）。

## 文件说明

### 1. DifficultySettings.cs
- **功能**: 定义三个难度档位的阈值配置
- **类型**: ScriptableObject
- **位置**: 需要在 Unity 中创建资源文件

### 2. DifficultySelectionUI.cs
- **功能**: 难度选择界面控制器，支持眼动交互
- **挂载**: 挂载到练习场景的UI Canvas上

### 3. PoseScorer.cs (已修改)
- **修改内容**: 
  - 添加了难度配置支持
  - 新增 `SetStabilityDifficulty()` 和 `SetScoringDifficulty()` 方法
  - 所有阈值判断改为使用难度配置

### 4. PracticeSessionController.cs (已修改)
- **修改内容**:
  - 集成难度选择界面
  - 在开始练习前显示难度选择
  - 将用户选择的难度应用到 PoseScorer

## Unity 设置步骤

### 步骤 1: 创建 DifficultySettings 资源
1. 在 Unity 项目窗口中，右键点击 `Assets/Resources` 文件夹（如果没有就创建一个）
2. 选择 `Create > Spine Recovery > Difficulty Settings`
3. 命名为 `DifficultySettings`
4. 在 Inspector 中可以调整三个难度档位的具体数值

### 步骤 2: 创建难度选择UI界面

#### 2.1 在练习场景中创建UI Canvas
```
Hierarchy:
├── DifficultySelectionCanvas
│   ├── DifficultySelectionPanel
│   │   ├── TitleText ("选择练习难度")
│   │   ├── StabilitySection
│   │   │   ├── StabilityLabel ("稳定性要求")
│   │   │   ├── StabilityLevelText (显示当前选择)
│   │   │   ├── StabilityDescriptionText (显示描述)
│   │   │   ├── StabilityButtons
│   │   │   │   ├── RelaxedButton ("宽松")
│   │   │   │   ├── MediumButton ("中等")
│   │   │   │   └── StrictButton ("严格")
│   │   ├── ScoringSection
│   │   │   ├── ScoringLabel ("评分严格度")
│   │   │   ├── ScoringLevelText (显示当前选择)
│   │   │   ├── ScoringDescriptionText (显示描述)
│   │   │   ├── ScoringButtons
│   │   │   │   ├── RelaxedButton ("宽松")
│   │   │   │   ├── MediumButton ("中等")
│   │   │   │   └── StrictButton ("严格")
│   │   └── ConfirmButton ("确认开始")
```

#### 2.2 配置 DifficultySelectionUI 组件
1. 将 `DifficultySelectionUI` 脚本挂载到 `DifficultySelectionCanvas` 上
2. 在 Inspector 中配置引用:
   - **Difficulty Settings**: 拖入步骤1创建的 `DifficultySettings` 资源
   - **Selection Panel**: 拖入 `DifficultySelectionPanel`
   - **Stability Buttons**: 分别拖入稳定性的三个按钮
   - **Scoring Buttons**: 分别拖入评分的三个按钮
   - **Confirm Button**: 拖入确认按钮
   - **显示文本**: (可选) 拖入各个Text组件

### 步骤 3: 配置 PracticeSessionController
1. 找到练习场景中的 `PracticeSessionController` 对象
2. 在 Inspector 中配置:
   - **Difficulty Selection UI**: 拖入步骤2创建的UI Canvas
   - **Pose Scorer**: 拖入场景中的 PoseScorer 对象
   - **Show Difficulty Selection Before Practice**: 勾选（默认显示难度选择）

### 步骤 4: 配置眼动交互 (GazeDwellClicker)
1. 确保场景中有 `GazeDwellClicker` 组件
2. 配置:
   - **Scan All Active Canvases**: 勾选（自动扫描所有Canvas上的按钮）
   - **Dwell Seconds**: 设置为 1-2 秒（注视时长）
   - **Progress Ring**: 拖入一个环形进度条Image（显示注视进度）

## 难度配置说明

### 宽松模式 (Relaxed)
适合初学者，容错率高：
- **稳定性漂移阈值**: 8°/秒 (允许更大晃动)
- **对齐角度阈值**: 45° (一般骨骼)
- **腿部对齐阈值**: 60° (腿部骨骼)
- **最大角度误差**: 40° (评分容错范围大)
- **完美角度阈值**: 8° (满分要求低)

### 中等模式 (Medium) - 推荐
标准难度，平衡挑战性：
- **稳定性漂移阈值**: 5°/秒
- **对齐角度阈值**: 35°
- **腿部对齐阈值**: 50°
- **最大角度误差**: 30°
- **完美角度阈值**: 5°

### 严格模式 (Strict)
适合熟练者，要求精确：
- **稳定性漂移阈值**: 3°/秒 (需要非常稳定)
- **对齐角度阈值**: 25° (对齐要求高)
- **腿部对齐阈值**: 35° (腿部也要求严格)
- **最大角度误差**: 20° (评分容错范围小)
- **完美角度阈值**: 3° (满分要求高)

## 使用流程

1. 用户点击"开始练习"按钮
2. 系统显示难度选择界面
3. 用户通过眼动交互选择:
   - 稳定性要求档位（宽松/中等/严格）
   - 评分严格度档位（宽松/中等/严格）
4. 用户注视"确认开始"按钮完成选择
5. 系统应用选择的难度配置到 PoseScorer
6. 开始练习，使用选定的难度进行评分

## 调试和测试

### 查看应用的难度
在 Console 中会输出:
```
PoseScorer: 稳定性难度已设置 - 漂移阈值=5°/s, 对齐阈值=35°, 腿部阈值=50°
PoseScorer: 评分难度已设置 - 最大误差=30°, 完美阈值=5°
PracticeSessionController: 难度已应用 - 稳定性=中等, 评分=中等
```

### 测试不同难度
- **宽松模式**: 应该更容易通过稳定性检测，得分更高
- **严格模式**: 需要更精确的姿势，更难得高分
- **混合设置**: 可以单独设置稳定性宽松+评分严格，或其他组合

## 注意事项

1. **确保 Resources 文件夹**: DifficultySettings 必须在 Resources 文件夹中才能被自动加载
2. **眼动交互**: 确保按钮大小合适，方便眼动选择（建议至少 100x100 像素）
3. **UI层级**: 难度选择面板应该在其他UI之上，避免被遮挡
4. **默认值**: 如果没有设置难度配置，PoseScorer 会使用 Inspector 中的默认值（中等难度）
5. **跳过难度选择**: 如果想跳过难度选择直接开始，取消勾选 `Show Difficulty Selection Before Practice`

## 扩展建议

1. **记住上次选择**: 可以使用 PlayerPrefs 保存用户上次的选择
2. **动态调整**: 根据用户表现自动推荐难度
3. **更多档位**: 可以在 DifficultySettings 中添加更多难度档位
4. **单独设置**: 为不同的练习动作设置不同的默认难度
