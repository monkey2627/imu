敲鼓 · 正面鼓面 · 自主节奏 —— Unity 使用说明

1. 导入 TaikoSelf.unitypackage（Unity 2021.3 及以上，内置管线或 URP 均可）
2. 菜单 Tools > Taiko Self > Build Prefab + Demo Scene
   - 自动把 Sprites 设为 Sprite、200 PPU（2x 美术，1 设计 px = 0.01 unit）
   - 自动组装并生成 Prefabs/TaikoSelf.prefab 与 Scenes/TaikoSelf_Demo.unity
3. 打开 TaikoSelf_Demo，Game 视图设为手机横屏比例（如 2532x1170），点 Play
   演示由 SimulatedPatient 驱动：抬起 1.2 秒 → 保持 2.8 秒 → 慢慢放下 5 秒 → 休息，循环

设计尺寸 844x390 设计 px（与圆环 C 相同），鼓面中心 (422, 215)。

核心思路：没有预设时间，全部由患者动作驱动
  SelfPacedDrum.height：每帧写入归一化的抬臀高度（0 = 躺平，1 = 标定的最高点），来自 IMU / 姿态识别
  组件内部用阈值自动判断阶段：
    休息 → 抬起：height > startThreshold（默认 0.05）
    抬起 → 保持：height ≥ topThreshold（默认 0.92）
    保持 → 慢慢放下：height 比保持阶段的最高点下降超过 releaseDrop（默认 0.02）
    慢慢放下 → 休息：height ≤ restThreshold（默认 0.04）
  正式使用时关闭 SimulatedPatient 组件，改为每帧写入 height 即可；阈值按设备噪声在 Inspector 里调

三个色块依次亮起，互不重叠（上一块在 gap = 0.12 秒内完全变暗后，下一块才开始）
  粉 · 抬起（80°，最小）   色块和外圈进度弧随抬起高度变亮、走进度
  黄 · 保持（120°）        六颗星每 0.4 秒点亮一颗，保持多久就亮多少；色块随星星变亮；鼓心发金光
  紫 · 慢慢放下（160°，最大）带彗星拖尾的光点沿弧移动，拖尾长度 = 放下进度；色块随同一进度变亮；
                           光点经过时音符依次点亮并放出光圈，星尘向下飘落
  奖励：本次放下用时比抬起长，右上星座多亮一颗星（保持到下一次抬起）

层级结构（TaikoSelf 预制体）
  Background   夜空、星芒、星座亮星、流星、孔明灯、外圈虚线环自转、鼓边与铆钉
  DrumHead
    Lift       粉色块、进度弧（被 LiftProgressMask 半圆遮罩裁出，遮罩绕鼓心旋转到进度前沿）
    Hold       黄色块、6 个星槽 + 6 颗星
    Return     紫色块、ReturnClipMask（静态扇形遮罩）+ 彗星拖尾（随光点绕鼓心旋转）、6 个音符与光圈、
               DotPivot/Dot 光点、12 颗星尘
    Centre     鼓心金光、鼓心、鼓面高光、星座奖励
  HUD_Sprites  暂停按钮、阶段提示文字（抬起 / 保持 / 慢慢放下）
  HUD          分数与 COMBO（UI Text，Fredoka）

脚本（Scripts/，命名空间 TaikoSelf）
  SelfPacedDrum     阶段判断与全部视觉映射（输入：height）
  SimulatedPatient  演示用模拟数据
  SelfLayout        由设计稿自动生成的图层坐标、层级、星星 / 音符 / 星尘位置与角度
  Twinkle / MeteorLoop / FloatBob / SpinForever   背景循环动效
  DesignSpace       坐标换算与缓动

字体 Fredoka 为 SIL OFL 开源字体（见 Fonts/OFL.txt）；中文文字已烘焙进图片。
