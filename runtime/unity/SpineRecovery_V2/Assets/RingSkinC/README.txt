Ring Skin C · 星轨光环 —— Unity 使用说明

1. 导入 RingSkinC.unitypackage（Unity 2021.3 及以上，内置管线或 URP 均可）
2. 菜单 Tools > Ring Skin C > Build Prefab + Demo Scene
   - 自动把 Sprites 设为 Sprite、300 PPU（3x 美术，1 设计 px = 0.01 unit）
   - 自动组装并生成 Prefabs/RingSkinC.prefab 与 Scenes/RingSkinC_Demo.unity
3. 打开 RingSkinC_Demo，Game 视图设为横屏手机分辨率（如 2532x1170），点 Play

层级结构（RingSkinC 预制体）
  Background   夜空、星星/月亮/星座、闪烁星 ×5、星座亮星呼吸、流星（每 4 秒一次）、山峦、云海、孔明灯（6 秒浮动）
  Ring         刻度环（40 秒一圈）、四个 Zone_*（底 + 弧线发光 + 命中闪亮）、判定环节拍光、Core 光球
  HUD          暂停按钮、曲名、分数、COMBO（随拍弹跳）
  RingDemoChart（挂在根节点）演示谱面：音符飞入、命中波纹、区域闪亮、PERFECT 上浮、中心收缩圈与核心脉冲

脚本（Scripts/，命名空间 RingSkinC，可与 RingSkinD 同时导入同一工程）
  RingDemoChart      演示用 4 秒循环谱面，正式版替换为真实谱面数据
  DesignSpace.Now    所有动画的时间源，正式版改为音乐播放时间
  其余：SpinForever / BeatPulseAlpha / BeatBump / Twinkle / MeteorLoop / FloatBob

字体 Fredoka 为 SIL OFL 开源字体（见 Fonts/OFL.txt），仅含拉丁字符；
曲名中文使用 Unity 内置字体回退到系统字体。
