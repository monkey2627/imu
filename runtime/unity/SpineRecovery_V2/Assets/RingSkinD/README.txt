Ring Skin D · 赛博声场 —— Unity 使用说明

1. 导入 RingSkinD.unitypackage（Unity 2021.3 及以上，内置管线或 URP 均可）
2. 菜单 Tools > Ring Skin D > Build Prefab + Demo Scene
   - 自动把 Sprites 设为 Sprite、300 PPU（3x 美术，1 设计 px = 0.01 unit）
   - 自动组装并生成 Prefabs/RingSkinD.prefab 与 Scenes/RingSkinD_Demo.unity
3. 打开 RingSkinD_Demo，Game 视图设为横屏手机分辨率（如 2532x1170），点 Play

层级结构（RingSkinD 预制体）
  Background   背景、地平线光（随拍脉冲）、城市剪影、地面网格、地面横线动画、楼顶闪灯
  Ring         刻度环（40 秒一圈）、四个 Zone_*（底 + LED1–4 + 蓄满闪光）、判定环与色差副本、Core
  HUD          暂停按钮、曲名、进度条、分数、SYNC 标签（随拍弹跳）
  RingDemoChart（挂在根节点）演示谱面：音符飞入、命中菱形、PERFECT、中心收缩圈与核心脉冲

脚本（Scripts/，命名空间 RingSkinD）
  LedChargeQuadrant  5 秒逐层蓄力，满后闪一次；斜对侧两区一组，第二组晚 3 秒
  RingDemoChart      演示用 4 秒循环谱面，正式版替换为真实谱面数据
  DesignSpace.Now    所有动画的时间源，正式版改为音乐播放时间
  其余：SpinForever / BeatPulseAlpha / BeatBump / ChromaGlitch / FloorLines / BlinkLight

字体 Orbitron、JetBrains Mono 为 SIL OFL 开源字体（见 Fonts/OFL.txt），仅含拉丁字符；
曲名中文使用 Unity 内置字体回退到系统字体。
