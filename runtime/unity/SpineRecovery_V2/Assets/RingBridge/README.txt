圆环 C · 臀桥三段 —— Unity 使用说明

本项目正式游戏已接入：打开 Assets/Scenes/HipThrust_RingBridge_Recorded.unity。
新录制的帧位分段、正式控制方式及验证说明见同目录 INTEGRATION.md。
以下保留交付素材的原始演示使用说明；正式场景不使用固定 7 秒循环。

1. 导入 RingBridge.unitypackage（Unity 2021.3 及以上，内置管线或 URP 均可）
2. 菜单 Tools > Ring Bridge > Build Prefab + Demo Scene
   - 自动把 Sprites 设为 Sprite、200 PPU（2x 美术，1 设计 px = 0.01 unit）
   - 自动组装并生成 Prefabs/RingBridge.prefab 与 Scenes/RingBridge_Demo.unity
3. 打开 RingBridge_Demo，Game 视图设为手机横屏比例（如 2532x1170），点 Play

设计尺寸 844x390 设计 px，圆环中心 (422, 215)。

层级结构（RingBridge 预制体）
  Background   夜空背景、5 颗闪烁星芒、星座亮星呼吸、流星、小孔明灯浮动、外圈虚线环自转
  Ring
    Section_lift (抬起)    左上 1/4，粉
    Section_hold (保持)    右上 1/4，金
    Section_lower (放下)   下半环，蓝
      每段：fill 色块 / static 描边与内弧 / track 外圈底轨 / prog 外圈进度（被 ProgressMask 半圆遮罩裁出）
            / done 完成闪光 / HeadPivot 进度头光点 / ink 小人+文字
    Centre       中心圆盘、臀桥人物（13 帧，躺平 -> 抬臀）、保持时的金色光线
  HUD          暂停按钮（Sprite）、分数与 COMBO（UI Text，Fredoka）

动效逻辑（RingBridgeChart，挂在根节点，7 秒循环）
  0.5 – 2.0 秒  抬起：外圈从左往上走进度，色块 / 底轨 / 小人文字随进度从暗到亮
  2.0 – 5.0 秒  保持：外圈从顶部往右走 3 秒进度，中心人物保持抬臀，上方金光闪动
  5.0 – 6.5 秒  放下：外圈从右经过底部走到左边
  每段走满：外圈闪光、小人文字轻弹一下、中心圆盘跳一下，然后 0.4 秒内暗回去
  三段依次进行，同一时间只有一段亮
  中心人物：抬起阶段连续抬臀，保持阶段停在最高，放下阶段连续放平

外圈进度的实现
  prog 是整段完整的发光弧；ProgressMask 是一个半圆 SpriteMask，绕圆环中心旋转到进度前沿，
  只露出“起点到当前进度”的那部分弧。每段的遮罩只作用于自己的 prog（Custom Range 排序区间），互不干扰。

接入正式玩法
  - 把 RingBridgeChart.autoPlay 关掉，按动作识别结果调用：
      SetProgress(段号, 0..1)    段号 0 抬起 / 1 保持 / 2 放下
      Complete(段号, 完成后经过的秒数)   播放完成闪光与变暗
  - 或保留 autoPlay，把 DesignSpace.Now 换成训练计时
  - 各段时长、起止时间在 Inspector 的 sections 数组里改

脚本（Scripts/，命名空间 RingBridge）
  BridgeLayout      由设计稿自动生成的图层中心坐标与层级
  RingBridgeChart   三段进度、色块亮度、完成反馈、中心人物
  Twinkle / MeteorLoop / FloatBob / SpinForever   背景循环动效
  DesignSpace       坐标换算与缓动

字体 Fredoka 为 SIL OFL 开源字体（见 Fonts/OFL.txt）；中文文字已烘焙进图片。
