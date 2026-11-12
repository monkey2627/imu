using UnityEngine;

/// <summary>
/// 教练动作的播放后端。把 SegmentedCoachController 对 Animator 的直接依赖
/// 收敛到这个接口，从而支持两种驱动源：
///   - AnimatorCoachMotionBackend：编辑期导入的 AnimatorController（现有动作）
///   - KeyframeCoachMotionBackend：运行时下发的 CoachMotionPackage（服务器动作）
/// </summary>
public interface ICoachMotionBackend
{
    /// <summary>后端是否可用（缺 Animator 或数据时为 false）</summary>
    bool IsReady { get; }

    /// <summary>分段总数</summary>
    int SegmentCount { get; }

    /// <summary>
    /// 是否需要运行时合成 Dead Bug 准备姿势的双臂。
    /// Animator 后端为 true（源 clip 末帧姿势不合用，需要现场拼）；
    /// 关键帧后端为 false（导出时已把修正烧进数据）。
    /// </summary>
    bool RequiresRuntimeArmComposition { get; }

    /// <summary>从头播放第 index 段</summary>
    void PlaySegment(int index);

    /// <summary>每帧推进播放。Animator 后端为空实现（Animator 自己走时间）。</summary>
    void Tick(float deltaTime);

    /// <summary>当前段是否已播到末帧</summary>
    bool HasReachedSegmentEnd();

    /// <summary>精确吸附到当前段末帧姿势，使评分目标每次完全一致</summary>
    void SnapToSegmentEnd();

    /// <summary>冻结播放（保持当前姿势）</summary>
    void Freeze();

    /// <summary>恢复播放</summary>
    void Resume();

    /// <summary>回到 Idle 状态</summary>
    void ReturnIdle();

    /// <summary>取得段名，用于日志和与旧配置对照</summary>
    string GetSegmentSourceName(int index);
}
