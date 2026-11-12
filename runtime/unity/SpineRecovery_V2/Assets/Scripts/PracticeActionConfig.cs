using System;
using UnityEngine;

/// <summary>
/// 单个练习动作的完整配置，包括动画控制器、分段定义、场景引用等。
/// 用于驱动统一的练习场景（testPractice），避免为每个动作复制整个场景。
/// </summary>
[CreateAssetMenu(fileName = "PracticeAction", menuName = "Spine Recovery/Practice Action Config")]
public class PracticeActionConfig : ScriptableObject
{
    [Header("Identity")]
    [Tooltip("动作唯一标识符，如 \"deadbug\", \"birddog\"")]
    public string actionId;

    [Tooltip("显示名称，如 \"Dead Bug 练习\"")]
    public string displayName;

    [Header("Animation")]
    [Tooltip("教练 Animator 使用的控制器（包含所有动作状态）")]
    public RuntimeAnimatorController animatorController;

    [Header("Segments")]
    [Tooltip("动作的分段序列（开始、抬起、放下、结束等）。复用现有的 ActionSegment 类型。")]
    public ActionSegment[] segments = Array.Empty<ActionSegment>();

    [Tooltip("整个动作序列重复的组数")]
    [Min(1)]
    public int setCount = 4;

    [Header("Dead Bug Specific (Optional)")]
    [Tooltip("是否使用 Dead Bug 特殊的准备姿势手臂合成（两臂举过头顶）")]
    public bool composeDeadBugPreparationArms;

    [Tooltip("Dead Bug 准备姿势状态名")]
    public string deadBugPreparationStateName = "deadbug_start";

    [Tooltip("Dead Bug 右臂伸展状态名")]
    public string deadBugRightArmExtensionStateName = "deadbug_1_down";

    [Tooltip("Dead Bug 左臂伸展状态名")]
    public string deadBugLeftArmExtensionStateName = "deadbug_2_down";

    [Header("Scene References")]
    [Tooltip("对应的游戏模式场景名称，如 \"DeadBugGaming\"")]
    public string gamingSceneName;

    [Tooltip("对应的教练选择场景名称，如 \"DeadBug\"")]
    public string coachingSceneName;

    [Tooltip("对应的原练习场景名称（兼容性回退），如 \"DeadBugPractice\"")]
    public string legacyPracticeSceneName;

    [Header("Scoring (Optional)")]
    [Tooltip("默认评分持续时间（秒）。如果为 0，使用 PoseScorer 的默认值。")]
    [Min(0f)]
    public float defaultScoringDuration = 0f;

    /// <summary>
    /// 验证配置完整性
    /// </summary>
    public bool Validate(out string error)
    {
        if (string.IsNullOrWhiteSpace(actionId))
        {
            error = "actionId 不能为空";
            return false;
        }

        if (animatorController == null)
        {
            error = "animatorController 未设置";
            return false;
        }

        if (segments == null || segments.Length == 0)
        {
            error = "segments 数组为空";
            return false;
        }

        if (string.IsNullOrWhiteSpace(gamingSceneName))
        {
            error = "gamingSceneName 未设置";
            return false;
        }

        error = null;
        return true;
    }
}
