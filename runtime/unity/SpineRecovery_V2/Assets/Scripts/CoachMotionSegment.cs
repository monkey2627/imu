using System;
using UnityEngine;

/// <summary>
/// 动作包里的一段子动作。对应原来 Animator 的一个 state（如 deadbug_1_up）。
/// 播到末帧后冻结，等待 PoseScorer 评分完成再前进。
/// </summary>
[Serializable]
public class CoachMotionSegment
{
    /// <summary>UI / 日志显示名，如 "抬腿1"</summary>
    public string label;

    /// <summary>源 Animator state 名，仅用于溯源和与旧配置对照</summary>
    public string sourceStateName;

    /// <summary>该段重复次数</summary>
    public int repeatCount = 1;

    /// <summary>打分宽松倍数。1 = 正常；>1 更宽松</summary>
    public float scoreLeniency = 1f;

    /// <summary>该段打分持续时间（秒）</summary>
    public float scoringDuration = 6f;

    /// <summary>是否套用正式动作的评分映射（准备段和过渡段不要开）</summary>
    public bool applyFormalActionScoreMapping;

    /// <summary>Action-agnostic corrective guidance serialized with the segment.</summary>
    public SegmentPoseGuidanceRule poseGuidanceRule;

    /// <summary>语音提示的资源名（可空）。运行时按名查找已打包的 AudioClip。</summary>
    public string voiceClipName;

    /// <summary>该段的逐帧姿势。最后一帧即评分目标姿势。</summary>
    public CoachMotionFrame[] frames = Array.Empty<CoachMotionFrame>();

    public bool IsValid(out string error)
    {
        if (frames == null || frames.Length == 0)
        {
            error = "frames 为空";
            return false;
        }

        for (int i = 0; i < frames.Length; i++)
        {
            if (frames[i].muscles == null ||
                frames[i].muscles.Length != CoachMotionFrame.MuscleCount)
            {
                int actual = frames[i].muscles?.Length ?? 0;
                error = $"frame[{i}] muscles 长度为 {actual}，应为 {CoachMotionFrame.MuscleCount}";
                return false;
            }
        }

        error = null;
        return true;
    }
}
