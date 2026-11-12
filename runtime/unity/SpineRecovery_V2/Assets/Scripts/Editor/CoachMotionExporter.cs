using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

/// <summary>
/// 把 FBX 的 Humanoid 子片段按分段配置采样导出成 CoachMotionPackage JSON。
/// 这是"服务器发布"链路的上游：在 Unity 里录制/切分好，导出后上传。
/// </summary>
public static class CoachMotionExporter
{
    /// <summary>
    /// 采样单个 AnimationClip 的全长，返回逐帧 HumanPose。
    /// sampleTarget 必须是带 Humanoid Avatar 的实例。
    /// </summary>
    public static CoachMotionFrame[] SampleClip(
        AnimationClip clip,
        GameObject sampleTarget,
        Animator sampleAnimator,
        float fps)
    {
        if (clip == null || sampleAnimator == null || sampleAnimator.avatar == null)
        {
            return new CoachMotionFrame[0];
        }

        var handler = new HumanPoseHandler(sampleAnimator.avatar, sampleAnimator.transform);
        var pose = new HumanPose();
        var frames = new List<CoachMotionFrame>();

        float step = 1f / Mathf.Max(1f, fps);
        // 末帧必须精确落在 clip.length 上：它是评分目标姿势。
        int stepCount = Mathf.Max(1, Mathf.CeilToInt(clip.length / step));

        for (int i = 0; i <= stepCount; i++)
        {
            float time = Mathf.Min(i * step, clip.length);
            clip.SampleAnimation(sampleTarget, time);
            handler.GetHumanPose(ref pose);
            frames.Add(CoachMotionFrame.FromHumanPose(time, pose));

            if (time >= clip.length)
            {
                break;
            }
        }

        handler.Dispose();
        return frames.ToArray();
    }
}
