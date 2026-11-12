using System;
using UnityEngine;

/// <summary>
/// 关键帧后端：由运行时下发的 CoachMotionPackage 驱动，不依赖 AnimatorController。
/// 用 HumanPoseHandler 写 muscle space 姿势，因此同一份数据可驱动男女不同 avatar。
/// 与 MotionPlaybackController 同一思路：禁用 Animator，直接摆骨架。
/// </summary>
public class KeyframeCoachMotionBackend : ICoachMotionBackend, IDisposable
{
    private readonly Animator animator;
    private readonly CoachMotionPackage package;
    private readonly HumanPoseHandler poseHandler;
    private readonly bool lockRootPosition;
    private readonly Vector3 initialBodyPosition;

    private HumanPose pose;
    private int activeSegmentIndex = -1;
    private float cursorTime;
    private bool frozen;
    private bool reachedEnd;

    public KeyframeCoachMotionBackend(
        Animator animator,
        CoachMotionPackage package,
        bool lockRootPosition = false)
    {
        this.animator = animator;
        this.package = package;
        this.lockRootPosition = lockRootPosition;

        if (animator == null || package == null)
        {
            return;
        }

        if (animator.avatar == null || !animator.avatar.isHuman)
        {
            Debug.LogError(
                "KeyframeCoachMotionBackend 需要 Humanoid avatar，无法驱动该模型。",
                animator);
            return;
        }

        // Animator 会在每帧覆盖骨骼姿势，必须禁用后自行摆骨架。
        animator.enabled = false;

        poseHandler = new HumanPoseHandler(animator.avatar, animator.transform);
        pose = new HumanPose();
        poseHandler.GetHumanPose(ref pose);
        initialBodyPosition = pose.bodyPosition;
    }

    public bool IsReady =>
        poseHandler != null &&
        package != null &&
        package.segments != null &&
        package.segments.Length > 0;

    public int SegmentCount => package?.segments?.Length ?? 0;

    // 导出时已把 Dead Bug 准备姿势的双臂修正烧进数据，运行时无需再合成。
    public bool RequiresRuntimeArmComposition => false;

    public void PlaySegment(int index)
    {
        if (!IsReady || index < 0 || index >= package.segments.Length)
        {
            return;
        }

        activeSegmentIndex = index;
        cursorTime = 0f;
        frozen = false;
        reachedEnd = false;
        ApplyPoseAt(cursorTime);
    }

    public void Tick(float deltaTime)
    {
        if (!IsReady || frozen || reachedEnd || activeSegmentIndex < 0)
        {
            return;
        }

        cursorTime += deltaTime;

        float duration = GetActiveSegmentDuration();
        if (cursorTime >= duration)
        {
            cursorTime = duration;
            reachedEnd = true;
        }

        ApplyPoseAt(cursorTime);
    }

    public bool HasReachedSegmentEnd() => reachedEnd;

    public void SnapToSegmentEnd()
    {
        if (!IsReady || activeSegmentIndex < 0)
        {
            return;
        }

        cursorTime = GetActiveSegmentDuration();
        reachedEnd = true;
        ApplyPoseAt(cursorTime);
    }

    public void Freeze() => frozen = true;

    public void Resume() => frozen = false;

    public void ReturnIdle()
    {
        activeSegmentIndex = -1;
        cursorTime = 0f;
        frozen = false;
        reachedEnd = false;

        // 关键帧包不含 Idle 段；回到首段首帧作为静止姿势。
        if (IsReady)
        {
            activeSegmentIndex = 0;
            ApplyPoseAt(0f);
            activeSegmentIndex = -1;
        }
    }

    public string GetSegmentSourceName(int index)
    {
        if (package?.segments == null || index < 0 || index >= package.segments.Length)
        {
            return null;
        }

        return package.segments[index].sourceStateName;
    }

    private float GetActiveSegmentDuration()
    {
        CoachMotionSegment segment = package.segments[activeSegmentIndex];
        CoachMotionFrame[] frames = segment.frames;
        return frames.Length > 0 ? frames[frames.Length - 1].time : 0f;
    }

    /// <summary>按时间在关键帧间插值，写入 muscle space 姿势</summary>
    private void ApplyPoseAt(float time)
    {
        if (activeSegmentIndex < 0)
        {
            return;
        }

        CoachMotionFrame[] frames = package.segments[activeSegmentIndex].frames;
        if (frames == null || frames.Length == 0)
        {
            return;
        }

        if (frames.Length == 1)
        {
            WritePose(frames[0]);
            return;
        }

        // 定位到 time 所在的帧区间
        int upper = 1;
        while (upper < frames.Length && frames[upper].time < time)
        {
            upper++;
        }

        if (upper >= frames.Length)
        {
            WritePose(frames[frames.Length - 1]);
            return;
        }

        CoachMotionFrame a = frames[upper - 1];
        CoachMotionFrame b = frames[upper];
        float span = b.time - a.time;
        float t = span > Mathf.Epsilon ? Mathf.Clamp01((time - a.time) / span) : 0f;

        WriteInterpolatedPose(a, b, t);
    }

    private void WritePose(CoachMotionFrame frame)
    {
        frame.ApplyTo(ref pose);
        FinalizeAndWrite();
    }

    private void WriteInterpolatedPose(CoachMotionFrame a, CoachMotionFrame b, float t)
    {
        if (pose.muscles == null || pose.muscles.Length != CoachMotionFrame.MuscleCount)
        {
            pose.muscles = new float[CoachMotionFrame.MuscleCount];
        }

        pose.bodyPosition = Vector3.Lerp(a.bodyPosition, b.bodyPosition, t);
        pose.bodyRotation = Quaternion.Slerp(a.bodyRotation, b.bodyRotation, t);

        for (int i = 0; i < CoachMotionFrame.MuscleCount; i++)
        {
            pose.muscles[i] = Mathf.Lerp(a.muscles[i], b.muscles[i], t);
        }

        FinalizeAndWrite();
    }

    private void FinalizeAndWrite()
    {
        if (lockRootPosition)
        {
            pose.bodyPosition = initialBodyPosition;
        }

        poseHandler.SetHumanPose(ref pose);
    }

    public void Dispose()
    {
        poseHandler?.Dispose();
    }
}
