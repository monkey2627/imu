using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 现有行为的后端实现：由编辑期导入的 AnimatorController 驱动。
/// 逐字保留原 SegmentedCoachController 的 Animator 调用语义，
/// 以确保已有的 deadbug / birddog 动作行为不变。
/// </summary>
public class AnimatorCoachMotionBackend : ICoachMotionBackend
{
    private readonly Animator animator;
    private readonly List<ActionSegment> segments;
    private readonly string idleStateName;
    private readonly float transitionBlendSeconds;

    private string activeStateName;
    private bool hasPlayedSegment;

    public AnimatorCoachMotionBackend(
        Animator animator,
        List<ActionSegment> segments,
        string idleStateName,
        float transitionBlendSeconds = 0f)
    {
        this.animator = animator;
        this.segments = segments;
        this.idleStateName = idleStateName;
        this.transitionBlendSeconds = Mathf.Max(0f, transitionBlendSeconds);
    }

    public bool IsReady => animator != null && segments != null && segments.Count > 0;

    public int SegmentCount => segments?.Count ?? 0;

    public bool RequiresRuntimeArmComposition => true;

    /// <summary>当前正在播放的 state 名。Dead Bug 手臂合成需要据此判断。</summary>
    public string ActiveStateName => activeStateName;

    public void PlaySegment(int index)
    {
        if (!IsReady || index < 0 || index >= segments.Count)
        {
            return;
        }

        activeStateName = segments[index].clipStateName;
        animator.speed = 1f;

        if (!string.IsNullOrEmpty(activeStateName))
        {
            if (hasPlayedSegment && transitionBlendSeconds > 0f)
            {
                animator.CrossFadeInFixedTime(
                    activeStateName,
                    transitionBlendSeconds,
                    0,
                    0f);
            }
            else
            {
                animator.Play(activeStateName, 0, 0f);
            }

            hasPlayedSegment = true;
        }
        else
        {
            Debug.LogWarning(
                $"AnimatorCoachMotionBackend: segment {index} " +
                $"'{segments[index].label}' has no clipStateName.");
        }
    }

    public void Tick(float deltaTime)
    {
        // Animator 自己推进时间，无需外部驱动。
    }

    public bool HasReachedSegmentEnd()
    {
        if (!IsReady)
        {
            return false;
        }

        AnimatorStateInfo state = animator.GetCurrentAnimatorStateInfo(0);

        // 调用 Play() 的当帧，Animator 可能仍报告上一个 state。
        // 等到目标子片段真正成为当前 state 再判断。
        if (!string.IsNullOrEmpty(activeStateName) && !state.IsName(activeStateName))
        {
            return false;
        }

        return state.normalizedTime >= 1f;
    }

    public void SnapToSegmentEnd()
    {
        if (!IsReady || string.IsNullOrEmpty(activeStateName))
        {
            return;
        }

        animator.Play(activeStateName, 0, 1f);
        animator.Update(0f);
    }

    public void Freeze()
    {
        if (animator != null)
        {
            animator.speed = 0f;
        }
    }

    public void Resume()
    {
        if (animator != null)
        {
            animator.speed = 1f;
        }
    }

    public void ReturnIdle()
    {
        if (animator == null)
        {
            return;
        }

        animator.speed = 1f;
        activeStateName = null;
        hasPlayedSegment = false;

        if (!string.IsNullOrEmpty(idleStateName))
        {
            animator.Play(idleStateName, 0, 0f);
        }
    }

    public string GetSegmentSourceName(int index)
    {
        if (segments == null || index < 0 || index >= segments.Count)
        {
            return null;
        }

        return segments[index].clipStateName;
    }
}
