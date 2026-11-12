using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public sealed class JsonPracticeCoachController : MonoBehaviour
{
    public event Action<SegmentHoldInfo> OnSegmentHold;
    public event Action OnAllComplete;

    private CoachMotionPackage package;
    private KeyframeCoachMotionBackend backend;
    private Coroutine playbackRoutine;
    private bool waitingForAdvance;
    private bool complete;
    private int setIndex;
    private int segmentIndex;
    private int repeatIndex;
    private MotionRecorder motionRecorder;
    private Animator coachAnimator;
    private VoicePromptManager voicePromptManager;
    private DeadBugStartMenuController presentationController;
    private bool introductionStarted;

    // 介绍阶段的 Renderer 状态缓存
    private Renderer[] introductionCoachRenderers = System.Array.Empty<Renderer>();
    private Renderer[] introductionUserRenderers = System.Array.Empty<Renderer>();
    private readonly Dictionary<Renderer, bool> introductionRendererStates = new Dictionary<Renderer, bool>();

    public int TotalSets => Mathf.Max(1, package?.setCount ?? 1);
    public int TotalSegments => package?.segments?.Length ?? 0;
    public bool IsComplete => complete;
    public bool IsHoldingAtSegment => waitingForAdvance;

    public void Initialize(
        Animator animator,
        CoachMotionPackage motionPackage,
        MotionRecorder recorder = null)
    {
        coachAnimator = animator;
        package = motionPackage;
        motionRecorder = recorder;

        // 不要立即创建 backend，等到播放时再创建
        // 这样教练模型在介绍阶段保持 T-pose（Animator 驱动的 Idle 状态）
        backend?.Dispose();
        backend = null;

        waitingForAdvance = false;
        complete = false;
        setIndex = 0;
        segmentIndex = 0;
        repeatIndex = 0;
        introductionStarted = false;

        // 查找 VoicePromptManager 和 PresentationController
        voicePromptManager = FindObjectOfType<VoicePromptManager>(true);
        presentationController = FindObjectOfType<DeadBugStartMenuController>(true);

        Debug.Log($"[JsonPracticeCoachController] 已初始化，package='{package.actionId}'，推迟创建 backend 直到播放时");
    }

    /// <summary>
    /// 创建并初始化 backend（在播放前调用）
    /// </summary>
    private void EnsureBackendReady()
    {
        if (backend != null)
        {
            return;
        }

        if (coachAnimator == null || package == null)
        {
            Debug.LogError("[JsonPracticeCoachController] 无法创建 backend：animator 或 package 为空");
            return;
        }

        backend = new KeyframeCoachMotionBackend(coachAnimator, package, true);
        Debug.Log("[JsonPracticeCoachController] Backend 已创建并准备就绪");
    }

    public void PlayAction()
    {
        if (backend == null || !backend.IsReady)
        {
            Debug.LogError("JsonPracticeCoachController needs a valid motion package.", this);
            return;
        }

        if (playbackRoutine != null)
        {
            StopCoroutine(playbackRoutine);
        }

        waitingForAdvance = false;
        complete = false;
        setIndex = 0;
        segmentIndex = 0;
        repeatIndex = 0;
        introductionStarted = false;

        // 通知 presentation controller 开始播放前的准备
        presentationController?.BeginPracticePlaybackPresentation();

        playbackRoutine = StartCoroutine(PlayWithIntroduction());
    }

    /// <summary>
    /// 直接播放动作，不播放介绍（由 PracticeSessionController 在介绍后调用）
    /// </summary>
    public void PlayActionWithoutIntroduction()
    {
        // 现在创建 backend，接管 Animator
        EnsureBackendReady();

        if (backend == null || !backend.IsReady)
        {
            Debug.LogError("JsonPracticeCoachController needs a valid motion package.", this);
            return;
        }

        if (playbackRoutine != null)
        {
            StopCoroutine(playbackRoutine);
        }

        waitingForAdvance = false;
        complete = false;
        setIndex = 0;
        segmentIndex = 0;
        repeatIndex = 0;

        Debug.Log("[JsonPracticeCoachController] 倒数结束，开始创建 backend 并播放 JSON 驱动的动作");

        // 直接开始播放动作
        playbackRoutine = StartCoroutine(PlayPackage());
    }

    public void AdvanceToNext()
    {
        if (waitingForAdvance)
        {
            waitingForAdvance = false;
        }
    }

    public void TogglePause()
    {
        if (Time.timeScale > 0f)
        {
            Time.timeScale = 0f;
            backend?.Freeze();
        }
        else
        {
            Time.timeScale = 1f;
            backend?.Resume();
        }
    }

    public void StopPractice()
    {
        if (playbackRoutine != null)
        {
            StopCoroutine(playbackRoutine);
            playbackRoutine = null;
        }
        waitingForAdvance = false;
        complete = false;
        CancelPracticeIntroduction();
        motionRecorder?.CancelRecording();
        backend?.ReturnIdle();
        Time.timeScale = 1f;
    }

    public void ReturnIdle() => StopPractice();

    private IEnumerator PlayWithIntroduction()
    {
        // 保存介绍阶段的 Renderer 状态
        CaptureIntroductionRendererStates();

        // 确保教练模型处于 Idle 状态（T-pose）
        backend.ReturnIdle();

        // 开始介绍阶段
        if (voicePromptManager != null)
        {
            voicePromptManager.StopAll();
            bool introductionPlayed = voicePromptManager.PlayPracticeIntroduction(
                ShowCoachIntroductionStage,
                ShowUserIntroductionStage,
                ShowBothIntroductionStage,
                BeginFormalPlayback);

            if (introductionPlayed)
            {
                Debug.Log("[JsonPracticeCoachController] 开始播放介绍和倒数");
                // 等待介绍完成（BeginFormalPlayback 会被调用）
                yield break;
            }
        }

        // 如果没有语音管理器或介绍失败，直接开始播放
        Debug.LogWarning("[JsonPracticeCoachController] 未找到 VoicePromptManager，跳过介绍直接开始");
        ShowBothIntroductionStage();
        BeginFormalPlayback();
    }

    private void CaptureIntroductionRendererStates()
    {
        RestoreIntroductionRendererStates();

        Animator coachAnim = motionRecorder != null ? motionRecorder.CoachAnimator : coachAnimator;
        Animator userAnim = motionRecorder != null ? motionRecorder.UserAnimator : null;

        introductionCoachRenderers = GetAvatarRenderers(coachAnim);
        introductionUserRenderers = GetAvatarRenderers(userAnim);

        RememberRendererStates(introductionCoachRenderers);
        RememberRendererStates(introductionUserRenderers);
    }

    private static Renderer[] GetAvatarRenderers(Animator animator)
    {
        return animator != null
            ? animator.GetComponentsInChildren<Renderer>(true)
            : System.Array.Empty<Renderer>();
    }

    private void RememberRendererStates(Renderer[] renderers)
    {
        foreach (Renderer avatarRenderer in renderers)
        {
            if (avatarRenderer != null &&
                !introductionRendererStates.ContainsKey(avatarRenderer))
            {
                introductionRendererStates.Add(
                    avatarRenderer,
                    avatarRenderer.enabled);
            }
        }
    }

    private void ShowCoachIntroductionStage()
    {
        SetIntroductionAvatarVisible(introductionCoachRenderers, true);
        SetIntroductionAvatarVisible(introductionUserRenderers, false);
    }

    private void ShowUserIntroductionStage()
    {
        SetIntroductionAvatarVisible(introductionCoachRenderers, false);
        SetIntroductionAvatarVisible(introductionUserRenderers, true);
    }

    private void ShowBothIntroductionStage()
    {
        SetIntroductionAvatarVisible(introductionCoachRenderers, true);
        SetIntroductionAvatarVisible(introductionUserRenderers, true);
    }

    private void SetIntroductionAvatarVisible(Renderer[] renderers, bool visible)
    {
        foreach (Renderer avatarRenderer in renderers)
        {
            if (avatarRenderer == null ||
                !introductionRendererStates.TryGetValue(avatarRenderer, out bool wasEnabled))
            {
                continue;
            }

            avatarRenderer.enabled = visible && wasEnabled;
        }
    }

    private void RestoreIntroductionRendererStates()
    {
        foreach (KeyValuePair<Renderer, bool> rendererState in introductionRendererStates)
        {
            if (rendererState.Key != null)
            {
                rendererState.Key.enabled = rendererState.Value;
            }
        }

        introductionRendererStates.Clear();
        introductionCoachRenderers = System.Array.Empty<Renderer>();
        introductionUserRenderers = System.Array.Empty<Renderer>();
    }

    private void CancelPracticeIntroduction()
    {
        voicePromptManager?.CancelPracticeIntroduction();
        RestoreIntroductionRendererStates();
    }

    private void BeginFormalPlayback()
    {
        if (introductionStarted)
        {
            return;
        }

        introductionStarted = true;

        // 恢复原始 Renderer 状态
        RestoreIntroductionRendererStates();

        Debug.Log("[JsonPracticeCoachController] 介绍完成，开始正式训练");

        // 开始 SpineFlow 训练会话
        SpineFlowTrainingSession.BeginCoachTraining();

        // 开始录制
        if (motionRecorder != null &&
            !motionRecorder.TryBeginRecording(out string recordingError))
        {
            Debug.LogError(
                $"JSON practice replay could not start: {recordingError}",
                this);
            return;
        }

        // 开始播放动作
        if (playbackRoutine != null)
        {
            StopCoroutine(playbackRoutine);
        }
        playbackRoutine = StartCoroutine(PlayPackage());
    }

    private IEnumerator PlayPackage()
    {
        if (backend == null || !backend.IsReady)
        {
            Debug.LogError("JsonPracticeCoachController needs a valid motion package.", this);
            yield break;
        }

        for (setIndex = 0; setIndex < TotalSets; setIndex++)
        {
            for (segmentIndex = 0; segmentIndex < package.segments.Length; segmentIndex++)
            {
                CoachMotionSegment segment = package.segments[segmentIndex];
                int repeatCount = Mathf.Max(1, segment.repeatCount);
                for (repeatIndex = 0; repeatIndex < repeatCount; repeatIndex++)
                {
                    backend.PlaySegment(segmentIndex);
                    while (!backend.HasReachedSegmentEnd())
                    {
                        backend.Tick(Time.deltaTime);
                        yield return null;
                    }

                    backend.SnapToSegmentEnd();
                    backend.Freeze();
                    waitingForAdvance = true;
                    OnSegmentHold?.Invoke(new SegmentHoldInfo(
                        segmentIndex,
                        repeatIndex,
                        setIndex,
                        string.IsNullOrEmpty(segment.label) ? segment.sourceStateName : segment.label,
                        Mathf.Max(0.1f, segment.scoreLeniency),
                        Mathf.Max(1f, segment.scoringDuration),
                        segment.applyFormalActionScoreMapping,
                        segment.poseGuidanceRule));
                    while (waitingForAdvance)
                    {
                        yield return null;
                    }
                    backend.Resume();
                }
            }
        }

        complete = true;
        playbackRoutine = null;
        motionRecorder?.RequestFinish();
        OnAllComplete?.Invoke();
    }

    private void OnDestroy()
    {
        Time.timeScale = 1f;
        backend?.Dispose();
    }
}
