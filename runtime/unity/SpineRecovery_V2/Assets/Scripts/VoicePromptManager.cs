using System;
using System.Collections;
using UnityEngine;

/// <summary>
/// Serializes coach-mode voice prompts and scoring beats so state changes cannot
/// restart or overlap the same announcement every frame.
/// </summary>
public class VoicePromptManager : MonoBehaviour
{
    public const float StabilityPromptGapSeconds = 0.15f;
    public const float StabilityNumberGapSeconds = 0.15f;
    public const float StabilityMissingNumberStepSeconds = 0.65f;

    private enum PromptKind
    {
        None,
        PracticeIntroduction,
        StabilityCountdown,
        ScoringBeats,
        ScoreFinished,
        NextAction,
        FollowCoach
    }

    [Header("语音文件路径 (Resources目录相对路径)")]
    [SerializeField] private string voiceFolder = "Audio/VoicePrompts";

    [Header("语音片段")]
    [Tooltip("'请保持住' 语音")]
    [SerializeField] private AudioClip keepStill;
    [Tooltip("'2' 语音")]
    [SerializeField] private AudioClip count2;
    [Tooltip("'1' 语音")]
    [SerializeField] private AudioClip count1;
    [Tooltip("'开始' 语音")]
    [SerializeField] private AudioClip start;
    [Tooltip("'请跟随教练前往下一个动作' 语音")]
    [SerializeField] private AudioClip nextAction;
    [Tooltip("'请根据教练演示调整动作' 语音")]
    [SerializeField] private AudioClip adjustPose;

    [Header("Practice Introduction")]
    [Tooltip("Introduces the coach while only the coach avatar is visible.")]
    [SerializeField] private AudioClip coachIntroduction;
    [Tooltip("Introduces the tracked avatar while only the user's model is visible.")]
    [SerializeField] private AudioClip userModelIntroduction;
    [Tooltip("Plays the ready prompt and 3-2-1 countdown while both avatars are visible.")]
    [SerializeField] private AudioClip readyCountdown;
    [Tooltip("Short visual lead-in before each introduction line begins.")]
    [SerializeField] [Min(0f)] private float introductionVisualLeadSeconds = 0.3f;
    [Tooltip("Pause between the coach and user introduction stages.")]
    [SerializeField] [Min(0f)] private float introductionStageGapSeconds = 0.35f;
    [Tooltip("Keeps a stage visible when its audio clip is unavailable.")]
    [SerializeField] [Min(0f)] private float missingIntroductionClipSeconds = 1.25f;

    [Header("节拍音效")]
    [Tooltip("节拍音 'deng'")]
    [SerializeField] private AudioClip beatDeng;
    [Tooltip("结束音 'ding'")]
    [SerializeField] private AudioClip beatDing;

    [Header("播放设置")]
    [SerializeField] [Range(0f, 1f)] private float voiceVolume = 0.8f;
    [SerializeField] [Range(0f, 1f)] private float beatVolume = 0.6f;
    [Tooltip("同一种提示被取消后，允许再次触发前至少等待的秒数。")]
    [SerializeField] [Min(0f)] private float repeatGuardSeconds = 1.5f;

    private AudioSource voiceSource;
    private AudioSource beatSource;
    private Coroutine currentSequence;
    private PromptKind activePrompt;
    private PromptKind lastStartedPrompt;
    private float lastPromptStartedAt = float.NegativeInfinity;
    private PromptKind lastEndedPrompt;
    private float lastPromptEndedAt = float.NegativeInfinity;

    public bool IsBusy => activePrompt != PromptKind.None;
    public bool IsPlayingPracticeIntroduction =>
        activePrompt == PromptKind.PracticeIntroduction;
    public bool IsPlayingStabilityCountdown => activePrompt == PromptKind.StabilityCountdown;
    public string ActivePromptName => activePrompt.ToString();

    private void Awake()
    {
        voiceSource = CreateAudioSource();
        beatSource = CreateAudioSource();
    }

    /// <summary>
    /// Introduces the two coach-mode avatars, then counts down before formal
    /// playback starts. Visual callbacks are synchronized with the voice stages.
    /// </summary>
    public bool PlayPracticeIntroduction(
        Action showCoach,
        Action showUserModel,
        Action showBoth,
        Action onComplete)
    {
        if (activePrompt != PromptKind.None)
        {
            return false;
        }

        BeginSequence(
            PromptKind.PracticeIntroduction,
            PracticeIntroductionSequence(
                showCoach,
                showUserModel,
                showBoth,
                onComplete));
        return true;
    }

    public void CancelPracticeIntroduction()
    {
        if (activePrompt == PromptKind.PracticeIntroduction)
        {
            StopCurrentSequence(true);
        }
    }

    /// <summary>
    /// Plays "请保持住，2，1，开始" once. Returns false when an identical
    /// prompt is already active or still inside the repeat guard window.
    /// </summary>
    public bool PlayStabilityCountdown(Action onComplete = null)
    {
        // This prompt is entered once by the flow state machine after a full
        // continuous-stability window. A repeat-time guard here could strand
        // PreparingScore without a completion callback.
        if (activePrompt != PromptKind.None)
        {
            return false;
        }

        BeginSequence(PromptKind.StabilityCountdown, StabilityCountdownSequence(onComplete));
        return true;
    }

    public void CancelStabilityCountdown()
    {
        if (activePrompt == PromptKind.StabilityCountdown)
        {
            StopCurrentSequence(true);
        }
    }

    /// <summary>
    /// Plays only the repeating scoring beats. The completion sound and optional
    /// next-action announcement are owned by FinishScoring so they cannot overlap.
    /// </summary>
    public bool PlayScoringBeats(
        float duration,
        Action onBeat = null,
        Action onComplete = null)
    {
        if (activePrompt != PromptKind.None)
        {
            return false;
        }

        BeginSequence(
            PromptKind.ScoringBeats,
            ScoringBeatsSequence(duration, onBeat, onComplete));
        return true;
    }

    /// <summary>
    /// Stops the scoring beats, plays the completion sound once, then optionally
    /// announces the next action. The final checkpoint passes false.
    /// </summary>
    public void FinishScoring(bool announceNextAction)
    {
        if (activePrompt != PromptKind.None &&
            activePrompt != PromptKind.ScoringBeats)
        {
            return;
        }

        BeginSequence(PromptKind.ScoreFinished, ScoreFinishedSequence(announceNextAction));
    }

    public bool PlayNextAction()
    {
        return PlaySinglePrompt(PromptKind.NextAction, nextAction, voiceVolume);
    }

    public bool PlayAdjustPose()
    {
        return PlayFollowCoach();
    }

    /// <summary>
    /// Announces the user-following phase once. Per-frame repeats are rejected
    /// by the active and repeat guards.
    /// </summary>
    public bool PlayFollowCoach()
    {
        return PlaySinglePrompt(PromptKind.FollowCoach, adjustPose, voiceVolume);
    }

    public void StopAll()
    {
        StopCurrentSequence(true);
    }

    private AudioSource CreateAudioSource()
    {
        AudioSource source = gameObject.AddComponent<AudioSource>();
        source.playOnAwake = false;
        source.spatialBlend = 0f;
        return source;
    }

    private bool PlaySinglePrompt(PromptKind kind, AudioClip clip, float volume)
    {
        if (clip == null || !CanStart(kind))
        {
            return false;
        }

        BeginSequence(kind, SinglePromptSequence(clip, volume));
        return true;
    }

    private bool CanStart(PromptKind kind)
    {
        // Ordinary prompts never interrupt one another and are never queued.
        // If their event has already passed when audio becomes idle, the stale
        // prompt is simply omitted instead of playing late.
        if (activePrompt != PromptKind.None)
        {
            return false;
        }

        bool startGuardPassed =
            lastStartedPrompt != kind ||
            Time.unscaledTime - lastPromptStartedAt >= repeatGuardSeconds;
        bool endGuardPassed =
            lastEndedPrompt != kind ||
            Time.unscaledTime - lastPromptEndedAt >= repeatGuardSeconds;
        return startGuardPassed && endGuardPassed;
    }

    private void BeginSequence(PromptKind kind, IEnumerator sequence)
    {
        StopCurrentSequence(true);
        activePrompt = kind;
        lastStartedPrompt = kind;
        lastPromptStartedAt = Time.unscaledTime;
        Debug.Log($"[CoachVoice] start={kind}", this);
        currentSequence = StartCoroutine(sequence);
    }

    private void StopCurrentSequence(bool stopAudio)
    {
        PromptKind stoppedPrompt = activePrompt;
        if (currentSequence != null)
        {
            StopCoroutine(currentSequence);
            currentSequence = null;
        }

        activePrompt = PromptKind.None;

        if (!stopAudio)
        {
            return;
        }

        if (voiceSource != null)
        {
            voiceSource.Stop();
            voiceSource.clip = null;
        }

        if (beatSource != null)
        {
            beatSource.Stop();
            beatSource.clip = null;
        }

        if (stoppedPrompt != PromptKind.None)
        {
            lastEndedPrompt = stoppedPrompt;
            lastPromptEndedAt = Time.unscaledTime;
            Debug.Log($"[CoachVoice] stop={stoppedPrompt}", this);
        }
    }

    private IEnumerator StabilityCountdownSequence(Action onComplete)
    {
        if (keepStill != null)
        {
            yield return PlayAndWait(voiceSource, keepStill, voiceVolume);
            yield return new WaitForSeconds(StabilityPromptGapSeconds);
        }

        if (count2 != null)
        {
            yield return PlayAndWait(voiceSource, count2, voiceVolume);
            yield return new WaitForSeconds(StabilityNumberGapSeconds);
        }
        else
        {
            yield return new WaitForSeconds(
                StabilityMissingNumberStepSeconds);
        }

        if (count1 != null)
        {
            yield return PlayAndWait(voiceSource, count1, voiceVolume);
            yield return new WaitForSeconds(StabilityNumberGapSeconds);
        }
        else
        {
            yield return new WaitForSeconds(
                StabilityMissingNumberStepSeconds);
        }

        if (start != null)
        {
            yield return PlayAndWait(voiceSource, start, voiceVolume);
        }

        CompleteSequence(onComplete);
    }

    private IEnumerator PracticeIntroductionSequence(
        Action showCoach,
        Action showUserModel,
        Action showBoth,
        Action onComplete)
    {
        showCoach?.Invoke();
        yield return WaitRealtime(introductionVisualLeadSeconds);
        yield return PlayIntroductionClipOrFallback(coachIntroduction);
        yield return WaitRealtime(introductionStageGapSeconds);

        showUserModel?.Invoke();
        yield return WaitRealtime(introductionVisualLeadSeconds);
        yield return PlayIntroductionClipOrFallback(userModelIntroduction);
        yield return WaitRealtime(introductionStageGapSeconds);

        showBoth?.Invoke();
        yield return WaitRealtime(introductionVisualLeadSeconds);
        yield return PlayIntroductionClipOrFallback(readyCountdown);

        CompleteSequence(onComplete);
    }

    private IEnumerator PlayIntroductionClipOrFallback(AudioClip clip)
    {
        if (voiceSource == null || clip == null)
        {
            yield return WaitRealtime(missingIntroductionClipSeconds);
            yield break;
        }

        PlayClip(voiceSource, clip, voiceVolume);
        yield return WaitRealtime(clip.length);
    }

    private static IEnumerator WaitRealtime(float seconds)
    {
        float duration = Mathf.Max(0f, seconds);
        if (duration > 0f)
        {
            yield return new WaitForSecondsRealtime(duration);
        }
    }

    private IEnumerator ScoringBeatsSequence(
        float duration,
        Action onBeat,
        Action onComplete)
    {
        float safeDuration = Mathf.Max(0f, duration);
        int beatCount = Mathf.Max(1, Mathf.RoundToInt(safeDuration));
        float interval = beatCount > 0 ? safeDuration / beatCount : safeDuration;

        for (int i = 0; i < beatCount; i++)
        {
            if (beatDeng != null)
            {
                PlayClip(beatSource, beatDeng, beatVolume);
            }
            onBeat?.Invoke();
            yield return new WaitForSeconds(interval);
        }

        CompleteSequence(onComplete);
    }

    private IEnumerator ScoreFinishedSequence(bool announceNextAction)
    {
        if (beatDing != null)
        {
            yield return PlayAndWait(beatSource, beatDing, beatVolume);
        }

        if (announceNextAction && nextAction != null)
        {
            lastStartedPrompt = PromptKind.NextAction;
            lastPromptStartedAt = Time.unscaledTime;
            yield return PlayAndWait(voiceSource, nextAction, voiceVolume);
        }

        CompleteSequence(null);
    }

    private IEnumerator SinglePromptSequence(AudioClip clip, float volume)
    {
        yield return PlayAndWait(voiceSource, clip, volume);
        CompleteSequence(null);
    }

    private IEnumerator PlayAndWait(AudioSource source, AudioClip clip, float volume)
    {
        if (source == null || clip == null)
        {
            yield break;
        }

        PlayClip(source, clip, volume);
        yield return new WaitForSeconds(clip.length);
    }

    private static void PlayClip(AudioSource source, AudioClip clip, float volume)
    {
        source.Stop();
        source.clip = clip;
        source.volume = volume;
        source.Play();
    }

    private void CompleteSequence(Action onComplete)
    {
        PromptKind completedPrompt = activePrompt;
        currentSequence = null;
        activePrompt = PromptKind.None;
        lastEndedPrompt = completedPrompt;
        lastPromptEndedAt = Time.unscaledTime;
        Debug.Log($"[CoachVoice] complete={completedPrompt}", this);
        onComplete?.Invoke();
    }

    private void OnDisable()
    {
        StopAll();
    }
}
