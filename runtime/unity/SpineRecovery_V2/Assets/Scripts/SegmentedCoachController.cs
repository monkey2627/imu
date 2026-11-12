using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 播放后端类型。Animator = 编辑期导入的 AnimatorController（现有动作）；
/// Keyframe = 运行时下发的 CoachMotionPackage JSON（服务器新动作）。
/// </summary>
public enum CoachMotionBackendType
{
    Animator,
    Keyframe
}

// One small, independent sub-clip of a training action. The full action is a
// sequence of segments. Scoring segments are played to their end (Loop Time
// OFF), then held so the user can match them before advancing. Finish segments
// are presentation-only: their animation plays fully and then advances without
// opening an IMU scoring checkpoint.
[Serializable]
public class ActionSegment
{
    public string label;                 // e.g. "抬右腿" - shown in UI / logs
    public string clipStateName;         // Animator state name of this sub-clip
    [Min(1)] public int repeatCount = 1; // how many times this segment repeats
    public AudioClip voiceClip;          // played when this segment's end pose is reached

    [Tooltip("打分宽松倍数。1 = 正常严格；>1 更宽松（如放下手脚的段设 1.5~2）。放大容许的角度/位置误差。")]
    [Min(0.1f)] public float scoreLeniency = 1f;

    [Tooltip("该段的打分持续时间（秒）。关键帧动作用 5s，准备与回位过渡统一用 2s。")]
    [Min(1f)] public float scoringDuration = 5f;

    [Tooltip("Coach limbs highlighted while this segment plays. Keep preparation, " +
        "return and finish segments at None.")]
    public CoachGuidanceLimb coachGuidanceLimbs = CoachGuidanceLimb.None;

    [Tooltip("Generic corrective guidance for this hold. Auto preserves the " +
        "legacy motion-inference fallback; Explicit uses the two region masks; " +
        "Disabled suppresses corrective cues.")]
    public SegmentPoseGuidanceRule poseGuidanceRule;

    [Tooltip("Compatibility flag used by server-authored motion packages. " +
        "The practice flow derives score mapping from the segment kind.")]
    public bool applyFormalActionScoreMapping;
}

public enum TrainingSegmentKind
{
    Preparation,
    CoreAction,
    ReturnTransition,
    Finish
}

[Flags]
public enum CoachGuidanceLimb
{
    None = 0,
    LeftArm = 1 << 0,
    RightArm = 1 << 1,
    LeftLeg = 1 << 2,
    RightLeg = 1 << 3
}

public struct TrainingSegmentPlanEntry
{
    public int segmentIndex;
    public int setIndex;
    public int repetitionIndex;
    public TrainingSegmentKind kind;
    public bool completesRepetition;
    public bool completesSet;

    public TrainingSegmentPlanEntry(
        int segmentIndex,
        int setIndex,
        int repetitionIndex,
        TrainingSegmentKind kind,
        bool completesRepetition,
        bool completesSet)
    {
        this.segmentIndex = segmentIndex;
        this.setIndex = setIndex;
        this.repetitionIndex = repetitionIndex;
        this.kind = kind;
        this.completesRepetition = completesRepetition;
        this.completesSet = completesSet;
    }
}

// Passed to OnSegmentHold when the coach freezes at a segment-end pose. This is
// the moment a scoring system should sample the IMU wearer's pose and compare it
// to the coach's (now-frozen, exact) target pose.
public struct SegmentHoldInfo
{
    public int segmentIndex;
    public int repeatIndex;   // 0-based repetition of this segment
    public int setIndex;      // 0-based set of the whole sequence
    public string label;
    public float scoreLeniency; // >1 = more lenient scoring for this segment
    public float scoringDuration; // seconds to score this segment
    public TrainingSegmentKind segmentKind;
    public bool applyFormalActionScoreMapping;
    public SegmentPoseGuidanceRule poseGuidanceRule;

    public SegmentHoldInfo(
        int segmentIndex,
        int repeatIndex,
        int setIndex,
        string label,
        float scoreLeniency,
        float scoringDuration,
        TrainingSegmentKind segmentKind = TrainingSegmentKind.Preparation)
        : this(
            segmentIndex,
            repeatIndex,
            setIndex,
            label,
            scoreLeniency,
            scoringDuration,
            segmentKind,
            SegmentPoseGuidanceRule.Auto)
    {
    }

    public SegmentHoldInfo(
        int segmentIndex,
        int repeatIndex,
        int setIndex,
        string label,
        float scoreLeniency,
        float scoringDuration,
        TrainingSegmentKind segmentKind,
        SegmentPoseGuidanceRule poseGuidanceRule)
    {
        this.segmentIndex = segmentIndex;
        this.repeatIndex = repeatIndex;
        this.setIndex = setIndex;
        this.label = label;
        this.scoreLeniency = scoreLeniency;
        this.scoringDuration = scoringDuration;
        this.segmentKind = segmentKind;
        this.poseGuidanceRule = poseGuidanceRule;
        applyFormalActionScoreMapping =
            segmentKind == TrainingSegmentKind.CoreAction;
    }

    public SegmentHoldInfo(
        int segmentIndex,
        int repeatIndex,
        int setIndex,
        string label,
        float scoreLeniency,
        float scoringDuration,
        bool applyFormalActionScoreMapping)
        : this(
            segmentIndex,
            repeatIndex,
            setIndex,
            label,
            scoreLeniency,
            scoringDuration,
            applyFormalActionScoreMapping,
            SegmentPoseGuidanceRule.Auto)
    {
    }

    public SegmentHoldInfo(
        int segmentIndex,
        int repeatIndex,
        int setIndex,
        string label,
        float scoreLeniency,
        float scoringDuration,
        bool applyFormalActionScoreMapping,
        SegmentPoseGuidanceRule poseGuidanceRule)
        : this(
            segmentIndex,
            repeatIndex,
            setIndex,
            label,
            scoreLeniency,
            scoringDuration,
            applyFormalActionScoreMapping
                ? TrainingSegmentKind.CoreAction
                : TrainingSegmentKind.ReturnTransition,
            poseGuidanceRule)
    {
        this.applyFormalActionScoreMapping = applyFormalActionScoreMapping;
    }
}

public struct PracticeSegmentPlaybackInfo
{
    public int segmentIndex;
    public int repetitionIndex;
    public int setIndex;
    public string label;
    public TrainingSegmentKind segmentKind;
    public CoachGuidanceLimb guidanceLimbs;

    public PracticeSegmentPlaybackInfo(
        int segmentIndex,
        int repetitionIndex,
        int setIndex,
        string label,
        TrainingSegmentKind segmentKind,
        CoachGuidanceLimb guidanceLimbs)
    {
        this.segmentIndex = segmentIndex;
        this.repetitionIndex = repetitionIndex;
        this.setIndex = setIndex;
        this.label = label;
        this.segmentKind = segmentKind;
        this.guidanceLimbs = guidanceLimbs;
    }
}

// Practice-mode coach controller: plays an action as a sequence of segment
// sub-clips, holding at each segment-end pose for IMU scoring. Kept separate
// from the gaming/demo CoachActionController so neither interferes with the other.
public class SegmentedCoachController : MonoBehaviour
{
    [SerializeField] private Animator coachAnimator;
    [SerializeField] private string idleStateName = "Idle";

    [Header("Action Segments")]
    [Tooltip("Ordered sub-clips of the action. Each plays to its end, holds, then waits for AdvanceToNext(). Each segment's clip should have Loop Time OFF.")]
    [SerializeField] private List<ActionSegment> segments = new List<ActionSegment>();
    [Tooltip("How many times the whole segment sequence runs (sets).")]
    [Min(1)] [SerializeField] private int setCount = 1;
    [Tooltip("How many complete up/down action pairs the user performs in each set.")]
    [Min(1)] [SerializeField] private int repetitionsPerSet = 4;
    [Tooltip("Legacy per-set boundaries. When enabled, preparation and finish " +
        "segments surround every set. Disable for one continuous training flow.")]
    [SerializeField] private bool repeatBoundarySegmentsPerSet;
    [Tooltip("Fixed-time pose blend used between adjacent training segments. " +
        "This keeps down-to-up set boundaries and the final down-to-finish " +
        "transition visually continuous even when the imported clips do not " +
        "share an identical boundary pose.")]
    [Min(0f)] [SerializeField] private float segmentTransitionBlendSeconds = 0.12f;
    [SerializeField] private bool enableSegmentPausing = true;

    [Header("Dead Bug Preparation Target")]
    [Tooltip("Build the lying preparation target with both arms extended overhead. " +
        "The source clip ends with an unsuitable wide-arm pose, so each arm is " +
        "taken from the matching return-to-preparation endpoint.")]
    [SerializeField] private bool composeDeadBugPreparationArms;
    [SerializeField] private string deadBugPreparationStateName = "deadbug_start";
    [SerializeField] private string deadBugRightArmExtensionStateName = "deadbug_1_down";
    [SerializeField] private string deadBugLeftArmExtensionStateName = "deadbug_2_down";

    [Header("Voice Settings")]
    [SerializeField] private float voiceVolume = 1f;

    // Fired when the coach reaches (and freezes at) a segment-end pose - the scoring checkpoint.
    public event Action<SegmentHoldInfo> OnSegmentHold;
    // Fired when movement begins so guidance is visible during the whole clip.
    public event Action<PracticeSegmentPlaybackInfo> OnSegmentStarted;
    // Fired whenever another full set finishes. Values are completed/total.
    public event Action<int, int> OnSetProgressChanged;
    // Fired after the final segment of the final set is advanced past.
    public event Action OnAllComplete;

    private bool hasStarted;
    private bool isPaused;          // global pause (TogglePause)
    private bool isHolding;         // frozen at a segment-end pose, waiting for advance
    private bool isComplete;        // whole action (all sets) done
    private int currentSetIndex;
    private int currentSegmentIndex;
    private int currentRepeat;
    private int completedSetCount;
    private int completedRepetitionCount;
    private int currentPlanIndex;
    private AudioSource audioSource;
    private readonly List<TrainingSegmentPlanEntry> playbackPlan =
        new List<TrainingSegmentPlanEntry>();
    private readonly Dictionary<HumanBodyBones, Quaternion> deadBugPreparationArmRotations =
        new Dictionary<HumanBodyBones, Quaternion>();

    // 播放后端（Animator 或 Keyframe），由 SetMotionBackend() 注入
    private ICoachMotionBackend motionBackend;
    private CoachMotionBackendType backendType;
    private KeyframeCoachMotionBackend keyframeBackend; // 留引用以便 Dispose

    private static readonly HumanBodyBones[] LeftArmBones =
    {
        HumanBodyBones.LeftShoulder,
        HumanBodyBones.LeftUpperArm,
        HumanBodyBones.LeftLowerArm,
        HumanBodyBones.LeftHand
    };

    private static readonly HumanBodyBones[] RightArmBones =
    {
        HumanBodyBones.RightShoulder,
        HumanBodyBones.RightUpperArm,
        HumanBodyBones.RightLowerArm,
        HumanBodyBones.RightHand
    };

    private void Start()
    {
        EnsureDefaultBackend();
    }

    private void Update()
    {
        if (!hasStarted || isPaused || isComplete || isHolding || !enableSegmentPausing)
        {
            return;
        }

        // 关键帧后端需要每帧驱动播放进度
        motionBackend?.Tick(Time.deltaTime);

        CheckSegmentEnd();
    }

    private void OnDestroy()
    {
        keyframeBackend?.Dispose();
    }

    private void CheckSegmentEnd()
    {
        if (motionBackend == null || !motionBackend.IsReady)
        {
            return;
        }

        if (motionBackend.HasReachedSegmentEnd())
        {
            HoldAtSegmentEnd();
        }
    }

    private void HoldAtSegmentEnd()
    {
        if (motionBackend == null)
        {
            return;
        }

        // Snap exactly to the segment-end pose so the scoring target is precise
        // and identical every run, then freeze.
        motionBackend.SnapToSegmentEnd();
        motionBackend.Freeze();
        isHolding = true;

        PlaySegmentVoice(currentSegmentIndex);

        string label = GetSegmentLabel(currentSegmentIndex);
        if (!IsScoringCheckpoint(CurrentSegmentKind))
        {
            Debug.Log(
                $"SegmentedCoachController: Finished guidance-only segment " +
                $"{currentSegmentIndex + 1}/{segments.Count} '{label}'.");
            AdvanceToNext();
            return;
        }

        float leniency = Mathf.Max(0.1f, segments[currentSegmentIndex].scoreLeniency);
        float duration = Mathf.Max(1f, segments[currentSegmentIndex].scoringDuration);
        Debug.Log($"SegmentedCoachController: Holding segment {currentSegmentIndex + 1}/{segments.Count} (rep {currentRepeat + 1}, set {currentSetIndex + 1}/{TotalSets}) '{label}' leniency={leniency} duration={duration}s");
        OnSegmentHold?.Invoke(new SegmentHoldInfo(
            currentSegmentIndex,
            currentRepeat,
            currentSetIndex,
            label,
            leniency,
            duration,
            CurrentSegmentKind,
            segments[currentSegmentIndex].poseGuidanceRule));
    }

    private void PlaySegment(int index)
    {
        if (motionBackend == null || !motionBackend.IsReady ||
            index < 0 || index >= motionBackend.SegmentCount)
        {
            return;
        }

        isHolding = false;
        motionBackend.PlaySegment(index);

        OnSegmentStarted?.Invoke(new PracticeSegmentPlaybackInfo(
            index,
            currentRepeat,
            currentSetIndex,
            GetSegmentLabel(index),
            CurrentSegmentKind,
            segments[index].coachGuidanceLimbs));
    }

    private void PlaySegmentVoice(int index)
    {
        if (audioSource == null || segments == null || index < 0 || index >= segments.Count)
        {
            return;
        }

        AudioClip clip = segments[index].voiceClip;
        if (clip != null)
        {
            audioSource.PlayOneShot(clip);
        }
    }

    // Advance from the current held pose according to the explicit training
    // plan. One repetition is a complete up/down pair, not an entire imported
    // four-variant animation sequence.
    public void AdvanceToNext()
    {
        if (!isHolding ||
            currentPlanIndex < 0 ||
            currentPlanIndex >= playbackPlan.Count)
        {
            return;
        }

        isHolding = false;
        motionBackend?.Resume();
        TrainingSegmentPlanEntry completedEntry = playbackPlan[currentPlanIndex];
        if (completedEntry.completesRepetition)
        {
            completedRepetitionCount = Mathf.Min(
                completedRepetitionCount + 1,
                TotalRepetitions);
        }

        if (completedEntry.completesSet)
        {
            completedSetCount = Mathf.Min(completedSetCount + 1, TotalSets);
            OnSetProgressChanged?.Invoke(completedSetCount, TotalSets);
        }

        if (currentPlanIndex + 1 < playbackPlan.Count)
        {
            currentPlanIndex++;
            ApplyPlanEntry(playbackPlan[currentPlanIndex]);
            PlaySegment(currentSegmentIndex);
            return;
        }

        // Everything done - keep coach frozen on the final pose.
        isComplete = true;
        motionBackend?.Freeze();

        Debug.Log("SegmentedCoachController: All segments and sets complete.");
        OnAllComplete?.Invoke();
    }

    // Backward-compatible aliases (older UI / wiring).
    public void AdvanceToNextStage() => AdvanceToNext();
    public void AdvanceToNextKeyframe() => AdvanceToNext();

    public void PlayAction1()
    {
        TryPlayAction1(out _);
    }

    public bool TryPlayAction1(out string error)
    {
        EnsureDefaultBackend();
        if (motionBackend == null || !motionBackend.IsReady)
        {
            error = "SegmentedCoachController: motion backend not ready.";
            Debug.LogError(error);
            return false;
        }

        if (segments == null || segments.Count == 0)
        {
            error = "SegmentedCoachController has no segments configured.";
            Debug.LogError(error);
            return false;
        }

        List<TrainingSegmentPlanEntry> configuredPlan = BuildTrainingPlan(
            segments.Count,
            TotalSets,
            RepetitionsPerSet,
            repeatBoundarySegmentsPerSet);
        if (configuredPlan.Count == 0)
        {
            error =
                "SegmentedCoachController needs one preparation segment, one " +
                "finish segment, and an even number of core up/down segments.";
            Debug.LogError(error, this);
            return false;
        }

        // 确保 Animator 启用。CoachActionController 在初始化时会禁用 Animator，
        // 即使它本身被禁用，Animator 也可能仍处于禁用状态，导致动画无法播放。
        if (backendType == CoachMotionBackendType.Animator &&
            coachAnimator != null &&
            !coachAnimator.enabled)
        {
            coachAnimator.enabled = true;
        }

        CacheDeadBugPreparationArms();
        currentSetIndex = 0;
        currentSegmentIndex = 0;
        currentRepeat = 0;
        completedSetCount = 0;
        completedRepetitionCount = 0;
        currentPlanIndex = 0;
        isHolding = false;
        isComplete = false;
        isPaused = false;
        hasStarted = true;
        playbackPlan.Clear();
        playbackPlan.AddRange(configuredPlan);
        ApplyPlanEntry(playbackPlan[0]);

        OnSetProgressChanged?.Invoke(0, TotalSets);
        PlaySegment(currentSegmentIndex);
        error = null;
        return true;
    }

    private void EnsureDefaultBackend()
    {
        if (audioSource == null)
        {
            audioSource = GetComponent<AudioSource>();
            if (audioSource == null)
            {
                audioSource = gameObject.AddComponent<AudioSource>();
            }

            audioSource.playOnAwake = false;
            audioSource.volume = voiceVolume;
        }

        // Public entry points can be invoked before Start (for example by a
        // scene bootstrapper or EditMode validation). Build the serialized
        // Animator backend lazily so the result never depends on call order.
        if (motionBackend == null &&
            coachAnimator != null &&
            segments != null &&
            segments.Count > 0)
        {
            motionBackend = new AnimatorCoachMotionBackend(
                coachAnimator,
                segments,
                idleStateName,
                segmentTransitionBlendSeconds);
            backendType = CoachMotionBackendType.Animator;
        }
    }

    private void ApplyPlanEntry(TrainingSegmentPlanEntry entry)
    {
        currentSegmentIndex = entry.segmentIndex;
        currentSetIndex = entry.setIndex;
        currentRepeat = entry.repetitionIndex;
    }

    public static List<TrainingSegmentPlanEntry> BuildTrainingPlan(
        int totalSegmentCount,
        int sets,
        int repetitionsPerSet,
        bool repeatBoundariesPerSet)
    {
        var plan = new List<TrainingSegmentPlanEntry>();
        int safeSets = Mathf.Max(1, sets);
        int safeRepetitions = Mathf.Max(1, repetitionsPerSet);
        int coreSegmentCount = totalSegmentCount - 2;
        if (totalSegmentCount < 4 ||
            coreSegmentCount < 2 ||
            coreSegmentCount % 2 != 0)
        {
            return plan;
        }

        int variantCount = coreSegmentCount / 2;
        int globalRepetition = 0;
        if (!repeatBoundariesPerSet)
        {
            plan.Add(new TrainingSegmentPlanEntry(
                0,
                0,
                0,
                TrainingSegmentKind.Preparation,
                false,
                false));
        }

        for (int setIndex = 0; setIndex < safeSets; setIndex++)
        {
            if (repeatBoundariesPerSet)
            {
                plan.Add(new TrainingSegmentPlanEntry(
                    0,
                    setIndex,
                    0,
                    TrainingSegmentKind.Preparation,
                    false,
                    false));
            }

            for (int repetitionIndex = 0;
                 repetitionIndex < safeRepetitions;
                 repetitionIndex++)
            {
                int variantIndex = globalRepetition % variantCount;
                int actionSegmentIndex = 1 + variantIndex * 2;
                bool completesSet =
                    !repeatBoundariesPerSet &&
                    repetitionIndex == safeRepetitions - 1;

                plan.Add(new TrainingSegmentPlanEntry(
                    actionSegmentIndex,
                    setIndex,
                    repetitionIndex,
                    TrainingSegmentKind.CoreAction,
                    false,
                    false));
                plan.Add(new TrainingSegmentPlanEntry(
                    actionSegmentIndex + 1,
                    setIndex,
                    repetitionIndex,
                    TrainingSegmentKind.ReturnTransition,
                    true,
                    completesSet));
                globalRepetition++;
            }

            if (repeatBoundariesPerSet)
            {
                plan.Add(new TrainingSegmentPlanEntry(
                    totalSegmentCount - 1,
                    setIndex,
                    safeRepetitions - 1,
                    TrainingSegmentKind.Finish,
                    false,
                    true));
            }
        }

        if (!repeatBoundariesPerSet)
        {
            plan.Add(new TrainingSegmentPlanEntry(
                totalSegmentCount - 1,
                safeSets - 1,
                safeRepetitions - 1,
                TrainingSegmentKind.Finish,
                false,
                false));
        }

        return plan;
    }

    public static bool IsScoringCheckpoint(TrainingSegmentKind segmentKind)
    {
        return segmentKind != TrainingSegmentKind.Finish;
    }

    private void CacheDeadBugPreparationArms()
    {
        deadBugPreparationArmRotations.Clear();

        // 关键帧后端导出时已把修正烧进数据，无需运行时合成
        if (!composeDeadBugPreparationArms ||
            motionBackend == null ||
            !motionBackend.RequiresRuntimeArmComposition)
        {
            return;
        }

        // Animator 后端走原有逻辑：播放两个手臂伸展状态、抓取旋转
        var animBackend = motionBackend as AnimatorCoachMotionBackend;
        if (animBackend == null || coachAnimator == null)
        {
            return;
        }

        if (!TryResolveStateHash(
                deadBugRightArmExtensionStateName,
                out int rightStateHash) ||
            !TryResolveStateHash(
                deadBugLeftArmExtensionStateName,
                out int leftStateHash))
        {
            Debug.LogWarning(
                "SegmentedCoachController cannot build the Dead Bug preparation " +
                "target because one of the arm-extension states is missing.",
                this);
            return;
        }

        coachAnimator.speed = 1f;
        coachAnimator.Play(rightStateHash, 0, 1f);
        coachAnimator.Update(0f);
        CaptureArmRotations(RightArmBones);

        coachAnimator.Play(leftStateHash, 0, 1f);
        coachAnimator.Update(0f);
        CaptureArmRotations(LeftArmBones);

        Debug.Log(
            $"SegmentedCoachController: built overhead preparation target from " +
            $"{deadBugRightArmExtensionStateName} and " +
            $"{deadBugLeftArmExtensionStateName}.",
            this);
    }

    private bool TryResolveStateHash(string stateName, out int stateHash)
    {
        stateHash = Animator.StringToHash(stateName);
        if (coachAnimator.HasState(0, stateHash))
        {
            return true;
        }

        string layerName = coachAnimator.GetLayerName(0);
        stateHash = Animator.StringToHash($"{layerName}.{stateName}");
        return coachAnimator.HasState(0, stateHash);
    }

    private void CaptureArmRotations(IEnumerable<HumanBodyBones> bones)
    {
        foreach (HumanBodyBones bone in bones)
        {
            Transform transform = coachAnimator.GetBoneTransform(bone);
            if (transform != null)
            {
                deadBugPreparationArmRotations[bone] = transform.localRotation;
            }
        }
    }

    private void ApplyDeadBugPreparationArms()
    {
        if (!composeDeadBugPreparationArms ||
            motionBackend == null ||
            !string.Equals(
                motionBackend.GetSegmentSourceName(currentSegmentIndex),
                deadBugPreparationStateName,
                StringComparison.Ordinal) ||
            deadBugPreparationArmRotations.Count == 0)
        {
            return;
        }

        foreach (KeyValuePair<HumanBodyBones, Quaternion> pair in
                 deadBugPreparationArmRotations)
        {
            Transform transform = coachAnimator.GetBoneTransform(pair.Key);
            if (transform != null)
            {
                transform.localRotation = pair.Value;
            }
        }
    }

    /// <summary>
    /// Captures the target used only by anatomical guidance. A known imported
    /// preparation-pose correction is applied and restored inside this call,
    /// so the Animator pose consumed by scoring is never changed.
    /// </summary>
    public Dictionary<HumanBodyBones, Vector3> CaptureGuidanceTargetPose(
        AnatomicalMotionScoringEngine guidanceEngine)
    {
        if (guidanceEngine == null || coachAnimator == null)
        {
            return new Dictionary<HumanBodyBones, Vector3>();
        }

        bool needsCorrection =
            isHolding &&
            composeDeadBugPreparationArms &&
            motionBackend != null &&
            string.Equals(
                motionBackend.GetSegmentSourceName(currentSegmentIndex),
                deadBugPreparationStateName,
                StringComparison.Ordinal) &&
            deadBugPreparationArmRotations.Count > 0;
        if (!needsCorrection)
        {
            return guidanceEngine.CaptureRawPose(coachAnimator);
        }

        var originalRotations =
            new Dictionary<HumanBodyBones, Quaternion>(
                deadBugPreparationArmRotations.Count);
        foreach (HumanBodyBones bone in deadBugPreparationArmRotations.Keys)
        {
            Transform transform = coachAnimator.GetBoneTransform(bone);
            if (transform != null)
            {
                originalRotations[bone] = transform.localRotation;
            }
        }

        ApplyDeadBugPreparationArms();
        Dictionary<HumanBodyBones, Vector3> targetPose =
            guidanceEngine.CaptureRawPose(coachAnimator);

        foreach (KeyValuePair<HumanBodyBones, Quaternion> pair in
                 originalRotations)
        {
            Transform transform = coachAnimator.GetBoneTransform(pair.Key);
            if (transform != null)
            {
                transform.localRotation = pair.Value;
            }
        }

        return targetPose;
    }

    public void SetTrainingVolume(int sets, int repetitions)
    {
        if (hasStarted)
        {
            Debug.LogWarning(
                "SegmentedCoachController cannot change training volume after playback starts.",
                this);
            return;
        }

        setCount = Mathf.Max(1, sets);
        repetitionsPerSet = Mathf.Max(1, repetitions);
    }

    public void SetTotalSequenceCycles(int cycleCount) =>
        SetTrainingVolume(cycleCount, RepetitionsPerSet);

    /// <summary>
    /// 设置动作分段序列（用于统一练习场景的运行时配置）。
    /// 必须在 PlayAction1() 之前调用。
    /// </summary>
    public void SetSegments(ActionSegment[] newSegments)
    {
        if (hasStarted)
        {
            Debug.LogWarning(
                "SegmentedCoachController cannot change segments after playback starts.",
                this);
            return;
        }

        if (newSegments == null || newSegments.Length == 0)
        {
            Debug.LogWarning("SegmentedCoachController.SetSegments: empty segments array", this);
            return;
        }

        segments = new List<ActionSegment>(newSegments);

        // segments 变了，Animator 后端要重建
        if (backendType == CoachMotionBackendType.Animator && coachAnimator != null)
        {
            motionBackend = new AnimatorCoachMotionBackend(
                coachAnimator,
                segments,
                idleStateName,
                segmentTransitionBlendSeconds);
        }

        Debug.Log($"SegmentedCoachController: segments updated to {segments.Count} items", this);
    }

    /// <summary>
    /// 设置关键帧播放后端（服务器下发动作）。segments 从 package 读取。
    /// 必须在 PlayAction1() 之前调用。
    /// </summary>
    public void SetMotionBackend(CoachMotionPackage package, bool lockRootPosition = false)
    {
        if (hasStarted)
        {
            Debug.LogWarning(
                "SegmentedCoachController cannot change backend after playback starts.",
                this);
            return;
        }

        if (package == null)
        {
            Debug.LogError("SegmentedCoachController: motion package is null", this);
            return;
        }

        if (!package.IsValid(out string error))
        {
            Debug.LogError($"SegmentedCoachController: invalid motion package - {error}", this);
            return;
        }

        // 清理旧后端
        keyframeBackend?.Dispose();
        keyframeBackend = null;

        keyframeBackend = new KeyframeCoachMotionBackend(coachAnimator, package, lockRootPosition);
        motionBackend = keyframeBackend;
        backendType = CoachMotionBackendType.Keyframe;

        // 从 package 拉取 segments 参数（复制到 ActionSegment，PoseScorer 等会读）
        segments.Clear();
        foreach (CoachMotionSegment seg in package.segments)
        {
            segments.Add(new ActionSegment
            {
                label = seg.label,
                clipStateName = seg.sourceStateName,
                repeatCount = seg.repeatCount,
                scoreLeniency = seg.scoreLeniency,
                scoringDuration = seg.scoringDuration,
                applyFormalActionScoreMapping = seg.applyFormalActionScoreMapping,
                poseGuidanceRule = seg.poseGuidanceRule,
                voiceClip = null // 语音按名查找，运行时加载
            });
        }

        setCount = Mathf.Max(1, package.setCount);

        Debug.Log(
            $"SegmentedCoachController: switched to Keyframe backend, " +
            $"{segments.Count} segments, {setCount} sets.",
            this);
    }

    public void ReturnIdle()
    {
        motionBackend?.ReturnIdle();

        hasStarted = false;
        isPaused = false;
        isHolding = false;
        isComplete = false;
        currentSetIndex = 0;
        currentSegmentIndex = 0;
        currentRepeat = 0;
        completedSetCount = 0;
        completedRepetitionCount = 0;
        currentPlanIndex = 0;
        playbackPlan.Clear();

        OnSetProgressChanged?.Invoke(0, TotalSets);
    }

    public void TogglePause()
    {
        if (coachAnimator == null || !hasStarted || isComplete)
        {
            return;
        }

        // Never un-freeze while holding at a segment - use AdvanceToNext() for that.
        if (isHolding)
        {
            return;
        }

        isPaused = !isPaused;
        if (isPaused)
        {
            motionBackend.Freeze();
        }
        else
        {
            motionBackend.Resume();
        }
    }

    // --- Read-only progress for UI / scoring ---

    public bool IsHoldingAtSegment => isHolding;
    public bool IsPreparationHold =>
        isHolding && CurrentSegmentKind == TrainingSegmentKind.Preparation;
    public bool UsesDeadBugPreparationPose => composeDeadBugPreparationArms;
    public bool IsDeadBugPreparationHold =>
        IsPreparationHold &&
        composeDeadBugPreparationArms &&
        motionBackend != null &&
        motionBackend.RequiresRuntimeArmComposition &&
        string.Equals(
            motionBackend.GetSegmentSourceName(currentSegmentIndex),
            deadBugPreparationStateName,
            StringComparison.Ordinal);
    public bool IsWaitingForNext => isHolding; // backward-compatible alias
    public bool IsComplete => isComplete;
    public bool HasStarted => hasStarted;
    public Animator CoachAnimator => coachAnimator;

    public bool HasNextAfterCurrentHold
    {
        get
        {
            if (!isHolding || isComplete || segments == null || segments.Count == 0)
            {
                return false;
            }

            return currentPlanIndex + 1 < playbackPlan.Count;
        }
    }

    public int CurrentSetIndex => currentSetIndex;
    public int TotalSets => Mathf.Max(1, setCount);
    public int CompletedSets => completedSetCount;
    public int RepetitionsPerSet => Mathf.Max(1, repetitionsPerSet);
    public int TotalRepetitions
    {
        get
        {
            long total = (long)TotalSets * RepetitionsPerSet;
            return total > int.MaxValue ? int.MaxValue : (int)total;
        }
    }
    public int CompletedRepetitions => completedRepetitionCount;

    public int CurrentSegmentIndex => currentSegmentIndex;
    public int TotalSegments => segments?.Count ?? 0;

    public int CurrentRepeat => currentRepeat;
    public int CurrentSegmentRepeatCount =>
        currentPlanIndex >= 0 &&
        currentPlanIndex < playbackPlan.Count &&
        (playbackPlan[currentPlanIndex].kind == TrainingSegmentKind.CoreAction ||
         playbackPlan[currentPlanIndex].kind == TrainingSegmentKind.ReturnTransition)
            ? RepetitionsPerSet
            : 1;
    public TrainingSegmentKind CurrentSegmentKind =>
        currentPlanIndex >= 0 && currentPlanIndex < playbackPlan.Count
            ? playbackPlan[currentPlanIndex].kind
            : TrainingSegmentKind.Preparation;
    public bool RepeatsBoundarySegmentsPerSet => repeatBoundarySegmentsPerSet;
    public float SegmentTransitionBlendSeconds =>
        Mathf.Max(0f, segmentTransitionBlendSeconds);

    public string CurrentSegmentLabel => GetSegmentLabel(currentSegmentIndex);

    public string GetSegmentLabel(int index)
    {
        if (segments == null || index < 0 || index >= segments.Count)
        {
            return string.Empty;
        }

        return segments[index].label ?? string.Empty;
    }

    public CoachGuidanceLimb GetCoachGuidanceLimbs(int index)
    {
        return segments != null && index >= 0 && index < segments.Count
            ? segments[index].coachGuidanceLimbs
            : CoachGuidanceLimb.None;
    }

}
