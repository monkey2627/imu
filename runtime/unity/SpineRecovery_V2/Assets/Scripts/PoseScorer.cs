using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

// Result of scoring one segment hold.
public struct SegmentScore
{
    public int segmentIndex;
    public int repeatIndex;
    public int setIndex;
    public TrainingSegmentKind segmentKind;
    public string label;
    // Compatibility aliases for persisted reports. Practice scoring currently
    // keeps them equal to averageScore/bestScore so UI and statistics agree.
    public float rawAverageScore;
    public float rawBestScore;
    public float averageScore;  // 0-100 averaged over the scoring window
    public float bestScore;     // best single frame
    public int frameCount;
}

// World-space joints used by the deliberately broad Dead Bug preparation
// check. Keeping this as plain geometry makes the rule deterministic and easy
// to regression-test without a live tracker or a particular avatar rig.
public struct DeadBugPreparationPoseGeometry
{
    public Vector3 hips;
    public Vector3 upperTorso;
    public Vector3 leftShoulder;
    public Vector3 leftHand;
    public Vector3 rightShoulder;
    public Vector3 rightHand;
    public Vector3 leftUpperLeg;
    public Vector3 leftKnee;
    public Vector3 leftFoot;
    public Vector3 rightUpperLeg;
    public Vector3 rightKnee;
    public Vector3 rightFoot;
    public Vector3 worldUp;
}

[Serializable]
public sealed class FormalActionScoreMappingSettings
{
    [Tooltip("Raw scores at or below this value are not boosted.")]
    [Range(0f, 100f)] public float unchangedThrough = 50f;
    [Tooltip("The minimum acceptable raw formal-action score maps to an encouraging result.")]
    [Range(0f, 100f)] public float sourceEncouragement = 60f;
    [Range(0f, 100f)] public float mappedEncouragement = 70f;
    [Tooltip("A raw formal-action score at this point maps to the first target score.")]
    [Range(0f, 100f)] public float sourceGood = 70f;
    [Range(0f, 100f)] public float mappedGood = 80f;
    [Tooltip("A strong raw score enters the high 80s without being compressed into the high 90s.")]
    [Range(0f, 100f)] public float sourceVeryGood = 80f;
    [Range(0f, 100f)] public float mappedVeryGood = 88f;
    [Tooltip("A genuinely excellent raw score enters the low 90s.")]
    [Range(0f, 100f)] public float sourceExcellent = 90f;
    [Range(0f, 100f)] public float mappedExcellent = 93f;
    [Tooltip("High-end anchors preserve genuine 98, 99, and 100 scores.")]
    [Range(0f, 100f)] public float sourceOutstanding = 95f;
    [Range(0f, 100f)] public float mappedOutstanding = 96f;
    [Range(0f, 100f)] public float sourceElite = 98f;
    [Range(0f, 100f)] public float mappedElite = 98f;
    [Range(0f, 100f)] public float sourceNearPerfect = 99f;
    [Range(0f, 100f)] public float mappedNearPerfect = 99f;

    public float Map(float score)
    {
        float clamped = Mathf.Clamp(score, 0f, 100f);
        float floor = Mathf.Clamp(unchangedThrough, 0f, 99.92f);
        float encouragementSource = Mathf.Clamp(
            sourceEncouragement,
            floor + 0.01f,
            99.93f);
        float goodSource = Mathf.Clamp(
            sourceGood,
            encouragementSource + 0.01f,
            99.94f);
        float veryGoodSource = Mathf.Clamp(
            sourceVeryGood,
            goodSource + 0.01f,
            99.95f);
        float excellentSource = Mathf.Clamp(
            sourceExcellent,
            veryGoodSource + 0.01f,
            99.96f);
        float outstandingSource = Mathf.Clamp(
            sourceOutstanding,
            excellentSource + 0.01f,
            99.97f);
        float eliteSource = Mathf.Clamp(
            sourceElite,
            outstandingSource + 0.01f,
            99.98f);
        float nearPerfectSource = Mathf.Clamp(
            sourceNearPerfect,
            eliteSource + 0.01f,
            99.99f);
        float encouragementTarget = Mathf.Clamp(
            mappedEncouragement,
            encouragementSource,
            100f);
        float goodTarget = Mathf.Clamp(mappedGood, goodSource, 100f);
        goodTarget = Mathf.Max(goodTarget, encouragementTarget);
        float veryGoodTarget = Mathf.Clamp(
            mappedVeryGood,
            goodTarget,
            100f);
        veryGoodTarget = Mathf.Max(veryGoodTarget, veryGoodSource);
        float excellentTarget = Mathf.Clamp(
            mappedExcellent,
            veryGoodTarget,
            100f);
        excellentTarget = Mathf.Max(excellentTarget, excellentSource);
        float outstandingTarget = Mathf.Clamp(
            mappedOutstanding,
            excellentTarget,
            100f);
        outstandingTarget = Mathf.Max(
            outstandingTarget,
            outstandingSource);
        float eliteTarget = Mathf.Clamp(
            mappedElite,
            outstandingTarget,
            100f);
        eliteTarget = Mathf.Max(eliteTarget, eliteSource);
        float nearPerfectTarget = Mathf.Clamp(
            mappedNearPerfect,
            eliteTarget,
            100f);
        nearPerfectTarget = Mathf.Max(
            nearPerfectTarget,
            nearPerfectSource);

        if (clamped <= floor)
        {
            return clamped;
        }
        if (clamped <= encouragementSource)
        {
            return Mathf.Lerp(
                floor,
                encouragementTarget,
                Mathf.InverseLerp(
                    floor,
                    encouragementSource,
                    clamped));
        }
        if (clamped <= goodSource)
        {
            return Mathf.Lerp(
                encouragementTarget,
                goodTarget,
                Mathf.InverseLerp(
                    encouragementSource,
                    goodSource,
                    clamped));
        }
        if (clamped <= veryGoodSource)
        {
            return Mathf.Lerp(
                goodTarget,
                veryGoodTarget,
                Mathf.InverseLerp(goodSource, veryGoodSource, clamped));
        }
        if (clamped <= excellentSource)
        {
            return Mathf.Lerp(
                veryGoodTarget,
                excellentTarget,
                Mathf.InverseLerp(
                    veryGoodSource,
                    excellentSource,
                    clamped));
        }
        if (clamped <= outstandingSource)
        {
            return Mathf.Lerp(
                excellentTarget,
                outstandingTarget,
                Mathf.InverseLerp(
                    excellentSource,
                    outstandingSource,
                    clamped));
        }
        if (clamped <= eliteSource)
        {
            return Mathf.Lerp(
                outstandingTarget,
                eliteTarget,
                Mathf.InverseLerp(
                    outstandingSource,
                    eliteSource,
                    clamped));
        }
        if (clamped <= nearPerfectSource)
        {
            return Mathf.Lerp(
                eliteTarget,
                nearPerfectTarget,
                Mathf.InverseLerp(
                    eliteSource,
                    nearPerfectSource,
                    clamped));
        }
        return Mathf.Lerp(
            nearPerfectTarget,
            100f,
            Mathf.InverseLerp(nearPerfectSource, 100f, clamped));
    }

    // A near-perfect decimal score must not be rounded into a displayed 100.
    // Genuine mapped 98 and 99 remain available; only an actual 100 is shown
    // as the full score.
    public static int RoundForDisplay(float score)
    {
        float clamped = Mathf.Clamp(score, 0f, 100f);
        return clamped >= 100f
            ? 100
            : Mathf.Min(99, Mathf.RoundToInt(clamped));
    }
}

// Scores the IMU user's pose against the coach's frozen target pose at each
// segment hold. Formal actions first reach a coarse profile-driven pose, then
// accumulate a short low-motion hold before scoring. A bounded follow window
// can bypass noisy stillness, but never the coarse active-region requirement.
public class PoseScorer : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private SegmentedCoachController coachController;
    [SerializeField] private Animator coachAnimator;
    [SerializeField] private Animator userAnimator;
    [SerializeField] private VoicePromptManager voiceManager; // 语音提示管理器
    [SerializeField] private ScorePopup scorePopup; // 分数弹出显示UI
    [SerializeField] private BodyPartFeedbackUI bodyPartFeedback; // 身体部位提示UI
    [SerializeField] private AvatarBodyPartHighlighter modelBodyPartHighlighter;
    [SerializeField] private bool enableModelSurfaceHighlight = true;
    [Tooltip("Dead Bug and Bird Dog use body-plane geometry instead of the " +
        "recorded coach pose. Generic preserves the reusable target comparer.")]
    [SerializeField] private PoseGuidanceExercise guidanceExercise =
        PoseGuidanceExercise.Generic;
    [SerializeField] private TargetedExercisePoseGuidanceSettings
        targetedExerciseGuidance = new TargetedExercisePoseGuidanceSettings();
    [Tooltip("VMC receiver that drives the user avatar. The scorer resolves it " +
        "from the avatar when this reference is empty.")]
    [SerializeField] private AvatarMotionSource guidanceVmcReceiver;
    [Tooltip("Fail closed when the global VMC stream is unavailable or stale.")]
    [SerializeField] private bool enableVmcFreshnessGate = true;
    [Tooltip("Maximum time without a packet frame or advancing VMC remote time.")]
    [SerializeField, Min(0.01f)] private float vmcFreshnessTimeoutSeconds = 0.25f;
    [HideInInspector]
    [Tooltip("Legacy serialized value retained for scene migration. The new " +
        "guidance path does not use regional scores.")]
    [SerializeField, Range(0f, 100f)] private float guidanceHighlightBelowScore = 70f;
    [HideInInspector]
    [SerializeField, Range(0f, 100f)] private float guidanceClearAtOrAboveScore = 75f;
    [HideInInspector]
    [SerializeField, Min(0.1f)] private float guidanceScoreLeniency = 1.5f;
    [Tooltip("Generic fallback IMU uncertainty, target-axis selection, and " +
        "diagnostic thresholds. Targeted Dead Bug / Bird Dog guidance does " +
        "not use these coach-target settings.")]
    [SerializeField] private PoseHighlightDiagnosticSettings guidanceDiagnostics =
        new PoseHighlightDiagnosticSettings();
    [Tooltip("A mismatch must persist for this long before its surface is highlighted.")]
    [SerializeField, Min(0f)] private float guidanceHighlightConfirmationSeconds = 0.5f;
    [Tooltip("A recovered or untracked region must remain clear for this long " +
        "before its surface highlight is removed.")]
    [SerializeField, Min(0f)] private float guidanceHighlightReleaseSeconds = 0.25f;
    [Tooltip("Every confirmed corrective highlight uses one stable red color.")]
    [SerializeField] private Color guidanceHighlightColor =
        new Color(1f, 0f, 0f, 0.95f);
    [Tooltip("Maximum regions shown together. Targeted Dead Bug / Bird Dog " +
        "scenes use four so every incorrectly positioned limb can be shown; " +
        "generic guidance can retain a lower visual cap.")]
    [SerializeField, Range(1, 5)] private int maximumSimultaneousGuidanceParts = 2;
    [HideInInspector]
    [Tooltip("Legacy serialized value retained for compatibility. Visible " +
        "regions are no longer replaced by severity challengers.")]
    [SerializeField, Min(0f)] private float guidanceDisplaySwitchMargin = 0.2f;

    [Header("UI (Optional - 可以完全不配置)")]
    [Tooltip("可选：用于调试的文字显示")]
    [SerializeField] private Text statusText;
    [SerializeField] private Text scoreText;
    [SerializeField] private Button nextButton;

    [Header("Difficulty Configuration")]
    [Tooltip("难度配置 - 如果为空则使用下方的默认值")]
    [SerializeField] private DifficultyConfig stabilityDifficultyConfig;
    [SerializeField] private DifficultyConfig scoringDifficultyConfig;

    [Header("Stability Detection")]
    [Tooltip("Bones checked for stability (user must hold these still before scoring starts).")]
    [SerializeField] private HumanBodyBones[] stabilityBones = new HumanBodyBones[]
    {
        HumanBodyBones.Hips,
        HumanBodyBones.Spine,
        HumanBodyBones.Chest,
        HumanBodyBones.LeftUpperLeg,
        HumanBodyBones.RightUpperLeg,
        HumanBodyBones.LeftLowerLeg,
        HumanBodyBones.RightLowerLeg,
        HumanBodyBones.LeftUpperArm,
        HumanBodyBones.RightUpperArm
    };
    [Tooltip("Max rotation drift (degrees/second) for a bone to be considered stable. (默认值 - 可被难度配置覆盖)")]
    [SerializeField] private float stabilityThresholdDegPerSec = 5f;
    [Tooltip("How long the user must remain stable before scoring starts (seconds).")]
    [SerializeField] private float stabilityDuration = 2f;
    [Header("Dead Bug Preparation")]
    [Tooltip("The relaxed preparation pose only needs a short continuous confirmation. " +
        "It does not wait for the stricter IMU stillness gate used by formal actions.")]
    [Min(0.1f)] [SerializeField] private float preparationConfirmationDuration = 0.35f;
    [Tooltip("Maximum deviation from horizontal that still counts as lying down.")]
    [Range(0f, 90f)] [SerializeField] private float preparationMaxTorsoTiltFromHorizontal = 45f;
    [Tooltip("Each hand must extend this many torso lengths past its shoulder toward the head.")]
    [Min(0f)] [SerializeField] private float preparationMinHandExtensionTorsoRatio = 0.15f;
    [Tooltip("Maximum bend at each knee that still counts as a straight leg.")]
    [Range(0f, 90f)] [SerializeField] private float preparationMaxKneeBend = 40f;
    [Tooltip("For Bird Dog, hands and knees must drop at least this many torso lengths toward the floor.")]
    [Min(0f)] [SerializeField] private float preparationMinSupportDropTorsoRatio = 0.15f;
    [Tooltip("Legacy arm alignment setting retained for difficulty configuration; surface guidance uses the scorer-derived thresholds above.")]
    [SerializeField] private float stabilityAlignmentThreshold = 35f;
    [Tooltip("Legacy leg alignment setting retained for difficulty configuration; surface guidance uses the scorer-derived thresholds above.")]
    [SerializeField] private float stabilityAlignmentThresholdLegs = 50f;
    [Header("Scoring")]
    [Tooltip("Duration of the scoring countdown (seconds).")]
    [SerializeField] private float scoringDuration = 5f;
    [Tooltip("Bones compared for scoring. Rotation weight is higher than position.")]
    [SerializeField] private HumanBodyBones[] scoringBones = new HumanBodyBones[]
    {
        HumanBodyBones.Hips,
        HumanBodyBones.Spine,
        HumanBodyBones.Chest,
        HumanBodyBones.LeftUpperLeg,
        HumanBodyBones.RightUpperLeg,
        HumanBodyBones.LeftLowerLeg,
        HumanBodyBones.RightLowerLeg,
        HumanBodyBones.LeftUpperArm,
        HumanBodyBones.RightUpperArm,
        HumanBodyBones.LeftLowerArm,
        HumanBodyBones.RightLowerArm
    };

    [Header("Scoring Weights")]
    [Tooltip("Only rotation error is used for scoring (IMU tracking is more accurate for rotation than position).")]
    [SerializeField] private bool useRotationOnly = true;
    [Tooltip("Max angle difference (degrees) that maps to score=0. Smaller angles get higher scores. (默认值 - 可被难度配置覆盖)")]
    [SerializeField] private float maxAngleError = 30f;
    [Tooltip("Angle difference (degrees) that maps to score=100. Perfect alignment. (默认值 - 可被难度配置覆盖)")]
    [SerializeField] private float perfectAngleThreshold = 5f;

    [Header("Noise-Tolerant Scoring")]
    [SerializeField] private RobustPoseScoringSettings robustScoring = new RobustPoseScoringSettings();
    [Tooltip("Presentation-only remapping applied after main's scorer and robust " +
        "aggregation complete, and only on segments explicitly marked as formal actions.")]
    [SerializeField] private FormalActionScoreMappingSettings formalActionScoreMapping =
        new FormalActionScoreMappingSettings();
    [Tooltip("Compare normalized Humanoid muscle values so the user VMC rig " +
        "and coach animation rig can differ without producing whole-body " +
        "false errors. Root horizontal heading is excluded.")]
    [SerializeField] private bool useHumanoidMuscleSpace = true;
    [Tooltip("Remove fixed avatar and sensor-mounting offsets captured when training starts. " +
        "This is part of main's original robust scoring path.")]
    [SerializeField] private bool useSessionStartCalibration = true;
    [Tooltip("Stability is measured over this interval instead of frame-to-frame, which suppresses IMU jitter.")]
    [Min(0.05f)] [SerializeField] private float stabilitySampleInterval = 0.18f;
    [Tooltip("Additional IMU noise allowance added to the selected stability difficulty. " +
        "The difficulty still changes the gate, while normal wearable jitter no longer blocks every hold.")]
    [Min(0f)] [SerializeField] private float stabilityNoiseAllowanceDegPerSec = 7f;
    [Tooltip("A small fraction of noisy trackers may move without blocking the user.")]
    [Range(0f, 0.5f)] [SerializeField] private float allowedUnstableBoneFraction = 0.3f;
    [Tooltip("Hard cap for noisy bones allowed by the fraction above.")]
    [Min(0)] [SerializeField] private int maximumUnstableBoneCount = 2;
    [Tooltip("Every active Profile region must roughly match for this long before stillness can advance scoring.")]
    [Min(0f)] [SerializeField] private float coarseActionConfirmationSeconds = 0.3f;
    [Tooltip("An already-ready active region must remain obviously wrong for this long before its coarse marker clears.")]
    [Min(0f)] [SerializeField] private float coarseActionReleaseSeconds = 0.5f;
    [Tooltip("Ignore a short tracking spike without advancing or erasing the accumulated stable hold.")]
    [Min(0f)] [SerializeField] private float stabilityGraceDuration = 0.45f;
    [Tooltip("When sustained movement is detected, accumulated stable time decays at this rate instead of resetting to zero.")]
    [Min(0f)] [SerializeField] private float stabilityProgressDecayRate = 0.35f;
    [Tooltip("Stable time that must be re-earned if the user moves during the spoken scoring countdown.")]
    [Min(0.1f)] [SerializeField] private float stabilityRecoveryDuration = 0.75f;
    [Tooltip("Maximum time spent in the follow-and-adjust state. When reached, the attempt is scored as-is instead of trapping the session forever.")]
    [Min(0f)] [SerializeField] private float maximumFollowingWaitSeconds = 10f;

    // Fired when a segment's scoring window finishes, with the averaged result.
    public event Action<SegmentScore> OnSegmentScored;
    // Fired when the whole action is complete, with all collected segment scores.
    public event Action<List<SegmentScore>> OnAllScored;

    private readonly List<SegmentScore> sessionScores = new List<SegmentScore>();
    private readonly AnimatorMotionEvidence sessionMotionEvidence = new AnimatorMotionEvidence();
    private RobustPoseScoringEngine scoringEngine;
    private PoseHighlightDiagnosticEngine guidanceDiagnosticEngine;
    private PoseFrameScore lastFrameResult;
    private readonly Dictionary<HumanBodyBones, Quaternion> coachCalibrationRotations =
        new Dictionary<HumanBodyBones, Quaternion>();
    private readonly Dictionary<HumanBodyBones, Quaternion> userCalibrationRotations =
        new Dictionary<HumanBodyBones, Quaternion>();
    private bool hasSessionStartCalibration;

    private enum State
    {
        Idle,               // not holding at a segment
        WaitingStable,      // holding, waiting for user to stabilize
        PreparingScore,     // stable confirmed; synchronized voice countdown
        Scoring,            // counting down and sampling once per scoring beat
    }

    private State currentState = State.Idle;
    private float stateTimer;
    private bool initialFollowInstructionAnnouncedEarly;
    private Dictionary<HumanBodyBones, Quaternion> prevRotations = new Dictionary<HumanBodyBones, Quaternion>();
    private float stabilitySampleElapsed;
    private float unstableElapsed;
    private float followingElapsed;
    private bool lastStabilityResult = true;
    private bool preparingBestEffortScore;
    private HumanBodyBones[] effectiveStabilityBones;

    // Current segment scoring results
    private List<float> frameScores = new List<float>();
    private readonly List<PoseBodyRegion> frameWorstRegions = new List<PoseBodyRegion>();
    private SegmentHoldInfo currentSegmentInfo;

    // Alignment feedback
    private static readonly HumanBodyBones[] GuidanceBones =
    {
        HumanBodyBones.LeftUpperArm,
        HumanBodyBones.LeftLowerArm,
        HumanBodyBones.RightUpperArm,
        HumanBodyBones.RightLowerArm,
        HumanBodyBones.LeftUpperLeg,
        HumanBodyBones.LeftLowerLeg,
        HumanBodyBones.RightUpperLeg,
        HumanBodyBones.RightLowerLeg,
        HumanBodyBones.Hips,
        HumanBodyBones.Spine,
        HumanBodyBones.Chest
    };
    private readonly Dictionary<BodyPart, float> guidancePartErrors =
        new Dictionary<BodyPart, float>
        {
            { BodyPart.LeftArm, 0f },
            { BodyPart.RightArm, 0f },
            { BodyPart.LeftLeg, 0f },
            { BodyPart.RightLeg, 0f },
            { BodyPart.Torso, 0f }
        };
    private readonly Dictionary<BodyPart, float> guidancePartScores =
        new Dictionary<BodyPart, float>
        {
            { BodyPart.LeftArm, 100f },
            { BodyPart.RightArm, 100f },
            { BodyPart.LeftLeg, 100f },
            { BodyPart.RightLeg, 100f },
            { BodyPart.Torso, 100f }
        };
    private readonly Dictionary<BodyPart, float> guidancePartRawScores =
        new Dictionary<BodyPart, float>
        {
            { BodyPart.LeftArm, 100f },
            { BodyPart.RightArm, 100f },
            { BodyPart.LeftLeg, 100f },
            { BodyPart.RightLeg, 100f },
            { BodyPart.Torso, 100f }
        };
    private readonly Dictionary<BodyPart, float> guidancePartThresholds =
        new Dictionary<BodyPart, float>
        {
            { BodyPart.LeftArm, 0f },
            { BodyPart.RightArm, 0f },
            { BodyPart.LeftLeg, 0f },
            { BodyPart.RightLeg, 0f },
            { BodyPart.Torso, 0f }
        };
    private readonly Dictionary<BodyPart, float> guidancePartSeverities =
        new Dictionary<BodyPart, float>
        {
            { BodyPart.LeftArm, 0f },
            { BodyPart.RightArm, 0f },
            { BodyPart.LeftLeg, 0f },
            { BodyPart.RightLeg, 0f },
            { BodyPart.Torso, 0f }
        };
    private readonly Dictionary<BodyPart, bool> guidancePartValidity =
        new Dictionary<BodyPart, bool>
        {
            { BodyPart.LeftArm, false },
            { BodyPart.RightArm, false },
            { BodyPart.LeftLeg, false },
            { BodyPart.RightLeg, false },
            { BodyPart.Torso, false }
        };
    private readonly Dictionary<BodyPart, bool> guidancePartShouldClear =
        new Dictionary<BodyPart, bool>
        {
            { BodyPart.LeftArm, true },
            { BodyPart.RightArm, true },
            { BodyPart.LeftLeg, true },
            { BodyPart.RightLeg, true },
            { BodyPart.Torso, true }
        };
    private static readonly BodyPart[] AllGuidanceParts =
    {
        BodyPart.LeftArm,
        BodyPart.RightArm,
        BodyPart.LeftLeg,
        BodyPart.RightLeg,
        BodyPart.Torso
    };
    private readonly List<BodyPart> guidanceCandidateParts = new List<BodyPart>(5);
    private readonly List<BodyPart> guidancePartsToHighlight = new List<BodyPart>(5);
    private readonly Dictionary<BodyPart, Color> guidancePartColors =
        new Dictionary<BodyPart, Color>();
    private readonly HashSet<BodyPart> confirmedGuidanceParts = new HashSet<BodyPart>();
    private readonly Dictionary<BodyPart, float> guidanceMismatchSeconds =
        new Dictionary<BodyPart, float>();
    private readonly Dictionary<BodyPart, float> guidanceClearSeconds =
        new Dictionary<BodyPart, float>();
    private readonly List<PoseGuidanceRegionEvidence> guidanceEvidence =
        new List<PoseGuidanceRegionEvidence>(5);
    private PoseGuidanceRegionLatch guidanceRegionLatch;
    private CoarseActionReadinessLatch coarseActionReadinessLatch;
    private VmcPoseDataFreshnessMonitor vmcFreshnessMonitor;
    private bool guidanceInputIsFresh = true;
    private bool guidanceInputWasFresh;
    private bool hasLoggedMissingGuidanceReceiver;
#if UNITY_EDITOR || DEVELOPMENT_BUILD
    private readonly HashSet<BodyPart> lastReportedGuidanceParts = new HashSet<BodyPart>();
    private bool hasReportedGuidanceState;
#endif

    private void Start()
    {
        scoringEngine = new RobustPoseScoringEngine(robustScoring);
        guidanceDiagnosticEngine = new PoseHighlightDiagnosticEngine(
            guidanceDiagnostics);
        EnsureGuidanceState();
        ResolveGuidanceVmcReceiver();
        BuildEffectiveStabilityBones();
        // 自动查找 VoicePromptManager
        if (voiceManager == null)
        {
            voiceManager = FindObjectOfType<VoicePromptManager>();
        }

        if (coachController == null)
        {
            coachController = FindObjectOfType<SegmentedCoachController>();
        }

        // 自动查找 ScorePopup
        if (scorePopup == null)
        {
            scorePopup = FindObjectOfType<ScorePopup>();
        }

        // 自动查找 BodyPartFeedbackUI
        if (bodyPartFeedback == null)
        {
            bodyPartFeedback = FindObjectOfType<BodyPartFeedbackUI>();
        }

        if (enableModelSurfaceHighlight && userAnimator != null)
        {
            if (modelBodyPartHighlighter == null)
            {
                modelBodyPartHighlighter =
                    userAnimator.GetComponent<AvatarBodyPartHighlighter>();
            }

            if (modelBodyPartHighlighter == null)
            {
                modelBodyPartHighlighter =
                    userAnimator.gameObject.AddComponent<AvatarBodyPartHighlighter>();
            }

            modelBodyPartHighlighter.Configure(userAnimator);
            // Diagnostic severity colors must remain visually stable in the
            // see-through display; pulsing changes perceived color/brightness.
            modelBodyPartHighlighter.SetPulseEnabled(false);
        }

        if (coachController != null)
        {
            coachController.OnSegmentHold += HandleSegmentHold;
            coachController.OnAllComplete += HandleAllComplete;
        }

        if (nextButton != null)
        {
            nextButton.interactable = false;
        }

        UpdateStatusText("准备中...");
    }

    /// <summary>
    /// 设置稳定性难度配置（由DifficultySelectionUI调用）
    /// </summary>
    public void SetStabilityDifficulty(DifficultyConfig config)
    {
        if (config != null)
        {
            stabilityDifficultyConfig = config;
            Debug.Log($"PoseScorer: 稳定性难度已设置 - 漂移阈值={config.stabilityThresholdDegPerSec}°/s, " +
                      $"对齐阈值={config.stabilityAlignmentThreshold}°, 腿部阈值={config.stabilityAlignmentThresholdLegs}°");
        }
    }

    /// <summary>
    /// 设置评分难度配置（由DifficultySelectionUI调用）
    /// </summary>
    public void SetScoringDifficulty(DifficultyConfig config)
    {
        if (config != null)
        {
            scoringDifficultyConfig = config;
            Debug.Log($"PoseScorer: 评分难度已设置 - 最大误差={config.maxAngleError}°, 完美阈值={config.perfectAngleThreshold}°");
        }
    }

    /// <summary>
    /// 获取当前有效的稳定性漂移阈值
    /// </summary>
    private float GetEffectiveStabilityThreshold()
    {
        return stabilityDifficultyConfig != null
            ? stabilityDifficultyConfig.stabilityThresholdDegPerSec
            : stabilityThresholdDegPerSec;
    }

    /// <summary>
    /// 获取当前有效的稳定性对齐阈值
    /// </summary>
    private float GetEffectiveStabilityAlignment()
    {
        return stabilityDifficultyConfig != null
            ? stabilityDifficultyConfig.stabilityAlignmentThreshold
            : stabilityAlignmentThreshold;
    }

    /// <summary>
    /// 获取当前有效的腿部稳定性对齐阈值
    /// </summary>
    private float GetEffectiveStabilityAlignmentLegs()
    {
        return stabilityDifficultyConfig != null
            ? stabilityDifficultyConfig.stabilityAlignmentThresholdLegs
            : stabilityAlignmentThresholdLegs;
    }

    /// <summary>
    /// 获取当前有效的最大角度误差
    /// </summary>
    private float GetEffectiveMaxAngleError()
    {
        return scoringDifficultyConfig != null
            ? scoringDifficultyConfig.maxAngleError
            : maxAngleError;
    }

    /// <summary>
    /// 获取当前有效的完美角度阈值
    /// </summary>
    private float GetEffectivePerfectAngleThreshold()
    {
        return scoringDifficultyConfig != null
            ? scoringDifficultyConfig.perfectAngleThreshold
            : perfectAngleThreshold;
    }

    private void OnDestroy()
    {
        if (coachController != null)
        {
            coachController.OnSegmentHold -= HandleSegmentHold;
            coachController.OnAllComplete -= HandleAllComplete;
        }

        scoringEngine?.Dispose();
        scoringEngine = null;
        guidanceDiagnosticEngine?.Dispose();
        guidanceDiagnosticEngine = null;
    }

    private void OnDisable()
    {
        ResetVisualGuidanceEvidence();
        modelBodyPartHighlighter?.Hide();
    }

    private void Update()
    {
        sessionMotionEvidence.Sample();

        if (currentState == State.Idle)
        {
            return;
        }

        if (currentState == State.WaitingStable)
        {
            UpdateWaitingStable();
        }
        else if (currentState == State.PreparingScore)
        {
            UpdatePreparingScore();
        }
        else if (currentState == State.Scoring)
        {
            stateTimer -= Time.deltaTime;
            UpdateScoring();
        }

    }

    private void HandleSegmentHold(SegmentHoldInfo info)
    {
        bool isFirstHold =
            info.segmentIndex == 0 && info.repeatIndex == 0 && info.setIndex == 0;

        // First hold of a fresh run — clear previous session results.
        if (isFirstHold)
        {
            sessionScores.Clear();
        }

        currentSegmentInfo = info;
        ResetVisualGuidanceEvidence();
        frameScores.Clear();
        frameWorstRegions.Clear();
        prevRotations.Clear();
        voiceManager?.CancelStabilityCountdown();
        scoringEngine?.Reset();
        if (guidanceDiagnosticEngine == null)
        {
            guidanceDiagnosticEngine = new PoseHighlightDiagnosticEngine(
                guidanceDiagnostics);
        }
        if (UsesTargetedExerciseGuidance)
        {
            // Targeted Dead Bug / Bird Dog guidance must not inherit any
            // recorded coach-pose error.
            guidanceDiagnosticEngine.ResetTarget();
        }
        else
        {
            guidanceDiagnosticEngine.BeginTarget(
                coachAnimator,
                GuidanceBones,
                info.segmentKind,
                info.poseGuidanceRule);
        }

        EnterFollowingPhase(
            "coach demonstration complete",
            !isFirstHold || !initialFollowInstructionAnnouncedEarly,
            RequiresRelaxedPreparation()
                ? preparationConfirmationDuration
                : -1f);

        if (isFirstHold)
        {
            initialFollowInstructionAnnouncedEarly = false;
        }

        if (nextButton != null)
        {
            nextButton.interactable = false;
        }

        UpdateStatusText("等待姿势稳定...");
        Debug.Log($"PoseScorer: Segment {info.segmentIndex} hold (duration {info.scoringDuration}s), waiting for user to stabilize.");
    }

    private void HandleAllComplete()
    {
        sessionMotionEvidence.Stop();
        TransitionTo(State.Idle, "all segments complete");
        // The final transition can be shorter than the queued completion cue.
        // Let VoicePromptManager finish naturally; the completion controls stay
        // hidden until it is idle.
        ResetVisualGuidanceEvidence();
        modelBodyPartHighlighter?.Hide();

        float sessionAvg = 0f;
        if (sessionScores.Count > 0)
        {
            foreach (SegmentScore s in sessionScores)
            {
                sessionAvg += s.averageScore;
            }
            sessionAvg /= sessionScores.Count;
        }

        Debug.Log($"PoseScorer: 训练完成！总平均分 {sessionAvg:F1} ({sessionScores.Count} 个片段)");
        UpdateStatusText(
            $"训练完成！总分 " +
            $"{FormalActionScoreMappingSettings.RoundForDisplay(sessionAvg)}");
        OnAllScored?.Invoke(sessionScores);

        if (nextButton != null)
        {
            nextButton.interactable = false;
        }
    }

    private void BuildEffectiveStabilityBones()
    {
        var bones = new List<HumanBodyBones>();
        var seen = new HashSet<HumanBodyBones>();

        if (stabilityBones != null)
        {
            foreach (HumanBodyBones bone in stabilityBones)
            {
                if (seen.Add(bone))
                {
                    bones.Add(bone);
                }
            }
        }

        // Include all guided limbs in motion tracking. These bones only answer
        // whether the user has finished moving; coarse action readiness is
        // evaluated separately from profile-driven region evidence.
        foreach (HumanBodyBones bone in GuidanceBones)
        {
            if (seen.Add(bone))
            {
                bones.Add(bone);
            }
        }

        effectiveStabilityBones = bones.ToArray();
    }

    private void EnterFollowingPhase(
        string reason,
        bool announce,
        float requiredStableSeconds = -1f,
        bool resetFollowingWait = true)
    {
        voiceManager?.CancelStabilityCountdown();
        unstableElapsed = 0f;
        preparingBestEffortScore = false;
        if (resetFollowingWait)
        {
            followingElapsed = 0f;
        }
        stateTimer = requiredStableSeconds > 0f
            ? Mathf.Clamp(requiredStableSeconds, 0.1f, stabilityDuration)
            : stabilityDuration;
        TransitionTo(State.WaitingStable, reason);
        CaptureCurrentRotations();

        if (announce)
        {
            voiceManager?.PlayFollowCoach();
        }
    }

    private void TransitionTo(State nextState, string reason)
    {
        State previousState = currentState;
        currentState = nextState;
        Debug.Log(
            $"[CoachFlow] phase={nextState}, from={previousState}, reason={reason}, " +
            $"stableRemaining={Mathf.Max(0f, stateTimer):0.00}s",
            this);
    }

    private void UpdateWaitingStable()
    {
        UpdateStabilityGuidance();
        bool motionIsStable = CheckUserStability();
        bool requiresRelaxedPreparation = RequiresRelaxedPreparation();
        bool coarseActionIsReady = IsCoarseActionReady();
        bool isStable = requiresRelaxedPreparation
            ? IsCurrentUserInRelaxedPreparationPose()
            : motionIsStable && coarseActionIsReady;
        followingElapsed += Time.deltaTime;

        bool hasHighlightedMismatch = guidancePartsToHighlight.Count > 0;
        bool maximumWaitReached =
            maximumFollowingWaitSeconds > 0f &&
            followingElapsed >= maximumFollowingWaitSeconds;

        if (maximumWaitReached)
        {
            if (!requiresRelaxedPreparation && !coarseActionIsReady)
            {
                UpdateStatusText("请先把高亮提示的主动部位做到大致位置...");
                return;
            }

            if (TryBeginScoringCountdown(true))
            {
                UpdateStatusText("本次跟做时间已到，按当前动作进入评分...");
            }
            else
            {
                UpdateStatusText("语音提示中，播放完后按当前动作进入评分...");
            }
            return;
        }

        if (isStable)
        {
            unstableElapsed = 0f;
            stateTimer -= Time.deltaTime;

            if (stateTimer <= 0f)
            {
                if (!TryBeginScoringCountdown(false))
                {
                    UpdateStatusText("已达到稳定要求，语音播放完后进入倒计时...");
                }
            }
            else
            {
                UpdateStatusText(
                    hasHighlightedMismatch
                        ? $"根据高亮调整并保持 {Mathf.CeilToInt(stateTimer)}s..."
                        : $"保持姿势 {Mathf.CeilToInt(stateTimer)}s...");
            }

            return;
        }

        if (!requiresRelaxedPreparation && !coarseActionIsReady)
        {
            unstableElapsed += Time.deltaTime;
            if (unstableElapsed > stabilityGraceDuration)
            {
                stateTimer = Mathf.Min(
                    stabilityDuration,
                    stateTimer + Time.deltaTime * stabilityProgressDecayRate);
            }

            UpdateStatusText("请先把高亮提示的主动部位做到大致位置...");
            return;
        }

        if (requiresRelaxedPreparation)
        {
            // Preparation is deliberately semantic and forgiving: once the
            // user is lying down with both hands overhead and both legs
            // straight, it counts as stable regardless of normal IMU drift.
            // Reset on a missing condition so the short confirmation remains
            // continuous instead of accumulating across unrelated poses.
            unstableElapsed = 0f;
            stateTimer = preparationConfirmationDuration;
            UpdateStatusText(
                coachController != null &&
                coachController.UsesDeadBugPreparationPose
                    ? "请躺下，双手举过头顶并伸直双腿..."
                    : "请进入趴姿，用双手和双膝稳定支撑...");
            return;
        }

        unstableElapsed += Time.deltaTime;
        if (unstableElapsed <= stabilityGraceDuration)
        {
            // A brief noisy sample pauses progress. It does not count as stable
            // time and does not erase stable time the user has already earned.
            UpdateStatusText("检测到轻微晃动，请继续保持...");
            return;
        }

        // Sustained movement slowly erodes progress instead of resetting the
        // entire hold. This lets a real user converge across noisy IMU samples
        // while someone who keeps moving still cannot advance.
        stateTimer = Mathf.Min(
            stabilityDuration,
            stateTimer + Time.deltaTime * stabilityProgressDecayRate);
        float required = Mathf.Max(0.1f, stabilityDuration);
        float progress = 1f - Mathf.Clamp01(stateTimer / required);
        UpdateStatusText($"请稳定姿势，已保持 {progress * 100f:0}%");
    }

    private bool TryBeginScoringCountdown(bool bestEffort)
    {
        // Do not queue a delayed announcement or interrupt a sentence belonging
        // to the current event. WaitingStable retries after speech finishes and
        // rechecks the live stability/highlight gate first.
        if (voiceManager != null && voiceManager.IsBusy)
        {
            return false;
        }

        preparingBestEffortScore = bestEffort;
        TransitionTo(
            State.PreparingScore,
            bestEffort
                ? "maximum follow time reached; scoring current attempt"
                : "accumulated stability requirement met");
        UpdateStatusText("准备评分...");

        if (voiceManager == null)
        {
            StartScoring();
            return true;
        }

        if (!voiceManager.PlayStabilityCountdown(HandleScoringCountdownCompleted))
        {
            // An unexpected voice race must not strand PreparingScore. Return
            // to the live gate; no prompt is queued for delayed playback.
            preparingBestEffortScore = false;
            TransitionTo(State.WaitingStable, "voice became busy before countdown started");
            return false;
        }

        return true;
    }

    private void UpdatePreparingScore()
    {
        UpdateStabilityGuidance();

        if (RequiresRelaxedPreparation())
        {
            // The relaxed preparation pose already completed its continuous
            // confirmation before this spoken countdown started. Do not let a
            // later tracker sample cancel "2, 1, start" and loop back to the
            // follow-coach prompt.
            unstableElapsed = 0f;
            UpdateStatusText("请保持姿势，准备评分...");
            return;
        }

        bool coarseActionIsReady = IsCoarseActionReady();

        if (preparingBestEffortScore)
        {
            if (!coarseActionIsReady)
            {
                EnterFollowingPhase(
                    "coarse action readiness lost during best-effort countdown",
                    true,
                    stabilityRecoveryDuration,
                    false);
                UpdateStatusText("主动部位偏离，请重新做到大致位置...");
                return;
            }

            UpdateStatusText("请保持当前动作，准备评分...");
            return;
        }

        bool motionIsStable = CheckUserStability();
        bool isStable = RequiresRelaxedPreparation()
            ? IsCurrentUserInRelaxedPreparationPose()
            : motionIsStable && coarseActionIsReady;
        if (isStable)
        {
            unstableElapsed = 0f;
            UpdateStatusText("请保持姿势，准备评分...");
            return;
        }

        unstableElapsed += Time.deltaTime;
        if (unstableElapsed <= stabilityGraceDuration)
        {
            UpdateStatusText(
                coarseActionIsReady
                    ? "请继续保持稳定..."
                    : "主动部位偏离，请恢复到大致位置...");
            return;
        }

        EnterFollowingPhase(
            coarseActionIsReady
                ? "pre-score stability interrupted"
                : "pre-score coarse action readiness lost",
            true,
            stabilityRecoveryDuration,
            false);
        UpdateStatusText(
            coarseActionIsReady
                ? "检测到持续移动，请重新稳定片刻..."
                : "主动部位偏离，请重新做到大致位置...");
    }

    private void HandleScoringCountdownCompleted()
    {
        if (currentState == State.PreparingScore)
        {
            // The timeout may bypass noisy motion, but active regions must
            // still retain their low-threshold coarse readiness marker.
            bool requiresRelaxedPreparation = RequiresRelaxedPreparation();
            bool coarseActionIsReady = false;
            bool motionIsStable = false;
            if (!requiresRelaxedPreparation)
            {
                coarseActionIsReady = IsCoarseActionReady();
                motionIsStable = CheckUserStability();
            }

            bool isStable = CanCompleteScoringCountdown(
                requiresRelaxedPreparation,
                coarseActionIsReady,
                motionIsStable,
                preparingBestEffortScore);
            if (isStable)
            {
                StartScoring();
            }
            else
            {
                EnterFollowingPhase(
                    "not stable when scoring countdown completed",
                    true,
                    stabilityRecoveryDuration,
                    false);
            }
        }
    }

    private void UpdateScoring()
    {
        // The original scorer evaluates every rendered frame. Audio beats are
        // presentation only and must not define the statistical sample cadence.
        float score = CaptureScoringSample(Time.deltaTime);
        int remainingSec = Mathf.CeilToInt(stateTimer);
        UpdateStatusText($"评分中 {remainingSec}s");
        UpdateScoreText(
            $"当前得分 {FormalActionScoreMappingSettings.RoundForDisplay(score)}");
        scorePopup?.ShowLiveScore(score);

        if (stateTimer <= 0f)
        {
            FinishScoring();
        }
    }

    private void StartScoring()
    {
        TransitionTo(State.Scoring, "stability voice countdown completed");
        preparingBestEffortScore = false;
        // Use segment-specific duration if available, otherwise fall back to default
        float duration = currentSegmentInfo.scoringDuration > 0f ? currentSegmentInfo.scoringDuration : scoringDuration;
        stateTimer = duration;
        frameScores.Clear();
        frameWorstRegions.Clear();
        scoringEngine?.Reset();
        // 隐藏身体部位提示图片
        if (bodyPartFeedback != null)
        {
            bodyPartFeedback.Hide();
        }
        modelBodyPartHighlighter?.Hide();
        ResetVisualGuidanceEvidence();
#if UNITY_EDITOR || DEVELOPMENT_BUILD
        ReportGuidanceChange();
#endif

        // Beats remain audible countdown cues; UpdateScoring owns per-frame
        // evaluation and timer-based completion.
        voiceManager?.PlayScoringBeats(duration);

        UpdateStatusText($"评分中 {Mathf.CeilToInt(stateTimer)}s");
        Debug.Log(
            $"PoseScorer: User stabilized, starting {duration}s scoring " +
            "countdown with per-frame score sampling.");
    }

    private float CaptureScoringSample(float sampleDeltaTime)
    {
        float score = ScoreCurrentPose(sampleDeltaTime);
        // Final statistics intentionally use the exact score shown in the UI.
        frameScores.Add(score);
        frameWorstRegions.Add(lastFrameResult.worstRegion);
        return score;
    }

    private void FinishScoring()
    {
        float avgScore = 0f;
        float bestScore = 0f;
        if (frameScores.Count > 0)
        {
            foreach (float s in frameScores)
            {
                avgScore += s;
                if (s > bestScore)
                {
                    bestScore = s;
                }
            }
            avgScore = RobustPoseScoringEngine.CalculateRobustAverage(
                frameScores,
                robustScoring != null ? robustScoring.finalScoreTrimFraction : 0.1f);
        }

        SegmentScore result = new SegmentScore
        {
            segmentIndex = currentSegmentInfo.segmentIndex,
            repeatIndex = currentSegmentInfo.repeatIndex,
            setIndex = currentSegmentInfo.setIndex,
            segmentKind = currentSegmentInfo.segmentKind,
            label = currentSegmentInfo.label,
            // Retained for report-schema compatibility. Practice scoring now
            // has one authoritative score stream: the values shown in the UI.
            rawAverageScore = avgScore,
            rawBestScore = bestScore,
            averageScore = avgScore,
            bestScore = bestScore,
            frameCount = frameScores.Count
        };
        sessionScores.Add(result);

        Debug.Log(
            $"PoseScorer: Segment {result.segmentIndex} '{result.label}' " +
            $"完成。平均分 {avgScore:F1}, 最高分 {bestScore:F1} " +
            $"({result.frameCount} 帧采样)");

        OnSegmentScored?.Invoke(result);
        UpdateStatusText(
            $"本段得分 {FormalActionScoreMappingSettings.RoundForDisplay(avgScore)}");
        UpdateScoreText(
            $"平均 {FormalActionScoreMappingSettings.RoundForDisplay(avgScore)} / " +
            $"最高 {FormalActionScoreMappingSettings.RoundForDisplay(bestScore)}");

        // 使用ScorePopup显示分数
        if (scorePopup != null)
        {
            PoseBodyRegion feedbackRegion = GetDominantFeedbackRegion();
            scorePopup.ShowAverageScore(
                avgScore,
                RobustPoseScoringEngine.GetEncouragingFeedback(avgScore, feedbackRegion));
        }
        else
        {
            Debug.LogWarning("PoseScorer: ScorePopup 未配置！");
        }

        bool hasNextCheckpoint = coachController != null &&
            coachController.HasNextAfterCurrentHold;

        // The manager serializes ding -> next action and skips the next-action
        // announcement at the final checkpoint.
        voiceManager?.FinishScoring(hasNextCheckpoint);

        // Auto-advance to next segment
        TransitionTo(State.Idle, "segment scoring complete");
        if (coachController != null)
        {
            coachController.AdvanceToNext();
        }
    }

    private bool RequiresRelaxedPreparation()
    {
        return currentSegmentInfo.segmentKind == TrainingSegmentKind.Preparation &&
               coachController != null &&
               coachController.IsPreparationHold;
    }

    private bool IsCurrentUserInRelaxedPreparationPose()
    {
        if (coachController != null &&
            coachController.UsesDeadBugPreparationPose)
        {
            if (!TryCapturePreparationPose(
                    out DeadBugPreparationPoseGeometry deadBugPose))
            {
                return false;
            }

            return MeetsRelaxedDeadBugPreparationPose(
                deadBugPose,
                preparationMaxTorsoTiltFromHorizontal,
                preparationMinHandExtensionTorsoRatio,
                preparationMaxKneeBend);
        }

        if (!TryCapturePreparationPose(
                out DeadBugPreparationPoseGeometry birdDogPose))
        {
            return false;
        }

        return MeetsRelaxedBirdDogPreparationPose(
            birdDogPose,
            preparationMaxTorsoTiltFromHorizontal,
            preparationMinSupportDropTorsoRatio);
    }

    private bool TryCapturePreparationPose(
        out DeadBugPreparationPoseGeometry pose)
    {
        return TryCapturePreparationPose(
            userAnimator,
            Vector3.up,
            out pose);
    }

    public static bool TryCapturePreparationPose(
        Animator animator,
        Vector3 worldUp,
        out DeadBugPreparationPoseGeometry pose)
    {
        pose = new DeadBugPreparationPoseGeometry();
        if (animator == null)
        {
            return false;
        }

        Transform hips = animator.GetBoneTransform(HumanBodyBones.Hips);
        Transform upperTorso =
            animator.GetBoneTransform(HumanBodyBones.UpperChest) ??
            animator.GetBoneTransform(HumanBodyBones.Chest) ??
            animator.GetBoneTransform(HumanBodyBones.Spine);
        Transform leftShoulder =
            animator.GetBoneTransform(HumanBodyBones.LeftShoulder) ??
            animator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
        Transform rightShoulder =
            animator.GetBoneTransform(HumanBodyBones.RightShoulder) ??
            animator.GetBoneTransform(HumanBodyBones.RightUpperArm);
        Transform leftHand = animator.GetBoneTransform(HumanBodyBones.LeftHand);
        Transform rightHand = animator.GetBoneTransform(HumanBodyBones.RightHand);
        Transform leftUpperLeg = animator.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
        Transform leftKnee = animator.GetBoneTransform(HumanBodyBones.LeftLowerLeg);
        Transform leftFoot = animator.GetBoneTransform(HumanBodyBones.LeftFoot);
        Transform rightUpperLeg = animator.GetBoneTransform(HumanBodyBones.RightUpperLeg);
        Transform rightKnee = animator.GetBoneTransform(HumanBodyBones.RightLowerLeg);
        Transform rightFoot = animator.GetBoneTransform(HumanBodyBones.RightFoot);

        if (hips == null || upperTorso == null ||
            leftShoulder == null || leftHand == null ||
            rightShoulder == null || rightHand == null ||
            leftUpperLeg == null || leftKnee == null ||
            rightUpperLeg == null || rightKnee == null)
        {
            return false;
        }

        pose = new DeadBugPreparationPoseGeometry
        {
            hips = hips.position,
            upperTorso = upperTorso.position,
            leftShoulder = leftShoulder.position,
            leftHand = leftHand.position,
            rightShoulder = rightShoulder.position,
            rightHand = rightHand.position,
            leftUpperLeg = leftUpperLeg.position,
            leftKnee = leftKnee.position,
            leftFoot = leftFoot != null
                ? leftFoot.position
                : leftKnee.position,
            rightUpperLeg = rightUpperLeg.position,
            rightKnee = rightKnee.position,
            rightFoot = rightFoot != null
                ? rightFoot.position
                : rightKnee.position,
            worldUp = worldUp.sqrMagnitude > 1e-6f
                ? worldUp.normalized
                : Vector3.up
        };
        return true;
    }

    public static bool MeetsRelaxedDeadBugDirectionLockPose(
        DeadBugPreparationPoseGeometry pose,
        float maxTorsoTiltFromHorizontal,
        float minimumHandExtensionTorsoRatio)
    {
        Vector3 torso = pose.upperTorso - pose.hips;
        Vector3 up = pose.worldUp;
        if (torso.sqrMagnitude < 1e-6f || up.sqrMagnitude < 1e-6f)
        {
            return false;
        }

        float torsoLength = torso.magnitude;
        Vector3 headward = torso / torsoLength;
        float verticalShare = Mathf.Abs(
            Vector3.Dot(headward, up.normalized));
        float maximumVerticalShare = Mathf.Sin(
            Mathf.Clamp(maxTorsoTiltFromHorizontal, 0f, 90f) *
            Mathf.Deg2Rad);
        if (verticalShare > maximumVerticalShare)
        {
            return false;
        }

        float minimumHandExtension =
            Mathf.Max(0f, minimumHandExtensionTorsoRatio) * torsoLength;
        return Vector3.Dot(
                   pose.leftHand - pose.leftShoulder,
                   headward) >= minimumHandExtension &&
               Vector3.Dot(
                   pose.rightHand - pose.rightShoulder,
                   headward) >= minimumHandExtension;
    }

    public static bool MeetsRelaxedDeadBugPreparationPose(
        DeadBugPreparationPoseGeometry pose,
        float maxTorsoTiltFromHorizontal,
        float minimumHandExtensionTorsoRatio,
        float maximumKneeBend)
    {
        bool bothLegsStraight =
            IsLegStraight(
                pose.leftUpperLeg,
                pose.leftKnee,
                pose.leftFoot,
                maximumKneeBend) &&
            IsLegStraight(
                pose.rightUpperLeg,
                pose.rightKnee,
                pose.rightFoot,
                maximumKneeBend);

        return MeetsRelaxedDeadBugDirectionLockPose(
                   pose,
                   maxTorsoTiltFromHorizontal,
                   minimumHandExtensionTorsoRatio) &&
               bothLegsStraight;
    }

    public static bool MeetsRelaxedBirdDogPreparationPose(
        DeadBugPreparationPoseGeometry pose,
        float maxTorsoTiltFromHorizontal,
        float minimumSupportDropTorsoRatio)
    {
        Vector3 torso = pose.upperTorso - pose.hips;
        Vector3 up = pose.worldUp;
        if (torso.sqrMagnitude < 1e-6f || up.sqrMagnitude < 1e-6f)
        {
            return false;
        }

        float torsoLength = torso.magnitude;
        Vector3 normalizedUp = up.normalized;
        float verticalShare = Mathf.Abs(
            Vector3.Dot(torso / torsoLength, normalizedUp));
        float maximumVerticalShare = Mathf.Sin(
            Mathf.Clamp(maxTorsoTiltFromHorizontal, 0f, 90f) *
            Mathf.Deg2Rad);
        bool torsoIsHorizontal = verticalShare <= maximumVerticalShare;

        float minimumSupportDrop =
            Mathf.Max(0f, minimumSupportDropTorsoRatio) * torsoLength;
        Vector3 floorward = -normalizedUp;
        bool bothHandsSupport =
            Vector3.Dot(pose.leftHand - pose.leftShoulder, floorward) >=
                minimumSupportDrop &&
            Vector3.Dot(pose.rightHand - pose.rightShoulder, floorward) >=
                minimumSupportDrop;
        bool bothKneesSupport =
            Vector3.Dot(pose.leftKnee - pose.leftUpperLeg, floorward) >=
                minimumSupportDrop &&
            Vector3.Dot(pose.rightKnee - pose.rightUpperLeg, floorward) >=
                minimumSupportDrop;

        return torsoIsHorizontal && bothHandsSupport && bothKneesSupport;
    }

    public static bool CanCompleteScoringCountdown(
        bool requiresRelaxedPreparation,
        bool coarseActionReady,
        bool motionIsStable,
        bool bestEffort)
    {
        // Reaching a preparation countdown means the exercise-specific pose
        // has already passed its continuous confirmation. Sampling it again at
        // the end can trap noisy avatar rigs in an endless countdown loop.
        if (requiresRelaxedPreparation)
        {
            return true;
        }

        return coarseActionReady && (bestEffort || motionIsStable);
    }

    private static bool IsLegStraight(
        Vector3 upperLeg,
        Vector3 knee,
        Vector3 foot,
        float maximumKneeBend)
    {
        Vector3 upperSection = knee - upperLeg;
        Vector3 lowerSection = foot - knee;
        if (upperSection.sqrMagnitude < 1e-6f || lowerSection.sqrMagnitude < 1e-6f)
        {
            return false;
        }

        return Vector3.Angle(upperSection, lowerSection) <=
            Mathf.Clamp(maximumKneeBend, 0f, 90f);
    }

    private bool CheckUserStability()
    {
        if (userAnimator == null)
        {
            return false;
        }

        stabilitySampleElapsed += Time.deltaTime;
        if (stabilitySampleElapsed < Mathf.Max(0.05f, stabilitySampleInterval))
        {
            return lastStabilityResult;
        }

        float sampleDuration = stabilitySampleElapsed;
        stabilitySampleElapsed = 0f;
        int checkedBoneCount = 0;
        int unstableBoneCount = 0;
        // Keep Relaxed/Medium/Strict distinct while accounting for the normal
        // baseline jitter of wearable IMU trackers.
        float effectiveThreshold = Mathf.Max(
            0.1f,
            GetEffectiveStabilityThreshold() + stabilityNoiseAllowanceDegPerSec);

        foreach (HumanBodyBones bone in effectiveStabilityBones)
        {
            Transform t = userAnimator.GetBoneTransform(bone);
            if (t == null)
            {
                continue;
            }

            Quaternion currentRot = t.localRotation;

            if (prevRotations.TryGetValue(bone, out Quaternion prevRot))
            {
                float angleDiff = Quaternion.Angle(prevRot, currentRot);
                float angularVelocity = angleDiff / Mathf.Max(sampleDuration, 1e-4f);

                if (angularVelocity > effectiveThreshold)
                {
                    unstableBoneCount++;
                }
            }

            checkedBoneCount++;

            // Roll the reference forward at the fixed sampling interval.
            prevRotations[bone] = currentRot;
        }

        int allowedUnstable = CalculateAllowedUnstableBoneCount(
            checkedBoneCount,
            allowedUnstableBoneFraction,
            maximumUnstableBoneCount);
        lastStabilityResult =
            checkedBoneCount > 0 &&
            unstableBoneCount <= allowedUnstable;
        return lastStabilityResult;
    }

    public static int CalculateAllowedUnstableBoneCount(
        int checkedBoneCount,
        float allowedFraction,
        int maximumCount)
    {
        int checkedCount = Mathf.Max(0, checkedBoneCount);
        int proportionalAllowance = Mathf.FloorToInt(
            checkedCount * Mathf.Clamp01(allowedFraction));
        return Mathf.Min(
            proportionalAllowance,
            Mathf.Max(0, maximumCount));
    }

    private void CaptureCurrentRotations()
    {
        if (userAnimator == null)
        {
            return;
        }

        prevRotations.Clear();
        stabilitySampleElapsed = 0f;
        lastStabilityResult = false;
        foreach (HumanBodyBones bone in effectiveStabilityBones)
        {
            Transform t = userAnimator.GetBoneTransform(bone);
            if (t != null)
            {
                prevRotations[bone] = t.localRotation;
            }
        }
    }

    // Compares user's current pose to coach's frozen target pose. Returns a score 0-100.
    private float ScoreCurrentPose(float sampleDeltaTime)
    {
        if (userAnimator == null || coachAnimator == null)
        {
            return 0f;
        }

        // Per-segment leniency widens the tolerated error (e.g. transition poses).
        float leniency = Mathf.Max(0.1f, currentSegmentInfo.scoreLeniency);
        float effectiveMaxAngle = GetEffectiveMaxAngleError();
        float effectivePerfectAngle = GetEffectivePerfectAngleThreshold();
        if (scoringEngine == null)
        {
            scoringEngine = new RobustPoseScoringEngine(robustScoring);
        }

        lastFrameResult = scoringEngine.Evaluate(
            userAnimator,
            coachAnimator,
            scoringBones,
            effectivePerfectAngle,
            effectiveMaxAngle,
            leniency,
            userBaselines: useSessionStartCalibration && hasSessionStartCalibration
                ? userCalibrationRotations
                : null,
            coachBaselines: useSessionStartCalibration && hasSessionStartCalibration
                ? coachCalibrationRotations
                : null,
            deltaTime: Mathf.Max(0.0001f, sampleDeltaTime),
            ignoreHipsHeading: true,
            useHumanoidMuscleSpace: useHumanoidMuscleSpace);
        return ApplyFormalActionScoreMapping(
            lastFrameResult.score,
            currentSegmentInfo.segmentKind,
            formalActionScoreMapping);
    }

    public static float ApplyFormalActionScoreMapping(
        float rawScore,
        TrainingSegmentKind segmentKind,
        FormalActionScoreMappingSettings mapping)
    {
        return segmentKind == TrainingSegmentKind.CoreAction &&
               mapping != null
            ? mapping.Map(rawScore)
            : Mathf.Clamp(rawScore, 0f, 100f);
    }

    private void UpdateStatusText(string message)
    {
        if (statusText != null)
        {
            statusText.text = message;
        }
    }

    private void UpdateScoreText(string message)
    {
        if (scoreText != null)
        {
            scoreText.text = message;
        }
    }

    // Public API for debugging / manual control
    public float GetCurrentScore()
    {
        if (frameScores.Count == 0)
        {
            return 0f;
        }

        return frameScores[frameScores.Count - 1];
    }

    public float GetAverageScore()
    {
        if (frameScores.Count == 0)
        {
            return 0f;
        }

        float sum = 0f;
        foreach (float s in frameScores)
        {
            sum += s;
        }
        return sum / frameScores.Count;
    }

    // All segment scores collected so far this session (read-only view).
    public IReadOnlyList<SegmentScore> SessionScores => sessionScores;
    public PoseFrameScore LastFrameScore => lastFrameResult;
    public float MaximumUserMotionDegrees => sessionMotionEvidence.MaximumUserMotionDegrees;
    public float MaximumCoachMotionDegrees => sessionMotionEvidence.MaximumCoachMotionDegrees;
    public string CurrentFlowPhase => currentState.ToString();
    public float StabilityRemainingSeconds => Mathf.Max(0f, stateTimer);
    public bool CoarseActionReady => IsCoarseActionReady();

    public void BeginTrainingMeasurement()
    {
        if (effectiveStabilityBones == null || effectiveStabilityBones.Length == 0)
        {
            BuildEffectiveStabilityBones();
        }

        voiceManager?.StopAll();
        scorePopup?.HideImmediately();
        TransitionTo(State.Idle, "training session started");
        stateTimer = stabilityDuration;
        followingElapsed = 0f;
        unstableElapsed = 0f;
        preparingBestEffortScore = false;
        initialFollowInstructionAnnouncedEarly = false;
        sessionScores.Clear();
        CaptureSessionStartCalibration();
        scoringEngine?.Reset();
        if (guidanceDiagnosticEngine == null)
        {
            guidanceDiagnosticEngine = new PoseHighlightDiagnosticEngine(
                guidanceDiagnostics);
        }
        if (UsesTargetedExerciseGuidance)
        {
            guidanceDiagnosticEngine.ResetTarget();
        }
        else
        {
            guidanceDiagnosticEngine.CaptureSessionReference(
                userAnimator,
                coachAnimator,
                GuidanceBones);
        }
        sessionMotionEvidence.Begin(coachAnimator, userAnimator, scoringBones);
    }

    /// <summary>
    /// Tells the user to follow as soon as the first coach demonstration begins.
    /// The first hold checkpoint then suppresses its usual duplicate announcement.
    /// </summary>
    public void AnnounceInitialFollowInstruction()
    {
        initialFollowInstructionAnnouncedEarly =
            voiceManager != null && voiceManager.PlayFollowCoach();
    }

    public void StopTrainingMeasurement()
    {
        sessionMotionEvidence.Stop();
        voiceManager?.StopAll();
        scorePopup?.HideImmediately();
        TransitionTo(State.Idle, "training session stopped");
        stateTimer = stabilityDuration;
        followingElapsed = 0f;
        unstableElapsed = 0f;
        preparingBestEffortScore = false;
        initialFollowInstructionAnnouncedEarly = false;
        prevRotations.Clear();
        frameScores.Clear();
        frameWorstRegions.Clear();
        ResetVisualGuidanceEvidence();
        bodyPartFeedback?.Hide();
        modelBodyPartHighlighter?.Hide();
        scoringEngine?.Reset();
        guidanceDiagnosticEngine?.ResetTarget();
    }

#if UNITY_EDITOR
    /// <summary>
    /// Starts a deterministic hold checkpoint for the local VMC simulator.
    /// This bypasses the coach clip lead-in but keeps the production
    /// stabilization, guidance, countdown, and scoring state machine intact.
    /// </summary>
    public void BeginEditorGuidanceCheckpoint()
    {
        if (coachAnimator == null || userAnimator == null)
        {
            Debug.LogWarning("PoseScorer: cannot start editor guidance checkpoint without both animators.");
            return;
        }

        coachAnimator.speed = 0f;
        CaptureSessionStartCalibration();
        if (guidanceDiagnosticEngine == null)
        {
            guidanceDiagnosticEngine = new PoseHighlightDiagnosticEngine(
                guidanceDiagnostics);
        }
        guidanceDiagnosticEngine.CaptureSessionReference(
            userAnimator,
            coachAnimator,
            GuidanceBones);
        HandleSegmentHold(new SegmentHoldInfo(
            0,
            0,
            0,
            "Editor guidance probe",
            1f,
            Mathf.Max(3f, scoringDuration),
            TrainingSegmentKind.CoreAction));
        Debug.Log("PoseScorer: editor guidance checkpoint started.");
    }
#endif

    private PoseBodyRegion GetDominantFeedbackRegion()
    {
        var counts = new Dictionary<PoseBodyRegion, int>();
        PoseBodyRegion dominant = PoseBodyRegion.None;
        int dominantCount = 0;
        foreach (PoseBodyRegion region in frameWorstRegions)
        {
            if (region == PoseBodyRegion.None)
            {
                continue;
            }

            counts.TryGetValue(region, out int count);
            count++;
            counts[region] = count;
            if (count > dominantCount)
            {
                dominant = region;
                dominantCount = count;
            }
        }

        return dominant;
    }

    private void CaptureSessionStartCalibration()
    {
        coachCalibrationRotations.Clear();
        userCalibrationRotations.Clear();
        hasSessionStartCalibration = false;
        if (!useSessionStartCalibration || coachAnimator == null || userAnimator == null)
        {
            return;
        }

        foreach (HumanBodyBones bone in scoringBones)
        {
            Transform coachBone = coachAnimator.GetBoneTransform(bone);
            Transform userBone = userAnimator.GetBoneTransform(bone);
            if (coachBone == null || userBone == null)
            {
                continue;
            }

            coachCalibrationRotations[bone] = coachBone.localRotation;
            userCalibrationRotations[bone] = userBone.localRotation;
        }

        hasSessionStartCalibration = coachCalibrationRotations.Count > 0;
        Debug.Log($"PoseScorer: captured neutral calibration for {coachCalibrationRotations.Count} bones.");
    }

    private void UpdateStabilityGuidance()
    {
        guidanceInputIsFresh = false;
        ShowMisalignedBodyParts();
        UpdateCoarseActionReadiness(Time.deltaTime);
    }

    private bool RequiresCoarseActionReadiness =>
        currentSegmentInfo.segmentKind == TrainingSegmentKind.CoreAction &&
        currentSegmentInfo.poseGuidanceRule.mode ==
            PoseGuidanceProfileMode.Explicit;

    private bool IsCoarseActionReady()
    {
        EnsureGuidanceState();
        return !RequiresCoarseActionReadiness ||
            coarseActionReadinessLatch.IsReady;
    }

    private void UpdateCoarseActionReadiness(float deltaTime)
    {
        EnsureGuidanceState();
        coarseActionReadinessLatch.Configure(
            coarseActionConfirmationSeconds,
            coarseActionReleaseSeconds);
        if (!RequiresCoarseActionReadiness)
        {
            coarseActionReadinessLatch.Reset();
            return;
        }

        coarseActionReadinessLatch.Update(
            currentSegmentInfo.poseGuidanceRule,
            guidanceEvidence,
            deltaTime,
            guidanceInputIsFresh);
    }

    /// <summary>
    /// Evaluates either targeted Dead Bug / Bird Dog body-plane geometry or
    /// the generic target-salient Humanoid comparer. Formal scores and score
    /// mappings are intentionally absent from both corrective paths.
    /// </summary>
    private void ShowMisalignedBodyParts()
    {
        if (userAnimator == null ||
            (!UsesTargetedExerciseGuidance && coachAnimator == null))
        {
            bodyPartFeedback?.Hide();
            modelBodyPartHighlighter?.Hide();
            return;
        }

        if (modelBodyPartHighlighter != null &&
            modelBodyPartHighlighter.TargetAnimator != userAnimator)
        {
            modelBodyPartHighlighter.Configure(userAnimator);
            ResetGuidanceLatchState();
            ResolveGuidanceVmcReceiver();
        }

        // EditMode validators drive HumanPose directly and intentionally have
        // no live VMC receiver. Runtime/play mode keeps the full stale-input
        // gate; editor-side pose diagnostics remain deterministic.
        guidanceInputIsFresh = !Application.isPlaying ||
            UpdateGuidanceVmcFreshness(Time.unscaledDeltaTime);

        PoseHighlightFrameDiagnostics diagnostics;
        if (UsesTargetedExerciseGuidance)
        {
            TryCaptureTargetedExerciseGeometry(
                out TargetedExercisePoseGeometry geometry);
            diagnostics = TargetedExercisePoseGuidance.Evaluate(
                guidanceExercise,
                currentSegmentInfo.segmentKind,
                currentSegmentInfo.poseGuidanceRule,
                geometry,
                targetedExerciseGuidance);
        }
        else
        {
            if (guidanceDiagnosticEngine == null)
            {
                guidanceDiagnosticEngine = new PoseHighlightDiagnosticEngine(
                    guidanceDiagnostics);
            }

            if (!guidanceDiagnosticEngine.HasTarget &&
                !guidanceDiagnosticEngine.BeginTarget(
                    coachAnimator,
                    GuidanceBones,
                    currentSegmentInfo.segmentKind,
                    currentSegmentInfo.poseGuidanceRule))
            {
                ResetVisualGuidanceEvidence();
                bodyPartFeedback?.Hide();
                modelBodyPartHighlighter?.Hide();
                return;
            }

            diagnostics = guidanceDiagnosticEngine.Evaluate(
                userAnimator,
                GuidanceBones,
                Time.deltaTime);
        }

        guidanceCandidateParts.Clear();
        UpdateGuidancePart(BodyPart.LeftArm, diagnostics.leftArm);
        UpdateGuidancePart(BodyPart.RightArm, diagnostics.rightArm);
        UpdateGuidancePart(BodyPart.LeftLeg, diagnostics.leftLeg);
        UpdateGuidancePart(BodyPart.RightLeg, diagnostics.rightLeg);
        UpdateGuidancePart(BodyPart.Torso, diagnostics.torso);
        UpdateConfirmedGuidanceParts(Time.deltaTime);

        UpdateGuidancePartColors();

#if UNITY_EDITOR || DEVELOPMENT_BUILD
        ReportGuidanceChange();
#endif

        if (guidancePartsToHighlight.Count == 0)
        {
            bodyPartFeedback?.Hide();
            modelBodyPartHighlighter?.Hide();
            return;
        }

        BodyPart mostMisalignedPart = guidancePartsToHighlight[0];
        bool modelGuidanceShown = enableModelSurfaceHighlight &&
            modelBodyPartHighlighter != null &&
            modelBodyPartHighlighter.Show(guidancePartColors);
        if (modelGuidanceShown)
        {
            // The surface overlay replaces the old picture card.
            bodyPartFeedback?.Hide();
        }
        else
        {
            // Keep the picture only as a fallback if an avatar mesh cannot
            // produce overlays (for example, an unsupported runtime model).
            modelBodyPartHighlighter?.Hide();
            bodyPartFeedback?.ShowBodyPart(mostMisalignedPart);
        }
    }

    private bool UsesTargetedExerciseGuidance =>
        guidanceExercise == PoseGuidanceExercise.DeadBug ||
        guidanceExercise == PoseGuidanceExercise.BirdDog;

    private bool TryCaptureTargetedExerciseGeometry(
        out TargetedExercisePoseGeometry geometry)
    {
        geometry = new TargetedExercisePoseGeometry();
        if (userAnimator == null)
        {
            return false;
        }

        Transform hips = userAnimator.GetBoneTransform(HumanBodyBones.Hips);
        Transform upperTorso =
            userAnimator.GetBoneTransform(HumanBodyBones.UpperChest) ??
            userAnimator.GetBoneTransform(HumanBodyBones.Chest) ??
            userAnimator.GetBoneTransform(HumanBodyBones.Spine);
        Transform leftUpperArm =
            userAnimator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
        Transform rightUpperArm =
            userAnimator.GetBoneTransform(HumanBodyBones.RightUpperArm);
        Transform leftShoulder =
            userAnimator.GetBoneTransform(HumanBodyBones.LeftShoulder) ??
            leftUpperArm;
        Transform rightShoulder =
            userAnimator.GetBoneTransform(HumanBodyBones.RightShoulder) ??
            rightUpperArm;
        Transform leftLowerArm =
            userAnimator.GetBoneTransform(HumanBodyBones.LeftLowerArm);
        Transform rightLowerArm =
            userAnimator.GetBoneTransform(HumanBodyBones.RightLowerArm);
        Transform leftHand =
            userAnimator.GetBoneTransform(HumanBodyBones.LeftHand);
        Transform rightHand =
            userAnimator.GetBoneTransform(HumanBodyBones.RightHand);
        Transform leftUpperLeg =
            userAnimator.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
        Transform rightUpperLeg =
            userAnimator.GetBoneTransform(HumanBodyBones.RightUpperLeg);
        Transform leftLowerLeg =
            userAnimator.GetBoneTransform(HumanBodyBones.LeftLowerLeg);
        Transform rightLowerLeg =
            userAnimator.GetBoneTransform(HumanBodyBones.RightLowerLeg);
        Transform leftFoot =
            userAnimator.GetBoneTransform(HumanBodyBones.LeftFoot);
        Transform rightFoot =
            userAnimator.GetBoneTransform(HumanBodyBones.RightFoot);

        float torsoLength = hips != null && upperTorso != null
            ? Vector3.Distance(hips.position, upperTorso.position)
            : 0f;

        geometry.bodyPlaneTracked = TryBuildBodyPlaneNormal(
            hips,
            upperTorso,
            leftUpperArm,
            rightUpperArm,
            leftUpperLeg,
            rightUpperLeg,
            out geometry.bodyPlaneNormal);
        geometry.torsoTiltTracked = TryGetTorsoTiltFromHorizontal(
            hips,
            upperTorso,
            out geometry.torsoTiltFromHorizontalDegrees);
        geometry.leftArmTracked = TryGetBoneDirection(
            leftUpperArm,
            leftLowerArm,
            out geometry.leftArmDirection);
        geometry.leftArmLineTracked = TryGetBoneDirection(
            leftShoulder,
            leftHand,
            out geometry.leftArmLineDirection);
        geometry.leftArmSupportTracked = TryGetSupportDropTorsoRatio(
            leftShoulder,
            leftHand,
            torsoLength,
            out geometry.leftArmSupportDropTorsoRatio);
        geometry.leftArmBendTracked = TryGetJointBendDegrees(
            leftUpperArm,
            leftLowerArm,
            leftHand,
            out geometry.leftArmBendDegrees);
        geometry.rightArmTracked = TryGetBoneDirection(
            rightUpperArm,
            rightLowerArm,
            out geometry.rightArmDirection);
        geometry.rightArmLineTracked = TryGetBoneDirection(
            rightShoulder,
            rightHand,
            out geometry.rightArmLineDirection);
        geometry.rightArmSupportTracked = TryGetSupportDropTorsoRatio(
            rightShoulder,
            rightHand,
            torsoLength,
            out geometry.rightArmSupportDropTorsoRatio);
        geometry.rightArmBendTracked = TryGetJointBendDegrees(
            rightUpperArm,
            rightLowerArm,
            rightHand,
            out geometry.rightArmBendDegrees);
        geometry.leftLegTracked = TryGetBoneDirection(
            leftUpperLeg,
            leftLowerLeg,
            out geometry.leftLegDirection);
        geometry.leftLegSupportTracked = TryGetSupportDropTorsoRatio(
            leftUpperLeg,
            leftLowerLeg,
            torsoLength,
            out geometry.leftLegSupportDropTorsoRatio);
        geometry.leftLegBendTracked = TryGetJointBendDegrees(
            leftUpperLeg,
            leftLowerLeg,
            leftFoot,
            out geometry.leftLegBendDegrees);
        geometry.rightLegTracked = TryGetBoneDirection(
            rightUpperLeg,
            rightLowerLeg,
            out geometry.rightLegDirection);
        geometry.rightLegSupportTracked = TryGetSupportDropTorsoRatio(
            rightUpperLeg,
            rightLowerLeg,
            torsoLength,
            out geometry.rightLegSupportDropTorsoRatio);
        geometry.rightLegBendTracked = TryGetJointBendDegrees(
            rightUpperLeg,
            rightLowerLeg,
            rightFoot,
            out geometry.rightLegBendDegrees);
        return geometry.bodyPlaneTracked;
    }

    private static bool TryBuildBodyPlaneNormal(
        Transform hips,
        Transform upperTorso,
        Transform leftUpperArm,
        Transform rightUpperArm,
        Transform leftUpperLeg,
        Transform rightUpperLeg,
        out Vector3 normal)
    {
        normal = Vector3.zero;
        if (hips == null || upperTorso == null)
        {
            return false;
        }

        Vector3 longitudinal = upperTorso.position - hips.position;
        if (longitudinal.sqrMagnitude <= 0.000001f)
        {
            return false;
        }

        Vector3 lateral = Vector3.zero;
        int lateralSamples = 0;
        AddLateralAxis(
            leftUpperArm,
            rightUpperArm,
            ref lateral,
            ref lateralSamples);
        AddLateralAxis(
            leftUpperLeg,
            rightUpperLeg,
            ref lateral,
            ref lateralSamples);
        if (lateralSamples == 0)
        {
            return false;
        }

        longitudinal.Normalize();
        lateral -= Vector3.Project(lateral, longitudinal);
        if (lateral.sqrMagnitude <= 0.000001f)
        {
            return false;
        }

        normal = Vector3.Cross(longitudinal, lateral.normalized);
        if (normal.sqrMagnitude <= 0.000001f)
        {
            normal = Vector3.zero;
            return false;
        }

        normal.Normalize();
        return true;
    }

    private static void AddLateralAxis(
        Transform left,
        Transform right,
        ref Vector3 sum,
        ref int sampleCount)
    {
        if (left == null || right == null)
        {
            return;
        }

        Vector3 candidate = right.position - left.position;
        if (candidate.sqrMagnitude <= 0.000001f)
        {
            return;
        }

        candidate.Normalize();
        if (sampleCount > 0 && Vector3.Dot(sum, candidate) < 0f)
        {
            candidate = -candidate;
        }

        sum += candidate;
        sampleCount++;
    }

    private static bool TryGetBoneDirection(
        Transform proximal,
        Transform distal,
        out Vector3 direction)
    {
        direction = Vector3.zero;
        if (proximal == null || distal == null)
        {
            return false;
        }

        direction = distal.position - proximal.position;
        return direction.sqrMagnitude > 0.000001f;
    }

    private static bool TryGetTorsoTiltFromHorizontal(
        Transform hips,
        Transform upperTorso,
        out float tiltDegrees)
    {
        tiltDegrees = 0f;
        if (hips == null || upperTorso == null)
        {
            return false;
        }

        Vector3 torso = upperTorso.position - hips.position;
        if (torso.sqrMagnitude <= 0.000001f)
        {
            return false;
        }

        float verticalShare = Mathf.Clamp01(Mathf.Abs(Vector3.Dot(
            torso.normalized,
            Vector3.up)));
        tiltDegrees = Mathf.Asin(verticalShare) * Mathf.Rad2Deg;
        return true;
    }

    private static bool TryGetSupportDropTorsoRatio(
        Transform proximal,
        Transform endpoint,
        float torsoLength,
        out float dropRatio)
    {
        dropRatio = 0f;
        if (proximal == null || endpoint == null || torsoLength <= 0.000001f)
        {
            return false;
        }

        dropRatio = Vector3.Dot(
            endpoint.position - proximal.position,
            Vector3.down) / torsoLength;
        return true;
    }

    private static bool TryGetJointBendDegrees(
        Transform proximal,
        Transform joint,
        Transform distal,
        out float bendDegrees)
    {
        bendDegrees = 0f;
        if (proximal == null || joint == null || distal == null)
        {
            return false;
        }

        Vector3 towardProximal = proximal.position - joint.position;
        Vector3 towardDistal = distal.position - joint.position;
        if (towardProximal.sqrMagnitude <= 0.000001f ||
            towardDistal.sqrMagnitude <= 0.000001f)
        {
            return false;
        }

        bendDegrees = 180f - Vector3.Angle(towardProximal, towardDistal);
        return true;
    }

    private void UpdateGuidancePart(
        BodyPart part,
        PoseHighlightRegionDiagnostic diagnostic)
    {
        guidancePartValidity[part] = diagnostic.isValid;
        guidancePartShouldClear[part] = diagnostic.shouldClear;
        if (!diagnostic.isValid)
        {
            guidancePartErrors[part] = 0f;
            guidancePartRawScores[part] = 100f;
            guidancePartScores[part] = 100f;
            guidancePartThresholds[part] = 0f;
            guidancePartSeverities[part] = 0f;
            return;
        }

        guidancePartErrors[part] = diagnostic.worstRawErrorDegrees;
        guidancePartSeverities[part] = diagnostic.severity;

        // Keep the historical score-shaped debug properties available for
        // tooling during migration. They are derived telemetry and never feed
        // back into the highlighter decision.
        float compatibilityScore = Mathf.Clamp(
            100f - diagnostic.severity * 30f,
            0f,
            100f);
        guidancePartRawScores[part] = compatibilityScore;
        guidancePartScores[part] = compatibilityScore;
        guidancePartThresholds[part] = 70f;
        if (diagnostic.shouldActivate)
        {
            guidanceCandidateParts.Add(part);
        }
    }

    private void UpdateConfirmedGuidanceParts(float deltaTime)
    {
        EnsureGuidanceState();
        guidanceRegionLatch.Configure(
            guidanceHighlightConfirmationSeconds,
            guidanceHighlightReleaseSeconds,
            maximumSimultaneousGuidanceParts);

        guidanceEvidence.Clear();
        foreach (BodyPart part in AllGuidanceParts)
        {
            guidanceEvidence.Add(new PoseGuidanceRegionEvidence(
                part,
                guidancePartValidity[part],
                guidanceCandidateParts.Contains(part),
                guidancePartShouldClear[part],
                guidancePartSeverities[part]));
        }

        guidanceRegionLatch.Update(
            guidanceEvidence,
            deltaTime,
            guidanceInputIsFresh);
        SyncGuidanceLatchState();
    }

    private void SelectVisibleGuidanceParts()
    {
        EnsureGuidanceState();
        guidanceRegionLatch.Configure(
            guidanceHighlightConfirmationSeconds,
            guidanceHighlightReleaseSeconds,
            maximumSimultaneousGuidanceParts);
        SyncGuidanceLatchState();
    }

    private void UpdateGuidancePartColors()
    {
        guidancePartColors.Clear();
        foreach (BodyPart part in guidancePartsToHighlight)
        {
            guidancePartColors[part] = guidanceHighlightColor;
        }
    }

    private void EnsureGuidanceState()
    {
        if (guidanceRegionLatch == null)
        {
            guidanceRegionLatch = new PoseGuidanceRegionLatch(
                guidanceHighlightConfirmationSeconds,
                guidanceHighlightReleaseSeconds,
                maximumSimultaneousGuidanceParts);
        }

        if (coarseActionReadinessLatch == null)
        {
            coarseActionReadinessLatch = new CoarseActionReadinessLatch(
                coarseActionConfirmationSeconds,
                coarseActionReleaseSeconds);
        }

        if (vmcFreshnessMonitor == null)
        {
            vmcFreshnessMonitor = new VmcPoseDataFreshnessMonitor(
                vmcFreshnessTimeoutSeconds);
        }
    }

    private void SyncGuidanceLatchState()
    {
        confirmedGuidanceParts.Clear();
        foreach (BodyPart part in guidanceRegionLatch.ConfirmedParts)
        {
            confirmedGuidanceParts.Add(part);
        }

        guidancePartsToHighlight.Clear();
        foreach (BodyPart part in guidanceRegionLatch.VisibleParts)
        {
            guidancePartsToHighlight.Add(part);
        }

        guidanceMismatchSeconds.Clear();
        guidanceClearSeconds.Clear();
        foreach (BodyPart part in AllGuidanceParts)
        {
            float mismatch = guidanceRegionLatch.GetMismatchSeconds(part);
            if (mismatch > 0f)
            {
                guidanceMismatchSeconds[part] = mismatch;
            }

            float clear = guidanceRegionLatch.GetClearSeconds(part);
            if (clear > 0f)
            {
                guidanceClearSeconds[part] = clear;
            }
        }
    }

    private void ResetGuidanceLatchState()
    {
        EnsureGuidanceState();
        guidanceRegionLatch.Reset();
        coarseActionReadinessLatch.Reset();
        confirmedGuidanceParts.Clear();
        guidancePartsToHighlight.Clear();
        guidanceMismatchSeconds.Clear();
        guidanceClearSeconds.Clear();
    }

    private bool UpdateGuidanceVmcFreshness(float deltaTime)
    {
        EnsureGuidanceState();
        vmcFreshnessMonitor.Configure(vmcFreshnessTimeoutSeconds);
        if (!enableVmcFreshnessGate)
        {
            if (!guidanceInputWasFresh)
            {
                ResetGuidanceLatchState();
            }

            guidanceInputWasFresh = true;
            return true;
        }

        ResolveGuidanceVmcReceiver();
        if (guidanceVmcReceiver == null)
        {
            if (!hasLoggedMissingGuidanceReceiver)
            {
                Debug.LogError(
                    "PoseScorer: VMC freshness is enabled but no receiver " +
                    "driving the user avatar could be resolved. Corrective " +
                    "guidance is disabled until a receiver is available.",
                    this);
                hasLoggedMissingGuidanceReceiver = true;
            }

            vmcFreshnessMonitor.Reset();
            guidanceInputWasFresh = false;
            return false;
        }

        hasLoggedMissingGuidanceReceiver = false;
        bool receiverAvailable = guidanceVmcReceiver.isActiveAndEnabled &&
            guidanceVmcReceiver.GetAvailable() > 0;
        bool isFresh = vmcFreshnessMonitor.Update(
            receiverAvailable,
            guidanceVmcReceiver.GetRemoteTime(),
            guidanceVmcReceiver.LastPacketframeCounterInFrame,
            deltaTime);
        if (isFresh && !guidanceInputWasFresh)
        {
            // A reconnect must earn confirmation from a clean state.
            ResetGuidanceLatchState();
        }

        guidanceInputWasFresh = isFresh;
        return isFresh;
    }

    private void ResolveGuidanceVmcReceiver()
    {
        if (ReceiverDrivesGuidanceAnimator(
                guidanceVmcReceiver,
                userAnimator))
        {
            return;
        }

        guidanceVmcReceiver = null;
        if (userAnimator == null)
        {
            return;
        }

        foreach (AvatarMotionSource candidate in
            FindObjectsOfType<AvatarMotionSource>(true))
        {
            if (ReceiverDrivesGuidanceAnimator(candidate, userAnimator))
            {
                guidanceVmcReceiver = candidate;
                return;
            }
        }
    }

    private static bool ReceiverDrivesGuidanceAnimator(
        AvatarMotionSource receiver,
        Animator animator)
    {
        if (receiver == null || animator == null || receiver.Model == null)
        {
            return false;
        }

        Transform model = receiver.Model.transform;
        Transform target = animator.transform;
        return target == model || target.IsChildOf(model);
    }

    private void ResetVisualGuidanceEvidence()
    {
        guidanceCandidateParts.Clear();
        guidancePartColors.Clear();
        ResetGuidanceLatchState();
        EnsureGuidanceState();
        vmcFreshnessMonitor.Reset();
        guidanceInputWasFresh = false;
        guidanceInputIsFresh = true;
        foreach (BodyPart part in AllGuidanceParts)
        {
            guidancePartValidity[part] = false;
            guidancePartShouldClear[part] = true;
            guidancePartErrors[part] = 0f;
            guidancePartSeverities[part] = 0f;
            guidancePartScores[part] = 100f;
            guidancePartRawScores[part] = 100f;
            guidancePartThresholds[part] = 0f;
        }
    }

    public float GuidanceHighlightBelowScore =>
        guidanceHighlightBelowScore;
    public float GuidanceClearAtOrAboveScore =>
        Mathf.Max(guidanceHighlightBelowScore, guidanceClearAtOrAboveScore);
    public float GuidanceScoreLeniency => guidanceScoreLeniency;
    public IReadOnlyDictionary<BodyPart, float> CurrentGuidancePartErrors =>
        guidancePartErrors;
    public IReadOnlyDictionary<BodyPart, float> CurrentGuidancePartRawScores =>
        guidancePartRawScores;
    public IReadOnlyDictionary<BodyPart, float> CurrentGuidancePartScores =>
        guidancePartScores;
    public IReadOnlyDictionary<BodyPart, float> CurrentGuidancePartThresholds =>
        guidancePartThresholds;
    public IReadOnlyDictionary<BodyPart, float> CurrentGuidancePartSeverities =>
        guidancePartSeverities;
    public IReadOnlyDictionary<BodyPart, bool> CurrentGuidancePartValidity =>
        guidancePartValidity;
#if UNITY_EDITOR || DEVELOPMENT_BUILD
    private void ReportGuidanceChange()
    {
        bool unchanged = hasReportedGuidanceState &&
            lastReportedGuidanceParts.Count == guidancePartsToHighlight.Count;
        if (unchanged)
        {
            foreach (BodyPart part in guidancePartsToHighlight)
            {
                if (!lastReportedGuidanceParts.Contains(part))
                {
                    unchanged = false;
                    break;
                }
            }
        }

        if (unchanged)
        {
            return;
        }

        hasReportedGuidanceState = true;
        lastReportedGuidanceParts.Clear();
        foreach (BodyPart part in guidancePartsToHighlight)
        {
            lastReportedGuidanceParts.Add(part);
        }

        string highlighted = guidancePartsToHighlight.Count > 0
            ? string.Join(", ", guidancePartsToHighlight)
            : "None";
        string errors = string.Join(
            ", ",
            guidancePartErrors.Select(pair =>
                $"{pair.Key}=valid {guidancePartValidity[pair.Key]}, " +
                $"worst error {pair.Value:0.0}\u00b0, " +
                $"severity {guidancePartSeverities[pair.Key]:0.00}"));
        Debug.Log(
            $"[PoseGuidance] phase={currentState}, highlighted={highlighted}, " +
            $"exercise={guidanceExercise}, diagnostics: {errors}",
            this);
    }
#endif
}
