using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

[DefaultExecutionOrder(20000)]
public class PracticeSessionController : MonoBehaviour
{
    private const float DirectionLockDeadZoneDegrees = 0.25f;
    public const float DefaultMaximumLyingPlaneAlignmentOffset = 0.1f;
    private static readonly HumanBodyBones[] LyingPlaneReferenceBones =
    {
        HumanBodyBones.Hips,
        HumanBodyBones.Spine,
        HumanBodyBones.Chest,
        HumanBodyBones.UpperChest
    };

    // Practice mode drives the SEGMENTED coach (pauses at each segment-end pose
    // for IMU scoring), not the continuous gaming/demo CoachActionController.
    [SerializeField] private SegmentedCoachController coachActionController;
    [SerializeField] private MotionRecorder motionRecorder;

    [Header("Difficulty Selection")]
    [Tooltip("Legacy threshold panel retained only as a serialized settings source. " +
        "The panel stays disabled while coach mode uses the white relaxed preset.")]
    [SerializeField] private DifficultySelectionUI difficultySelectionUI;
    [SerializeField] private DifficultySettings difficultySettings;
    [SerializeField] private DifficultyLevel defaultDifficulty = DifficultyLevel.Relaxed;
    [SerializeField] private PoseScorer poseScorer;
    [SerializeField] private DeadBugStartMenuController presentationController;

    [Header("Play Button Integration")]
    [Tooltip("场景中的播放按钮（可选）- 如果配置，会自动添加点击事件")]
    [SerializeField] private Button scenePlayButton;

    [Header("Coach Mode Introduction")]
    [Tooltip("Serializes the coach, user-model and ready-countdown voice prompts.")]
    [SerializeField] private VoicePromptManager voicePromptManager;
    [Tooltip("The world-space Coach label that follows the coach avatar's visibility.")]
    [SerializeField] private GameObject coachLabel;

    [Header("Avatar Direction")]
    [Tooltip("Keep the user avatar's rendered direction aligned with the coach " +
        "inside their shared camera-facing presentation space, independent of " +
        "the tracked person's room heading.")]
    [SerializeField] private bool alignUserDirectionToCoachOnStart = true;
    [Tooltip("Wait until the tracked user's torso is in the horizontal exercise " +
        "preparation pose before changing the rendered avatar direction. This " +
        "prevents a standing user from being forced into the coach's lying pose.")]
    [SerializeField] private bool requirePreparationPoseBeforeDirectionLock = true;
    [Tooltip("Largest torso tilt away from horizontal that still counts as a " +
        "ready Dead Bug or Bird Dog preparation pose.")]
    [Range(5f, 60f)]
    [SerializeField] private float preparationMaximumTorsoTiltDegrees = 45f;
    [Tooltip("Minimum overhead hand reach for Dead Bug, measured in torso lengths.")]
    [Min(0f)]
    [SerializeField] private float preparationMinimumHandReachRatio = 0.1f;
    [Tooltip("Minimum downward hand/knee reach for Bird Dog, measured in torso lengths.")]
    [Min(0f)]
    [SerializeField] private float preparationMinimumSupportDropRatio = 0.1f;
    [Tooltip("How long the relaxed preparation pose must remain detected before " +
        "direction alignment is captured.")]
    [Min(0f)]
    [SerializeField] private float preparationPoseHoldSeconds = 0.3f;
    [Tooltip("Brief pose-tracking dropouts shorter than this do not discard the " +
        "accumulated preparation confirmation.")]
    [Min(0f)]
    [SerializeField] private float preparationPoseGraceSeconds = 0.15f;
    [Tooltip("The VMC receiver that drives the user avatar. It is resolved " +
        "automatically from MotionRecorder when left empty.")]
    [SerializeField] private AvatarMotionSource userTrackerReceiver;
    [Tooltip("Maximum time without Humanoid bone packets before Tracker data " +
        "is considered unavailable. Direction lock never runs without it.")]
    [Min(0f)]
    [SerializeField] private float trackerDataTimeoutSeconds = 0.35f;
    [Tooltip("Maximum one-time vertical correction applied to the coach after " +
        "both lying torso planes are measured. This keeps different avatar rigs " +
        "on the same visible floor without affecting the standing introduction.")]
    [Min(0f)]
    [SerializeField] private float maximumLyingPlaneAlignmentOffset =
        DefaultMaximumLyingPlaneAlignmentOffset;

    [Header("Completion Controls")]
    [Tooltip("Wait for the tracked user's torso to become upright before moving " +
        "the four completion controls into the current headset view.")]
    [SerializeField] private bool requireUprightPoseForCompletionControls = true;
    [Tooltip("Largest torso tilt away from gravity-up that still counts as " +
        "upright for completion interaction.")]
    [Range(5f, 60f)]
    [SerializeField] private float completionMaximumTorsoTiltDegrees = 35f;
    [Tooltip("How long the user's torso must remain upright before the controls " +
        "are recentered.")]
    [Min(0f)]
    [SerializeField] private float completionUprightHoldSeconds = 0.4f;

    // 是否已经配置过难度
    private bool difficultyConfigured;
    private bool practiceRunning;
    private Transform avatarDirectionReference;
    private bool preparationDirectionAlignmentActive;
    private bool hasAlignedDuringPreparation;
    private bool hasLockedUserRotation;
    private float preparationPoseReadySeconds;
    private float preparationPoseMissSeconds;
    private float secondsSinceFreshTrackerBones = float.PositiveInfinity;
    private bool waitingForCompletionUprightPose;
    private float completionUprightReadySeconds;
    private Quaternion lockedUserRotationInReference = Quaternion.identity;
    private Vector3 lockedCoachHeadwardInReference = Vector3.forward;
    private float lockedCoachVerticalAlignmentOffset;
    private Transform directionLockedUserRoot;
    private Transform directionLockedUserOriginalParent;
    private Quaternion userLocalRotationBeforeDirectionLock = Quaternion.identity;
    private bool hasUserRotationBeforeDirectionLock;
    private bool trainingPlaybackStarted;
    private Renderer[] introductionCoachRenderers = System.Array.Empty<Renderer>();
    private Renderer[] introductionUserRenderers = System.Array.Empty<Renderer>();
    private readonly Dictionary<Renderer, bool> introductionRendererStates =
        new Dictionary<Renderer, bool>();
    private bool introductionCoachLabelStateCaptured;
    private bool introductionCoachLabelWasActive;
    private Transform authoredScaleUserRoot;
    private Vector3 authoredUserLocalScale;
    private bool hasAuthoredUserLocalScale;

    public bool IsLyingOverheadPresentationReady =>
        ShouldUseLyingOverheadPresentation(
            practiceRunning,
            hasLockedUserRotation,
            coachActionController != null,
            coachActionController != null
                ? coachActionController.CurrentSegmentKind
                : TrainingSegmentKind.Preparation);
    public Transform LyingPresentationAnchor =>
        motionRecorder != null && motionRecorder.CoachAnimator != null
            ? motionRecorder.CoachAnimator.transform
            : null;
    public float LyingCoachVerticalAlignmentOffset =>
        hasLockedUserRotation
            ? lockedCoachVerticalAlignmentOffset
            : 0f;

    public static bool ShouldUseLyingOverheadPresentation(
        bool isPracticeRunning,
        bool hasLockedReadyUserDirection,
        bool hasCoachPlayback,
        TrainingSegmentKind currentSegmentKind)
    {
        return isPracticeRunning &&
            hasLockedReadyUserDirection &&
            hasCoachPlayback &&
            currentSegmentKind != TrainingSegmentKind.Finish;
    }

    private void Awake()
    {
        // 自动查找组件
        if (difficultySelectionUI == null)
        {
            difficultySelectionUI = FindObjectOfType<DifficultySelectionUI>(true);
        }

        if (poseScorer == null)
        {
            poseScorer = FindObjectOfType<PoseScorer>();
        }

        if (difficultySettings == null && difficultySelectionUI != null)
        {
            difficultySettings = difficultySelectionUI.Settings;
        }

        ApplyDefaultDifficulty();
        DisableLegacyDifficultySelectionPanel();

        ResolvePresentationController();
        ResolveVoicePromptManager();
        ResolveCoachLabel();
        ResolveUserTrackerReceiver();
        CaptureAuthoredUserScale();
        DisableTrackerRootScaleSynchronization();

        if (coachActionController != null)
        {
            coachActionController.OnAllComplete += HandlePracticeCompleted;
            coachActionController.OnSegmentStarted +=
                HandleDirectionLockSegmentStarted;
            coachActionController.OnSegmentHold +=
                HandleDirectionLockSegmentHold;
        }

        // 如果配置了场景播放按钮，绑定点击事件
        if (scenePlayButton != null)
        {
            scenePlayButton.onClick.AddListener(OnScenePlayButtonClicked);
        }
    }

    private void OnDestroy()
    {
        CancelPracticeIntroduction();
        RestoreUserRotationBeforeDirectionLock();

        if (coachActionController != null)
        {
            coachActionController.OnAllComplete -= HandlePracticeCompleted;
            coachActionController.OnSegmentStarted -=
                HandleDirectionLockSegmentStarted;
            coachActionController.OnSegmentHold -=
                HandleDirectionLockSegmentHold;
        }

        if (scenePlayButton != null)
        {
            scenePlayButton.onClick.RemoveListener(OnScenePlayButtonClicked);
        }
    }

#if UNITY_EDITOR
    private void Update()
    {
        // Fast local loop for the VMC simulator. Production/XR keeps the normal
        // Play -> difficulty selection -> automatic training flow.
        if (Input.GetKeyDown(KeyCode.F8) && !practiceRunning)
        {
            StartPracticeDirectly();
        }
        else if (Input.GetKeyDown(KeyCode.F7) && practiceRunning)
        {
            poseScorer?.BeginEditorGuidanceCheckpoint();
        }
        else if (Input.GetKeyDown(KeyCode.F9) && practiceRunning)
        {
            StopPractice();
        }
    }
#endif

    private void LateUpdate()
    {
        PreserveAuthoredUserScale();
        UpdateUserTrackerFreshness(Time.unscaledDeltaTime);
        UpdateCompletionPostureGate(Time.unscaledDeltaTime);

        if (!practiceRunning || !alignUserDirectionToCoachOnStart)
        {
            return;
        }

        if (preparationDirectionAlignmentActive && !hasLockedUserRotation)
        {
            TryLockReadyUserDirection(Time.unscaledDeltaTime, false);
        }

        if (hasLockedUserRotation)
        {
            EnforceLockedUserRotation();
        }
    }

    private void CaptureAuthoredUserScale()
    {
        Animator userAnimator = motionRecorder != null
            ? motionRecorder.UserAnimator
            : null;
        if (userAnimator == null)
        {
            return;
        }

        authoredScaleUserRoot = userAnimator.transform;
        authoredUserLocalScale = authoredScaleUserRoot.localScale;
        hasAuthoredUserLocalScale = true;
    }

    private void PreserveAuthoredUserScale()
    {
        DisableTrackerRootScaleSynchronization();
        if (!hasAuthoredUserLocalScale || authoredScaleUserRoot == null)
        {
            CaptureAuthoredUserScale();
        }

        if (hasAuthoredUserLocalScale && authoredScaleUserRoot != null)
        {
            authoredScaleUserRoot.localScale = authoredUserLocalScale;
        }
    }

    private void DisableTrackerRootScaleSynchronization()
    {
        if (userTrackerReceiver == null)
        {
            ResolveUserTrackerReceiver();
        }

        if (userTrackerReceiver == null)
        {
            return;
        }

        userTrackerReceiver.RootScaleOffsetSynchronize = false;
        userTrackerReceiver.SyncCalibrationModeWithScaleOffsetSynchronize = false;
    }

    /// <summary>
    /// 场景播放按钮点击事件
    /// </summary>
    private void OnScenePlayButtonClicked()
    {
        if (!difficultyConfigured)
        {
            // 第一次点击：显示难度选择界面
            ApplyDefaultDifficulty();
            StartPracticeDirectly();
        }
        else
        {
            // 已配置难度，直接开始练习
            StartPracticeDirectly();
        }
    }

    /// <summary>
    /// 显示难度选择界面
    /// </summary>
    private void ApplyDefaultDifficulty()
    {
        DifficultyConfig config = difficultySettings != null
            ? difficultySettings.GetConfig(defaultDifficulty)
            : DifficultySettings.CreateBuiltInConfig(defaultDifficulty);
        if (poseScorer != null)
        {
            poseScorer.SetStabilityDifficulty(config);
            poseScorer.SetScoringDifficulty(config);
        }

        DifficultySettings.CurrentDifficulty = defaultDifficulty;
        difficultyConfigured = true;
        Debug.Log(
            $"PracticeSessionController: coach mode uses the fixed " +
            $"{DifficultySettings.GetDifficultyName(defaultDifficulty)} threshold preset.",
            this);
    }

    /// <summary>
    /// 难度确认后的回调 - 应用难度设置并开始练习
    /// </summary>
    private void OnDifficultyConfirmed()
    {
        if (difficultySelectionUI != null && poseScorer != null)
        {
            // 获取用户选择的难度配置
            DifficultyConfig stabilityConfig = difficultySelectionUI.GetStabilityConfig();
            DifficultyConfig scoringConfig = difficultySelectionUI.GetScoringConfig();

            // 应用到PoseScorer
            poseScorer.SetStabilityDifficulty(stabilityConfig);
            poseScorer.SetScoringDifficulty(scoringConfig);

            Debug.Log($"PracticeSessionController: 难度已应用 - " +
                      $"稳定性={DifficultySettings.GetDifficultyName(difficultySelectionUI.GetSelectedStabilityLevel())}, " +
                      $"评分={DifficultySettings.GetDifficultyName(difficultySelectionUI.GetSelectedScoringLevel())}");
        }

        // 标记已配置难度
        difficultyConfigured = true;

        // Difficulty selection only configures the session. The user starts the
        // coach demonstration explicitly with the Play button.
    }

    /// <summary>
    /// 开始练习（可被外部UI按钮调用）
    /// </summary>
    public void StartPractice()
    {
        if (!difficultyConfigured)
        {
            // 还没配置难度，显示难度选择界面
            ApplyDefaultDifficulty();
            StartPracticeDirectly();
        }
        else
        {
            // 已配置难度，直接开始练习
            StartPracticeDirectly();
        }
    }

    /// <summary>
    /// 直接开始练习（不显示难度选择界面）
    /// </summary>
    private void StartPracticeDirectly()
    {
        if (practiceRunning)
        {
            return;
        }

        if (motionRecorder == null || coachActionController == null)
        {
            Debug.LogError("PracticeSessionController references are missing.");
            return;
        }

        ResolvePresentationController();
        presentationController?.BeginPracticePlaybackPresentation();
        ResolveVoicePromptManager();

        CancelCompletionPostureGate();
        practiceRunning = true;
        trainingPlaybackStarted = false;
        Debug.Log(
            "PracticeSessionController: starting coach-mode introduction.",
            this);
        ResetUserDirectionLock();
        SpineFlowTrainingVolume trainingVolume =
            SpineFlowTrainingSession.GetPlannedTrainingVolume(
                "coach",
                coachActionController.TotalSets,
                coachActionController.RepetitionsPerSet);
        coachActionController.SetTrainingVolume(
            trainingVolume.sets,
            trainingVolume.repetitionsPerSet);

        BeginPracticeIntroduction();
    }

    private void BeginPracticeIntroduction()
    {
        CaptureIntroductionRendererStates();

        if (voicePromptManager != null)
        {
            voicePromptManager.StopAll();
            if (voicePromptManager.PlayPracticeIntroduction(
                ShowCoachIntroductionStage,
                ShowUserIntroductionStage,
                ShowBothIntroductionStage,
                BeginFormalPracticePlayback))
            {
                return;
            }
        }

        Debug.LogWarning(
            "PracticeSessionController could not start the voice introduction; " +
            "formal playback will begin with both avatars visible.",
            this);
        ShowBothIntroductionStage();
        BeginFormalPracticePlayback();
    }

    private void BeginFormalPracticePlayback()
    {
        if (!practiceRunning || trainingPlaybackStarted)
        {
            return;
        }

        RestoreIntroductionRendererStates();
        trainingPlaybackStarted = true;
        Debug.Log(
            "PracticeSessionController: introduction complete; starting formal training.",
            this);
        SpineFlowTrainingSession.BeginCoachTraining();
        if (poseScorer != null)
        {
            poseScorer.BeginTrainingMeasurement();
        }

        // The coach backend must validate and accept the action before a
        // replay session exists. ReplayRecorder then captures frame zero in
        // LateUpdate, after this frame's Animator/tracking evaluation.
        if (!coachActionController.TryPlayAction1(out string actionStartError))
        {
            practiceRunning = false;
            trainingPlaybackStarted = false;
            poseScorer?.StopTrainingMeasurement();
            Debug.LogError(
                $"Practice action failed to start; replay was not created: {actionStartError}",
                this);
            return;
        }

        if (!motionRecorder.TryBeginRecording(out string recordingError))
        {
            practiceRunning = false;
            trainingPlaybackStarted = false;
            poseScorer?.StopTrainingMeasurement();
            coachActionController.ReturnIdle();
            Debug.LogError(
                $"Practice recording failed to start: {recordingError}",
                this);
            return;
        }

        poseScorer?.AnnounceInitialFollowInstruction();
    }

    private void CaptureIntroductionRendererStates()
    {
        RestoreIntroductionRendererStates();

        Animator coachAnimator = motionRecorder != null
            ? motionRecorder.CoachAnimator
            : null;
        Animator userAnimator = motionRecorder != null
            ? motionRecorder.UserAnimator
            : null;
        introductionCoachRenderers = GetAvatarRenderers(coachAnimator);
        introductionUserRenderers = GetAvatarRenderers(userAnimator);

        RememberRendererStates(introductionCoachRenderers);
        RememberRendererStates(introductionUserRenderers);
        ResolveCoachLabel();
        if (coachLabel != null)
        {
            introductionCoachLabelWasActive = coachLabel.activeSelf;
            introductionCoachLabelStateCaptured = true;
        }
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
        SetIntroductionCoachLabelVisible(true);
    }

    private void ShowUserIntroductionStage()
    {
        SetIntroductionAvatarVisible(introductionCoachRenderers, false);
        SetIntroductionAvatarVisible(introductionUserRenderers, true);
        SetIntroductionCoachLabelVisible(false);
    }

    private void ShowBothIntroductionStage()
    {
        SetIntroductionAvatarVisible(introductionCoachRenderers, true);
        SetIntroductionAvatarVisible(introductionUserRenderers, true);
        SetIntroductionCoachLabelVisible(true);
    }

    private void SetIntroductionCoachLabelVisible(bool visible)
    {
        if (coachLabel != null && introductionCoachLabelStateCaptured)
        {
            coachLabel.SetActive(visible && introductionCoachLabelWasActive);
        }
    }

    private void SetIntroductionAvatarVisible(
        Renderer[] renderers,
        bool visible)
    {
        foreach (Renderer avatarRenderer in renderers)
        {
            if (avatarRenderer == null ||
                !introductionRendererStates.TryGetValue(
                    avatarRenderer,
                    out bool wasEnabled))
            {
                continue;
            }

            avatarRenderer.enabled = visible && wasEnabled;
        }
    }

    private void RestoreIntroductionRendererStates()
    {
        foreach (KeyValuePair<Renderer, bool> rendererState in
            introductionRendererStates)
        {
            if (rendererState.Key != null)
            {
                rendererState.Key.enabled = rendererState.Value;
            }
        }

        introductionRendererStates.Clear();
        introductionCoachRenderers = System.Array.Empty<Renderer>();
        introductionUserRenderers = System.Array.Empty<Renderer>();

        if (coachLabel != null && introductionCoachLabelStateCaptured)
        {
            coachLabel.SetActive(introductionCoachLabelWasActive);
        }

        introductionCoachLabelStateCaptured = false;
        introductionCoachLabelWasActive = false;
    }

    private void CancelPracticeIntroduction()
    {
        voicePromptManager?.CancelPracticeIntroduction();
        RestoreIntroductionRendererStates();
    }

    private void AlignUserDirectionToCoach()
    {
        if (!alignUserDirectionToCoachOnStart)
        {
            return;
        }

        preparationDirectionAlignmentActive = true;
        preparationPoseReadySeconds = 0f;
        preparationPoseMissSeconds = 0f;
    }

    private void HandleDirectionLockSegmentStarted(
        PracticeSegmentPlaybackInfo segment)
    {
        if (!practiceRunning || !alignUserDirectionToCoachOnStart)
        {
            return;
        }

        if (segment.segmentKind == TrainingSegmentKind.Preparation)
        {
            // Preparation can be entered again between sets. Keep the first
            // stable display-space lock for the whole session so the avatars
            // never flip when the tracked person shifts on the floor.
            if (!hasLockedUserRotation)
            {
                AlignUserDirectionToCoach();
            }
            return;
        }

        if (!preparationDirectionAlignmentActive)
        {
            return;
        }

        bool alignmentSucceededAtSegmentStart =
            hasLockedUserRotation || TryLockReadyUserDirection(0f, true);
        preparationDirectionAlignmentActive =
            !alignmentSucceededAtSegmentStart;
        if (hasLockedUserRotation)
        {
            EnforceLockedUserRotation();
        }

        if (!alignmentSucceededAtSegmentStart && !hasAlignedDuringPreparation)
        {
            Debug.LogWarning(
                "PracticeSessionController did not detect the relaxed exercise " +
                "preparation pose before the formal action started and will " +
                "keep monitoring until the user is ready.",
                this);
        }
    }

    private void HandleDirectionLockSegmentHold(SegmentHoldInfo segment)
    {
        if (practiceRunning &&
            alignUserDirectionToCoachOnStart &&
            segment.segmentKind == TrainingSegmentKind.Preparation)
        {
            AlignUserDirectionToCoach();
        }
    }

    private bool TryLockReadyUserDirection(float deltaTime, bool reportResult)
    {
        if (motionRecorder == null || motionRecorder.UserAnimator == null)
        {
            return false;
        }

        bool hasFreshUserTrackerData = HasFreshUserTrackerData();

        if (requirePreparationPoseBeforeDirectionLock)
        {
            Animator userAnimator = motionRecorder.UserAnimator;
            Animator coachAnimator = motionRecorder.CoachAnimator;
            ResolveAvatarDirectionReference(userAnimator, coachAnimator);
            bool poseReady = IsUserInRelaxedDirectionLockPose(userAnimator);
            if (!CanUseUserPoseForDirectionLock(
                    hasFreshUserTrackerData,
                    poseReady))
            {
                UpdateDirectionPreparationReadiness(false, deltaTime);
                if (reportResult)
                {
                    Debug.Log(
                        "PracticeSessionController left the user avatar " +
                        "transform untouched because neither fresh Tracker " +
                        "telemetry nor a live preparation pose is available.",
                        this);
                }

                return false;
            }

            UpdateDirectionPreparationReadiness(poseReady, deltaTime);
            if (!poseReady || preparationPoseReadySeconds + Mathf.Epsilon <
                Mathf.Max(0f, preparationPoseHoldSeconds))
            {
                return false;
            }
        }
        else if (!hasFreshUserTrackerData)
        {
            if (reportResult)
            {
                Debug.Log(
                    "PracticeSessionController left the user avatar transform " +
                    "untouched because no fresh Humanoid Tracker data is " +
                    "available.",
                    this);
            }

            return false;
        }

        bool alignmentSucceeded = ApplyUserDirectionLock(reportResult);
        if (alignmentSucceeded)
        {
            Debug.Log(
                "PracticeSessionController detected the relaxed exercise " +
                "preparation pose, then aligned and locked the user " +
                "display direction to the coach.",
                this);
        }

        return alignmentSucceeded;
    }

    private bool IsUserInRelaxedDirectionLockPose(Animator userAnimator)
    {
        bool isDeadBug = coachActionController != null &&
            coachActionController.UsesDeadBugPreparationPose;
        if (isDeadBug)
        {
            return PoseScorer.TryCapturePreparationPose(
                    userAnimator,
                    ResolveWorldUp(),
                    out DeadBugPreparationPoseGeometry deadBugPose) &&
                PoseScorer.MeetsRelaxedDeadBugDirectionLockPose(
                    deadBugPose,
                    preparationMaximumTorsoTiltDegrees,
                    preparationMinimumHandReachRatio);
        }

        return PoseScorer.TryCapturePreparationPose(
                userAnimator,
                ResolveWorldUp(),
                out DeadBugPreparationPoseGeometry birdDogPose) &&
            PoseScorer.MeetsRelaxedBirdDogPreparationPose(
                birdDogPose,
                Mathf.Min(preparationMaximumTorsoTiltDegrees, 35f),
                preparationMinimumSupportDropRatio);
    }

    private void UpdateDirectionPreparationReadiness(
        bool poseReady,
        float deltaTime)
    {
        float safeDeltaTime = Mathf.Max(0f, deltaTime);
        if (poseReady)
        {
            preparationPoseReadySeconds += safeDeltaTime;
            preparationPoseMissSeconds = 0f;
            return;
        }

        preparationPoseMissSeconds += safeDeltaTime;
        if (preparationPoseMissSeconds + Mathf.Epsilon >=
            Mathf.Max(0f, preparationPoseGraceSeconds))
        {
            preparationPoseReadySeconds = 0f;
        }
    }

    private bool ApplyUserDirectionLock(bool reportResult)
    {
        if (motionRecorder == null)
        {
            return false;
        }

        Animator userAnimator = motionRecorder.UserAnimator;
        Animator coachAnimator = motionRecorder.CoachAnimator;
        if (userAnimator == null || coachAnimator == null)
        {
            return false;
        }

        ResolveAvatarDirectionReference(userAnimator, coachAnimator);

        if (avatarDirectionReference == null ||
            !TryGetBodyHeadwardDirection(
                userAnimator,
                out Vector3 userHeadwardWorld) ||
            !TryGetBodyHeadwardDirection(
                coachAnimator,
                out Vector3 coachHeadwardWorld))
        {
            if (reportResult)
            {
                Debug.LogWarning(
                    "PracticeSessionController could not lock the user display " +
                    "direction because a reliable hip-to-torso direction is " +
                    "unavailable.",
                    this);
            }

            return false;
        }

        Vector3 userHeadwardInReference =
            avatarDirectionReference.InverseTransformDirection(
                userHeadwardWorld);
        Vector3 coachHeadwardInReference =
            avatarDirectionReference.InverseTransformDirection(
                coachHeadwardWorld);
        if (!TryCalculateYawAlignmentDegrees(
                userHeadwardInReference,
                coachHeadwardInReference,
                Vector3.up,
                out float yawCorrectionDegrees))
        {
            if (reportResult)
            {
                Debug.LogWarning(
                    "PracticeSessionController could not lock the user display " +
                    "direction because the lying head-to-feet direction is " +
                    "not horizontal enough to resolve.",
                    this);
            }

            return false;
        }

        Transform userRoot = userAnimator.transform;
        CaptureUserRotationBeforeDirectionLock(userRoot);
        userRoot.rotation = Quaternion.AngleAxis(
            Mathf.Abs(yawCorrectionDegrees) > DirectionLockDeadZoneDegrees
                ? yawCorrectionDegrees
                : 0f,
            avatarDirectionReference.up) * userRoot.rotation;

        Quaternion referenceRotation = avatarDirectionReference.rotation;
        lockedUserRotationInReference =
            Quaternion.Inverse(referenceRotation) *
            userRoot.rotation;
        lockedCoachHeadwardInReference = Vector3.ProjectOnPlane(
            coachHeadwardInReference,
            Vector3.up).normalized;
        lockedCoachVerticalAlignmentOffset =
            TryGetLyingBodyPlaneHeightInReference(
                userAnimator,
                avatarDirectionReference,
                out float userPlaneHeight) &&
            TryGetLyingBodyPlaneHeightInReference(
                coachAnimator,
                avatarDirectionReference,
                out float coachPlaneHeight)
                ? CalculateCoachVerticalAlignmentOffset(
                    userPlaneHeight,
                    coachPlaneHeight,
                    maximumLyingPlaneAlignmentOffset)
                : 0f;
        hasLockedUserRotation = true;
        hasAlignedDuringPreparation = true;
        preparationDirectionAlignmentActive = false;

        if (reportResult)
        {
            Debug.Log(
                "PracticeSessionController aligned only the user's horizontal " +
                "lying direction to the coach and captured a stable " +
                "display-space rotation lock. Coach lying-plane correction: " +
                $"{lockedCoachVerticalAlignmentOffset:F3} m.",
                this);
        }

        return true;
    }

    private Transform ResolveAvatarDirectionReference(
        Animator userAnimator,
        Animator coachAnimator)
    {
        if (avatarDirectionReference != null)
        {
            return avatarDirectionReference;
        }

        avatarDirectionReference = FindSharedDisplayReference(
            userAnimator != null ? userAnimator.transform : null,
            coachAnimator != null ? coachAnimator.transform : null);
        if (avatarDirectionReference == null)
        {
            Camera activeCamera = Camera.main;
            avatarDirectionReference =
                activeCamera != null ? activeCamera.transform : null;
        }

        return avatarDirectionReference;
    }

    private bool EnforceLockedUserRotation()
    {
        if (!hasLockedUserRotation ||
            avatarDirectionReference == null ||
            motionRecorder == null ||
            motionRecorder.UserAnimator == null)
        {
            return false;
        }

        Transform userRoot = motionRecorder.UserAnimator.transform;
        userRoot.rotation =
            avatarDirectionReference.rotation *
            lockedUserRotationInReference;

        // A real VMC/IMU stream carries the person's room heading in the Hips
        // bone, not in the avatar root. Restoring only the root therefore does
        // not hold the displayed direction when a person lies down at another
        // heading. Re-evaluate the latest tracked torso after restoring the
        // authored display root and cancel only its horizontal yaw. Tracker
        // pitch/roll and all exercise articulation remain untouched.
        if (TryGetBodyHeadwardDirection(
                motionRecorder.UserAnimator,
                out Vector3 userHeadwardWorld) &&
            TryCalculateYawAlignmentDegrees(
                avatarDirectionReference.InverseTransformDirection(
                    userHeadwardWorld),
                lockedCoachHeadwardInReference,
                Vector3.up,
                out float yawCorrectionDegrees) &&
            Mathf.Abs(yawCorrectionDegrees) > DirectionLockDeadZoneDegrees)
        {
            userRoot.rotation = Quaternion.AngleAxis(
                yawCorrectionDegrees,
                avatarDirectionReference.up) * userRoot.rotation;
        }

        return true;
    }

    public static float CalculateYawAlignmentDegrees(
        Vector3 userBodyRight,
        Vector3 coachBodyRight)
    {
        return CalculateYawAlignmentDegrees(
            userBodyRight,
            coachBodyRight,
            Vector3.up);
    }

    public static float CalculateYawAlignmentDegrees(
        Vector3 userBodyRight,
        Vector3 coachBodyRight,
        Vector3 worldUp)
    {
        return TryCalculateYawAlignmentDegrees(
            userBodyRight,
            coachBodyRight,
            worldUp,
            out float correctionDegrees)
                ? correctionDegrees
                : 0f;
    }

    public static bool TryCalculateYawAlignmentDegrees(
        Vector3 userDirection,
        Vector3 coachDirection,
        Vector3 rotationAxis,
        out float correctionDegrees)
    {
        correctionDegrees = 0f;
        Vector3 safeUp = rotationAxis.sqrMagnitude > 0.0001f
            ? rotationAxis.normalized
            : Vector3.up;
        Vector3 horizontalUserRight =
            Vector3.ProjectOnPlane(userDirection, safeUp);
        Vector3 horizontalCoachRight =
            Vector3.ProjectOnPlane(coachDirection, safeUp);
        if (horizontalUserRight.sqrMagnitude < 0.0001f ||
            horizontalCoachRight.sqrMagnitude < 0.0001f)
        {
            return false;
        }

        correctionDegrees = Vector3.SignedAngle(
            horizontalUserRight.normalized,
            horizontalCoachRight.normalized,
            safeUp);
        return true;
    }

    public static float CalculateCoachVerticalAlignmentOffset(
        float userPlaneHeight,
        float coachPlaneHeight,
        float maximumAbsoluteOffset)
    {
        float safeMaximum = Mathf.Max(0f, maximumAbsoluteOffset);
        return Mathf.Clamp(
            userPlaneHeight - coachPlaneHeight,
            -safeMaximum,
            safeMaximum);
    }

    private static bool TryGetLyingBodyPlaneHeightInReference(
        Animator animator,
        Transform reference,
        out float planeHeight)
    {
        planeHeight = 0f;
        if (animator == null || reference == null)
        {
            return false;
        }

        int measuredBoneCount = 0;
        foreach (HumanBodyBones bone in LyingPlaneReferenceBones)
        {
            Transform boneTransform = animator.GetBoneTransform(bone);
            if (boneTransform == null)
            {
                continue;
            }

            planeHeight += reference
                .InverseTransformPoint(boneTransform.position)
                .y;
            measuredBoneCount++;
        }

        if (measuredBoneCount == 0)
        {
            planeHeight = 0f;
            return false;
        }

        planeHeight /= measuredBoneCount;
        return true;
    }

    public static float CalculateStableYawCorrectionDegrees(
        Vector3 userBodyRight,
        Vector3 lockedCoachBodyRight,
        Vector3 worldUp)
    {
        float correction = CalculateYawAlignmentDegrees(
            userBodyRight,
            lockedCoachBodyRight,
            worldUp);
        return Mathf.Abs(correction) > DirectionLockDeadZoneDegrees
            ? correction
            : 0f;
    }

    public static float CalculateDisplaySpaceYawCorrectionDegrees(
        Vector3 userWorldRight,
        Vector3 lockedCoachRightInDisplaySpace,
        Quaternion displayRotation)
    {
        Vector3 userRightInDisplaySpace =
            Quaternion.Inverse(displayRotation) * userWorldRight;
        return CalculateStableYawCorrectionDegrees(
            userRightInDisplaySpace,
            lockedCoachRightInDisplaySpace,
            Vector3.up);
    }

    public static bool IsHorizontalPreparationPose(
        Vector3 bodyHeadward,
        Vector3 worldUp,
        float maximumTiltFromHorizontalDegrees)
    {
        if (bodyHeadward.sqrMagnitude < 0.0001f ||
            worldUp.sqrMagnitude < 0.0001f)
        {
            return false;
        }

        float verticalAlignment = Mathf.Abs(Vector3.Dot(
            bodyHeadward.normalized,
            worldUp.normalized));
        float tiltFromHorizontalDegrees =
            Mathf.Asin(Mathf.Clamp01(verticalAlignment)) * Mathf.Rad2Deg;
        return tiltFromHorizontalDegrees <=
            Mathf.Clamp(maximumTiltFromHorizontalDegrees, 0f, 90f);
    }

    public static bool IsUprightCompletionPose(
        Vector3 bodyHeadward,
        Vector3 worldUp,
        float maximumTiltFromUprightDegrees)
    {
        if (bodyHeadward.sqrMagnitude < 0.0001f ||
            worldUp.sqrMagnitude < 0.0001f)
        {
            return false;
        }

        float verticalAlignment = Mathf.Clamp(Vector3.Dot(
            bodyHeadward.normalized,
            worldUp.normalized), -1f, 1f);
        float tiltFromUprightDegrees =
            Mathf.Acos(verticalAlignment) * Mathf.Rad2Deg;
        return tiltFromUprightDegrees <=
            Mathf.Clamp(maximumTiltFromUprightDegrees, 0f, 90f);
    }

    public static float UpdatePreparationPoseReadySeconds(
        float previousReadySeconds,
        bool poseReady,
        float deltaTime)
    {
        return poseReady
            ? Mathf.Max(0f, previousReadySeconds) + Mathf.Max(0f, deltaTime)
            : 0f;
    }

    private static Vector3 ResolveWorldUp()
    {
        return Physics.gravity.sqrMagnitude > 0.0001f
            ? -Physics.gravity.normalized
            : Vector3.up;
    }

    private static bool TryGetBodyHeadwardDirection(
        Animator animator,
        out Vector3 bodyHeadward)
    {
        bodyHeadward = Vector3.zero;
        if (animator == null)
        {
            return false;
        }

        Transform hips = animator.GetBoneTransform(HumanBodyBones.Hips);
        Transform torso = animator.GetBoneTransform(HumanBodyBones.UpperChest) ??
            animator.GetBoneTransform(HumanBodyBones.Chest) ??
            animator.GetBoneTransform(HumanBodyBones.Head);
        if (hips == null || torso == null)
        {
            return false;
        }

        bodyHeadward = torso.position - hips.position;
        return bodyHeadward.sqrMagnitude >= 0.0001f;
    }

    private static bool TryGetAnatomicalFrameInReference(
        Animator animator,
        Transform displayReference,
        out Quaternion frame)
    {
        frame = Quaternion.identity;
        if (animator == null || displayReference == null)
        {
            return false;
        }

        Transform left =
            animator.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
        Transform right =
            animator.GetBoneTransform(HumanBodyBones.RightUpperLeg);
        Transform hips = animator.GetBoneTransform(HumanBodyBones.Hips);
        Transform torso = animator.GetBoneTransform(HumanBodyBones.UpperChest) ??
            animator.GetBoneTransform(HumanBodyBones.Chest) ??
            animator.GetBoneTransform(HumanBodyBones.Head);
        if (left == null || right == null || hips == null || torso == null)
        {
            return false;
        }

        Vector3 bodyRightInReference =
            displayReference.InverseTransformDirection(
                right.position - left.position);
        Vector3 bodyHeadwardInReference =
            displayReference.InverseTransformDirection(
                torso.position - hips.position);
        return TryBuildAnatomicalFrame(
            bodyRightInReference,
            bodyHeadwardInReference,
            out frame);
    }

    public static bool TryBuildAnatomicalFrame(
        Vector3 bodyRight,
        Vector3 bodyHeadward,
        out Quaternion frame)
    {
        frame = Quaternion.identity;
        if (bodyRight.sqrMagnitude < 0.0001f ||
            bodyHeadward.sqrMagnitude < 0.0001f)
        {
            return false;
        }

        Vector3 normalizedRight = bodyRight.normalized;
        Vector3 orthogonalHeadward =
            Vector3.ProjectOnPlane(bodyHeadward, normalizedRight);
        if (orthogonalHeadward.sqrMagnitude < 0.0001f)
        {
            return false;
        }

        orthogonalHeadward.Normalize();
        Vector3 bodyForward =
            Vector3.Cross(normalizedRight, orthogonalHeadward);
        if (bodyForward.sqrMagnitude < 0.0001f)
        {
            return false;
        }

        frame = Quaternion.LookRotation(
            bodyForward.normalized,
            orthogonalHeadward);
        return true;
    }

    private static Transform FindSharedDisplayReference(
        Transform userRoot,
        Transform coachRoot)
    {
        for (Transform userAncestor = userRoot != null
                 ? userRoot.parent
                 : null;
             userAncestor != null;
             userAncestor = userAncestor.parent)
        {
            for (Transform coachAncestor = coachRoot != null
                     ? coachRoot.parent
                     : null;
                 coachAncestor != null;
                 coachAncestor = coachAncestor.parent)
            {
                if (userAncestor == coachAncestor)
                {
                    return userAncestor;
                }
            }
        }

        return null;
    }

    public void StopPractice()
    {
        practiceRunning = false;
        trainingPlaybackStarted = false;
        CancelPracticeIntroduction();
        CancelCompletionPostureGate();
        ResetUserDirectionLock();
        poseScorer?.StopTrainingMeasurement();
        if (motionRecorder != null)
        {
            motionRecorder.FinishAndSave(false);
        }

        if (coachActionController != null)
        {
            coachActionController.ReturnIdle();
        }
    }

    private void HandlePracticeCompleted()
    {
        practiceRunning = false;
        trainingPlaybackStarted = false;
        preparationDirectionAlignmentActive = false;
        BeginCompletionPostureGate();
        if (motionRecorder != null)
        {
            motionRecorder.RequestFinish();
        }

        if (poseScorer != null)
        {
            SpineFlowTrainingSession.CompleteCoachTraining(
                poseScorer.SessionScores,
                poseScorer.MaximumUserMotionDegrees,
                poseScorer.MaximumCoachMotionDegrees,
                coachActionController != null
                    ? coachActionController.CompletedRepetitions
                    : 0);
        }
    }

    private void BeginCompletionPostureGate()
    {
        ResolvePresentationController();
        presentationController?.BeginPracticeCompletionPostureGate();
        completionUprightReadySeconds = 0f;

        if (!requireUprightPoseForCompletionControls)
        {
            waitingForCompletionUprightPose = false;
            ResetUserDirectionLock();
            presentationController?.NotifyPracticeCompletionUserUpright();
            return;
        }

        waitingForCompletionUprightPose = true;
    }

    private void UpdateCompletionPostureGate(float deltaTime)
    {
        if (!waitingForCompletionUprightPose)
        {
            return;
        }

        Animator userAnimator = motionRecorder != null
            ? motionRecorder.UserAnimator
            : null;
        bool upright =
            TryGetBodyHeadwardDirection(userAnimator, out Vector3 bodyHeadward) &&
            IsUprightCompletionPose(
                bodyHeadward,
                ResolveWorldUp(),
                completionMaximumTorsoTiltDegrees);
        completionUprightReadySeconds = UpdatePreparationPoseReadySeconds(
            completionUprightReadySeconds,
            upright,
            deltaTime);
        if (!upright || completionUprightReadySeconds + Mathf.Epsilon <
            Mathf.Max(0f, completionUprightHoldSeconds))
        {
            return;
        }

        waitingForCompletionUprightPose = false;
        ResetUserDirectionLock();
        ResolvePresentationController();
        presentationController?.NotifyPracticeCompletionUserUpright();
        Debug.Log(
            "PracticeSessionController detected a stable upright user and " +
            "released the completion controls into the current headset view.",
            this);
    }

    private void CancelCompletionPostureGate()
    {
        waitingForCompletionUprightPose = false;
        completionUprightReadySeconds = 0f;
    }

    private void UpdateUserTrackerFreshness(float deltaTime)
    {
        ResolveUserTrackerReceiver();
        if (!IsTrackerReceiverOperational(userTrackerReceiver))
        {
            secondsSinceFreshTrackerBones = float.PositiveInfinity;
            return;
        }

        if (userTrackerReceiver.LastBonePacketCounterInFrame > 0)
        {
            secondsSinceFreshTrackerBones = 0f;
            return;
        }

        if (!float.IsPositiveInfinity(secondsSinceFreshTrackerBones))
        {
            secondsSinceFreshTrackerBones += Mathf.Max(0f, deltaTime);
        }
    }

    private bool HasFreshUserTrackerData()
    {
        ResolveUserTrackerReceiver();
        return IsFreshTrackerBoneData(
            IsTrackerReceiverOperational(userTrackerReceiver),
            userTrackerReceiver != null
                ? userTrackerReceiver.LastBonePacketCounterInFrame
                : 0,
            secondsSinceFreshTrackerBones,
            trackerDataTimeoutSeconds);
    }

    public static bool IsFreshTrackerBoneData(
        bool receiverOperational,
        int bonePacketsInLatestFrame,
        float secondsSinceFreshBones,
        float timeoutSeconds)
    {
        return receiverOperational &&
            (bonePacketsInLatestFrame > 0 ||
             Mathf.Max(0f, secondsSinceFreshBones) <=
                  Mathf.Max(0f, timeoutSeconds));
    }

    public static bool CanUseUserPoseForDirectionLock(
        bool hasFreshTrackerTelemetry,
        bool exercisePreparationPoseReady)
    {
        // Some Android receiver paths update Humanoid bones without updating
        // the per-frame packet counter. A recognized live preparation pose is
        // sufficient evidence that the displayed rig can be aligned.
        return hasFreshTrackerTelemetry || exercisePreparationPoseReady;
    }

    private static bool IsTrackerReceiverOperational(
        AvatarMotionSource receiver)
    {
        return receiver != null &&
            receiver.isActiveAndEnabled &&
            !receiver.Freeze;
    }

    private void ResolveUserTrackerReceiver()
    {
        Animator userAnimator = motionRecorder != null
            ? motionRecorder.UserAnimator
            : null;
        if (userTrackerReceiver != null &&
            userTrackerReceiver.gameObject.scene == gameObject.scene &&
            ReceiverDrivesAnimator(userTrackerReceiver, userAnimator))
        {
            return;
        }

        userTrackerReceiver = null;
        if (userAnimator == null)
        {
            return;
        }

        foreach (AvatarMotionSource candidate in
            FindObjectsOfType<AvatarMotionSource>(true))
        {
            if (candidate.gameObject.scene == gameObject.scene &&
                ReceiverDrivesAnimator(candidate, userAnimator))
            {
                userTrackerReceiver = candidate;
                return;
            }
        }
    }

    private static bool ReceiverDrivesAnimator(
        AvatarMotionSource receiver,
        Animator animator)
    {
        if (receiver == null || animator == null)
        {
            return false;
        }

        GameObject model = receiver.Model;
        return model != null &&
            (model == animator.gameObject ||
             animator.transform.IsChildOf(model.transform));
    }

    private void ResolvePresentationController()
    {
        if (presentationController != null)
        {
            return;
        }

        DeadBugStartMenuController[] candidates =
            FindObjectsOfType<DeadBugStartMenuController>(true);
        foreach (DeadBugStartMenuController candidate in candidates)
        {
            if (candidate.gameObject.scene == gameObject.scene)
            {
                presentationController = candidate;
                return;
            }
        }
    }

    private void ResolveVoicePromptManager()
    {
        if (voicePromptManager != null)
        {
            return;
        }

        VoicePromptManager[] candidates =
            FindObjectsOfType<VoicePromptManager>(true);
        foreach (VoicePromptManager candidate in candidates)
        {
            if (candidate.gameObject.scene == gameObject.scene)
            {
                voicePromptManager = candidate;
                return;
            }
        }
    }

    private void ResolveCoachLabel()
    {
        if (coachLabel != null)
        {
            return;
        }

        Animator coachAnimator = motionRecorder != null
            ? motionRecorder.CoachAnimator
            : null;
        if (coachAnimator == null)
        {
            return;
        }

        foreach (Transform candidate in
            coachAnimator.GetComponentsInChildren<Transform>(true))
        {
            if (candidate.name == "CoachLabel")
            {
                coachLabel = candidate.gameObject;
                return;
            }
        }
    }

    private void CaptureUserRotationBeforeDirectionLock(Transform userRoot)
    {
        if (hasUserRotationBeforeDirectionLock || userRoot == null)
        {
            return;
        }

        directionLockedUserRoot = userRoot;
        directionLockedUserOriginalParent = userRoot.parent;
        userLocalRotationBeforeDirectionLock = userRoot.localRotation;
        hasUserRotationBeforeDirectionLock = true;
    }

    private void RestoreUserRotationBeforeDirectionLock()
    {
        if (!hasUserRotationBeforeDirectionLock ||
            directionLockedUserRoot == null)
        {
            ClearCapturedUserRotation();
            return;
        }

        // The practice avatar hierarchy is authored once and remains stable.
        // Only restore when it is still the same hierarchy, avoiding a write to
        // a replacement model loaded by a motion source during the session.
        if (directionLockedUserRoot.parent ==
            directionLockedUserOriginalParent)
        {
            directionLockedUserRoot.localRotation =
                userLocalRotationBeforeDirectionLock;
        }

        ClearCapturedUserRotation();
    }

    private void ClearCapturedUserRotation()
    {
        directionLockedUserRoot = null;
        directionLockedUserOriginalParent = null;
        userLocalRotationBeforeDirectionLock = Quaternion.identity;
        hasUserRotationBeforeDirectionLock = false;
    }

    private void ResetUserDirectionLock()
    {
        RestoreUserRotationBeforeDirectionLock();
        preparationDirectionAlignmentActive = false;
        hasAlignedDuringPreparation = false;
        hasLockedUserRotation = false;
        preparationPoseReadySeconds = 0f;
        preparationPoseMissSeconds = 0f;
        secondsSinceFreshTrackerBones = float.PositiveInfinity;
        lockedUserRotationInReference = Quaternion.identity;
        lockedCoachHeadwardInReference = Vector3.forward;
        lockedCoachVerticalAlignmentOffset = 0f;
        avatarDirectionReference = null;
    }

    private void DisableLegacyDifficultySelectionPanel()
    {
        if (difficultySelectionUI == null)
        {
            return;
        }

        Canvas[] selectionCanvases =
            difficultySelectionUI.GetComponentsInParent<Canvas>(true);
        foreach (Canvas selectionCanvas in selectionCanvases)
        {
            selectionCanvas.enabled = false;
        }

        difficultySelectionUI?.Hide();
    }

    /// <summary>
    /// 重置难度配置状态（用于重新选择难度）
    /// </summary>
    public void ResetDifficultyConfiguration()
    {
        ApplyDefaultDifficulty();
        DisableLegacyDifficultySelectionPanel();
    }

    public DifficultyLevel DefaultDifficulty => defaultDifficulty;
    public bool IsDifficultyConfigured => difficultyConfigured;
    public bool IsIntroductionPlaying =>
        practiceRunning && !trainingPlaybackStarted;
    public bool HasTrainingPlaybackStarted => trainingPlaybackStarted;
}
