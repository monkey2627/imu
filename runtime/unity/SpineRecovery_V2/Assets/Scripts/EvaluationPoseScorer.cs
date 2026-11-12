using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Self-contained live test harness for the zEvaluation scene. It uses the same
/// robust scoring engine as practice/game modes and builds a world-space HUD at
/// runtime so the test scene does not depend on production UI objects.
/// </summary>
public sealed class EvaluationPoseScorer : MonoBehaviour
{
    [Header("Scene References")]
    [SerializeField] private Animator coachAnimator;
    [SerializeField] private Animator userAnimator;
    [SerializeField] private CoachActionController coachController;

    [Header("Evaluation Scoring")]
    [SerializeField] private AnatomicalMotionScoringSettings anatomicalScoring =
        new AnatomicalMotionScoringSettings();

    [Header("Single-Pose Calibration")]
    [Tooltip("Minimum valid humanoid bones required to calibrate.")]
    [Min(3)] [SerializeField] private int minimumTrackedBones = 8;
    [Tooltip("How long the action-start pose must remain stable before sampling starts.")]
    [Min(0.1f)] [SerializeField] private float calibrationStableDuration = 0.5f;
    [Tooltip("Multi-frame sampling duration. No calibration is taken from a single frame.")]
    [Min(0.5f)] [SerializeField] private float calibrationSampleDuration = 1.2f;
    [Tooltip("Wearable motion below this speed is considered stable.")]
    [Min(1f)] [SerializeField] private float calibrationMotionLimit = 45f;
    [Tooltip("Fraction of moving bones tolerated as normal Tracker jitter.")]
    [Range(0f, 0.6f)] [SerializeField] private float allowedMovingBoneFraction = 0.4f;
    [Min(0.05f)] [SerializeField] private float textUpdateInterval = 0.1f;
    [SerializeField] private HumanBodyBones[] scoringBones =
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

    [Header("HUD Placement")]
    [SerializeField] private Vector3 hudCameraLocalPosition = new Vector3(-0.38f, 0.13f, 1.35f);
    [Min(0.0001f)] [SerializeField] private float hudScale = 0.0011f;

    private readonly Dictionary<HumanBodyBones, Vector3> coachCalibrationDirections =
        new Dictionary<HumanBodyBones, Vector3>();
    private readonly Dictionary<HumanBodyBones, Vector3> userCalibrationDirections =
        new Dictionary<HumanBodyBones, Vector3>();
    private readonly Dictionary<HumanBodyBones, Quaternion> previousCalibrationRotations =
        new Dictionary<HumanBodyBones, Quaternion>();
    private readonly Dictionary<Button, bool> calibrationBlockedButtons =
        new Dictionary<Button, bool>();

    private sealed class DirectionAccumulator
    {
        private Vector3 sum;
        public int Count { get; private set; }

        public void Add(Vector3 value)
        {
            if (value.sqrMagnitude < 0.5f) return;
            sum += value.normalized;
            Count++;
        }

        public Vector3 Average()
        {
            return Count > 0 && sum.sqrMagnitude > 1e-8f ? sum.normalized : Vector3.zero;
        }
    }

    private AnatomicalMotionScoringEngine scoringEngine;
    private TMP_Text scoreText;
    private TMP_Text feedbackText;
    private TMP_Text detailText;
    private TMP_Text stateText;
    private Button recalibrateButton;
    private Coroutine calibrationRoutine;
    private bool calibrated;
    private bool actionActive;
    private bool checkpointActive;
    private bool excludedTransitionActive;
    private bool waitingForActionStart;
    private float nextTextUpdateTime;
    private AnatomicalPoseScore currentResult;
    private float calibrationMotionSampleElapsed;
    private bool lastCalibrationStability = true;

    private void Awake()
    {
        ResolveReferences();
        scoringEngine = new AnatomicalMotionScoringEngine(anatomicalScoring);
        BuildHud();
        SubscribeToCoach();
    }

    private void Start()
    {
        PrepareCalibrationPrompt();
    }

    private void PrepareCalibrationPrompt()
    {
        calibrated = false;
        actionActive = false;
        checkpointActive = false;
        excludedTransitionActive = false;
        waitingForActionStart = false;
        if (coachController != null) coachController.enabled = false;
        PrepareStandingCalibrationPose();
        SetSceneButtonsBlocked(true);
        if (scoreText != null) scoreText.text = "--";
        if (feedbackText != null) feedbackText.text = "请自然站立，然后点击“站立校准”";
        if (stateText != null) stateText.text = "等待站立校准 · 当前不评分";
        if (detailText != null) detailText.text = "校准完成前，播放和训练按钮暂不可用";
    }

    private void OnDestroy()
    {
        SetSceneButtonsBlocked(false);
        UnsubscribeFromCoach();
        if (recalibrateButton != null)
        {
            recalibrateButton.onClick.RemoveListener(BeginCalibration);
        }
    }

    private void LateUpdate()
    {
        if (Input.GetKeyDown(KeyCode.R) || Input.GetKeyDown(KeyCode.C))
        {
            BeginCalibration();
        }

        if (!calibrated || coachAnimator == null || userAnimator == null)
        {
            return;
        }

        if (excludedTransitionActive)
        {
            if (scoreText != null)
            {
                scoreText.text = "--";
                scoreText.color = new Color(0.55f, 0.8f, 1f);
            }
            if (feedbackText != null)
            {
                feedbackText.text = waitingForActionStart
                    ? "站立校准已完成，请点击播放并跟随教练躺下"
                    : "教练正在进入或离开躺姿，本阶段不评分";
            }
            if (stateText != null)
            {
                stateText.text = waitingForActionStart
                    ? "站立准备 · 等待动作开始"
                    : "姿势过渡 · 暂停评分";
            }
            return;
        }

        currentResult = EvaluateCurrentPose();

        if (Time.unscaledTime >= nextTextUpdateTime)
        {
            nextTextUpdateTime = Time.unscaledTime + textUpdateInterval;
            UpdateHud();
        }
    }

    public void BeginCalibration()
    {
        if (calibrationRoutine != null)
        {
            StopCoroutine(calibrationRoutine);
        }

        calibrationRoutine = StartCoroutine(CalibrationSequence());
    }

    private IEnumerator CalibrationSequence()
    {
        calibrated = false;
        actionActive = false;
        checkpointActive = false;
        excludedTransitionActive = false;
        waitingForActionStart = false;
        SetSceneButtonsBlocked(true);
        if (coachController != null) coachController.enabled = false;
        scoringEngine.ResetSession();
        if (scoreText != null)
        {
            scoreText.text = "--";
            scoreText.color = new Color(0.55f, 0.8f, 1f);
        }
        ResolveReferences();
        coachCalibrationDirections.Clear();
        userCalibrationDirections.Clear();
        if (coachAnimator == null || userAnimator == null)
        {
            if (stateText != null) stateText.text = "未找到教练或用户模型";
            calibrationRoutine = null;
            yield break;
        }

        PrepareStandingCalibrationPose();
        yield return null;
        yield return CollectCalibrationStage(
            "站立姿势校准",
            "请自然站立并模仿教练，保持身体放松稳定");

        scoringEngine.SetStandingCalibration(userCalibrationDirections, coachCalibrationDirections);
        calibrated = scoringEngine.CalibratedSegmentCount >= 6;
        if (coachController != null) coachController.enabled = true;
        SetSceneButtonsBlocked(false);
        waitingForActionStart = calibrated;
        excludedTransitionActive = calibrated;
        if (stateText != null)
        {
            stateText.text = calibrated
                ? $"校准完成 · 已建立 {scoringEngine.CalibratedSegmentCount} 个肢段基准"
                : "有效骨骼不足，请检查传感器";
        }
        if (feedbackText != null)
        {
            feedbackText.text = calibrated ? "现在可以点击播放动作开始测试" : "连接设备后点击重新校准";
        }
        calibrationRoutine = null;
    }

    private IEnumerator CollectCalibrationStage(
        string stageTitle,
        string instruction)
    {
        var coachSamples = new Dictionary<HumanBodyBones, DirectionAccumulator>();
        var userSamples = new Dictionary<HumanBodyBones, DirectionAccumulator>();
        previousCalibrationRotations.Clear();
        calibrationMotionSampleElapsed = 0f;
        lastCalibrationStability = true;
        float stableElapsed = 0f;
        float sampleElapsed = 0f;
        float unstableDuringSample = 0f;
        bool samplingStarted = false;

        while (sampleElapsed < calibrationSampleDuration)
        {
            bool hasPose = CalculateCalibrationPoseMatch(
                out float averageError,
                out float worstError,
                out PoseBodyRegion worstRegion,
                out int comparedSegments);
            int validBones = CountValidCalibrationBones();
            bool trackingReady = hasPose && comparedSegments >= 5 && validBones >= minimumTrackedBones;
            bool stable = CheckCalibrationStability();

            if (trackingReady && stable)
            {
                unstableDuringSample = 0f;
                stableElapsed += Time.unscaledDeltaTime;
                if (stableElapsed >= calibrationStableDuration)
                {
                    samplingStarted = true;
                    CaptureDirectionSample(coachSamples, userSamples);
                    sampleElapsed += Time.unscaledDeltaTime;
                }
            }
            else
            {
                if (samplingStarted && trackingReady)
                {
                    // Brief Tracker spikes pause sampling instead of throwing
                    // away all progress. Only sustained motion restarts it.
                    unstableDuringSample += Time.unscaledDeltaTime;
                    if (unstableDuringSample > 0.5f)
                    {
                        stableElapsed = 0f;
                        sampleElapsed = 0f;
                        samplingStarted = false;
                        coachSamples.Clear();
                        userSamples.Clear();
                    }
                }
                else
                {
                    stableElapsed = 0f;
                }
            }

            if (feedbackText != null)
            {
                if (!trackingReady)
                    feedbackText.text = $"Tracker 骨骼不足（{validBones}/{minimumTrackedBones}），请检查连接";
                else if (!stable)
                    feedbackText.text = $"{instruction}\n检测到晃动，请自然保持";
                else if (stableElapsed < calibrationStableDuration)
                    feedbackText.text = $"{instruction}\n稳定保持 {calibrationStableDuration - stableElapsed:0.0}s";
                else
                    feedbackText.text = $"正在进行多帧采样 {sampleElapsed / calibrationSampleDuration:P0}";
            }
            if (stateText != null)
                stateText.text = $"{stageTitle} · 平均偏差 {averageError:0}°";
            if (detailText != null)
            {
                detailText.text =
                    $"姿势参考偏差：平均 {averageError:0}°，主要在{GetRegionName(worstRegion)} {worstError:0}°\n" +
                    $"该偏差仅用于提示，不再阻止校准；当前有效骨骼 {validBones}/{scoringBones.Length}。";
            }
            yield return null;
        }

        StoreAveragedDirections(coachSamples, coachCalibrationDirections);
        StoreAveragedDirections(userSamples, userCalibrationDirections);
    }

    private int CountValidCalibrationBones()
    {
        int count = 0;
        foreach (HumanBodyBones bone in scoringBones)
        {
            if (coachAnimator.GetBoneTransform(bone) != null && userAnimator.GetBoneTransform(bone) != null)
                count++;
        }
        return count;
    }

    private void CaptureDirectionSample(
        Dictionary<HumanBodyBones, DirectionAccumulator> coachSamples,
        Dictionary<HumanBodyBones, DirectionAccumulator> userSamples)
    {
        Dictionary<HumanBodyBones, Vector3> coachPose = scoringEngine.CaptureRawPose(coachAnimator);
        Dictionary<HumanBodyBones, Vector3> userPose = scoringEngine.CaptureRawPose(userAnimator);
        foreach (KeyValuePair<HumanBodyBones, Vector3> entry in coachPose)
        {
            if (!userPose.TryGetValue(entry.Key, out Vector3 userDirection)) continue;

            if (!coachSamples.TryGetValue(entry.Key, out DirectionAccumulator coachAccumulator))
            {
                coachAccumulator = new DirectionAccumulator();
                coachSamples[entry.Key] = coachAccumulator;
            }
            if (!userSamples.TryGetValue(entry.Key, out DirectionAccumulator userAccumulator))
            {
                userAccumulator = new DirectionAccumulator();
                userSamples[entry.Key] = userAccumulator;
            }
            coachAccumulator.Add(entry.Value);
            userAccumulator.Add(userDirection);
        }
    }

    private static void StoreAveragedDirections(
        Dictionary<HumanBodyBones, DirectionAccumulator> samples,
        Dictionary<HumanBodyBones, Vector3> destination)
    {
        destination.Clear();
        foreach (KeyValuePair<HumanBodyBones, DirectionAccumulator> entry in samples)
        {
            if (entry.Value.Count > 0) destination[entry.Key] = entry.Value.Average();
        }
    }

    private bool CheckCalibrationStability()
    {
        calibrationMotionSampleElapsed += Time.unscaledDeltaTime;
        if (calibrationMotionSampleElapsed < 0.15f) return lastCalibrationStability;

        float duration = calibrationMotionSampleElapsed;
        calibrationMotionSampleElapsed = 0f;
        int checkedCount = 0;
        int movingCount = 0;
        foreach (HumanBodyBones bone in scoringBones)
        {
            Transform userBone = userAnimator.GetBoneTransform(bone);
            if (userBone == null) continue;
            Quaternion rotation = userBone.localRotation;
            if (previousCalibrationRotations.TryGetValue(bone, out Quaternion previous))
            {
                float speed = Quaternion.Angle(previous, rotation) / Mathf.Max(0.001f, duration);
                if (speed > calibrationMotionLimit) movingCount++;
            }
            previousCalibrationRotations[bone] = rotation;
            checkedCount++;
        }

        int allowedMoving = Mathf.Max(1, Mathf.CeilToInt(checkedCount * allowedMovingBoneFraction));
        lastCalibrationStability = checkedCount > 0 && movingCount <= allowedMoving;
        return lastCalibrationStability;
    }

    private bool CalculateCalibrationPoseMatch(
        out float averageError,
        out float worstError,
        out PoseBodyRegion worstRegion,
        out int comparedSegments)
    {
        averageError = 0f;
        worstError = 0f;
        worstRegion = PoseBodyRegion.None;
        comparedSegments = 0;
        Transform coachHips = coachAnimator.GetBoneTransform(HumanBodyBones.Hips);
        Transform userHips = userAnimator.GetBoneTransform(HumanBodyBones.Hips);
        if (coachHips == null || userHips == null) return false;

        // Remove room heading only. Pitch/roll must remain visible; otherwise a
        // standing user could incorrectly pass calibration against a lying coach.
        Quaternion alignUserToCoach = CalculateYawAlignment(userHips, coachHips);
        AddCalibrationAxisError(coachHips.up, alignUserToCoach * userHips.up, PoseBodyRegion.Torso,
            ref averageError, ref worstError, ref worstRegion, ref comparedSegments);
        AddCalibrationDirectionError(HumanBodyBones.Hips, HumanBodyBones.Chest, PoseBodyRegion.Torso,
            alignUserToCoach, ref averageError, ref worstError, ref worstRegion, ref comparedSegments);
        AddCalibrationDirectionError(HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm, PoseBodyRegion.LeftArm,
            alignUserToCoach, ref averageError, ref worstError, ref worstRegion, ref comparedSegments);
        AddCalibrationDirectionError(HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand, PoseBodyRegion.LeftArm,
            alignUserToCoach, ref averageError, ref worstError, ref worstRegion, ref comparedSegments);
        AddCalibrationDirectionError(HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, PoseBodyRegion.RightArm,
            alignUserToCoach, ref averageError, ref worstError, ref worstRegion, ref comparedSegments);
        AddCalibrationDirectionError(HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand, PoseBodyRegion.RightArm,
            alignUserToCoach, ref averageError, ref worstError, ref worstRegion, ref comparedSegments);
        AddCalibrationDirectionError(HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, PoseBodyRegion.LeftLeg,
            alignUserToCoach, ref averageError, ref worstError, ref worstRegion, ref comparedSegments);
        AddCalibrationDirectionError(HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot, PoseBodyRegion.LeftLeg,
            alignUserToCoach, ref averageError, ref worstError, ref worstRegion, ref comparedSegments);
        AddCalibrationDirectionError(HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, PoseBodyRegion.RightLeg,
            alignUserToCoach, ref averageError, ref worstError, ref worstRegion, ref comparedSegments);
        AddCalibrationDirectionError(HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot, PoseBodyRegion.RightLeg,
            alignUserToCoach, ref averageError, ref worstError, ref worstRegion, ref comparedSegments);

        if (comparedSegments > 0) averageError /= comparedSegments;
        return comparedSegments > 0;
    }

    private Quaternion CalculateYawAlignment(Transform userHips, Transform coachHips)
    {
        Vector3 userForward = Vector3.ProjectOnPlane(userHips.forward, Vector3.up);
        Vector3 coachForward = Vector3.ProjectOnPlane(coachHips.forward, Vector3.up);
        if (userForward.sqrMagnitude < 0.01f) userForward = Vector3.ProjectOnPlane(userAnimator.transform.forward, Vector3.up);
        if (coachForward.sqrMagnitude < 0.01f) coachForward = Vector3.ProjectOnPlane(coachAnimator.transform.forward, Vector3.up);
        if (userForward.sqrMagnitude < 0.01f || coachForward.sqrMagnitude < 0.01f) return Quaternion.identity;
        float yaw = Vector3.SignedAngle(userForward, coachForward, Vector3.up);
        return Quaternion.AngleAxis(yaw, Vector3.up);
    }

    private static void AddCalibrationAxisError(
        Vector3 coachDirection,
        Vector3 userDirection,
        PoseBodyRegion region,
        ref float totalError,
        ref float worstError,
        ref PoseBodyRegion worstRegion,
        ref int count)
    {
        float error = Vector3.Angle(coachDirection, userDirection);
        totalError += error;
        count++;
        if (error > worstError)
        {
            worstError = error;
            worstRegion = region;
        }
    }

    private void AddCalibrationDirectionError(
        HumanBodyBones parentBone,
        HumanBodyBones childBone,
        PoseBodyRegion region,
        Quaternion alignUserToCoach,
        ref float totalError,
        ref float worstError,
        ref PoseBodyRegion worstRegion,
        ref int count)
    {
        Transform coachParent = coachAnimator.GetBoneTransform(parentBone);
        Transform coachChild = coachAnimator.GetBoneTransform(childBone);
        Transform userParent = userAnimator.GetBoneTransform(parentBone);
        Transform userChild = userAnimator.GetBoneTransform(childBone);
        if (coachParent == null || coachChild == null || userParent == null || userChild == null) return;

        Vector3 coachDirection = (coachChild.position - coachParent.position).normalized;
        Vector3 userDirection = alignUserToCoach * (userChild.position - userParent.position).normalized;
        float error = Vector3.Angle(coachDirection, userDirection);
        totalError += error;
        count++;
        if (error > worstError)
        {
            worstError = error;
            worstRegion = region;
        }
    }

    private void PrepareStandingCalibrationPose()
    {
        if (coachAnimator == null || coachController == null) return;
        coachAnimator.enabled = true;
        coachAnimator.cullingMode = AnimatorCullingMode.AlwaysAnimate;
        coachAnimator.Rebind();
        coachAnimator.Update(0f);
        coachAnimator.speed = 0f;
        string stateName = coachController.StartSegmentStateName;
        if (!string.IsNullOrEmpty(stateName))
        {
            // The first frame of deadbug_start is standing. Calibration must be
            // captured here; the later standing-to-lying transition is excluded.
            coachAnimator.Play(stateName, 0, 0f);
            coachAnimator.Update(0f);
        }
        coachAnimator.speed = 0f;
    }

    private void SetSceneButtonsBlocked(bool blocked)
    {
        if (blocked)
        {
            if (calibrationBlockedButtons.Count > 0) return;
            Button[] buttons = FindObjectsOfType<Button>(true);
            foreach (Button button in buttons)
            {
                if (button == null || button == recalibrateButton) continue;
                calibrationBlockedButtons[button] = button.interactable;
                button.interactable = false;
            }
            return;
        }

        foreach (KeyValuePair<Button, bool> entry in calibrationBlockedButtons)
        {
            if (entry.Key != null) entry.Key.interactable = entry.Value;
        }
        calibrationBlockedButtons.Clear();
    }

    private AnatomicalPoseScore EvaluateCurrentPose()
    {
        return scoringEngine.Evaluate(userAnimator, coachAnimator, Time.deltaTime);
    }

    private void UpdateHud()
    {
        SetScoreVisual(currentResult.score);
        if (feedbackText != null)
        {
            feedbackText.text = GetAnatomicalFeedback(currentResult);
        }

        if (detailText != null)
        {
            detailText.text =
                $"主要调整：{GetSegmentName(currentResult.worstSegment)}    相差约 {currentResult.worstErrorDegrees:0}°\n" +
                $"有效肢段：{currentResult.validSegmentCount}/11    " +
                $"响应补偿：{currentResult.matchedDelay * 1000f:0}ms\n" +
                $"满分≤{anatomicalScoring.perfectAngle:0}°    " +
                $"严重错误≥{anatomicalScoring.zeroScoreAngle:0}°    " +
                $"抗抖区 {anatomicalScoring.sensorDeadZone:0}°";
        }

        if (stateText != null)
        {
            stateText.text = checkpointActive
                ? "目标保持中 · 正在检查最终姿势"
                : actionActive
                    ? "动作进行中 · 人体坐标肢段评分"
                    : "等待动作 · 当前不评分";
        }
    }

    private static string GetAnatomicalFeedback(AnatomicalPoseScore result)
    {
        string segment = GetSegmentName(result.worstSegment);
        if (result.score >= 90f) return "动作很稳定，继续保持当前节奏";
        if (result.score >= 78f) return $"整体很好，轻微调整{segment}即可";
        if (result.score >= 65f) return $"动作方向基本正确，优先调整{segment}";
        return $"先放慢动作并对齐{segment}，不必追求完全重合";
    }

    private static string GetSegmentName(HumanBodyBones segment)
    {
        switch (segment)
        {
            case HumanBodyBones.LeftUpperArm: return "左上臂";
            case HumanBodyBones.RightUpperArm: return "右上臂";
            case HumanBodyBones.LeftLowerArm: return "左小臂";
            case HumanBodyBones.RightLowerArm: return "右小臂";
            case HumanBodyBones.LeftHand: return "左手";
            case HumanBodyBones.RightHand: return "右手";
            case HumanBodyBones.LeftUpperLeg: return "左大腿";
            case HumanBodyBones.RightUpperLeg: return "右大腿";
            case HumanBodyBones.LeftLowerLeg: return "左小腿";
            case HumanBodyBones.RightLowerLeg: return "右小腿";
            case HumanBodyBones.Head: return "头部";
            default: return "主要肢段";
        }
    }

    private void SetScoreVisual(float score)
    {
        if (scoreText == null)
        {
            return;
        }

        scoreText.text = $"{score:0}";
        scoreText.color = score >= 85f
            ? new Color(0.25f, 1f, 0.55f)
            : score >= 65f
                ? new Color(1f, 0.85f, 0.25f)
                : new Color(1f, 0.48f, 0.35f);
    }

    private static string GetRegionName(PoseBodyRegion region)
    {
        switch (region)
        {
            case PoseBodyRegion.LeftArm: return "左臂";
            case PoseBodyRegion.RightArm: return "右臂";
            case PoseBodyRegion.LeftLeg: return "左腿";
            case PoseBodyRegion.RightLeg: return "右腿";
            case PoseBodyRegion.Torso: return "躯干";
            default: return "无明显偏差";
        }
    }

    private void ResolveReferences()
    {
        if (coachController == null)
        {
            coachController = FindObjectOfType<CoachActionController>(true);
        }

        if (coachAnimator != null && userAnimator != null)
        {
            return;
        }

        Animator[] animators = FindObjectsOfType<Animator>(true);
        foreach (Animator animator in animators)
        {
            string objectName = animator.gameObject.name.ToLowerInvariant();
            if (coachAnimator == null && objectName.Contains("coach")) coachAnimator = animator;
            if (userAnimator == null && objectName.Contains("user")) userAnimator = animator;
        }
    }

    private void SubscribeToCoach()
    {
        if (coachController == null)
        {
            return;
        }
        coachController.ScoringSessionStarted += HandleSessionStarted;
        coachController.SegmentPlaybackStarted += HandleSegmentPlaybackStarted;
        coachController.ScoringCheckpointStarted += HandleCheckpointStarted;
        coachController.ScoringCheckpointEnded += HandleCheckpointEnded;
        coachController.ScoringSessionCompleted += HandleSessionCompleted;
    }

    private void UnsubscribeFromCoach()
    {
        if (coachController == null)
        {
            return;
        }
        coachController.ScoringSessionStarted -= HandleSessionStarted;
        coachController.SegmentPlaybackStarted -= HandleSegmentPlaybackStarted;
        coachController.ScoringCheckpointStarted -= HandleCheckpointStarted;
        coachController.ScoringCheckpointEnded -= HandleCheckpointEnded;
        coachController.ScoringSessionCompleted -= HandleSessionCompleted;
    }

    private void HandleSessionStarted()
    {
        scoringEngine.ResetSession();
        actionActive = true;
        checkpointActive = false;
        excludedTransitionActive = false;
        waitingForActionStart = false;
    }

    private void HandleCheckpointStarted(CoachActionController.ScoringCheckpointInfo _)
    {
        checkpointActive = true;
    }

    private void HandleSegmentPlaybackStarted(CoachActionController.SegmentPlaybackInfo info)
    {
        if (string.Equals(info.stateName, coachController.StartSegmentStateName, System.StringComparison.Ordinal))
        {
            excludedTransitionActive = true;
            actionActive = false;
            waitingForActionStart = false;
        }
    }

    private void HandleCheckpointEnded(CoachActionController.ScoringCheckpointInfo _)
    {
        checkpointActive = false;
    }

    private void HandleSessionCompleted()
    {
        actionActive = false;
        checkpointActive = false;
        excludedTransitionActive = true;
        waitingForActionStart = false;
    }

    private void BuildHud()
    {
        Camera targetCamera = Camera.main;
        if (targetCamera == null)
        {
            Camera[] cameras = FindObjectsOfType<Camera>(true);
            foreach (Camera candidate in cameras)
            {
                if (candidate.isActiveAndEnabled)
                {
                    targetCamera = candidate;
                    break;
                }
            }
            if (targetCamera == null && cameras.Length > 0) targetCamera = cameras[0];
        }

        GameObject canvasObject = new GameObject(
            "EvaluationScoreHUD",
            typeof(RectTransform),
            typeof(Canvas),
            typeof(CanvasScaler),
            typeof(GraphicRaycaster));
        RectTransform canvasRect = canvasObject.GetComponent<RectTransform>();
        Canvas canvas = canvasObject.GetComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;
        canvas.sortingOrder = 200;
        canvasRect.sizeDelta = new Vector2(760f, 430f);
        canvasRect.localScale = Vector3.one * hudScale;
        if (targetCamera != null)
        {
            canvasRect.SetParent(targetCamera.transform, false);
            canvasRect.localPosition = hudCameraLocalPosition;
            canvasRect.localRotation = Quaternion.identity;
            canvas.worldCamera = targetCamera;
        }

        Image background = canvasObject.AddComponent<Image>();
        background.color = new Color(0.025f, 0.045f, 0.075f, 0.88f);
        background.raycastTarget = false;

        scoreText = CreateText(canvasRect, "Score", new Vector2(20f, -25f), new Vector2(250f, 180f), 132f);
        scoreText.alignment = TextAlignmentOptions.Center;
        feedbackText = CreateText(canvasRect, "Feedback", new Vector2(270f, -45f), new Vector2(460f, 90f), 35f);
        feedbackText.color = Color.white;
        detailText = CreateText(canvasRect, "Details", new Vector2(45f, -215f), new Vector2(665f, 90f), 25f);
        detailText.color = new Color(0.75f, 0.85f, 0.95f);
        stateText = CreateText(canvasRect, "State", new Vector2(45f, -320f), new Vector2(500f, 55f), 23f);
        stateText.color = new Color(0.55f, 0.8f, 1f);

        recalibrateButton = CreateButton(canvasRect, new Vector2(565f, -330f), new Vector2(150f, 55f));
        recalibrateButton.onClick.AddListener(BeginCalibration);
    }

    private static TMP_Text CreateText(
        RectTransform parent,
        string objectName,
        Vector2 anchoredPosition,
        Vector2 size,
        float fontSize)
    {
        GameObject textObject = new GameObject(objectName, typeof(RectTransform), typeof(TextMeshProUGUI));
        RectTransform rect = textObject.GetComponent<RectTransform>();
        rect.SetParent(parent, false);
        rect.anchorMin = new Vector2(0f, 1f);
        rect.anchorMax = new Vector2(0f, 1f);
        rect.pivot = new Vector2(0f, 1f);
        rect.anchoredPosition = anchoredPosition;
        rect.sizeDelta = size;
        TMP_Text text = textObject.GetComponent<TMP_Text>();
        text.fontSize = fontSize;
        text.enableWordWrapping = true;
        text.alignment = TextAlignmentOptions.MidlineLeft;
        text.raycastTarget = false;
        return text;
    }

    private static Button CreateButton(RectTransform parent, Vector2 anchoredPosition, Vector2 size)
    {
        GameObject buttonObject = new GameObject("Recalibrate", typeof(RectTransform), typeof(Image), typeof(Button));
        RectTransform rect = buttonObject.GetComponent<RectTransform>();
        rect.SetParent(parent, false);
        rect.anchorMin = new Vector2(0f, 1f);
        rect.anchorMax = new Vector2(0f, 1f);
        rect.pivot = new Vector2(0f, 1f);
        rect.anchoredPosition = anchoredPosition;
        rect.sizeDelta = size;
        Image image = buttonObject.GetComponent<Image>();
        image.color = new Color(0.12f, 0.42f, 0.68f, 0.95f);
        Button button = buttonObject.GetComponent<Button>();
        button.targetGraphic = image;
        TMP_Text label = CreateText(rect, "Label", Vector2.zero, size, 23f);
        label.text = "站立校准";
        label.alignment = TextAlignmentOptions.Center;
        label.color = Color.white;
        return button;
    }
}
