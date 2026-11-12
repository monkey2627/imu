using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Noise-tolerant pose scoring parameters shared by practice and game modes.
/// Values are time based, so results do not depend on the display frame rate.
/// </summary>
[Serializable]
public sealed class RobustPoseScoringSettings
{
    [Tooltip("Small IMU errors inside this angle are ignored.")]
    [Min(0f)] public float sensorNoiseDeadZone = 8f;

    [Tooltip("Time used to smooth wearable rotations. Lower values react faster.")]
    [Min(0.01f)] public float rotationSmoothingTime = 0.12f;

    [Tooltip("Maximum believable single-sensor rotation speed. Faster changes are limited as tracking spikes.")]
    [Min(90f)] public float maximumAngularSpeed = 540f;

    [Tooltip("Blend a robust median into the weighted mean so one noisy tracker cannot dominate the result.")]
    [Range(0f, 0.5f)] public float robustMedianBlend = 0.2f;

    [Tooltip("Live scores fall more slowly than they rise to avoid flicker from brief tracking noise.")]
    [Min(0.01f)] public float liveScoreRiseTime = 0.16f;

    [Min(0.01f)] public float liveScoreFallTime = 0.2f;

    [Tooltip("Fraction removed from each end when a hold is finalized.")]
    [Range(0f, 0.24f)] public float finalScoreTrimFraction = 0.1f;
}

public enum PoseBodyRegion
{
    None,
    Torso,
    LeftArm,
    RightArm,
    LeftLeg,
    RightLeg
}

public struct PoseFrameScore
{
    public float score;
    public float rawScore;
    public float worstErrorDegrees;
    public HumanBodyBones worstBone;
    public PoseBodyRegion worstRegion;
    public int validBoneCount;
}

public struct PoseRegionScore
{
    private float weightedSquaredErrorTotal;
    private float weightedScoreTotal;
    private float totalWeight;

    public float averageErrorDegrees =>
        totalWeight > 0f
            ? Mathf.Sqrt(weightedSquaredErrorTotal / totalWeight)
            : 0f;
    public float worstErrorDegrees { get; private set; }
    public float score =>
        totalWeight > 0f ? weightedScoreTotal / totalWeight : 100f;
    public int validBoneCount { get; private set; }
    public bool isValid => validBoneCount > 0 && totalWeight > 0f;

    internal void Add(float errorDegrees, float boneScore, float weight)
    {
        float safeWeight = Mathf.Max(0f, weight);
        if (safeWeight <= 0f)
        {
            return;
        }

        float safeError = Mathf.Max(0f, errorDegrees);
        weightedSquaredErrorTotal +=
            safeError * safeError * safeWeight;
        weightedScoreTotal += Mathf.Clamp01(boneScore) * 100f * safeWeight;
        totalWeight += safeWeight;
        worstErrorDegrees = Mathf.Max(
            worstErrorDegrees,
            safeError);
        validBoneCount++;
    }
}

public struct PoseRegionScores
{
    public PoseRegionScore torso;
    public PoseRegionScore leftArm;
    public PoseRegionScore rightArm;
    public PoseRegionScore leftLeg;
    public PoseRegionScore rightLeg;

    internal void Add(
        PoseBodyRegion region,
        float errorDegrees,
        float boneScore,
        float weight)
    {
        switch (region)
        {
            case PoseBodyRegion.Torso:
                torso.Add(errorDegrees, boneScore, weight);
                break;
            case PoseBodyRegion.LeftArm:
                leftArm.Add(errorDegrees, boneScore, weight);
                break;
            case PoseBodyRegion.RightArm:
                rightArm.Add(errorDegrees, boneScore, weight);
                break;
            case PoseBodyRegion.LeftLeg:
                leftLeg.Add(errorDegrees, boneScore, weight);
                break;
            case PoseBodyRegion.RightLeg:
                rightLeg.Add(errorDegrees, boneScore, weight);
                break;
        }
    }

    public bool TryGetScore(
        PoseBodyRegion region,
        out PoseRegionScore regionScore)
    {
        switch (region)
        {
            case PoseBodyRegion.Torso:
                regionScore = torso;
                break;
            case PoseBodyRegion.LeftArm:
                regionScore = leftArm;
                break;
            case PoseBodyRegion.RightArm:
                regionScore = rightArm;
                break;
            case PoseBodyRegion.LeftLeg:
                regionScore = leftLeg;
                break;
            case PoseBodyRegion.RightLeg:
                regionScore = rightLeg;
                break;
            default:
                regionScore = default(PoseRegionScore);
                return false;
        }

        return regionScore.isValid;
    }
}

/// <summary>
/// Stateful scorer that filters wearable rotations before comparing them with a
/// coach. Create one instance per tracked user and Reset it between sessions.
/// </summary>
public sealed class RobustPoseScoringEngine : IDisposable
{
    private readonly RobustPoseScoringSettings settings;
    private readonly Dictionary<HumanBodyBones, Quaternion> filteredUserRotations =
        new Dictionary<HumanBodyBones, Quaternion>();
    private readonly Dictionary<HumanBodyBones, Vector3> filteredUserMuscleAngles =
        new Dictionary<HumanBodyBones, Vector3>();
    private Animator muscleUserAnimator;
    private Animator muscleCoachAnimator;
    private HumanPoseHandler userPoseHandler;
    private HumanPoseHandler coachPoseHandler;
    private HumanPose userHumanPose;
    private HumanPose coachHumanPose;
    private float displayedScore;
    private bool hasDisplayedScore;

    private struct WeightedBoneScore
    {
        public float score;
        public float weight;
    }

    private struct EvaluatedBone
    {
        public HumanBodyBones bone;
        public PoseBodyRegion region;
        public float actualErrorDegrees;
        public float score;
        public float weight;
    }

    public RobustPoseScoringEngine(RobustPoseScoringSettings settings)
    {
        this.settings = settings ?? new RobustPoseScoringSettings();
    }

    public void Reset()
    {
        filteredUserRotations.Clear();
        filteredUserMuscleAngles.Clear();
        displayedScore = 0f;
        hasDisplayedScore = false;
    }

    public void Dispose()
    {
        userPoseHandler?.Dispose();
        coachPoseHandler?.Dispose();
        userPoseHandler = null;
        coachPoseHandler = null;
        muscleUserAnimator = null;
        muscleCoachAnimator = null;
    }

    public PoseFrameScore Evaluate(
        Animator userAnimator,
        Animator coachAnimator,
        IReadOnlyList<HumanBodyBones> bones,
        float perfectAngleThreshold,
        float maximumAngleError,
        float leniency = 1f,
        Func<HumanBodyBones, float> weightProvider = null,
        IReadOnlyDictionary<HumanBodyBones, Quaternion> userBaselines = null,
        IReadOnlyDictionary<HumanBodyBones, Quaternion> coachBaselines = null,
        float deltaTime = 0f,
        bool ignoreHipsHeading = false,
        bool useHumanoidMuscleSpace = false)
    {
        PoseFrameScore result = new PoseFrameScore { worstBone = HumanBodyBones.LastBone };
        List<EvaluatedBone> evaluatedBones = EvaluateBones(
            userAnimator,
            coachAnimator,
            bones,
            perfectAngleThreshold,
            maximumAngleError,
            leniency,
            weightProvider,
            userBaselines,
            coachBaselines,
            deltaTime,
            ignoreHipsHeading,
            useHumanoidMuscleSpace,
            out float dt);
        var boneScores =
            new List<WeightedBoneScore>(evaluatedBones.Count);
        float weightedTotal = 0f;
        float totalWeight = 0f;

        foreach (EvaluatedBone evaluated in evaluatedBones)
        {
            boneScores.Add(new WeightedBoneScore
            {
                score = evaluated.score,
                weight = evaluated.weight
            });
            weightedTotal += evaluated.score * evaluated.weight;
            totalWeight += evaluated.weight;
            result.validBoneCount++;

            if (evaluated.actualErrorDegrees > result.worstErrorDegrees)
            {
                result.worstErrorDegrees = evaluated.actualErrorDegrees;
                result.worstBone = evaluated.bone;
                result.worstRegion = evaluated.region;
            }
        }

        if (totalWeight <= 0f)
        {
            return result;
        }

        float mean = weightedTotal / totalWeight;
        float median = CalculateWeightedMedian(boneScores, totalWeight);
        result.rawScore = Mathf.Clamp01(Mathf.Lerp(mean, median, settings.robustMedianBlend)) * 100f;
        result.score = SmoothLiveScore(result.rawScore, dt);
        return result;
    }

    /// <summary>
    /// Uses the same calibration-relative local rotations, wearable filtering,
    /// dead zone, score curve, bone weights, and body-region mapping as the
    /// formal scorer, while exposing every region for visual guidance.
    /// </summary>
    public PoseRegionScores EvaluateRegionScores(
        Animator userAnimator,
        Animator coachAnimator,
        IReadOnlyList<HumanBodyBones> bones,
        float perfectAngleThreshold,
        float maximumAngleError,
        float leniency = 1f,
        Func<HumanBodyBones, float> weightProvider = null,
        IReadOnlyDictionary<HumanBodyBones, Quaternion> userBaselines = null,
        IReadOnlyDictionary<HumanBodyBones, Quaternion> coachBaselines = null,
        float deltaTime = 0f,
        bool ignoreHipsHeading = false,
        bool useHumanoidMuscleSpace = false)
    {
        PoseRegionScores result = default(PoseRegionScores);
        List<EvaluatedBone> evaluatedBones = EvaluateBones(
            userAnimator,
            coachAnimator,
            bones,
            perfectAngleThreshold,
            maximumAngleError,
            leniency,
            weightProvider,
            userBaselines,
            coachBaselines,
            deltaTime,
            ignoreHipsHeading,
            useHumanoidMuscleSpace,
            out _);
        foreach (EvaluatedBone evaluated in evaluatedBones)
        {
            result.Add(
                evaluated.region,
                evaluated.actualErrorDegrees,
                evaluated.score,
                evaluated.weight);
        }

        return result;
    }

    private List<EvaluatedBone> EvaluateBones(
        Animator userAnimator,
        Animator coachAnimator,
        IReadOnlyList<HumanBodyBones> bones,
        float perfectAngleThreshold,
        float maximumAngleError,
        float leniency,
        Func<HumanBodyBones, float> weightProvider,
        IReadOnlyDictionary<HumanBodyBones, Quaternion> userBaselines,
        IReadOnlyDictionary<HumanBodyBones, Quaternion> coachBaselines,
        float deltaTime,
        bool ignoreHipsHeading,
        bool useHumanoidMuscleSpace,
        out float dt)
    {
        dt = Mathf.Clamp(
            deltaTime > 0f ? deltaTime : Time.deltaTime,
            1f / 240f,
            0.1f);
        var result = new List<EvaluatedBone>(
            bones != null ? bones.Count : 0);
        if (userAnimator == null || coachAnimator == null || bones == null)
        {
            return result;
        }

        float toleranceScale = Mathf.Max(0.1f, leniency);
        float perfect = Mathf.Max(
            settings.sensorNoiseDeadZone,
            perfectAngleThreshold * toleranceScale);
        float maximum = Mathf.Max(
            perfect + 1f,
            maximumAngleError * toleranceScale);
        float effectivePerfect = Mathf.Max(
            0f,
            perfect - settings.sensorNoiseDeadZone);
        float effectiveMaximum = Mathf.Max(
            effectivePerfect + 1f,
            maximum - settings.sensorNoiseDeadZone);
        bool hasHumanoidMusclePoses =
            useHumanoidMuscleSpace &&
            TryCaptureHumanoidPoses(userAnimator, coachAnimator);

        for (int index = 0; index < bones.Count; index++)
        {
            HumanBodyBones bone = bones[index];
            PoseBodyRegion region = GetRegion(bone);
            float actualError;
            if (hasHumanoidMusclePoses)
            {
                if (bone == HumanBodyBones.Hips)
                {
                    Quaternion userTilt = GetHeadingInvariantTiltDelta(
                        Quaternion.identity,
                        userHumanPose.bodyRotation);
                    Quaternion coachTilt = GetHeadingInvariantTiltDelta(
                        Quaternion.identity,
                        coachHumanPose.bodyRotation);
                    userTilt = FilterRotation(bone, userTilt, dt);
                    actualError = Quaternion.Angle(userTilt, coachTilt);
                }
                else
                {
                    if (!TryGetBoneMuscleAngles(
                            userHumanPose,
                            bone,
                            out Vector3 userAngles) ||
                        !TryGetBoneMuscleAngles(
                            coachHumanPose,
                            bone,
                            out Vector3 coachAngles))
                    {
                        continue;
                    }

                    userAngles = FilterMuscleAngles(
                        bone,
                        userAngles,
                        dt);
                    actualError = Mathf.Min(
                        180f,
                        Vector3.Distance(userAngles, coachAngles));
                }
            }
            else
            {
                Transform userBone = userAnimator.GetBoneTransform(bone);
                Transform coachBone = coachAnimator.GetBoneTransform(bone);
                if (userBone == null || coachBone == null)
                {
                    continue;
                }

                Quaternion userRotation = userBone.localRotation;
                Quaternion coachRotation = coachBone.localRotation;
                Quaternion userBaseline = Quaternion.identity;
                Quaternion coachBaseline = Quaternion.identity;
                bool hasCalibration =
                    userBaselines != null &&
                    coachBaselines != null &&
                    userBaselines.TryGetValue(
                        bone,
                        out userBaseline) &&
                    coachBaselines.TryGetValue(
                        bone,
                        out coachBaseline);
                if (ignoreHipsHeading && bone == HumanBodyBones.Hips)
                {
                    userRotation = GetHeadingInvariantTiltDelta(
                        hasCalibration
                            ? userBaseline
                            : Quaternion.identity,
                        userRotation);
                    coachRotation = GetHeadingInvariantTiltDelta(
                        hasCalibration
                            ? coachBaseline
                            : Quaternion.identity,
                        coachRotation);
                }
                else if (hasCalibration)
                {
                    userRotation =
                        Quaternion.Inverse(userBaseline) * userRotation;
                    coachRotation =
                        Quaternion.Inverse(coachBaseline) * coachRotation;
                }

                userRotation = FilterRotation(
                    bone,
                    userRotation,
                    dt);
                actualError =
                    Quaternion.Angle(userRotation, coachRotation);
            }

            float errorAfterDeadZone = Mathf.Max(
                0f,
                actualError - settings.sensorNoiseDeadZone);
            float normalized = Mathf.InverseLerp(
                effectivePerfect,
                effectiveMaximum,
                errorAfterDeadZone);
            float boneScore =
                1f - Mathf.Pow(normalized, 1.7f);
            float weight = Mathf.Max(
                0f,
                weightProvider != null
                    ? weightProvider(bone)
                    : GetDefaultBoneWeight(bone));
            if (weight <= 0f)
            {
                continue;
            }

            result.Add(new EvaluatedBone
            {
                bone = bone,
                region = region,
                actualErrorDegrees = actualError,
                score = boneScore,
                weight = weight
            });
        }

        return result;
    }

    private bool TryCaptureHumanoidPoses(
        Animator userAnimator,
        Animator coachAnimator)
    {
        if (!IsUsableHumanoid(userAnimator) ||
            !IsUsableHumanoid(coachAnimator))
        {
            return false;
        }

        if (userPoseHandler == null ||
            coachPoseHandler == null ||
            muscleUserAnimator != userAnimator ||
            muscleCoachAnimator != coachAnimator)
        {
            Dispose();
            try
            {
                userPoseHandler = new HumanPoseHandler(
                    userAnimator.avatar,
                    userAnimator.transform);
                coachPoseHandler = new HumanPoseHandler(
                    coachAnimator.avatar,
                    coachAnimator.transform);
                muscleUserAnimator = userAnimator;
                muscleCoachAnimator = coachAnimator;
                userHumanPose = new HumanPose
                {
                    muscles = new float[HumanTrait.MuscleCount]
                };
                coachHumanPose = new HumanPose
                {
                    muscles = new float[HumanTrait.MuscleCount]
                };
            }
            catch
            {
                Dispose();
                return false;
            }
        }

        try
        {
            userPoseHandler.GetHumanPose(ref userHumanPose);
            coachPoseHandler.GetHumanPose(ref coachHumanPose);
            return userHumanPose.muscles != null &&
                coachHumanPose.muscles != null &&
                userHumanPose.muscles.Length == HumanTrait.MuscleCount &&
                coachHumanPose.muscles.Length == HumanTrait.MuscleCount;
        }
        catch
        {
            Dispose();
            return false;
        }
    }

    private static bool IsUsableHumanoid(Animator animator)
    {
        return animator != null &&
            animator.isHuman &&
            animator.avatar != null &&
            animator.avatar.isValid &&
            animator.avatar.isHuman;
    }

    private static bool TryGetBoneMuscleAngles(
        HumanPose pose,
        HumanBodyBones bone,
        out Vector3 angles)
    {
        angles = Vector3.zero;
        if (pose.muscles == null)
        {
            return false;
        }

        bool hasMuscle = false;
        for (int dof = 0; dof < 3; dof++)
        {
            int muscleIndex =
                HumanTrait.MuscleFromBone((int)bone, dof);
            if (muscleIndex < 0 ||
                muscleIndex >= pose.muscles.Length)
            {
                continue;
            }

            float normalized = Mathf.Clamp(
                pose.muscles[muscleIndex],
                -1f,
                1f);
            angles[dof] = normalized < 0f
                ? -normalized *
                    HumanTrait.GetMuscleDefaultMin(muscleIndex)
                : normalized *
                    HumanTrait.GetMuscleDefaultMax(muscleIndex);
            hasMuscle = true;
        }

        return hasMuscle;
    }

    private Vector3 FilterMuscleAngles(
        HumanBodyBones bone,
        Vector3 rawAngles,
        float deltaTime)
    {
        if (!filteredUserMuscleAngles.TryGetValue(
                bone,
                out Vector3 filtered))
        {
            filteredUserMuscleAngles[bone] = rawAngles;
            return rawAngles;
        }

        float maximumStep =
            settings.maximumAngularSpeed * deltaTime;
        Vector3 limited = Vector3.MoveTowards(
            filtered,
            rawAngles,
            maximumStep);
        float alpha = 1f - Mathf.Exp(
            -deltaTime /
            Mathf.Max(0.01f, settings.rotationSmoothingTime));
        filtered = Vector3.Lerp(filtered, limited, alpha);
        filteredUserMuscleAngles[bone] = filtered;
        return filtered;
    }

    /// <summary>
    /// Describes pelvis tilt relative to the avatar up axis while discarding
    /// rotation around that axis. Room heading therefore cannot become a torso
    /// error, while standing-versus-lying tilt remains measurable.
    /// </summary>
    public static Quaternion GetHeadingInvariantTiltDelta(
        Quaternion referenceRotation,
        Quaternion currentRotation)
    {
        Vector3 referenceUpInBone =
            Quaternion.Inverse(referenceRotation) * Vector3.up;
        Vector3 currentUpInBone =
            Quaternion.Inverse(currentRotation) * Vector3.up;
        if (referenceUpInBone.sqrMagnitude < 0.0001f ||
            currentUpInBone.sqrMagnitude < 0.0001f)
        {
            return Quaternion.identity;
        }

        return Quaternion.FromToRotation(
            referenceUpInBone.normalized,
            currentUpInBone.normalized);
    }

    /// <summary>
    /// Capture one filtered user pose in calibration-relative local rotations.
    /// The returned pose can be compared against a time-shifted coach snapshot.
    /// </summary>
    public Dictionary<HumanBodyBones, Quaternion> CaptureFilteredUserPose(
        Animator userAnimator,
        IReadOnlyList<HumanBodyBones> bones,
        IReadOnlyDictionary<HumanBodyBones, Quaternion> userBaselines = null,
        float deltaTime = 0f)
    {
        var pose = new Dictionary<HumanBodyBones, Quaternion>();
        if (userAnimator == null || bones == null) return pose;

        float dt = Mathf.Clamp(deltaTime > 0f ? deltaTime : Time.deltaTime, 1f / 240f, 0.1f);
        for (int index = 0; index < bones.Count; index++)
        {
            HumanBodyBones bone = bones[index];
            Transform boneTransform = userAnimator.GetBoneTransform(bone);
            if (boneTransform == null) continue;
            Quaternion rotation = boneTransform.localRotation;
            if (userBaselines != null && userBaselines.TryGetValue(bone, out Quaternion baseline))
                rotation = Quaternion.Inverse(baseline) * rotation;
            pose[bone] = FilterRotation(bone, rotation, dt);
        }
        return pose;
    }

    /// <summary>
    /// Score two already captured pose maps. This is used for latency-aware
    /// dynamic matching without filtering the same user frame multiple times.
    /// </summary>
    public PoseFrameScore EvaluatePoseMaps(
        IReadOnlyDictionary<HumanBodyBones, Quaternion> userPose,
        IReadOnlyDictionary<HumanBodyBones, Quaternion> coachPose,
        IReadOnlyList<HumanBodyBones> bones,
        float perfectAngleThreshold,
        float maximumAngleError,
        float leniency = 1f,
        Func<HumanBodyBones, float> weightProvider = null,
        float deltaTime = 0f,
        bool smoothLiveScore = true)
    {
        PoseFrameScore result = new PoseFrameScore { worstBone = HumanBodyBones.LastBone };
        if (userPose == null || coachPose == null || bones == null) return result;

        float toleranceScale = Mathf.Max(0.1f, leniency);
        float perfect = Mathf.Max(settings.sensorNoiseDeadZone, perfectAngleThreshold * toleranceScale);
        float maximum = Mathf.Max(perfect + 1f, maximumAngleError * toleranceScale);
        float effectivePerfect = Mathf.Max(0f, perfect - settings.sensorNoiseDeadZone);
        float effectiveMaximum = Mathf.Max(effectivePerfect + 1f, maximum - settings.sensorNoiseDeadZone);
        var boneScores = new List<WeightedBoneScore>(bones.Count);
        float weightedTotal = 0f;
        float totalWeight = 0f;

        for (int index = 0; index < bones.Count; index++)
        {
            HumanBodyBones bone = bones[index];
            if (!userPose.TryGetValue(bone, out Quaternion userRotation) ||
                !coachPose.TryGetValue(bone, out Quaternion coachRotation))
                continue;

            float actualError = Quaternion.Angle(userRotation, coachRotation);
            float error = Mathf.Max(0f, actualError - settings.sensorNoiseDeadZone);
            float normalized = Mathf.InverseLerp(effectivePerfect, effectiveMaximum, error);
            float boneScore = 1f - Mathf.Pow(normalized, 1.7f);
            float weight = Mathf.Max(0f, weightProvider != null
                ? weightProvider(bone)
                : GetDefaultBoneWeight(bone));
            if (weight <= 0f) continue;

            boneScores.Add(new WeightedBoneScore { score = boneScore, weight = weight });
            weightedTotal += boneScore * weight;
            totalWeight += weight;
            result.validBoneCount++;
            if (actualError > result.worstErrorDegrees)
            {
                result.worstErrorDegrees = actualError;
                result.worstBone = bone;
                result.worstRegion = GetRegion(bone);
            }
        }

        if (totalWeight <= 0f) return result;
        float mean = weightedTotal / totalWeight;
        float median = CalculateWeightedMedian(boneScores, totalWeight);
        result.rawScore = Mathf.Clamp01(Mathf.Lerp(mean, median, settings.robustMedianBlend)) * 100f;
        float dt = Mathf.Clamp(deltaTime > 0f ? deltaTime : Time.deltaTime, 1f / 240f, 0.1f);
        result.score = smoothLiveScore ? SmoothLiveScore(result.rawScore, dt) : result.rawScore;
        return result;
    }

    private Quaternion FilterRotation(HumanBodyBones bone, Quaternion rawRotation, float deltaTime)
    {
        if (!filteredUserRotations.TryGetValue(bone, out Quaternion filtered))
        {
            filteredUserRotations[bone] = rawRotation;
            return rawRotation;
        }

        float maximumStep = settings.maximumAngularSpeed * deltaTime;
        Quaternion limited = Quaternion.RotateTowards(filtered, rawRotation, maximumStep);
        float alpha = 1f - Mathf.Exp(-deltaTime / Mathf.Max(0.01f, settings.rotationSmoothingTime));
        filtered = Quaternion.Slerp(filtered, limited, alpha);
        filteredUserRotations[bone] = filtered;
        return filtered;
    }

    private float SmoothLiveScore(float rawScore, float deltaTime)
    {
        if (!hasDisplayedScore)
        {
            displayedScore = rawScore;
            hasDisplayedScore = true;
            return displayedScore;
        }

        float responseTime = rawScore >= displayedScore
            ? settings.liveScoreRiseTime
            : settings.liveScoreFallTime;
        float alpha = 1f - Mathf.Exp(-deltaTime / Mathf.Max(0.01f, responseTime));
        displayedScore = Mathf.Lerp(displayedScore, rawScore, alpha);
        return displayedScore;
    }

    private static float CalculateWeightedMedian(List<WeightedBoneScore> values, float totalWeight)
    {
        values.Sort((left, right) => left.score.CompareTo(right.score));
        float accumulated = 0f;
        float midpoint = totalWeight * 0.5f;
        foreach (WeightedBoneScore value in values)
        {
            accumulated += value.weight;
            if (accumulated >= midpoint)
            {
                return value.score;
            }
        }

        return values.Count > 0 ? values[values.Count - 1].score : 0f;
    }

    public static float CalculateRobustAverage(IReadOnlyList<float> scores, float trimFraction)
    {
        if (scores == null || scores.Count == 0)
        {
            return 0f;
        }

        var sorted = new List<float>(scores.Count);
        for (int index = 0; index < scores.Count; index++)
        {
            if (!float.IsNaN(scores[index]) && !float.IsInfinity(scores[index]))
            {
                sorted.Add(Mathf.Clamp(scores[index], 0f, 100f));
            }
        }

        if (sorted.Count == 0)
        {
            return 0f;
        }

        sorted.Sort();
        int trim = Mathf.Min(
            Mathf.FloorToInt(sorted.Count * Mathf.Clamp(trimFraction, 0f, 0.24f)),
            (sorted.Count - 1) / 2);
        float total = 0f;
        int count = 0;
        for (int index = trim; index < sorted.Count - trim; index++)
        {
            total += sorted[index];
            count++;
        }

        return count > 0 ? total / count : sorted[sorted.Count / 2];
    }

    public static float GetDefaultBoneWeight(HumanBodyBones bone)
    {
        switch (bone)
        {
            // Multiple torso bones often contain the same accumulated IMU drift;
            // down-weighting them avoids counting one sensor problem three times.
            case HumanBodyBones.Hips:
            case HumanBodyBones.Spine:
            case HumanBodyBones.Chest:
            case HumanBodyBones.UpperChest:
                return 0.65f;
            case HumanBodyBones.LeftLowerArm:
            case HumanBodyBones.RightLowerArm:
            case HumanBodyBones.LeftLowerLeg:
            case HumanBodyBones.RightLowerLeg:
                return 0.85f;
            default:
                return 1f;
        }
    }

    public static PoseBodyRegion GetRegion(HumanBodyBones bone)
    {
        switch (bone)
        {
            case HumanBodyBones.LeftUpperArm:
            case HumanBodyBones.LeftLowerArm:
            case HumanBodyBones.LeftHand:
            case HumanBodyBones.LeftShoulder:
                return PoseBodyRegion.LeftArm;
            case HumanBodyBones.RightUpperArm:
            case HumanBodyBones.RightLowerArm:
            case HumanBodyBones.RightHand:
            case HumanBodyBones.RightShoulder:
                return PoseBodyRegion.RightArm;
            case HumanBodyBones.LeftUpperLeg:
            case HumanBodyBones.LeftLowerLeg:
            case HumanBodyBones.LeftFoot:
                return PoseBodyRegion.LeftLeg;
            case HumanBodyBones.RightUpperLeg:
            case HumanBodyBones.RightLowerLeg:
            case HumanBodyBones.RightFoot:
                return PoseBodyRegion.RightLeg;
            case HumanBodyBones.Hips:
            case HumanBodyBones.Spine:
            case HumanBodyBones.Chest:
            case HumanBodyBones.UpperChest:
                return PoseBodyRegion.Torso;
            default:
                return PoseBodyRegion.None;
        }
    }

    public static string GetEncouragingFeedback(float score, PoseBodyRegion region)
    {
        if (score >= 90f) return "动作很标准，继续保持";
        if (score >= 80f) return "完成得很好，保持节奏";
        if (score >= 65f) return $"整体不错，{GetRegionAdjustment(region)}";
        return $"已经完成，下一次可尝试{GetRegionAdjustment(region)}";
    }

    private static string GetRegionAdjustment(PoseBodyRegion region)
    {
        switch (region)
        {
            case PoseBodyRegion.LeftArm: return "微调左臂位置";
            case PoseBodyRegion.RightArm: return "微调右臂位置";
            case PoseBodyRegion.LeftLeg: return "微调左腿位置";
            case PoseBodyRegion.RightLeg: return "微调右腿位置";
            case PoseBodyRegion.Torso: return "保持躯干稳定";
            default: return "放慢动作并稳定保持";
        }
    }
}
