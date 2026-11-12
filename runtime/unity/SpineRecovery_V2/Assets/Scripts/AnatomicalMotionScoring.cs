using System;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public sealed class AnatomicalMotionScoringSettings
{
    [Tooltip("Angular jitter ignored for every tracked segment.")]
    [Min(0f)] public float sensorDeadZone = 7f;
    [Tooltip("Errors up to this angle still receive full credit.")]
    [Min(0f)] public float perfectAngle = 15f;
    [Tooltip("A segment reaches zero only at a clearly incorrect angle.")]
    [Min(20f)] public float zeroScoreAngle = 60f;
    [Tooltip("Raises only the middle of the score curve. Perfect and clearly wrong poses remain unchanged.")]
    [Range(0f, 0.3f)] public float midRangeScoreBoost = 0.18f;
    [Tooltip("Short direction smoothing suppresses wearable jitter without making motion sluggish.")]
    [Min(0.01f)] public float directionSmoothingTime = 0.07f;
    [Tooltip("Rejects physically implausible one-frame Tracker jumps.")]
    [Min(90f)] public float maximumAngularSpeed = 720f;
    [Range(0f, 0.4f)] public float robustMedianBlend = 0.15f;
    [Min(0.01f)] public float scoreRiseTime = 0.18f;
    [Min(0.01f)] public float scoreFallTime = 0.55f;

    [Header("Human response delay")]
    [Min(0.1f)] public float coachHistorySeconds = 0.7f;
    [Range(0f, 0.7f)] public float maximumResponseDelay = 0.55f;
    [Range(0f, 0.7f)] public float initialResponseDelay = 0.2f;
    [Min(0f)] public float delayContinuityPenalty = 10f;
    [Min(0.05f)] public float delayAdaptationTime = 0.35f;

    [Header("Anatomical reliability weights")]
    [Range(0f, 2f)] public float upperArmWeight = 1f;
    [Range(0f, 1f)] public float forearmWeight = 0.25f;
    [Range(0f, 0.25f)] public float handWeight = 0.05f;
    [Range(0f, 2f)] public float upperLegWeight = 1.15f;
    [Range(0f, 1f)] public float lowerLegWeight = 0.35f;
    [Range(0f, 0.25f)] public float headWeight = 0.08f;
}

public struct AnatomicalPoseScore
{
    public float score;
    public float rawScore;
    public float matchedDelay;
    public float worstErrorDegrees;
    public float primarySegmentErrorDegrees;
    public float secondarySegmentErrorDegrees;
    public Vector3 correctionDirection;
    public HumanBodyBones worstSegment;
    public int validSegmentCount;
}

public enum AnatomicalLimb
{
    LeftArm,
    RightArm,
    LeftLeg,
    RightLeg
}

public struct AnatomicalLimbScores
{
    public AnatomicalPoseScore leftArm;
    public AnatomicalPoseScore rightArm;
    public AnatomicalPoseScore leftLeg;
    public AnatomicalPoseScore rightLeg;
}

/// <summary>
/// Scores anatomical segment directions in a body-relative coordinate frame.
/// Global room heading, standing/lying orientation, avatar proportions and bone
/// local-axis differences therefore do not directly affect the result.
/// </summary>
public sealed class AnatomicalMotionScoringEngine
{
    private sealed class CoachSample
    {
        public float timestamp;
        public Dictionary<HumanBodyBones, Vector3> directions;
    }

    private struct WeightedScore
    {
        public float score;
        public float weight;
    }

    private readonly AnatomicalMotionScoringSettings settings;
    private readonly Dictionary<HumanBodyBones, Quaternion> userCalibrationAlignments =
        new Dictionary<HumanBodyBones, Quaternion>();
    private readonly Dictionary<HumanBodyBones, Vector3> filteredUserDirections =
        new Dictionary<HumanBodyBones, Vector3>();
    private readonly List<CoachSample> coachHistory = new List<CoachSample>();
    private readonly List<WeightedScore> scoreScratch = new List<WeightedScore>(11);
    private readonly Dictionary<AnatomicalLimb, float> displayedLimbScores =
        new Dictionary<AnatomicalLimb, float>();
    private readonly Dictionary<AnatomicalLimb, float> estimatedLimbDelays =
        new Dictionary<AnatomicalLimb, float>();
    private float estimatedDelay;
    private float displayedScore;
    private bool hasDisplayedScore;

    public int CalibratedSegmentCount => userCalibrationAlignments.Count;

    public AnatomicalMotionScoringEngine(AnatomicalMotionScoringSettings settings)
    {
        this.settings = settings ?? new AnatomicalMotionScoringSettings();
        estimatedDelay = this.settings.initialResponseDelay;
    }

    public void SetStandingCalibration(
        IReadOnlyDictionary<HumanBodyBones, Vector3> userDirections,
        IReadOnlyDictionary<HumanBodyBones, Vector3> coachDirections,
        float maximumCorrectionDegrees = 180f)
    {
        userCalibrationAlignments.Clear();
        if (userDirections == null || coachDirections == null) return;
        foreach (KeyValuePair<HumanBodyBones, Vector3> entry in userDirections)
        {
            if (!coachDirections.TryGetValue(entry.Key, out Vector3 coachDirection)) continue;
            if (entry.Value.sqrMagnitude < 0.5f || coachDirection.sqrMagnitude < 0.5f) continue;
            Quaternion alignment = Quaternion.FromToRotation(
                entry.Value.normalized,
                coachDirection.normalized);
            userCalibrationAlignments[entry.Key] = Quaternion.RotateTowards(
                Quaternion.identity, alignment, Mathf.Max(0f, maximumCorrectionDegrees));
        }
        ResetSession();
    }

    /// <summary>
    /// Uses the already body-relative anatomical directions without applying a
    /// single-pose alignment. This is appropriate for guidance when the coach
    /// and user do not share a guaranteed neutral calibration frame.
    /// </summary>
    public void ClearCalibration()
    {
        userCalibrationAlignments.Clear();
        ResetSession();
    }

    public void ResetSession()
    {
        filteredUserDirections.Clear();
        coachHistory.Clear();
        displayedLimbScores.Clear();
        estimatedLimbDelays.Clear();
        estimatedDelay = Mathf.Clamp(settings.initialResponseDelay, 0f, settings.maximumResponseDelay);
        displayedScore = 0f;
        hasDisplayedScore = false;
    }

    public Dictionary<HumanBodyBones, Vector3> CaptureRawPose(Animator animator)
    {
        var result = new Dictionary<HumanBodyBones, Vector3>();
        if (animator == null || !TryGetBodyFrame(animator, out Quaternion bodyFrame)) return result;
        Quaternion worldToBody = Quaternion.Inverse(bodyFrame);

        AddSegment(animator, HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm, worldToBody, result);
        AddSegment(animator, HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand, worldToBody, result);
        AddOrientation(animator, HumanBodyBones.LeftHand, worldToBody, result);
        AddSegment(animator, HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, worldToBody, result);
        AddSegment(animator, HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand, worldToBody, result);
        AddOrientation(animator, HumanBodyBones.RightHand, worldToBody, result);
        AddSegment(animator, HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, worldToBody, result);
        AddSegment(animator, HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot, worldToBody, result);
        AddSegment(animator, HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, worldToBody, result);
        AddSegment(animator, HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot, worldToBody, result);
        AddOrientation(animator, HumanBodyBones.Head, worldToBody, result);
        return result;
    }
    public Dictionary<HumanBodyBones,Vector3> CaptureAlignedPose(Animator animator)
    {
        var pose=CaptureRawPose(animator);
        foreach(var bone in new List<HumanBodyBones>(pose.Keys))
            if(userCalibrationAlignments.TryGetValue(bone,out var alignment))pose[bone]=alignment*pose[bone];
        return pose;
    }

    public AnatomicalPoseScore Evaluate(Animator userAnimator, Animator coachAnimator, float deltaTime)
    {
        return EvaluateInternal(
            userAnimator,
            coachAnimator,
            null,
            deltaTime,
            false,
            out _);
    }

    public AnatomicalPoseScore Evaluate(
        Animator userAnimator,
        Animator coachAnimator,
        float deltaTime,
        out AnatomicalLimbScores limbScores)
    {
        return EvaluateInternal(
            userAnimator,
            coachAnimator,
            null,
            deltaTime,
            true,
            out limbScores);
    }

    /// <summary>
    /// Evaluates against an explicit body-relative coach pose. This lets
    /// guidance correct a known imported target without mutating the Animator
    /// pose consumed by the independent scoring system.
    /// </summary>
    public AnatomicalPoseScore Evaluate(
        Animator userAnimator,
        IReadOnlyDictionary<HumanBodyBones, Vector3> coachPose,
        float deltaTime,
        out AnatomicalLimbScores limbScores)
    {
        return EvaluateInternal(
            userAnimator,
            null,
            coachPose,
            deltaTime,
            true,
            out limbScores);
    }

    private AnatomicalPoseScore EvaluateInternal(
        Animator userAnimator,
        Animator coachAnimator,
        IReadOnlyDictionary<HumanBodyBones, Vector3> coachPoseOverride,
        float deltaTime,
        bool calculateLimbScores,
        out AnatomicalLimbScores limbScores)
    {
        float dt = Mathf.Clamp(deltaTime > 0f ? deltaTime : Time.deltaTime, 1f / 240f, 0.1f);
        Dictionary<HumanBodyBones, Vector3> rawUser = CaptureRawPose(userAnimator);
        Dictionary<HumanBodyBones, Vector3> coachPose =
            coachPoseOverride != null
                ? new Dictionary<HumanBodyBones, Vector3>(coachPoseOverride)
                : CaptureRawPose(coachAnimator);
        var userPose = new Dictionary<HumanBodyBones, Vector3>();

        foreach (KeyValuePair<HumanBodyBones, Vector3> entry in rawUser)
        {
            Vector3 filtered = FilterDirection(entry.Key, entry.Value, dt);
            if (userCalibrationAlignments.TryGetValue(entry.Key, out Quaternion alignment))
                filtered = alignment * filtered;
            userPose[entry.Key] = filtered.normalized;
        }

        float now = Time.unscaledTime;
        coachHistory.Add(new CoachSample { timestamp = now, directions = coachPose });
        TrimCoachHistory(now);

        CoachSample bestSample = coachHistory[coachHistory.Count - 1];
        float bestDelay = 0f;
        float bestObjective = float.NegativeInfinity;
        foreach (CoachSample sample in coachHistory)
        {
            float delay = now - sample.timestamp;
            if (delay > settings.maximumResponseDelay) continue;
            AnatomicalPoseScore candidate = ScorePose(userPose, sample.directions, null);
            float objective = candidate.rawScore -
                Mathf.Abs(delay - estimatedDelay) * settings.delayContinuityPenalty;
            if (objective <= bestObjective) continue;
            bestObjective = objective;
            bestDelay = delay;
            bestSample = sample;
        }

        float adaptation = 1f - Mathf.Exp(-dt / Mathf.Max(0.05f, settings.delayAdaptationTime));
        estimatedDelay = Mathf.Lerp(estimatedDelay, bestDelay, adaptation);
        AnatomicalPoseScore result = ScorePose(userPose, bestSample.directions, null);
        result.matchedDelay = bestDelay;
        result.score = SmoothScore(result.rawScore, dt);

        limbScores = calculateLimbScores
            ? new AnatomicalLimbScores
        {
            leftArm = ScoreLimb(userPose, now, AnatomicalLimb.LeftArm, dt),
            rightArm = ScoreLimb(userPose, now, AnatomicalLimb.RightArm, dt),
            leftLeg = ScoreLimb(userPose, now, AnatomicalLimb.LeftLeg, dt),
            rightLeg = ScoreLimb(userPose, now, AnatomicalLimb.RightLeg, dt)
        }
            : default(AnatomicalLimbScores);
        return result;
    }

    private AnatomicalPoseScore ScoreLimb(
        IReadOnlyDictionary<HumanBodyBones, Vector3> userPose,
        float now,
        AnatomicalLimb limb,
        float deltaTime)
    {
        float estimated = estimatedLimbDelays.TryGetValue(limb, out float previousEstimate)
            ? previousEstimate
            : Mathf.Clamp(settings.initialResponseDelay, 0f, settings.maximumResponseDelay);
        CoachSample bestSample = coachHistory[coachHistory.Count - 1];
        float bestDelay = 0f;
        float bestObjective = float.NegativeInfinity;

        foreach (CoachSample sample in coachHistory)
        {
            float delay = now - sample.timestamp;
            if (delay > settings.maximumResponseDelay) continue;
            AnatomicalPoseScore candidate = ScorePose(userPose, sample.directions, limb);
            float objective = candidate.rawScore -
                Mathf.Abs(delay - estimated) * settings.delayContinuityPenalty;
            if (objective <= bestObjective) continue;
            bestObjective = objective;
            bestDelay = delay;
            bestSample = sample;
        }

        float adaptation = 1f - Mathf.Exp(
            -deltaTime / Mathf.Max(0.05f, settings.delayAdaptationTime));
        estimatedLimbDelays[limb] = Mathf.Lerp(estimated, bestDelay, adaptation);

        AnatomicalPoseScore result = ScorePose(userPose, bestSample.directions, limb);
        result.matchedDelay = bestDelay;
        result.score = SmoothLimbScore(limb, result.rawScore, deltaTime);
        return result;
    }

    private AnatomicalPoseScore ScorePose(
        IReadOnlyDictionary<HumanBodyBones, Vector3> userPose,
        IReadOnlyDictionary<HumanBodyBones, Vector3> coachPose,
        AnatomicalLimb? limb)
    {
        var result = new AnatomicalPoseScore { worstSegment = HumanBodyBones.LastBone };
        scoreScratch.Clear();
        List<WeightedScore> values = scoreScratch;
        float weightedTotal = 0f;
        float totalWeight = 0f;
        float worstSignificance = -1f;

        foreach (KeyValuePair<HumanBodyBones, Vector3> entry in userPose)
        {
            if (limb.HasValue && !BelongsToLimb(entry.Key, limb.Value)) continue;
            if (!coachPose.TryGetValue(entry.Key, out Vector3 coachDirection)) continue;
            float weight = GetWeight(entry.Key);
            if (weight <= 0f) continue;
            float error = Vector3.Angle(entry.Value, coachDirection);
            if (limb.HasValue &&
                IsPrimarySegment(entry.Key, limb.Value))
            {
                result.primarySegmentErrorDegrees = error;
            }
            else if (limb.HasValue &&
                IsSecondarySegment(entry.Key, limb.Value))
            {
                result.secondarySegmentErrorDegrees = error;
            }
            float effectiveError = Mathf.Max(0f, error - settings.sensorDeadZone);
            float perfect = Mathf.Max(0f, settings.perfectAngle - settings.sensorDeadZone);
            float maximum = Mathf.Max(perfect + 1f, settings.zeroScoreAngle - settings.sensorDeadZone);
            float normalized = Mathf.InverseLerp(perfect, maximum, effectiveError);
            float eased = normalized * normalized * (3f - 2f * normalized);
            float baseScore = 1f - eased;
            // Add encouragement where normal motion/tracker differences live,
            // while mathematically preserving both endpoints: 0 stays 0 and 1 stays 1.
            float middleInfluence = 4f * baseScore * (1f - baseScore);
            float segmentScore = Mathf.Clamp01(
                baseScore + settings.midRangeScoreBoost * middleInfluence);

            values.Add(new WeightedScore { score = segmentScore, weight = weight });
            weightedTotal += segmentScore * weight;
            totalWeight += weight;
            result.validSegmentCount++;

            float significance = error * Mathf.Sqrt(weight);
            if (significance > worstSignificance)
            {
                worstSignificance = significance;
                result.worstErrorDegrees = error;
                result.correctionDirection = coachDirection - entry.Value;
                result.worstSegment = entry.Key;
            }
        }

        if (totalWeight <= 0f) return result;
        float mean = weightedTotal / totalWeight;
        float median = WeightedMedian(values, totalWeight);
        result.rawScore = Mathf.Clamp01(Mathf.Lerp(mean, median, settings.robustMedianBlend)) * 100f;
        return result;
    }

    private Vector3 FilterDirection(HumanBodyBones segment, Vector3 raw, float dt)
    {
        raw.Normalize();
        if (!filteredUserDirections.TryGetValue(segment, out Vector3 filtered))
        {
            filteredUserDirections[segment] = raw;
            return raw;
        }
        float angle = Vector3.Angle(filtered, raw);
        float limitedAngle = Mathf.Min(angle, settings.maximumAngularSpeed * dt);
        Vector3 limited = angle > 0.001f
            ? Vector3.Slerp(filtered, raw, limitedAngle / angle).normalized
            : raw;
        float alpha = 1f - Mathf.Exp(-dt / Mathf.Max(0.01f, settings.directionSmoothingTime));
        filtered = Vector3.Slerp(filtered, limited, alpha).normalized;
        filteredUserDirections[segment] = filtered;
        return filtered;
    }

    private float SmoothScore(float rawScore, float dt)
    {
        if (!hasDisplayedScore)
        {
            displayedScore = rawScore;
            hasDisplayedScore = true;
            return rawScore;
        }
        float response = rawScore >= displayedScore ? settings.scoreRiseTime : settings.scoreFallTime;
        float alpha = 1f - Mathf.Exp(-dt / Mathf.Max(0.01f, response));
        displayedScore = Mathf.Lerp(displayedScore, rawScore, alpha);
        return displayedScore;
    }

    private float SmoothLimbScore(AnatomicalLimb limb, float rawScore, float dt)
    {
        if (!displayedLimbScores.TryGetValue(limb, out float displayed))
        {
            displayedLimbScores[limb] = rawScore;
            return rawScore;
        }

        float response = rawScore >= displayed ? settings.scoreRiseTime : settings.scoreFallTime;
        float alpha = 1f - Mathf.Exp(-dt / Mathf.Max(0.01f, response));
        displayed = Mathf.Lerp(displayed, rawScore, alpha);
        displayedLimbScores[limb] = displayed;
        return displayed;
    }

    private static bool BelongsToLimb(HumanBodyBones segment, AnatomicalLimb limb)
    {
        switch (limb)
        {
            case AnatomicalLimb.LeftArm:
                return segment == HumanBodyBones.LeftUpperArm ||
                    segment == HumanBodyBones.LeftLowerArm ||
                    segment == HumanBodyBones.LeftHand;
            case AnatomicalLimb.RightArm:
                return segment == HumanBodyBones.RightUpperArm ||
                    segment == HumanBodyBones.RightLowerArm ||
                    segment == HumanBodyBones.RightHand;
            case AnatomicalLimb.LeftLeg:
                return segment == HumanBodyBones.LeftUpperLeg ||
                    segment == HumanBodyBones.LeftLowerLeg;
            case AnatomicalLimb.RightLeg:
                return segment == HumanBodyBones.RightUpperLeg ||
                    segment == HumanBodyBones.RightLowerLeg;
            default:
                return false;
        }
    }

    private static bool IsPrimarySegment(
        HumanBodyBones segment,
        AnatomicalLimb limb)
    {
        switch (limb)
        {
            case AnatomicalLimb.LeftArm:
                return segment == HumanBodyBones.LeftUpperArm;
            case AnatomicalLimb.RightArm:
                return segment == HumanBodyBones.RightUpperArm;
            case AnatomicalLimb.LeftLeg:
                return segment == HumanBodyBones.LeftUpperLeg;
            case AnatomicalLimb.RightLeg:
                return segment == HumanBodyBones.RightUpperLeg;
            default:
                return false;
        }
    }

    private static bool IsSecondarySegment(
        HumanBodyBones segment,
        AnatomicalLimb limb)
    {
        switch (limb)
        {
            case AnatomicalLimb.LeftArm:
                return segment == HumanBodyBones.LeftLowerArm;
            case AnatomicalLimb.RightArm:
                return segment == HumanBodyBones.RightLowerArm;
            case AnatomicalLimb.LeftLeg:
                return segment == HumanBodyBones.LeftLowerLeg;
            case AnatomicalLimb.RightLeg:
                return segment == HumanBodyBones.RightLowerLeg;
            default:
                return false;
        }
    }

    private void TrimCoachHistory(float now)
    {
        float oldest = now - Mathf.Max(settings.coachHistorySeconds, settings.maximumResponseDelay + 0.05f);
        int removeCount = 0;
        while (removeCount < coachHistory.Count && coachHistory[removeCount].timestamp < oldest) removeCount++;
        if (removeCount > 0) coachHistory.RemoveRange(0, removeCount);
    }

    private float GetWeight(HumanBodyBones segment)
    {
        switch (segment)
        {
            case HumanBodyBones.LeftUpperArm:
            case HumanBodyBones.RightUpperArm:
                return settings.upperArmWeight;
            case HumanBodyBones.LeftLowerArm:
            case HumanBodyBones.RightLowerArm:
                return settings.forearmWeight;
            case HumanBodyBones.LeftHand:
            case HumanBodyBones.RightHand:
                return settings.handWeight;
            case HumanBodyBones.LeftUpperLeg:
            case HumanBodyBones.RightUpperLeg:
                return settings.upperLegWeight;
            case HumanBodyBones.LeftLowerLeg:
            case HumanBodyBones.RightLowerLeg:
                return settings.lowerLegWeight;
            case HumanBodyBones.Head:
                return settings.headWeight;
            default:
                return 0f;
        }
    }

    private static float WeightedMedian(List<WeightedScore> values, float totalWeight)
    {
        values.Sort((left, right) => left.score.CompareTo(right.score));
        float accumulated = 0f;
        foreach (WeightedScore value in values)
        {
            accumulated += value.weight;
            if (accumulated >= totalWeight * 0.5f) return value.score;
        }
        return values.Count > 0 ? values[values.Count - 1].score : 0f;
    }

    public static bool TryGetBodyFrame(Animator animator, out Quaternion frame)
    {
        frame = Quaternion.identity;
        Transform hips = animator.GetBoneTransform(HumanBodyBones.Hips);
        Transform chest = animator.GetBoneTransform(HumanBodyBones.Chest) ??
                          animator.GetBoneTransform(HumanBodyBones.Spine);
        // Prefer the actual shoulder joints. Using upper-arm transforms as the
        // shoulder line lets an overhead arm pose rotate the reference frame
        // that the same arm is being measured against.
        Transform leftShoulder =
            animator.GetBoneTransform(HumanBodyBones.LeftShoulder) ??
            animator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
        Transform rightShoulder =
            animator.GetBoneTransform(HumanBodyBones.RightShoulder) ??
            animator.GetBoneTransform(HumanBodyBones.RightUpperArm);
        if (hips == null || chest == null || leftShoulder == null || rightShoulder == null) return false;

        Vector3 up = (chest.position - hips.position).normalized;
        Vector3 shoulderRight = rightShoulder.position - leftShoulder.position;
        if (up.sqrMagnitude < 0.5f || shoulderRight.sqrMagnitude < 1e-6f) return false;

        Vector3 right = shoulderRight.normalized;
        Transform leftHip = animator.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
        Transform rightHip = animator.GetBoneTransform(HumanBodyBones.RightUpperLeg);
        if (leftHip != null && rightHip != null)
        {
            Vector3 hipRight = rightHip.position - leftHip.position;
            if (hipRight.sqrMagnitude > 1e-6f)
            {
                hipRight.Normalize();
                if (Vector3.Dot(hipRight, right) < 0f)
                {
                    hipRight = -hipRight;
                }
                right = (right + hipRight).normalized;
            }
        }

        // Remove any shrug/retargeting skew so the lateral axis cannot borrow
        // headward motion from the arm pose.
        right = Vector3.ProjectOnPlane(right, up).normalized;
        Vector3 forward = Vector3.Cross(right, up).normalized;
        if (up.sqrMagnitude < 0.5f || right.sqrMagnitude < 0.5f || forward.sqrMagnitude < 0.5f) return false;
        right = Vector3.Cross(up, forward).normalized;
        up = Vector3.Cross(forward, right).normalized;
        frame = Quaternion.LookRotation(forward, up);
        return true;
    }

    private static void AddSegment(
        Animator animator,
        HumanBodyBones parent,
        HumanBodyBones child,
        Quaternion worldToBody,
        Dictionary<HumanBodyBones, Vector3> destination)
    {
        Transform parentTransform = animator.GetBoneTransform(parent);
        Transform childTransform = animator.GetBoneTransform(child);
        if (parentTransform == null || childTransform == null) return;
        Vector3 direction = childTransform.position - parentTransform.position;
        if (direction.sqrMagnitude < 1e-6f) return;
        destination[parent] = (worldToBody * direction.normalized).normalized;
    }

    private static void AddOrientation(
        Animator animator,
        HumanBodyBones bone,
        Quaternion worldToBody,
        Dictionary<HumanBodyBones, Vector3> destination)
    {
        Transform transform = animator.GetBoneTransform(bone);
        if (transform == null) return;
        destination[bone] = (worldToBody * transform.forward).normalized;
    }
}
