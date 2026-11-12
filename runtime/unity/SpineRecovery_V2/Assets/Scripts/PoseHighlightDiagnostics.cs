using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Product-wide, action-agnostic settings for corrective pose highlights.
/// These values describe the IMU system rather than any one exercise.
/// </summary>
[Serializable]
public sealed class PoseHighlightDiagnosticSettings
{
    [Tooltip("A target muscle axis must move this far from the preceding recorded target before it can produce a strict highlight.")]
    [Min(0f)] public float targetAxisMinimumDegrees = 18f;

    [Tooltip("A body region is treated as actively moving only when the recorded target changes by at least this much from the preceding target.")]
    [Min(0f)] public float activeRegionMotionMinimumDegrees = 20f;

    [Tooltip("Keep axes whose target motion is this fraction of the bone's dominant axis. This preserves diagonal actions while ignoring incidental twist and spacing.")]
    [Range(0.1f, 1f)] public float dominantAxisRatio = 0.45f;

    [Tooltip("Expected angular uncertainty for proximal arm and leg trackers.")]
    [Min(0f)] public float proximalLimbSensorAllowanceDegrees = 18f;

    [Tooltip("Expected angular uncertainty for noisier forearm and lower-leg trackers.")]
    [Min(0f)] public float distalLimbSensorAllowanceDegrees = 24f;

    [Tooltip("Expected angular uncertainty for pelvis and torso trackers.")]
    [Min(0f)] public float torsoSensorAllowanceDegrees = 15f;

    [Tooltip("A relevant limb error must exceed its sensor allowance by this amount before it becomes a normal highlight candidate.")]
    [Min(1f)] public float limbActivationExcessDegrees = 24f;

    [Tooltip("A highlighted limb is considered recovered below this excess error.")]
    [Min(0f)] public float limbClearExcessDegrees = 14f;

    [Tooltip("A torso error must exceed its sensor allowance by this amount before it becomes a highlight candidate.")]
    [Min(1f)] public float torsoActivationExcessDegrees = 22f;

    [Tooltip("A highlighted torso is considered recovered below this excess error.")]
    [Min(0f)] public float torsoClearExcessDegrees = 10f;

    [Tooltip("A single distal tracker must be this much worse than the normal boundary before it can highlight a whole region by itself.")]
    [Min(1f)] public float strongSingleBoneActivation = 1.35f;

    [Tooltip("Minimum fraction of relevant bones that must be tracked before the region is diagnosable.")]
    [Range(0f, 1f)] public float minimumTrackedCoverage = 0.5f;

    [Tooltip("Time constant for wearable pose smoothing used only by guidance.")]
    [Min(0.01f)] public float rotationSmoothingTime = 0.18f;

    [Tooltip("Maximum believable angular speed before a tracker update is treated as a spike.")]
    [Min(90f)] public float maximumAngularSpeed = 420f;

    [Tooltip("Additional uncertainty when Humanoid muscle data is unavailable and local-rotation fallback is used.")]
    [Min(0f)] public float localRotationFallbackAllowanceDegrees = 12f;

    [Tooltip("Extra tolerance for a region that is expected to remain stable rather than drive the action.")]
    [Min(0f)] public float stabilizerAdditionalAllowanceDegrees = 10f;

    [Tooltip("Extra tolerance for an arm or leg that only needs to remain generally extended.")]
    [Min(0f)] public float simpleExtensionAdditionalAllowanceDegrees = 12f;

    [Tooltip("Excess bend required before a generally extended arm or leg becomes a highlight candidate.")]
    [Min(1f)] public float simpleExtensionActivationExcessDegrees = 30f;

    [Tooltip("Excess bend below which a generally extended arm or leg is considered recovered.")]
    [Min(0f)] public float simpleExtensionClearExcessDegrees = 18f;
}

public enum PoseHighlightBoneRole
{
    ActiveMotion,
    Stabilizer,
    SimpleExtension,
    Ignore
}

public struct PoseHighlightBoneSample
{
    public HumanBodyBones bone;
    public PoseHighlightBoneRole role;
    public bool isTracked;
    public int availableAxisMask;
    public Vector3 errorDegrees;
    public Vector3 targetSalienceDegrees;
    public float additionalSensorAllowanceDegrees;
}

public struct PoseHighlightRegionDiagnostic
{
    public BodyPart part;
    public bool isValid;
    public bool shouldActivate;
    public bool shouldClear;
    public int relevantBoneCount;
    public int validBoneCount;
    public float trackingCoverage;
    public float worstRawErrorDegrees;
    public float severity;
}

public struct PoseHighlightFrameDiagnostics
{
    public PoseHighlightRegionDiagnostic leftArm;
    public PoseHighlightRegionDiagnostic rightArm;
    public PoseHighlightRegionDiagnostic leftLeg;
    public PoseHighlightRegionDiagnostic rightLeg;
    public PoseHighlightRegionDiagnostic torso;

    public PoseHighlightRegionDiagnostic Get(BodyPart part)
    {
        switch (part)
        {
            case BodyPart.LeftArm:
                return leftArm;
            case BodyPart.RightArm:
                return rightArm;
            case BodyPart.LeftLeg:
                return leftLeg;
            case BodyPart.RightLeg:
                return rightLeg;
            case BodyPart.Torso:
                return torso;
            default:
                return default(PoseHighlightRegionDiagnostic);
        }
    }
}

/// <summary>
/// Pure diagnostic aggregation. It deliberately consumes angular evidence,
/// never presentation scores, score mappings, or exercise-specific metadata.
/// </summary>
public static class PoseHighlightDiagnosticMath
{
    private struct RegionAccumulator
    {
        public BodyPart part;
        public int relevantBoneCount;
        public int validBoneCount;
        public int supportingErrorCount;
        public float worstRawError;
        public float maximumSeverity;
        public float maximumPrimarySeverity;
        public float maximumClearRatio;

        public void Add(
            PoseHighlightBoneSample sample,
            PoseHighlightDiagnosticSettings settings)
        {
            if (sample.role == PoseHighlightBoneRole.Ignore)
            {
                return;
            }

            bool torso = part == BodyPart.Torso;
            bool simpleExtension =
                sample.role == PoseHighlightBoneRole.SimpleExtension;
            bool stabilizer =
                sample.role == PoseHighlightBoneRole.Stabilizer;
            int selectedAxes = simpleExtension
                ? sample.availableAxisMask & 1
                : stabilizer
                    ? sample.availableAxisMask & 0x7
                    : SelectRelevantAxes(
                    sample.targetSalienceDegrees,
                    sample.availableAxisMask,
                    false,
                    settings.targetAxisMinimumDegrees,
                    settings.dominantAxisRatio);
            if (selectedAxes == 0)
            {
                return;
            }

            relevantBoneCount++;
            if (!sample.isTracked)
            {
                return;
            }

            validBoneCount++;
            float rawError = MaximumSelectedAxis(
                sample.errorDegrees,
                selectedAxes);
            float sensorAllowance = GetSensorAllowance(sample.bone, settings) +
                Mathf.Max(0f, sample.additionalSensorAllowanceDegrees) +
                (stabilizer
                    ? settings.stabilizerAdditionalAllowanceDegrees
                    : 0f) +
                (simpleExtension
                    ? settings.simpleExtensionAdditionalAllowanceDegrees
                    : 0f);
            float effectiveError = Mathf.Max(0f, rawError - sensorAllowance);
            float activationExcess = simpleExtension
                ? settings.simpleExtensionActivationExcessDegrees
                : torso
                    ? settings.torsoActivationExcessDegrees
                    : settings.limbActivationExcessDegrees;
            float clearExcess = simpleExtension
                ? settings.simpleExtensionClearExcessDegrees
                : torso
                    ? settings.torsoClearExcessDegrees
                    : settings.limbClearExcessDegrees;
            float severity = effectiveError /
                Mathf.Max(1f, activationExcess);
            float clearRatio = effectiveError /
                Mathf.Max(1f, clearExcess);

            worstRawError = Mathf.Max(worstRawError, rawError);
            maximumSeverity = Mathf.Max(maximumSeverity, severity);
            maximumClearRatio = Mathf.Max(maximumClearRatio, clearRatio);
            if (severity >= 0.8f)
            {
                supportingErrorCount++;
            }

            if (IsPrimaryBone(sample.bone))
            {
                maximumPrimarySeverity = Mathf.Max(
                    maximumPrimarySeverity,
                    severity);
            }
        }

        public PoseHighlightRegionDiagnostic Build(
            PoseHighlightDiagnosticSettings settings)
        {
            float coverage = relevantBoneCount > 0
                ? (float)validBoneCount / relevantBoneCount
                : 0f;
            bool valid = relevantBoneCount > 0 &&
                validBoneCount > 0 &&
                coverage >= settings.minimumTrackedCoverage;
            bool torso = part == BodyPart.Torso;
            bool activate = valid &&
                (maximumSeverity >= settings.strongSingleBoneActivation ||
                 (maximumSeverity >= 1f && supportingErrorCount >= 2) ||
                 (!torso && maximumPrimarySeverity >= 1f));

            return new PoseHighlightRegionDiagnostic
            {
                part = part,
                isValid = valid,
                shouldActivate = activate,
                // Unknown tracking must not leave a stale red region forever.
                // The caller still applies the normal release delay.
                shouldClear = !valid || maximumClearRatio <= 1f,
                relevantBoneCount = relevantBoneCount,
                validBoneCount = validBoneCount,
                trackingCoverage = coverage,
                worstRawErrorDegrees = worstRawError,
                severity = valid ? maximumSeverity : 0f
            };
        }
    }

    public static PoseHighlightFrameDiagnostics Evaluate(
        IReadOnlyList<PoseHighlightBoneSample> samples,
        PoseHighlightDiagnosticSettings settings)
    {
        settings = settings ?? new PoseHighlightDiagnosticSettings();
        RegionAccumulator leftArm = NewAccumulator(BodyPart.LeftArm);
        RegionAccumulator rightArm = NewAccumulator(BodyPart.RightArm);
        RegionAccumulator leftLeg = NewAccumulator(BodyPart.LeftLeg);
        RegionAccumulator rightLeg = NewAccumulator(BodyPart.RightLeg);
        RegionAccumulator torso = NewAccumulator(BodyPart.Torso);

        if (samples != null)
        {
            for (int index = 0; index < samples.Count; index++)
            {
                PoseHighlightBoneSample sample = samples[index];
                switch (GetBodyPart(sample.bone))
                {
                    case BodyPart.LeftArm:
                        leftArm.Add(sample, settings);
                        break;
                    case BodyPart.RightArm:
                        rightArm.Add(sample, settings);
                        break;
                    case BodyPart.LeftLeg:
                        leftLeg.Add(sample, settings);
                        break;
                    case BodyPart.RightLeg:
                        rightLeg.Add(sample, settings);
                        break;
                    case BodyPart.Torso:
                        torso.Add(sample, settings);
                        break;
                }
            }
        }

        return new PoseHighlightFrameDiagnostics
        {
            leftArm = leftArm.Build(settings),
            rightArm = rightArm.Build(settings),
            leftLeg = leftLeg.Build(settings),
            rightLeg = rightLeg.Build(settings),
            torso = torso.Build(settings)
        };
    }

    public static int SelectRelevantAxes(
        Vector3 salienceDegrees,
        int availableAxisMask,
        bool alwaysUseAvailableAxes,
        float minimumDegrees,
        float dominantRatio)
    {
        int available = availableAxisMask & 0x7;
        if (available == 0)
        {
            return 0;
        }

        if (alwaysUseAvailableAxes)
        {
            return available;
        }

        float x = (available & 1) != 0 ? Mathf.Abs(salienceDegrees.x) : 0f;
        float y = (available & 2) != 0 ? Mathf.Abs(salienceDegrees.y) : 0f;
        float z = (available & 4) != 0 ? Mathf.Abs(salienceDegrees.z) : 0f;
        float dominant = Mathf.Max(x, Mathf.Max(y, z));
        if (dominant < Mathf.Max(0f, minimumDegrees))
        {
            return 0;
        }

        float cutoff = Mathf.Max(
            Mathf.Max(0f, minimumDegrees),
            dominant * Mathf.Clamp01(dominantRatio));
        int selected = 0;
        if (x >= cutoff) selected |= 1;
        if (y >= cutoff) selected |= 2;
        if (z >= cutoff) selected |= 4;
        return selected;
    }

    public static BodyPart GetBodyPart(HumanBodyBones bone)
    {
        switch (bone)
        {
            case HumanBodyBones.LeftShoulder:
            case HumanBodyBones.LeftUpperArm:
            case HumanBodyBones.LeftLowerArm:
            case HumanBodyBones.LeftHand:
                return BodyPart.LeftArm;
            case HumanBodyBones.RightShoulder:
            case HumanBodyBones.RightUpperArm:
            case HumanBodyBones.RightLowerArm:
            case HumanBodyBones.RightHand:
                return BodyPart.RightArm;
            case HumanBodyBones.LeftUpperLeg:
            case HumanBodyBones.LeftLowerLeg:
            case HumanBodyBones.LeftFoot:
                return BodyPart.LeftLeg;
            case HumanBodyBones.RightUpperLeg:
            case HumanBodyBones.RightLowerLeg:
            case HumanBodyBones.RightFoot:
                return BodyPart.RightLeg;
            case HumanBodyBones.Hips:
            case HumanBodyBones.Spine:
            case HumanBodyBones.Chest:
            case HumanBodyBones.UpperChest:
                return BodyPart.Torso;
            default:
                return BodyPart.None;
        }
    }

    private static RegionAccumulator NewAccumulator(BodyPart part)
    {
        return new RegionAccumulator { part = part };
    }

    private static float MaximumSelectedAxis(Vector3 value, int mask)
    {
        float maximum = 0f;
        if ((mask & 1) != 0) maximum = Mathf.Max(maximum, Mathf.Abs(value.x));
        if ((mask & 2) != 0) maximum = Mathf.Max(maximum, Mathf.Abs(value.y));
        if ((mask & 4) != 0) maximum = Mathf.Max(maximum, Mathf.Abs(value.z));
        return maximum;
    }

    private static float GetSensorAllowance(
        HumanBodyBones bone,
        PoseHighlightDiagnosticSettings settings)
    {
        switch (bone)
        {
            case HumanBodyBones.Hips:
            case HumanBodyBones.Spine:
            case HumanBodyBones.Chest:
            case HumanBodyBones.UpperChest:
                return settings.torsoSensorAllowanceDegrees;
            case HumanBodyBones.LeftLowerArm:
            case HumanBodyBones.RightLowerArm:
            case HumanBodyBones.LeftLowerLeg:
            case HumanBodyBones.RightLowerLeg:
                return settings.distalLimbSensorAllowanceDegrees;
            default:
                return settings.proximalLimbSensorAllowanceDegrees;
        }
    }

    private static bool IsPrimaryBone(HumanBodyBones bone)
    {
        switch (bone)
        {
            case HumanBodyBones.Hips:
            case HumanBodyBones.Spine:
            case HumanBodyBones.Chest:
            case HumanBodyBones.UpperChest:
            case HumanBodyBones.LeftUpperArm:
            case HumanBodyBones.RightUpperArm:
            case HumanBodyBones.LeftUpperLeg:
            case HumanBodyBones.RightUpperLeg:
                return true;
            default:
                return false;
        }
    }
}

/// <summary>
/// Captures normalized Humanoid evidence for the pure diagnostic aggregator.
/// A target's active regions and relevant axes are inferred automatically from
/// adjacent recorded targets; no exercise profile is required.
/// </summary>
public sealed class PoseHighlightDiagnosticEngine : IDisposable
{
    private readonly PoseHighlightDiagnosticSettings settings;
    private readonly List<PoseHighlightBoneSample> samples =
        new List<PoseHighlightBoneSample>(16);
    private readonly Dictionary<HumanBodyBones, Vector3> sessionCoachAngles =
        new Dictionary<HumanBodyBones, Vector3>();
    private readonly Dictionary<HumanBodyBones, Vector3> previousCoachAngles =
        new Dictionary<HumanBodyBones, Vector3>();
    private readonly Dictionary<HumanBodyBones, Vector3> targetCoachAngles =
        new Dictionary<HumanBodyBones, Vector3>();
    private readonly Dictionary<HumanBodyBones, Vector3> targetSalienceAngles =
        new Dictionary<HumanBodyBones, Vector3>();
    private readonly Dictionary<HumanBodyBones, int> targetAxisMasks =
        new Dictionary<HumanBodyBones, int>();
    private readonly Dictionary<HumanBodyBones, PoseHighlightBoneRole> targetBoneRoles =
        new Dictionary<HumanBodyBones, PoseHighlightBoneRole>();
    private readonly Dictionary<HumanBodyBones, Vector3> filteredUserAngles =
        new Dictionary<HumanBodyBones, Vector3>();
    private readonly Dictionary<HumanBodyBones, Quaternion> sessionCoachLocal =
        new Dictionary<HumanBodyBones, Quaternion>();
    private readonly Dictionary<HumanBodyBones, Quaternion> sessionUserLocal =
        new Dictionary<HumanBodyBones, Quaternion>();
    private readonly Dictionary<HumanBodyBones, Quaternion> previousCoachLocal =
        new Dictionary<HumanBodyBones, Quaternion>();
    private readonly Dictionary<HumanBodyBones, Quaternion> targetCoachLocal =
        new Dictionary<HumanBodyBones, Quaternion>();
    private readonly Dictionary<HumanBodyBones, Quaternion> filteredUserLocal =
        new Dictionary<HumanBodyBones, Quaternion>();

    private HumanPoseHandler userPoseHandler;
    private HumanPoseHandler coachPoseHandler;
    private Animator poseUserAnimator;
    private Animator poseCoachAnimator;
    private HumanPose userPose;
    private HumanPose coachPose;
    private Quaternion targetCoachBodyRotation = Quaternion.identity;
    private Quaternion previousCoachBodyRotation = Quaternion.identity;
    private bool hasPreviousCoachBodyRotation;
    private bool targetUsesHumanoidMuscles;
    private TrainingSegmentKind currentTargetSegmentKind =
        TrainingSegmentKind.Preparation;
    private SegmentPoseGuidanceRule currentGuidanceRule =
        SegmentPoseGuidanceRule.Auto;

    public bool HasTarget { get; private set; }

    public PoseHighlightDiagnosticEngine(
        PoseHighlightDiagnosticSettings settings)
    {
        this.settings = settings ?? new PoseHighlightDiagnosticSettings();
    }

    public void CaptureSessionReference(
        Animator userAnimator,
        Animator coachAnimator,
        IReadOnlyList<HumanBodyBones> bones)
    {
        ResetAll();
        CaptureLocalReference(
            userAnimator,
            coachAnimator,
            bones);

        if (!TryCaptureCoachPose(coachAnimator))
        {
            return;
        }

        previousCoachBodyRotation = coachPose.bodyRotation;
        hasPreviousCoachBodyRotation = true;

        for (int index = 0; bones != null && index < bones.Count; index++)
        {
            HumanBodyBones bone = bones[index];
            if (bone == HumanBodyBones.Hips)
            {
                continue;
            }

            if (TryGetBoneMuscleAngles(
                    coachPose,
                    bone,
                    out Vector3 angles,
                    out _))
            {
                sessionCoachAngles[bone] = angles;
                previousCoachAngles[bone] = angles;
            }
        }
    }

    public bool BeginTarget(
        Animator coachAnimator,
        IReadOnlyList<HumanBodyBones> bones,
        TrainingSegmentKind segmentKind = TrainingSegmentKind.CoreAction)
    {
        return BeginTarget(
            coachAnimator,
            bones,
            segmentKind,
            SegmentPoseGuidanceRule.Auto);
    }

    public bool BeginTarget(
        Animator coachAnimator,
        IReadOnlyList<HumanBodyBones> bones,
        TrainingSegmentKind segmentKind,
        SegmentPoseGuidanceRule guidanceRule)
    {
        if (guidanceRule.mode == PoseGuidanceProfileMode.Disabled)
        {
            ResetTarget();
            currentTargetSegmentKind = segmentKind;
            currentGuidanceRule = guidanceRule;
            return false;
        }

        bool previousTargetUsedMuscles = targetUsesHumanoidMuscles;
        TrainingSegmentKind previousSegmentKind = currentTargetSegmentKind;
        SegmentPoseGuidanceRule previousGuidanceRule = currentGuidanceRule;
        Quaternion previousTargetBodyRotation = targetCoachBodyRotation;
        var previousTargetAngles = new Dictionary<HumanBodyBones, Vector3>(
            targetCoachAngles);
        var previousTargetLocalRotations =
            new Dictionary<HumanBodyBones, Quaternion>(targetCoachLocal);
        var previousTargetSalience = new Dictionary<HumanBodyBones, Vector3>(
            targetSalienceAngles);

        HasTarget = false;
        targetCoachAngles.Clear();
        targetSalienceAngles.Clear();
        targetAxisMasks.Clear();
        targetBoneRoles.Clear();
        targetCoachLocal.Clear();
        filteredUserAngles.Clear();
        filteredUserLocal.Clear();

        targetUsesHumanoidMuscles = TryCaptureCoachPose(coachAnimator);
        if (targetUsesHumanoidMuscles)
        {
            targetCoachBodyRotation = coachPose.bodyRotation;
            bool sameTarget = previousTargetAngles.Count > 0 &&
                previousTargetUsedMuscles &&
                previousSegmentKind == segmentKind &&
                GuidanceRulesEqual(previousGuidanceRule, guidanceRule) &&
                Quaternion.Angle(
                    previousTargetBodyRotation,
                    targetCoachBodyRotation) < 0.5f;
            Quaternion targetTilt =
                RobustPoseScoringEngine.GetHeadingInvariantTiltDelta(
                    Quaternion.identity,
                    targetCoachBodyRotation);
            Quaternion previousTilt = hasPreviousCoachBodyRotation
                ? RobustPoseScoringEngine.GetHeadingInvariantTiltDelta(
                    Quaternion.identity,
                    previousCoachBodyRotation)
                : Quaternion.identity;
            float bodySalience = Quaternion.Angle(previousTilt, targetTilt);
            for (int index = 0; bones != null && index < bones.Count; index++)
            {
                HumanBodyBones bone = bones[index];
                if (bone == HumanBodyBones.Hips)
                {
                    targetAxisMasks[bone] = 1;
                    targetSalienceAngles[bone] = new Vector3(
                        bodySalience,
                        0f,
                        0f);
                    continue;
                }

                if (!TryGetBoneMuscleAngles(
                        coachPose,
                        bone,
                        out Vector3 target,
                        out int axisMask))
                {
                    sameTarget = false;
                    continue;
                }

                if (!previousTargetAngles.TryGetValue(
                        bone,
                        out Vector3 previousTarget) ||
                    MaximumAvailableAxis(
                        target - previousTarget,
                        axisMask) >= 0.5f)
                {
                    sameTarget = false;
                }

                bool hasPrevious = previousCoachAngles.TryGetValue(
                    bone,
                    out Vector3 previous);
                Vector3 salience = hasPrevious
                    ? Abs(target - previous)
                    : Abs(target) * 0.5f;
                targetCoachAngles[bone] = target;
                targetSalienceAngles[bone] = salience;
                targetAxisMasks[bone] = axisMask;
                previousCoachAngles[bone] = target;
            }

            if (targetCoachAngles.Count != previousTargetAngles.Count)
            {
                sameTarget = false;
            }

            if (sameTarget)
            {
                RestoreTargetSalience(previousTargetSalience);
            }

            previousCoachBodyRotation = targetCoachBodyRotation;
            hasPreviousCoachBodyRotation = true;
            ConfigureBoneRolesWithRule(bones, segmentKind, guidanceRule);
            currentTargetSegmentKind = segmentKind;
            currentGuidanceRule = guidanceRule;
            HasTarget = targetAxisMasks.Count > 0;
            return HasTarget;
        }

        bool sameLocalTarget = previousTargetLocalRotations.Count > 0 &&
            !previousTargetUsedMuscles &&
            previousSegmentKind == segmentKind &&
            GuidanceRulesEqual(previousGuidanceRule, guidanceRule);
        for (int index = 0; bones != null && index < bones.Count; index++)
        {
            HumanBodyBones bone = bones[index];
            Transform coachBone = coachAnimator != null
                ? coachAnimator.GetBoneTransform(bone)
                : null;
            if (coachBone == null)
            {
                sameLocalTarget = false;
                continue;
            }

            Quaternion target = coachBone.localRotation;
            if (!previousTargetLocalRotations.TryGetValue(
                    bone,
                    out Quaternion previousTarget) ||
                Quaternion.Angle(previousTarget, target) >= 0.5f)
            {
                sameLocalTarget = false;
            }
            bool hasPrevious = previousCoachLocal.TryGetValue(
                bone,
                out Quaternion previous);
            float salience = hasPrevious
                ? Quaternion.Angle(previous, target)
                : Quaternion.Angle(Quaternion.identity, target) * 0.5f;

            targetCoachLocal[bone] = target;
            targetSalienceAngles[bone] = new Vector3(salience, 0f, 0f);
            targetAxisMasks[bone] = 1;
            previousCoachLocal[bone] = target;
        }

        if (targetCoachLocal.Count != previousTargetLocalRotations.Count)
        {
            sameLocalTarget = false;
        }

        if (sameLocalTarget)
        {
            RestoreTargetSalience(previousTargetSalience);
        }

        ConfigureBoneRolesWithRule(bones, segmentKind, guidanceRule);
        currentTargetSegmentKind = segmentKind;
        currentGuidanceRule = guidanceRule;
        HasTarget = targetCoachLocal.Count > 0;
        return HasTarget;
    }

    public PoseHighlightFrameDiagnostics Evaluate(
        Animator userAnimator,
        IReadOnlyList<HumanBodyBones> bones,
        float deltaTime)
    {
        samples.Clear();
        if (!HasTarget || userAnimator == null || bones == null)
        {
            return PoseHighlightDiagnosticMath.Evaluate(samples, settings);
        }

        float dt = Mathf.Clamp(
            deltaTime > 0f ? deltaTime : Time.deltaTime,
            1f / 240f,
            0.1f);
        bool hasUserMuscles = targetUsesHumanoidMuscles &&
            TryCaptureUserPose(userAnimator);

        for (int index = 0; index < bones.Count; index++)
        {
            HumanBodyBones bone = bones[index];
            if (!targetAxisMasks.TryGetValue(bone, out int targetMask) ||
                !targetSalienceAngles.TryGetValue(
                    bone,
                    out Vector3 targetSalience) ||
                !targetBoneRoles.TryGetValue(
                    bone,
                    out PoseHighlightBoneRole targetRole))
            {
                continue;
            }

            if (hasUserMuscles)
            {
                if (bone == HumanBodyBones.Hips)
                {
                    Quaternion userTilt =
                        RobustPoseScoringEngine.GetHeadingInvariantTiltDelta(
                            Quaternion.identity,
                            userPose.bodyRotation);
                    Quaternion coachTilt =
                        RobustPoseScoringEngine.GetHeadingInvariantTiltDelta(
                            Quaternion.identity,
                            targetCoachBodyRotation);
                    samples.Add(new PoseHighlightBoneSample
                    {
                        bone = bone,
                        role = targetRole,
                        isTracked = true,
                        availableAxisMask = 1,
                        errorDegrees = new Vector3(
                            Quaternion.Angle(userTilt, coachTilt),
                            0f,
                            0f),
                        targetSalienceDegrees = targetSalience
                    });
                    continue;
                }

                bool userValid = TryGetBoneMuscleAngles(
                    userPose,
                    bone,
                    out Vector3 userAngles,
                    out int userMask);
                if (userValid && targetCoachAngles.TryGetValue(
                        bone,
                        out Vector3 targetAngles))
                {
                    userAngles = FilterAngles(bone, userAngles, dt);
                    samples.Add(new PoseHighlightBoneSample
                    {
                        bone = bone,
                        role = targetRole,
                        isTracked = true,
                        availableAxisMask = targetMask & userMask,
                        errorDegrees = Abs(userAngles - targetAngles),
                        targetSalienceDegrees = targetSalience
                    });
                }
                else
                {
                    samples.Add(new PoseHighlightBoneSample
                    {
                        bone = bone,
                        role = targetRole,
                        isTracked = false,
                        availableAxisMask = targetMask,
                        targetSalienceDegrees = targetSalience
                    });
                }

                continue;
            }

            EvaluateLocalFallback(
                userAnimator,
                bone,
                targetMask,
                targetSalience,
                targetRole,
                dt);
        }

        return PoseHighlightDiagnosticMath.Evaluate(samples, settings);
    }

    public void ResetTarget()
    {
        HasTarget = false;
        // Retain the last target snapshot until BeginTarget consumes it. This
        // lets an explicit reset/re-evaluation of the same held pose preserve
        // its motion salience instead of silently turning every limb into an
        // ignored role. HasTarget remains false, so stale data is never read by
        // Evaluate between reset and the next BeginTarget call.
        filteredUserAngles.Clear();
        filteredUserLocal.Clear();
    }

    public void Dispose()
    {
        DisposePoseHandlers();
    }

    private void EvaluateLocalFallback(
        Animator userAnimator,
        HumanBodyBones bone,
        int targetMask,
        Vector3 targetSalience,
        PoseHighlightBoneRole targetRole,
        float deltaTime)
    {
        Transform userBone = userAnimator.GetBoneTransform(bone);
        if (userBone == null ||
            !targetCoachLocal.TryGetValue(bone, out Quaternion coachTarget))
        {
            samples.Add(new PoseHighlightBoneSample
            {
                bone = bone,
                role = targetRole,
                isTracked = false,
                availableAxisMask = targetMask,
                targetSalienceDegrees = targetSalience,
                additionalSensorAllowanceDegrees =
                    settings.localRotationFallbackAllowanceDegrees
            });
            return;
        }

        Quaternion userRotation = userBone.localRotation;
        if (sessionUserLocal.TryGetValue(bone, out Quaternion userBaseline) &&
            sessionCoachLocal.TryGetValue(bone, out Quaternion coachBaseline))
        {
            userRotation = Quaternion.Inverse(userBaseline) * userRotation;
            coachTarget = Quaternion.Inverse(coachBaseline) * coachTarget;
        }

        userRotation = FilterRotation(bone, userRotation, deltaTime);
        float error = bone == HumanBodyBones.Hips
            ? Quaternion.Angle(
                RobustPoseScoringEngine.GetHeadingInvariantTiltDelta(
                    Quaternion.identity,
                    userRotation),
                RobustPoseScoringEngine.GetHeadingInvariantTiltDelta(
                    Quaternion.identity,
                    coachTarget))
            : Quaternion.Angle(userRotation, coachTarget);
        samples.Add(new PoseHighlightBoneSample
        {
            bone = bone,
            role = targetRole,
            isTracked = true,
            availableAxisMask = 1,
            errorDegrees = new Vector3(error, 0f, 0f),
            targetSalienceDegrees = targetSalience,
            additionalSensorAllowanceDegrees =
                settings.localRotationFallbackAllowanceDegrees
        });
    }

    private void CaptureLocalReference(
        Animator userAnimator,
        Animator coachAnimator,
        IReadOnlyList<HumanBodyBones> bones)
    {
        for (int index = 0; bones != null && index < bones.Count; index++)
        {
            HumanBodyBones bone = bones[index];
            Transform coachBone = coachAnimator != null
                ? coachAnimator.GetBoneTransform(bone)
                : null;
            Transform userBone = userAnimator != null
                ? userAnimator.GetBoneTransform(bone)
                : null;
            if (coachBone != null)
            {
                sessionCoachLocal[bone] = coachBone.localRotation;
                previousCoachLocal[bone] = coachBone.localRotation;
            }

            if (userBone != null)
            {
                sessionUserLocal[bone] = userBone.localRotation;
            }
        }
    }

    private void ConfigureBoneRoles(
        IReadOnlyList<HumanBodyBones> bones,
        TrainingSegmentKind segmentKind)
    {
        ConfigureBoneRolesWithRule(
            bones,
            segmentKind,
            SegmentPoseGuidanceRule.Auto);
    }

    private void ConfigureBoneRolesWithRule(
        IReadOnlyList<HumanBodyBones> bones,
        TrainingSegmentKind segmentKind,
        SegmentPoseGuidanceRule guidanceRule)
    {
        targetBoneRoles.Clear();
        if (guidanceRule.mode == PoseGuidanceProfileMode.Explicit)
        {
            ConfigureExplicitBoneRoles(bones, guidanceRule);
            return;
        }

        bool isCoreAction = segmentKind == TrainingSegmentKind.CoreAction;
        var activeRegions = new HashSet<BodyPart>();
        float regionMotionMinimum = Mathf.Max(
            settings.targetAxisMinimumDegrees,
            settings.activeRegionMotionMinimumDegrees);

        if (isCoreAction)
        {
            foreach (KeyValuePair<HumanBodyBones, Vector3> pair in
                     targetSalienceAngles)
            {
                if (!targetAxisMasks.TryGetValue(pair.Key, out int mask) ||
                    MaximumAvailableAxis(pair.Value, mask) < regionMotionMinimum)
                {
                    continue;
                }

                BodyPart part = PoseHighlightDiagnosticMath.GetBodyPart(pair.Key);
                if (part != BodyPart.None)
                {
                    activeRegions.Add(part);
                }
            }
        }

        for (int index = 0; bones != null && index < bones.Count; index++)
        {
            HumanBodyBones bone = bones[index];
            if (!targetAxisMasks.TryGetValue(bone, out int mask) ||
                !targetSalienceAngles.TryGetValue(bone, out Vector3 salience))
            {
                continue;
            }

            BodyPart part = PoseHighlightDiagnosticMath.GetBodyPart(bone);
            bool regionIsActive = isCoreAction && activeRegions.Contains(part);
            bool boneIsActive = regionIsActive &&
                MaximumAvailableAxis(salience, mask) >=
                settings.targetAxisMinimumDegrees;

            targetBoneRoles[bone] = boneIsActive
                ? PoseHighlightBoneRole.ActiveMotion
                : IsRelaxedJointShapeBone(bone)
                    ? PoseHighlightBoneRole.SimpleExtension
                    : PoseHighlightBoneRole.Ignore;
        }

    }

    private void ConfigureExplicitBoneRoles(
        IReadOnlyList<HumanBodyBones> bones,
        SegmentPoseGuidanceRule guidanceRule)
    {
        for (int index = 0; bones != null && index < bones.Count; index++)
        {
            HumanBodyBones bone = bones[index];
            if (!targetAxisMasks.TryGetValue(bone, out int mask) ||
                !targetSalienceAngles.TryGetValue(bone, out Vector3 salience))
            {
                continue;
            }

            PoseGuidanceRegionRole regionRole = guidanceRule.GetRole(
                PoseHighlightDiagnosticMath.GetBodyPart(bone));
            if (regionRole == PoseGuidanceRegionRole.Stabilizer)
            {
                // Stabilizers remain diagnosable even when the reference pose
                // barely moved since the preceding segment.
                targetBoneRoles[bone] = PoseHighlightBoneRole.Stabilizer;
                continue;
            }

            if (regionRole != PoseGuidanceRegionRole.Active)
            {
                targetBoneRoles[bone] = PoseHighlightBoneRole.Ignore;
                continue;
            }

            bool boneIsTargetSalient =
                MaximumAvailableAxis(salience, mask) >=
                settings.targetAxisMinimumDegrees;
            targetBoneRoles[bone] = boneIsTargetSalient
                ? PoseHighlightBoneRole.ActiveMotion
                : IsRelaxedJointShapeBone(bone)
                    ? PoseHighlightBoneRole.SimpleExtension
                    : PoseHighlightBoneRole.Ignore;
        }
    }

    private static bool IsRelaxedJointShapeBone(HumanBodyBones bone)
    {
        switch (bone)
        {
            case HumanBodyBones.LeftLowerArm:
            case HumanBodyBones.RightLowerArm:
            case HumanBodyBones.LeftLowerLeg:
            case HumanBodyBones.RightLowerLeg:
                return true;
            default:
                return false;
        }
    }

    private static float MaximumAvailableAxis(Vector3 value, int mask)
    {
        float maximum = 0f;
        if ((mask & 1) != 0) maximum = Mathf.Max(maximum, Mathf.Abs(value.x));
        if ((mask & 2) != 0) maximum = Mathf.Max(maximum, Mathf.Abs(value.y));
        if ((mask & 4) != 0) maximum = Mathf.Max(maximum, Mathf.Abs(value.z));
        return maximum;
    }

    private static bool GuidanceRulesEqual(
        SegmentPoseGuidanceRule left,
        SegmentPoseGuidanceRule right)
    {
        return left.mode == right.mode &&
            left.activeRegions == right.activeRegions &&
            left.stabilizerRegions == right.stabilizerRegions;
    }

    private void RestoreTargetSalience(
        IReadOnlyDictionary<HumanBodyBones, Vector3> previousSalience)
    {
        foreach (KeyValuePair<HumanBodyBones, Vector3> pair in previousSalience)
        {
            if (targetSalienceAngles.ContainsKey(pair.Key))
            {
                targetSalienceAngles[pair.Key] = pair.Value;
            }
        }
    }

    private Vector3 FilterAngles(
        HumanBodyBones bone,
        Vector3 raw,
        float deltaTime)
    {
        if (!filteredUserAngles.TryGetValue(bone, out Vector3 filtered))
        {
            filteredUserAngles[bone] = raw;
            return raw;
        }

        float maximumStep = settings.maximumAngularSpeed * deltaTime;
        float alpha = 1f - Mathf.Exp(
            -deltaTime / Mathf.Max(0.01f, settings.rotationSmoothingTime));
        Vector3 smoothedTarget = Vector3.Lerp(filtered, raw, alpha);
        filtered = Vector3.MoveTowards(filtered, smoothedTarget, maximumStep);
        filteredUserAngles[bone] = filtered;
        return filtered;
    }

    private Quaternion FilterRotation(
        HumanBodyBones bone,
        Quaternion raw,
        float deltaTime)
    {
        if (!filteredUserLocal.TryGetValue(bone, out Quaternion filtered))
        {
            filteredUserLocal[bone] = raw;
            return raw;
        }

        float maximumStep = settings.maximumAngularSpeed * deltaTime;
        float alpha = 1f - Mathf.Exp(
            -deltaTime / Mathf.Max(0.01f, settings.rotationSmoothingTime));
        Quaternion smoothedTarget = Quaternion.Slerp(filtered, raw, alpha);
        filtered = Quaternion.RotateTowards(
            filtered,
            smoothedTarget,
            maximumStep);
        filteredUserLocal[bone] = filtered;
        return filtered;
    }

    private bool TryCaptureCoachPose(Animator animator)
    {
        if (!EnsureCoachPoseHandler(animator))
        {
            return false;
        }

        try
        {
            coachPoseHandler.GetHumanPose(ref coachPose);
            return IsCompletePose(coachPose);
        }
        catch
        {
            DisposeCoachPoseHandler();
            return false;
        }
    }

    private bool TryCaptureUserPose(Animator animator)
    {
        if (!EnsureUserPoseHandler(animator))
        {
            return false;
        }

        try
        {
            userPoseHandler.GetHumanPose(ref userPose);
            return IsCompletePose(userPose);
        }
        catch
        {
            DisposeUserPoseHandler();
            return false;
        }
    }

    private bool EnsureCoachPoseHandler(Animator animator)
    {
        if (!IsUsableHumanoid(animator))
        {
            return false;
        }

        if (coachPoseHandler != null && poseCoachAnimator == animator)
        {
            return true;
        }

        DisposeCoachPoseHandler();
        try
        {
            coachPoseHandler = new HumanPoseHandler(
                animator.avatar,
                animator.transform);
            poseCoachAnimator = animator;
            coachPose = new HumanPose
            {
                muscles = new float[HumanTrait.MuscleCount]
            };
            return true;
        }
        catch
        {
            DisposeCoachPoseHandler();
            return false;
        }
    }

    private bool EnsureUserPoseHandler(Animator animator)
    {
        if (!IsUsableHumanoid(animator))
        {
            return false;
        }

        if (userPoseHandler != null && poseUserAnimator == animator)
        {
            return true;
        }

        DisposeUserPoseHandler();
        try
        {
            userPoseHandler = new HumanPoseHandler(
                animator.avatar,
                animator.transform);
            poseUserAnimator = animator;
            userPose = new HumanPose
            {
                muscles = new float[HumanTrait.MuscleCount]
            };
            return true;
        }
        catch
        {
            DisposeUserPoseHandler();
            return false;
        }
    }

    private static bool TryGetBoneMuscleAngles(
        HumanPose pose,
        HumanBodyBones bone,
        out Vector3 angles,
        out int axisMask)
    {
        angles = Vector3.zero;
        axisMask = 0;
        if (pose.muscles == null)
        {
            return false;
        }

        for (int dof = 0; dof < 3; dof++)
        {
            int muscleIndex = HumanTrait.MuscleFromBone((int)bone, dof);
            if (muscleIndex < 0 || muscleIndex >= pose.muscles.Length)
            {
                continue;
            }

            float normalized = Mathf.Clamp(
                pose.muscles[muscleIndex],
                -1f,
                1f);
            angles[dof] = normalized < 0f
                ? -normalized * HumanTrait.GetMuscleDefaultMin(muscleIndex)
                : normalized * HumanTrait.GetMuscleDefaultMax(muscleIndex);
            axisMask |= 1 << dof;
        }

        return axisMask != 0;
    }

    private static bool IsUsableHumanoid(Animator animator)
    {
        return animator != null &&
            animator.isHuman &&
            animator.avatar != null &&
            animator.avatar.isValid &&
            animator.avatar.isHuman;
    }

    private static bool IsCompletePose(HumanPose pose)
    {
        return pose.muscles != null &&
            pose.muscles.Length == HumanTrait.MuscleCount;
    }

    private static Vector3 Abs(Vector3 value)
    {
        return new Vector3(
            Mathf.Abs(value.x),
            Mathf.Abs(value.y),
            Mathf.Abs(value.z));
    }

    private static Vector3 MaximumComponents(
        Vector3 first,
        Vector3 second,
        Vector3 third)
    {
        return new Vector3(
            Mathf.Max(first.x, Mathf.Max(second.x, third.x)),
            Mathf.Max(first.y, Mathf.Max(second.y, third.y)),
            Mathf.Max(first.z, Mathf.Max(second.z, third.z)));
    }

    private void ResetAll()
    {
        ResetTarget();
        targetCoachAngles.Clear();
        targetSalienceAngles.Clear();
        targetAxisMasks.Clear();
        targetBoneRoles.Clear();
        targetCoachLocal.Clear();
        sessionCoachAngles.Clear();
        previousCoachAngles.Clear();
        sessionCoachLocal.Clear();
        sessionUserLocal.Clear();
        previousCoachLocal.Clear();
        previousCoachBodyRotation = Quaternion.identity;
        hasPreviousCoachBodyRotation = false;
        currentTargetSegmentKind = TrainingSegmentKind.Preparation;
        currentGuidanceRule = SegmentPoseGuidanceRule.Auto;
    }

    private void DisposePoseHandlers()
    {
        DisposeUserPoseHandler();
        DisposeCoachPoseHandler();
    }

    private void DisposeUserPoseHandler()
    {
        userPoseHandler?.Dispose();
        userPoseHandler = null;
        poseUserAnimator = null;
    }

    private void DisposeCoachPoseHandler()
    {
        coachPoseHandler?.Dispose();
        coachPoseHandler = null;
        poseCoachAnimator = null;
    }
}
