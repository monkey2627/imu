using System;
using System.Collections.Generic;
using UnityEngine;

[Flags]
public enum PoseGuidanceRegionMask
{
    None = 0,
    LeftArm = 1 << 0,
    RightArm = 1 << 1,
    LeftLeg = 1 << 2,
    RightLeg = 1 << 3,
    Torso = 1 << 4
}

public enum PoseGuidanceProfileMode
{
    Auto,
    Explicit,
    Disabled
}

/// <summary>
/// Selects whether corrective guidance is generic or uses an exercise's
/// anatomical semantics. Targeted modes deliberately do not compare the user
/// with the recorded coach pose.
/// </summary>
public enum PoseGuidanceExercise
{
    Generic,
    DeadBug,
    BirdDog
}

[Serializable]
public sealed class TargetedExercisePoseGuidanceSettings
{
    [Tooltip("A required limb is wrong below this lift angle.")]
    [Range(0f, 90f)] public float requiredLiftActivationBelowDegrees = 35f;

    [Tooltip("A required limb is recovered at or above this lift angle.")]
    [Range(0f, 90f)] public float requiredLiftClearAtOrAboveDegrees = 45f;

    [Tooltip("A resting limb is wrong above this lift angle.")]
    [Range(0f, 90f)] public float unexpectedLiftActivationAboveDegrees = 35f;

    [Tooltip("A resting limb is recovered at or below this lift angle.")]
    [Range(0f, 90f)] public float unexpectedLiftClearAtOrBelowDegrees = 25f;

    [Tooltip("An active Bird Dog limb is wrong below this body-plane lift angle.")]
    [Range(0f, 90f)] public float birdDogActiveLiftActivationBelowDegrees = 35f;

    [Tooltip("An active Bird Dog limb recovers at or above this body-plane lift angle.")]
    [Range(0f, 90f)] public float birdDogActiveLiftClearAtOrAboveDegrees = 45f;

    [Tooltip("A supporting Bird Dog limb is clearly raised above this " +
        "body-plane lift angle.")]
    [Range(0f, 90f)] public float birdDogSupportLiftActivationAboveDegrees = 60f;

    [Tooltip("A supporting Bird Dog limb recovers at or below this " +
        "body-plane lift angle.")]
    [Range(0f, 90f)] public float birdDogSupportLiftClearAtOrBelowDegrees = 50f;

    [Tooltip("Bird Dog support is wrong when the hand or knee drops by less " +
        "than this fraction of torso length.")]
    [Min(0f)] public float birdDogSupportDropActivationBelowTorsoRatio = 0.08f;

    [Tooltip("Bird Dog support recovers when the hand or knee drops by at " +
        "least this fraction of torso length.")]
    [Min(0f)] public float birdDogSupportDropClearAtOrAboveTorsoRatio = 0.12f;

    [Tooltip("Bird Dog torso guidance activates above this tilt from horizontal.")]
    [Range(0f, 90f)] public float birdDogTorsoTiltActivationAboveDegrees = 50f;

    [Tooltip("Bird Dog torso guidance clears at or below this tilt from horizontal.")]
    [Range(0f, 90f)] public float birdDogTorsoTiltClearAtOrBelowDegrees = 40f;

    [Tooltip("An actively extended Bird Dog elbow or knee is clearly bent " +
        "above this angle.")]
    [Range(0f, 180f)] public float birdDogBendActivationAboveDegrees = 50f;

    [Tooltip("An actively extended Bird Dog elbow or knee recovers at or " +
        "below this bend angle.")]
    [Range(0f, 180f)] public float birdDogBendClearAtOrBelowDegrees = 35f;
}

/// <summary>
/// Action-local evidence derived from the user's Humanoid skeleton. Proximal
/// directions measure lift; Bird Dog also uses hand/knee drop and joint bend
/// so supporting and actively extended limbs can stay deliberately lenient.
/// </summary>
public struct TargetedExercisePoseGeometry
{
    public bool bodyPlaneTracked;
    public Vector3 bodyPlaneNormal;
    public bool torsoTiltTracked;
    public float torsoTiltFromHorizontalDegrees;

    public bool leftArmTracked;
    public Vector3 leftArmDirection;
    public bool leftArmLineTracked;
    public Vector3 leftArmLineDirection;
    public bool leftArmSupportTracked;
    public float leftArmSupportDropTorsoRatio;
    public bool leftArmBendTracked;
    public float leftArmBendDegrees;
    public bool rightArmTracked;
    public Vector3 rightArmDirection;
    public bool rightArmLineTracked;
    public Vector3 rightArmLineDirection;
    public bool rightArmSupportTracked;
    public float rightArmSupportDropTorsoRatio;
    public bool rightArmBendTracked;
    public float rightArmBendDegrees;
    public bool leftLegTracked;
    public Vector3 leftLegDirection;
    public bool leftLegSupportTracked;
    public float leftLegSupportDropTorsoRatio;
    public bool leftLegBendTracked;
    public float leftLegBendDegrees;
    public bool rightLegTracked;
    public Vector3 rightLegDirection;
    public bool rightLegSupportTracked;
    public float rightLegSupportDropTorsoRatio;
    public bool rightLegBendTracked;
    public float rightLegBendDegrees;
}

/// <summary>
/// Pure Dead Bug / Bird Dog corrective geometry. Dead Bug uses only the limb's
/// line-to-body-plane angle. Bird Dog uses a full-arm or thigh line for active
/// and supporting roles, plus endpoint drop and joint bend where relevant.
/// </summary>
public static class TargetedExercisePoseGuidance
{
    private const PoseGuidanceRegionMask LimbRegions =
        PoseGuidanceRegionMask.LeftArm |
        PoseGuidanceRegionMask.RightArm |
        PoseGuidanceRegionMask.LeftLeg |
        PoseGuidanceRegionMask.RightLeg;

    private const float DirectionEpsilon = 0.000001f;

    public static PoseHighlightFrameDiagnostics Evaluate(
        PoseGuidanceExercise exercise,
        TrainingSegmentKind segmentKind,
        SegmentPoseGuidanceRule guidanceRule,
        TargetedExercisePoseGeometry geometry,
        TargetedExercisePoseGuidanceSettings settings)
    {
        PoseHighlightFrameDiagnostics invalid = InvalidFrame();
        if ((exercise != PoseGuidanceExercise.DeadBug &&
             exercise != PoseGuidanceExercise.BirdDog) ||
            !IsGuidanceCheckpoint(segmentKind) ||
            !geometry.bodyPlaneTracked ||
            geometry.bodyPlaneNormal.sqrMagnitude <= DirectionEpsilon)
        {
            return invalid;
        }

        PoseGuidanceRegionMask expectedLifted = PoseGuidanceRegionMask.None;
        if (segmentKind == TrainingSegmentKind.CoreAction)
        {
            expectedLifted = guidanceRule.activeRegions & LimbRegions;
            if (guidanceRule.mode != PoseGuidanceProfileMode.Explicit ||
                !IsCrossBodyPair(expectedLifted))
            {
                // A malformed formal segment must fail closed instead of
                // guessing which diagonal should move.
                return invalid;
            }
        }

        settings = settings ?? new TargetedExercisePoseGuidanceSettings();
        if (exercise == PoseGuidanceExercise.BirdDog)
        {
            return EvaluateBirdDog(expectedLifted, geometry, settings);
        }

        return new PoseHighlightFrameDiagnostics
        {
            leftArm = EvaluateRegion(
                exercise,
                BodyPart.LeftArm,
                geometry.bodyPlaneNormal,
                geometry.leftArmDirection,
                geometry.leftArmTracked,
                (expectedLifted & PoseGuidanceRegionMask.LeftArm) != 0,
                settings),
            rightArm = EvaluateRegion(
                exercise,
                BodyPart.RightArm,
                geometry.bodyPlaneNormal,
                geometry.rightArmDirection,
                geometry.rightArmTracked,
                (expectedLifted & PoseGuidanceRegionMask.RightArm) != 0,
                settings),
            leftLeg = EvaluateRegion(
                exercise,
                BodyPart.LeftLeg,
                geometry.bodyPlaneNormal,
                geometry.leftLegDirection,
                geometry.leftLegTracked,
                (expectedLifted & PoseGuidanceRegionMask.LeftLeg) != 0,
                settings),
            rightLeg = EvaluateRegion(
                exercise,
                BodyPart.RightLeg,
                geometry.bodyPlaneNormal,
                geometry.rightLegDirection,
                geometry.rightLegTracked,
                (expectedLifted & PoseGuidanceRegionMask.RightLeg) != 0,
                settings),
            // These two exercises intentionally diagnose only the four limbs.
            torso = InvalidDiagnostic(BodyPart.Torso)
        };
    }

    private static PoseHighlightFrameDiagnostics EvaluateBirdDog(
        PoseGuidanceRegionMask expectedExtended,
        TargetedExercisePoseGeometry geometry,
        TargetedExercisePoseGuidanceSettings settings)
    {
        float torsoActivation = Mathf.Clamp(
            settings.birdDogTorsoTiltActivationAboveDegrees,
            0f,
            90f);
        bool limbGuidanceEnabled = geometry.torsoTiltTracked &&
            geometry.torsoTiltFromHorizontalDegrees <= torsoActivation;

        return new PoseHighlightFrameDiagnostics
        {
            leftArm = EvaluateBirdDogLimb(
                BodyPart.LeftArm,
                geometry.bodyPlaneNormal,
                geometry.leftArmLineDirection,
                geometry.leftArmLineTracked,
                geometry.leftArmSupportDropTorsoRatio,
                geometry.leftArmSupportTracked,
                geometry.leftArmBendDegrees,
                geometry.leftArmBendTracked,
                (expectedExtended & PoseGuidanceRegionMask.LeftArm) != 0,
                limbGuidanceEnabled,
                settings),
            rightArm = EvaluateBirdDogLimb(
                BodyPart.RightArm,
                geometry.bodyPlaneNormal,
                geometry.rightArmLineDirection,
                geometry.rightArmLineTracked,
                geometry.rightArmSupportDropTorsoRatio,
                geometry.rightArmSupportTracked,
                geometry.rightArmBendDegrees,
                geometry.rightArmBendTracked,
                (expectedExtended & PoseGuidanceRegionMask.RightArm) != 0,
                limbGuidanceEnabled,
                settings),
            leftLeg = EvaluateBirdDogLimb(
                BodyPart.LeftLeg,
                geometry.bodyPlaneNormal,
                geometry.leftLegDirection,
                geometry.leftLegTracked,
                geometry.leftLegSupportDropTorsoRatio,
                geometry.leftLegSupportTracked,
                geometry.leftLegBendDegrees,
                geometry.leftLegBendTracked,
                (expectedExtended & PoseGuidanceRegionMask.LeftLeg) != 0,
                limbGuidanceEnabled,
                settings),
            rightLeg = EvaluateBirdDogLimb(
                BodyPart.RightLeg,
                geometry.bodyPlaneNormal,
                geometry.rightLegDirection,
                geometry.rightLegTracked,
                geometry.rightLegSupportDropTorsoRatio,
                geometry.rightLegSupportTracked,
                geometry.rightLegBendDegrees,
                geometry.rightLegBendTracked,
                (expectedExtended & PoseGuidanceRegionMask.RightLeg) != 0,
                limbGuidanceEnabled,
                settings),
            torso = EvaluateBirdDogTorso(
                geometry.torsoTiltFromHorizontalDegrees,
                geometry.torsoTiltTracked,
                settings)
        };
    }

    private static PoseHighlightRegionDiagnostic EvaluateBirdDogLimb(
        BodyPart part,
        Vector3 bodyPlaneNormal,
        Vector3 limbDirection,
        bool limbTracked,
        float supportDropRatio,
        bool supportTracked,
        float bendDegrees,
        bool bendTracked,
        bool shouldExtend,
        bool guidanceEnabled,
        TargetedExercisePoseGuidanceSettings settings)
    {
        if (!guidanceEnabled)
        {
            return InvalidDiagnostic(part);
        }

        if (!shouldExtend)
        {
            return EvaluateBirdDogSupport(
                part,
                bodyPlaneNormal,
                limbDirection,
                limbTracked,
                supportDropRatio,
                supportTracked,
                settings);
        }

        if (!limbTracked ||
            !TryCalculateLiftDegrees(
                PoseGuidanceExercise.BirdDog,
                bodyPlaneNormal,
                limbDirection,
                out float liftDegrees))
        {
            return InvalidDiagnostic(part);
        }

        float liftActivation = Mathf.Clamp(
            settings.birdDogActiveLiftActivationBelowDegrees,
            0f,
            90f);
        float liftClear = Mathf.Clamp(
            Mathf.Max(
                liftActivation,
                settings.birdDogActiveLiftClearAtOrAboveDegrees),
            0f,
            90f);
        float bendClear = Mathf.Clamp(
            settings.birdDogBendClearAtOrBelowDegrees,
            0f,
            180f);
        float bendActivation = Mathf.Clamp(
            Mathf.Max(
                bendClear,
                settings.birdDogBendActivationAboveDegrees),
            0f,
            180f);

        bool liftWrong = liftDegrees < liftActivation;
        bool bendWrong = bendTracked && bendDegrees > bendActivation;
        bool liftClearEnough = liftDegrees >= liftClear;
        bool bendClearEnough = !bendTracked || bendDegrees <= bendClear;
        float liftSeverity = Mathf.Max(
            0f,
            (liftClear - liftDegrees) /
            Mathf.Max(1f, liftClear - liftActivation));
        float bendSeverity = bendTracked
            ? Mathf.Max(
                0f,
                (bendDegrees - bendClear) /
                Mathf.Max(1f, bendActivation - bendClear))
            : 0f;

        return new PoseHighlightRegionDiagnostic
        {
            part = part,
            isValid = true,
            shouldActivate = liftWrong || bendWrong,
            shouldClear = liftClearEnough && bendClearEnough,
            relevantBoneCount = bendTracked ? 2 : 1,
            validBoneCount = bendTracked ? 2 : 1,
            trackingCoverage = 1f,
            worstRawErrorDegrees = Mathf.Max(
                Mathf.Max(0f, liftClear - liftDegrees),
                bendTracked ? Mathf.Max(0f, bendDegrees - bendClear) : 0f),
            severity = Mathf.Max(liftSeverity, bendSeverity)
        };
    }

    private static PoseHighlightRegionDiagnostic EvaluateBirdDogSupport(
        BodyPart part,
        Vector3 bodyPlaneNormal,
        Vector3 limbDirection,
        bool limbTracked,
        float supportDropRatio,
        bool supportTracked,
        TargetedExercisePoseGuidanceSettings settings)
    {
        float liftDegrees = 0f;
        bool liftTracked = limbTracked && TryCalculateLiftDegrees(
            PoseGuidanceExercise.BirdDog,
            bodyPlaneNormal,
            limbDirection,
            out liftDegrees);
        if (!supportTracked && !liftTracked)
        {
            return InvalidDiagnostic(part);
        }

        float dropActivation = Mathf.Max(
            0f,
            settings.birdDogSupportDropActivationBelowTorsoRatio);
        float dropClear = Mathf.Max(
            dropActivation,
            settings.birdDogSupportDropClearAtOrAboveTorsoRatio);
        float liftClear = Mathf.Clamp(
            settings.birdDogSupportLiftClearAtOrBelowDegrees,
            0f,
            90f);
        float liftActivation = Mathf.Clamp(
            Mathf.Max(
                liftClear,
                settings.birdDogSupportLiftActivationAboveDegrees),
            0f,
            90f);

        bool dropWrong = supportTracked && supportDropRatio < dropActivation;
        bool liftWrong = liftTracked && liftDegrees > liftActivation;
        bool dropClearEnough = !supportTracked || supportDropRatio >= dropClear;
        bool liftClearEnough = !liftTracked || liftDegrees <= liftClear;
        float dropSeverity = supportTracked
            ? Mathf.Max(
                0f,
                (dropClear - supportDropRatio) /
                Mathf.Max(0.01f, dropClear - dropActivation))
            : 0f;
        float liftSeverity = liftTracked
            ? Mathf.Max(
                0f,
                (liftDegrees - liftClear) /
                Mathf.Max(1f, liftActivation - liftClear))
            : 0f;

        return new PoseHighlightRegionDiagnostic
        {
            part = part,
            isValid = true,
            shouldActivate = dropWrong || liftWrong,
            shouldClear = dropClearEnough && liftClearEnough,
            relevantBoneCount = supportTracked && liftTracked ? 2 : 1,
            validBoneCount = supportTracked && liftTracked ? 2 : 1,
            trackingCoverage = 1f,
            worstRawErrorDegrees = Mathf.Max(
                supportTracked
                    ? Mathf.Max(0f, dropClear - supportDropRatio) * 90f
                    : 0f,
                liftTracked ? Mathf.Max(0f, liftDegrees - liftClear) : 0f),
            severity = Mathf.Max(dropSeverity, liftSeverity)
        };
    }

    private static PoseHighlightRegionDiagnostic EvaluateBirdDogTorso(
        float tiltFromHorizontalDegrees,
        bool tracked,
        TargetedExercisePoseGuidanceSettings settings)
    {
        if (!tracked)
        {
            return InvalidDiagnostic(BodyPart.Torso);
        }

        float clear = Mathf.Clamp(
            settings.birdDogTorsoTiltClearAtOrBelowDegrees,
            0f,
            90f);
        float activation = Mathf.Clamp(
            Mathf.Max(
                clear,
                settings.birdDogTorsoTiltActivationAboveDegrees),
            0f,
            90f);
        float severity = Mathf.Max(
            0f,
            (tiltFromHorizontalDegrees - clear) /
            Mathf.Max(1f, activation - clear));

        return new PoseHighlightRegionDiagnostic
        {
            part = BodyPart.Torso,
            isValid = true,
            shouldActivate = tiltFromHorizontalDegrees > activation,
            shouldClear = tiltFromHorizontalDegrees <= clear,
            relevantBoneCount = 1,
            validBoneCount = 1,
            trackingCoverage = 1f,
            worstRawErrorDegrees = Mathf.Max(0f, tiltFromHorizontalDegrees - clear),
            severity = severity
        };
    }

    public static bool TryCalculateLiftDegrees(
        PoseGuidanceExercise exercise,
        Vector3 bodyPlaneNormal,
        Vector3 limbDirection,
        out float liftDegrees)
    {
        liftDegrees = 0f;
        if ((exercise != PoseGuidanceExercise.DeadBug &&
             exercise != PoseGuidanceExercise.BirdDog) ||
            bodyPlaneNormal.sqrMagnitude <= DirectionEpsilon ||
            limbDirection.sqrMagnitude <= DirectionEpsilon)
        {
            return false;
        }

        float normalAlignment = Mathf.Clamp01(Mathf.Abs(Vector3.Dot(
            bodyPlaneNormal.normalized,
            limbDirection.normalized)));
        float lineToPlaneDegrees = Mathf.Asin(normalAlignment) * Mathf.Rad2Deg;
        liftDegrees = exercise == PoseGuidanceExercise.DeadBug
            ? lineToPlaneDegrees
            : 90f - lineToPlaneDegrees;
        return true;
    }

    private static PoseHighlightRegionDiagnostic EvaluateRegion(
        PoseGuidanceExercise exercise,
        BodyPart part,
        Vector3 bodyPlaneNormal,
        Vector3 limbDirection,
        bool tracked,
        bool shouldBeLifted,
        TargetedExercisePoseGuidanceSettings settings)
    {
        if (!tracked ||
            !TryCalculateLiftDegrees(
                exercise,
                bodyPlaneNormal,
                limbDirection,
                out float liftDegrees))
        {
            return InvalidDiagnostic(part);
        }

        float requiredActivation = Mathf.Clamp(
            settings.requiredLiftActivationBelowDegrees,
            0f,
            90f);
        float requiredClear = Mathf.Clamp(
            Mathf.Max(
                requiredActivation,
                settings.requiredLiftClearAtOrAboveDegrees),
            0f,
            90f);
        float unexpectedClear = Mathf.Clamp(
            settings.unexpectedLiftClearAtOrBelowDegrees,
            0f,
            90f);
        float unexpectedActivation = Mathf.Clamp(
            Mathf.Max(
                unexpectedClear,
                settings.unexpectedLiftActivationAboveDegrees),
            0f,
            90f);

        bool activate;
        bool clear;
        float severity;
        float error;
        if (shouldBeLifted)
        {
            activate = liftDegrees < requiredActivation;
            clear = liftDegrees >= requiredClear;
            severity = Mathf.Max(
                0f,
                (requiredClear - liftDegrees) /
                Mathf.Max(1f, requiredClear - requiredActivation));
            error = Mathf.Max(0f, requiredClear - liftDegrees);
        }
        else
        {
            activate = liftDegrees > unexpectedActivation;
            clear = liftDegrees <= unexpectedClear;
            severity = Mathf.Max(
                0f,
                (liftDegrees - unexpectedClear) /
                Mathf.Max(1f, unexpectedActivation - unexpectedClear));
            error = liftDegrees;
        }

        return new PoseHighlightRegionDiagnostic
        {
            part = part,
            isValid = true,
            shouldActivate = activate,
            shouldClear = clear,
            relevantBoneCount = 1,
            validBoneCount = 1,
            trackingCoverage = 1f,
            worstRawErrorDegrees = error,
            severity = severity
        };
    }

    private static bool IsGuidanceCheckpoint(TrainingSegmentKind kind)
    {
        return kind == TrainingSegmentKind.Preparation ||
            kind == TrainingSegmentKind.CoreAction ||
            kind == TrainingSegmentKind.ReturnTransition;
    }

    private static bool IsCrossBodyPair(PoseGuidanceRegionMask regions)
    {
        return regions ==
                (PoseGuidanceRegionMask.LeftArm |
                 PoseGuidanceRegionMask.RightLeg) ||
            regions ==
                (PoseGuidanceRegionMask.RightArm |
                 PoseGuidanceRegionMask.LeftLeg);
    }

    private static PoseHighlightFrameDiagnostics InvalidFrame()
    {
        return new PoseHighlightFrameDiagnostics
        {
            leftArm = InvalidDiagnostic(BodyPart.LeftArm),
            rightArm = InvalidDiagnostic(BodyPart.RightArm),
            leftLeg = InvalidDiagnostic(BodyPart.LeftLeg),
            rightLeg = InvalidDiagnostic(BodyPart.RightLeg),
            torso = InvalidDiagnostic(BodyPart.Torso)
        };
    }

    private static PoseHighlightRegionDiagnostic InvalidDiagnostic(
        BodyPart part)
    {
        return new PoseHighlightRegionDiagnostic
        {
            part = part,
            isValid = false,
            shouldActivate = false,
            shouldClear = true,
            relevantBoneCount = 1,
            validBoneCount = 0,
            trackingCoverage = 0f,
            worstRawErrorDegrees = 0f,
            severity = 0f
        };
    }
}

public enum PoseGuidanceRegionRole
{
    Ignore,
    Active,
    Stabilizer
}

/// <summary>
/// Serializable, action-agnostic corrective guidance configuration for one
/// segment. The value is copied into SegmentHoldInfo so a live hold never
/// observes later edits to its source ActionSegment.
/// </summary>
[Serializable]
public struct SegmentPoseGuidanceRule
{
    public PoseGuidanceProfileMode mode;
    public PoseGuidanceRegionMask activeRegions;
    public PoseGuidanceRegionMask stabilizerRegions;

    public static SegmentPoseGuidanceRule Auto =>
        new SegmentPoseGuidanceRule
        {
            mode = PoseGuidanceProfileMode.Auto,
            activeRegions = PoseGuidanceRegionMask.None,
            stabilizerRegions = PoseGuidanceRegionMask.None
        };

    public PoseGuidanceRegionRole GetRole(BodyPart part)
    {
        if (mode == PoseGuidanceProfileMode.Disabled)
        {
            return PoseGuidanceRegionRole.Ignore;
        }

        if (mode == PoseGuidanceProfileMode.Auto)
        {
            return PoseGuidanceRegionRole.Active;
        }

        PoseGuidanceRegionMask region = ToMask(part);
        if ((activeRegions & region) != 0)
        {
            // Active wins when serialized masks overlap.
            return PoseGuidanceRegionRole.Active;
        }

        return (stabilizerRegions & region) != 0
            ? PoseGuidanceRegionRole.Stabilizer
            : PoseGuidanceRegionRole.Ignore;
    }

    public bool Allows(BodyPart part) =>
        GetRole(part) != PoseGuidanceRegionRole.Ignore;

    public static PoseGuidanceRegionMask ToMask(BodyPart part)
    {
        switch (part)
        {
            case BodyPart.LeftArm:
                return PoseGuidanceRegionMask.LeftArm;
            case BodyPart.RightArm:
                return PoseGuidanceRegionMask.RightArm;
            case BodyPart.LeftLeg:
                return PoseGuidanceRegionMask.LeftLeg;
            case BodyPart.RightLeg:
                return PoseGuidanceRegionMask.RightLeg;
            case BodyPart.Torso:
                return PoseGuidanceRegionMask.Torso;
            default:
                return PoseGuidanceRegionMask.None;
        }
    }
}

public struct PoseGuidanceRegionEvidence
{
    public BodyPart part;
    public bool isValid;
    public bool shouldActivate;
    public bool shouldClear;
    public float severity;

    public PoseGuidanceRegionEvidence(
        BodyPart part,
        bool isValid,
        bool shouldActivate,
        bool shouldClear,
        float severity)
    {
        this.part = part;
        this.isValid = isValid;
        this.shouldActivate = shouldActivate;
        this.shouldClear = shouldClear;
        this.severity = Mathf.Max(0f, severity);
    }
}

/// <summary>
/// Low-threshold, profile-driven evidence that every active region reached
/// roughly the requested action. It deliberately consumes the same coarse
/// region diagnostics as the highlighter instead of comparing against the
/// coach pose or introducing joint-angle rules.
/// </summary>
public sealed class CoarseActionReadinessLatch
{
    public const float MaximumEvidenceDeltaTime = 0.1f;

    private static readonly BodyPart[] AllParts =
    {
        BodyPart.LeftArm,
        BodyPart.RightArm,
        BodyPart.LeftLeg,
        BodyPart.RightLeg,
        BodyPart.Torso
    };

    private readonly HashSet<BodyPart> readyParts =
        new HashSet<BodyPart>();
    private readonly Dictionary<BodyPart, float> matchingSeconds =
        new Dictionary<BodyPart, float>();
    private readonly Dictionary<BodyPart, float> mismatchSeconds =
        new Dictionary<BodyPart, float>();

    private float confirmationSeconds;
    private float releaseSeconds;

    public CoarseActionReadinessLatch(
        float confirmationSeconds,
        float releaseSeconds)
    {
        Configure(confirmationSeconds, releaseSeconds);
    }

    public IReadOnlyCollection<BodyPart> ReadyParts => readyParts;
    public bool IsReady { get; private set; }
    public int MatchingTimerCount => matchingSeconds.Count;
    public int MismatchTimerCount => mismatchSeconds.Count;

    public void Configure(
        float confirmationDuration,
        float releaseDuration)
    {
        confirmationSeconds = Mathf.Max(0f, confirmationDuration);
        releaseSeconds = Mathf.Max(0f, releaseDuration);
    }

    public void Update(
        SegmentPoseGuidanceRule rule,
        IReadOnlyList<PoseGuidanceRegionEvidence> evidence,
        float deltaTime,
        bool inputValid = true)
    {
        float dt = Mathf.Clamp(
            float.IsNaN(deltaTime) ? 0f : deltaTime,
            0f,
            MaximumEvidenceDeltaTime);
        bool hasActiveRegion = false;
        bool everyActiveRegionReady = true;

        for (int index = 0; index < AllParts.Length; index++)
        {
            BodyPart part = AllParts[index];
            if (rule.GetRole(part) != PoseGuidanceRegionRole.Active)
            {
                readyParts.Remove(part);
                matchingSeconds.Remove(part);
                mismatchSeconds.Remove(part);
                continue;
            }

            hasActiveRegion = true;
            bool hasEvidence = TryFindEvidence(evidence, part, out var sample);
            bool roughlyMatches = inputValid &&
                hasEvidence &&
                sample.isValid &&
                !sample.shouldActivate;
            if (!readyParts.Contains(part))
            {
                mismatchSeconds.Remove(part);
                if (!roughlyMatches)
                {
                    matchingSeconds.Remove(part);
                    everyActiveRegionReady = false;
                    continue;
                }

                matchingSeconds.TryGetValue(part, out float matching);
                matching += dt;
                matchingSeconds[part] = matching;
                if (matching >= confirmationSeconds)
                {
                    readyParts.Add(part);
                    matchingSeconds.Remove(part);
                }
            }
            else
            {
                matchingSeconds.Remove(part);
                if (roughlyMatches)
                {
                    mismatchSeconds.Remove(part);
                }
                else
                {
                    mismatchSeconds.TryGetValue(part, out float mismatch);
                    mismatch += dt;
                    mismatchSeconds[part] = mismatch;
                    if (mismatch >= releaseSeconds)
                    {
                        readyParts.Remove(part);
                        mismatchSeconds.Remove(part);
                    }
                }
            }

            everyActiveRegionReady &= readyParts.Contains(part);
        }

        IsReady = hasActiveRegion && everyActiveRegionReady;
    }

    public void Reset()
    {
        readyParts.Clear();
        matchingSeconds.Clear();
        mismatchSeconds.Clear();
        IsReady = false;
    }

    private static bool TryFindEvidence(
        IReadOnlyList<PoseGuidanceRegionEvidence> evidence,
        BodyPart part,
        out PoseGuidanceRegionEvidence sample)
    {
        if (evidence != null)
        {
            for (int index = 0; index < evidence.Count; index++)
            {
                if (evidence[index].part == part)
                {
                    sample = evidence[index];
                    return true;
                }
            }
        }

        sample = default(PoseGuidanceRegionEvidence);
        return false;
    }
}

/// <summary>
/// Deterministic confirmation/release state for the five coarse regions.
/// Visible slots are sticky: severity can choose a free slot but can never
/// evict a region that has not actually recovered.
/// </summary>
public sealed class PoseGuidanceRegionLatch
{
    public const float MaximumEvidenceDeltaTime = 0.1f;

    private static readonly BodyPart[] AllParts =
    {
        BodyPart.LeftArm,
        BodyPart.RightArm,
        BodyPart.LeftLeg,
        BodyPart.RightLeg,
        BodyPart.Torso
    };

    private readonly HashSet<BodyPart> confirmedParts =
        new HashSet<BodyPart>();
    private readonly List<BodyPart> visibleParts = new List<BodyPart>(5);
    private readonly Dictionary<BodyPart, float> mismatchSeconds =
        new Dictionary<BodyPart, float>();
    private readonly Dictionary<BodyPart, float> clearSeconds =
        new Dictionary<BodyPart, float>();
    private readonly Dictionary<BodyPart, float> severities =
        new Dictionary<BodyPart, float>();
    private readonly List<BodyPart> fillCandidates = new List<BodyPart>(5);

    private float confirmationSeconds;
    private float releaseSeconds;
    private int maximumVisibleParts;

    public PoseGuidanceRegionLatch(
        float confirmationSeconds,
        float releaseSeconds,
        int maximumVisibleParts = 2)
    {
        Configure(
            confirmationSeconds,
            releaseSeconds,
            maximumVisibleParts);
    }

    public IReadOnlyCollection<BodyPart> ConfirmedParts => confirmedParts;
    public IReadOnlyList<BodyPart> VisibleParts => visibleParts;
    public int MismatchTimerCount => mismatchSeconds.Count;
    public int ClearTimerCount => clearSeconds.Count;

    public void Configure(
        float confirmationDuration,
        float releaseDuration,
        int maximumVisible)
    {
        confirmationSeconds = Mathf.Max(0f, confirmationDuration);
        releaseSeconds = Mathf.Max(0f, releaseDuration);
        maximumVisibleParts = Mathf.Clamp(maximumVisible, 1, AllParts.Length);
        TrimVisibleSlots();
    }

    public void Update(
        IReadOnlyList<PoseGuidanceRegionEvidence> evidence,
        float deltaTime,
        bool inputValid = true)
    {
        // A stalled frame must not create or release a cue immediately. Normal
        // frames still accumulate elapsed evidence, while each single sample is
        // capped so timing remains stable during a hitch.
        float dt = Mathf.Clamp(
            float.IsNaN(deltaTime) ? 0f : deltaTime,
            0f,
            MaximumEvidenceDeltaTime);

        for (int index = 0; index < AllParts.Length; index++)
        {
            BodyPart part = AllParts[index];
            bool hasEvidence = TryFindEvidence(evidence, part, out var sample);
            if (hasEvidence && sample.isValid)
            {
                severities[part] = Mathf.Max(0f, sample.severity);
            }

            bool isConfirmed = confirmedParts.Contains(part);
            if (!isConfirmed)
            {
                clearSeconds.Remove(part);
                bool isCandidate = inputValid &&
                    hasEvidence &&
                    sample.isValid &&
                    sample.shouldActivate;
                if (!isCandidate)
                {
                    mismatchSeconds.Remove(part);
                    continue;
                }

                mismatchSeconds.TryGetValue(part, out float mismatch);
                mismatch += dt;
                mismatchSeconds[part] = mismatch;
                if (mismatch >= confirmationSeconds)
                {
                    confirmedParts.Add(part);
                    mismatchSeconds.Remove(part);
                }

                continue;
            }

            mismatchSeconds.Remove(part);
            bool isClear = !inputValid ||
                !hasEvidence ||
                !sample.isValid ||
                sample.shouldClear;
            if (!isClear)
            {
                clearSeconds.Remove(part);
                continue;
            }

            clearSeconds.TryGetValue(part, out float clear);
            clear += dt;
            clearSeconds[part] = clear;
            if (clear >= releaseSeconds)
            {
                confirmedParts.Remove(part);
                clearSeconds.Remove(part);
                severities.Remove(part);
            }
        }

        SelectVisibleParts();
    }

    public void Reset()
    {
        confirmedParts.Clear();
        visibleParts.Clear();
        mismatchSeconds.Clear();
        clearSeconds.Clear();
        severities.Clear();
        fillCandidates.Clear();
    }

    public float GetMismatchSeconds(BodyPart part)
    {
        return mismatchSeconds.TryGetValue(part, out float value) ? value : 0f;
    }

    public float GetClearSeconds(BodyPart part)
    {
        return clearSeconds.TryGetValue(part, out float value) ? value : 0f;
    }

    private void SelectVisibleParts()
    {
        visibleParts.RemoveAll(part => !confirmedParts.Contains(part));
        TrimVisibleSlots();

        fillCandidates.Clear();
        foreach (BodyPart part in confirmedParts)
        {
            if (!visibleParts.Contains(part))
            {
                fillCandidates.Add(part);
            }
        }

        fillCandidates.Sort(CompareFillCandidates);
        for (int index = 0;
             index < fillCandidates.Count &&
             visibleParts.Count < maximumVisibleParts;
             index++)
        {
            visibleParts.Add(fillCandidates[index]);
        }
    }

    private void TrimVisibleSlots()
    {
        while (visibleParts.Count > maximumVisibleParts)
        {
            visibleParts.RemoveAt(visibleParts.Count - 1);
        }
    }

    private int CompareFillCandidates(BodyPart left, BodyPart right)
    {
        severities.TryGetValue(left, out float leftSeverity);
        severities.TryGetValue(right, out float rightSeverity);
        int severityOrder = rightSeverity.CompareTo(leftSeverity);
        return severityOrder != 0
            ? severityOrder
            : ((int)left).CompareTo((int)right);
    }

    private static bool TryFindEvidence(
        IReadOnlyList<PoseGuidanceRegionEvidence> evidence,
        BodyPart part,
        out PoseGuidanceRegionEvidence sample)
    {
        for (int index = 0; evidence != null && index < evidence.Count; index++)
        {
            if (evidence[index].part == part)
            {
                sample = evidence[index];
                return true;
            }
        }

        sample = default(PoseGuidanceRegionEvidence);
        return false;
    }
}

/// <summary>
/// Pure freshness state fed by the active motion-source adapter. Either a packet
/// frame or advancing remote time is evidence; a connected but frozen stream
/// expires after the configured timeout.
/// </summary>
public sealed class VmcPoseDataFreshnessMonitor
{
    private float timeoutSeconds;
    private float secondsSinceEvidence = float.PositiveInfinity;
    private float lastRemoteTime;
    private bool hasRemoteTime;
    private bool wasAvailable;

    public VmcPoseDataFreshnessMonitor(float timeoutSeconds = 0.25f)
    {
        Configure(timeoutSeconds);
    }

    public bool IsFresh { get; private set; }
    public float SecondsSinceEvidence => secondsSinceEvidence;

    public void Configure(float timeout)
    {
        timeoutSeconds = Mathf.Max(0.01f, timeout);
    }

    public bool Update(
        bool receiverAvailable,
        float remoteTime,
        int packetFramesInLatestFrame,
        float deltaTime)
    {
        if (!receiverAvailable)
        {
            wasAvailable = false;
            hasRemoteTime = false;
            secondsSinceEvidence = float.PositiveInfinity;
            IsFresh = false;
            return false;
        }

        bool remoteTimeUsable =
            !float.IsNaN(remoteTime) &&
            !float.IsInfinity(remoteTime) &&
            remoteTime > 0f;
        bool remoteAdvanced = remoteTimeUsable &&
            (!hasRemoteTime || !Mathf.Approximately(remoteTime, lastRemoteTime));
        bool hasNewEvidence = packetFramesInLatestFrame > 0 || remoteAdvanced;

        if (!wasAvailable)
        {
            secondsSinceEvidence = float.PositiveInfinity;
        }

        if (hasNewEvidence)
        {
            secondsSinceEvidence = 0f;
        }
        else if (!float.IsPositiveInfinity(secondsSinceEvidence))
        {
            float dt = float.IsNaN(deltaTime) ? 0f : Mathf.Max(0f, deltaTime);
            secondsSinceEvidence += dt;
        }

        if (remoteTimeUsable)
        {
            lastRemoteTime = remoteTime;
            hasRemoteTime = true;
        }

        wasAvailable = true;
        IsFresh = secondsSinceEvidence <= timeoutSeconds;
        return IsFresh;
    }

    public void Reset()
    {
        secondsSinceEvidence = float.PositiveInfinity;
        lastRemoteTime = 0f;
        hasRemoteTime = false;
        wasAvailable = false;
        IsFresh = false;
    }
}
