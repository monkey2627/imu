using System.Collections.Generic;
using UnityEngine;

// Geometry and quality weights ported from the delivered HipThrustHoldScorer.
// No retargeting or writes to the user's skeleton; arm/hand/foot rotations are untouched.
public sealed class HipThrustHoldGeometry
{
    public sealed class TargetPose
    {
        public readonly Dictionary<HumanBodyBones, Vector3> directions = new Dictionary<HumanBodyBones, Vector3>();
        public float leftHipAngle,rightHipAngle,leftKneeAngle,rightKneeAngle,bridgeLiftRatio;
        public bool hasBridgeLift,valid;
        public float bodyFrontUp,torsoVertical;
    }
    sealed class QualityRule
    {
        public float perfectHipAngleError=6,zeroHipAngleError=38;
        public float perfectLegDirectionError=10,zeroLegDirectionError=50;
        public float perfectKneeAngleError=8,zeroKneeAngleError=42;
        public float perfectSymmetryError=5,zeroSymmetryError=28;
    }
    readonly QualityRule rule=new QualityRule();
    readonly AnatomicalMotionScoringEngine poseCapture=new AnatomicalMotionScoringEngine(new AnatomicalMotionScoringSettings());
    static readonly HumanBodyBones[] scoredLegBones={HumanBodyBones.LeftUpperLeg,HumanBodyBones.LeftLowerLeg,HumanBodyBones.RightUpperLeg,HumanBodyBones.RightLowerLeg};
    public TargetPose restTarget,holdTarget;
    public static bool HasSupineSupport(TargetPose pose,TargetPose target) => pose!=null&&target!=null&&
        pose.torsoVertical<.7f&&Mathf.Abs(pose.bodyFrontUp)>.35f&&pose.bodyFrontUp*target.bodyFrontUp>0;
    public float currentLeftLegQuality,currentRightLegQuality,currentTorsoQuality;
    public float CalculateProgress(TargetPose pose)
    {
        // Hip and knee angles alone are ambiguous for a glute bridge: a user
        // can keep almost identical joint angles while moving the pelvis from
        // the mat to the raised position. Measure the pelvis distance from the
        // support plane formed by the upper torso and both feet instead. The
        // value is normalized by body span, so it transfers across avatars.
        if (pose.hasBridgeLift && restTarget.hasBridgeLift &&
            holdTarget.hasBridgeLift)
        {
            float lowerLift = Mathf.Min(
                restTarget.bridgeLiftRatio,
                holdTarget.bridgeLiftRatio);
            float upperLift = Mathf.Max(
                restTarget.bridgeLiftRatio,
                holdTarget.bridgeLiftRatio);
            float liftRange = upperLift - lowerLift;
            if (Mathf.Abs(liftRange) >= 0.02f)
            {
                return Mathf.Clamp01(
                    (pose.bridgeLiftRatio - lowerLift) /
                    liftRange);
            }
        }

        // Defensive fallback for malformed rigs whose feet and torso cannot
        // define a stable support plane.
        float fromRest = PoseDistance(pose, restTarget);
        float fromHold = PoseDistance(pose, holdTarget);
        return Mathf.Clamp01(fromRest / Mathf.Max(0.001f, fromRest + fromHold));
    }


    public float CalculatePoseQuality(
        TargetPose pose,
        TargetPose target,
        bool updateDiagnostics)
    {
        float leftHip = AngleScore(
            Mathf.Abs(pose.leftHipAngle - target.leftHipAngle),
            rule.perfectHipAngleError, rule.zeroHipAngleError);
        float rightHip = AngleScore(
            Mathf.Abs(pose.rightHipAngle - target.rightHipAngle),
            rule.perfectHipAngleError, rule.zeroHipAngleError);
        float leftKnee = AngleScore(
            Mathf.Abs(pose.leftKneeAngle - target.leftKneeAngle),
            rule.perfectKneeAngleError, rule.zeroKneeAngleError);
        float rightKnee = AngleScore(
            Mathf.Abs(pose.rightKneeAngle - target.rightKneeAngle),
            rule.perfectKneeAngleError, rule.zeroKneeAngleError);
        float leftDirection = DirectionScoreForLeg(
            pose,
            target,
            HumanBodyBones.LeftUpperLeg,
            HumanBodyBones.LeftLowerLeg);
        float rightDirection = DirectionScoreForLeg(
            pose,
            target,
            HumanBodyBones.RightUpperLeg,
            HumanBodyBones.RightLowerLeg);
        float hip = 0.5f * (leftHip + rightHip);
        float knees = 0.5f * (leftKnee + rightKnee);
        float legDirections = 0.5f * (leftDirection + rightDirection);
        float userSymmetry = Mathf.Abs(pose.leftHipAngle - pose.rightHipAngle);
        float targetSymmetry = Mathf.Abs(target.leftHipAngle - target.rightHipAngle);
        float symmetry = AngleScore(
            Mathf.Abs(userSymmetry - targetSymmetry),
            rule.perfectSymmetryError, rule.zeroSymmetryError);

        if (updateDiagnostics)
        {
            currentLeftLegQuality =
                0.45f * leftHip + 0.35f * leftDirection + 0.2f * leftKnee;
            currentRightLegQuality =
                0.45f * rightHip + 0.35f * rightDirection + 0.2f * rightKnee;
            currentTorsoQuality = 0.7f * hip + 0.3f * symmetry;
        }
        // Deliberately contains no arm, hand, neck or head term. Hip extension,
        // leg direction, knee angle and bilateral symmetry own the score.
        return 0.45f * hip + 0.3f * legDirections + 0.15f * knees + 0.1f * symmetry;
    }

    private float DirectionScoreForLeg(
        TargetPose pose,
        TargetPose target,
        HumanBodyBones upperLeg,
        HumanBodyBones lowerLeg)
    {
        float total = 0f;
        int count = 0;
        HumanBodyBones[] bones = { upperLeg, lowerLeg };
        foreach (HumanBodyBones bone in bones)
        {
            if (pose.directions.TryGetValue(bone, out Vector3 current) &&
                target.directions.TryGetValue(bone, out Vector3 expected))
            {
                total += AngleScore(
                    Vector3.Angle(current, expected),
                    rule.perfectLegDirectionError,
                    rule.zeroLegDirectionError);
                count++;
            }
        }
        return count > 0 ? total / count : 0f;
    }


    private static float PoseDistance(TargetPose left, TargetPose right)
    {
        float hipDistance = 0.5f * (
            Mathf.Abs(left.leftHipAngle - right.leftHipAngle) +
            Mathf.Abs(left.rightHipAngle - right.rightHipAngle));
        float kneeDistance = 0.5f * (
            Mathf.Abs(left.leftKneeAngle - right.leftKneeAngle) +
            Mathf.Abs(left.rightKneeAngle - right.rightKneeAngle));
        return 0.75f * hipDistance + 0.25f * kneeDistance;
    }

    public bool TryMeasurePose(Animator animator, out TargetPose pose)
    {
        pose = new TargetPose();
        if (animator == null || !animator.isHuman)
        {
            return false;
        }
        Transform hips = animator.GetBoneTransform(HumanBodyBones.Hips);
        Transform chest = animator.GetBoneTransform(HumanBodyBones.UpperChest) ??
            animator.GetBoneTransform(HumanBodyBones.Chest) ??
            animator.GetBoneTransform(HumanBodyBones.Spine);
        Transform leftKnee = animator.GetBoneTransform(HumanBodyBones.LeftLowerLeg);
        Transform rightKnee = animator.GetBoneTransform(HumanBodyBones.RightLowerLeg);
        Transform leftAnkle = animator.GetBoneTransform(HumanBodyBones.LeftFoot);
        Transform rightAnkle = animator.GetBoneTransform(HumanBodyBones.RightFoot);
        if (hips == null || chest == null || leftKnee == null || rightKnee == null ||
            leftAnkle == null || rightAnkle == null)
        {
            return false;
        }

        pose.leftHipAngle = Vector3.Angle(chest.position - hips.position,
            leftKnee.position - hips.position);
        var leftShoulder=animator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
        var rightShoulder=animator.GetBoneTransform(HumanBodyBones.RightUpperArm);
        if(leftShoulder==null||rightShoulder==null)return false;
        Vector3 torso=chest.position-hips.position;
        pose.torsoVertical=Mathf.Abs(Vector3.Dot(torso.normalized,Vector3.up));
        pose.bodyFrontUp=Vector3.Dot(Vector3.Cross(rightShoulder.position-leftShoulder.position,torso).normalized,Vector3.up);
        pose.rightHipAngle = Vector3.Angle(chest.position - hips.position,
            rightKnee.position - hips.position);
        pose.leftKneeAngle = Vector3.Angle(hips.position - leftKnee.position,
            leftAnkle.position - leftKnee.position);
        pose.rightKneeAngle = Vector3.Angle(hips.position - rightKnee.position,
            rightAnkle.position - rightKnee.position);
        Vector3 ankleMidpoint = 0.5f *
            (leftAnkle.position + rightAnkle.position);
        Vector3 ankleAxis = rightAnkle.position - leftAnkle.position;
        Vector3 torsoToFeet = ankleMidpoint - chest.position;
        Vector3 supportPlaneNormal = Vector3.Cross(ankleAxis, torsoToFeet);
        float bodySpan = torsoToFeet.magnitude;
        if (supportPlaneNormal.sqrMagnitude > 0.000001f && bodySpan > 0.001f)
        {
            pose.bridgeLiftRatio = Mathf.Abs(Vector3.Dot(
                hips.position - chest.position,
                supportPlaneNormal.normalized)) / bodySpan;
            pose.hasBridgeLift = true;
        }
        Dictionary<HumanBodyBones, Vector3> captured = poseCapture.CaptureRawPose(animator);
        foreach (KeyValuePair<HumanBodyBones, Vector3> entry in captured)
        {
            pose.directions[entry.Key] = entry.Value;
        }
        pose.valid = true;
        foreach (var bone in scoredLegBones) pose.valid &= pose.directions.ContainsKey(bone);
        return pose.valid;
    }


    private static float AngleScore(float error, float perfect, float zero)
    {
        float normalized = Mathf.InverseLerp(perfect, Mathf.Max(perfect + 0.01f, zero), error);
        float eased = normalized * normalized * (3f - 2f * normalized);
        return (1f - eased) * 100f;
    }

}
