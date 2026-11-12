using System.Collections.Generic;
using UnityEngine;

// Tracks real angular displacement from the pose at the beginning of a
// training mode. The result is used as evidence that both the coach animation
// and the IMU-driven user avatar moved during a measured session.
public sealed class AnimatorMotionEvidence
{
    private readonly Dictionary<HumanBodyBones, Quaternion> coachStartRotations =
        new Dictionary<HumanBodyBones, Quaternion>();
    private readonly Dictionary<HumanBodyBones, Quaternion> userStartRotations =
        new Dictionary<HumanBodyBones, Quaternion>();

    private Animator coachAnimator;
    private Animator userAnimator;
    private HumanBodyBones[] trackedBones;

    public float MaximumCoachMotionDegrees { get; private set; }
    public float MaximumUserMotionDegrees { get; private set; }
    public bool IsTracking { get; private set; }

    public void Begin(Animator coach, Animator user, HumanBodyBones[] bones)
    {
        coachAnimator = coach;
        userAnimator = user;
        trackedBones = bones;
        coachStartRotations.Clear();
        userStartRotations.Clear();
        MaximumCoachMotionDegrees = 0f;
        MaximumUserMotionDegrees = 0f;

        if (coachAnimator == null || userAnimator == null ||
            trackedBones == null || trackedBones.Length == 0)
        {
            IsTracking = false;
            return;
        }

        foreach (HumanBodyBones bone in trackedBones)
        {
            Transform coachBone = coachAnimator.GetBoneTransform(bone);
            Transform userBone = userAnimator.GetBoneTransform(bone);
            if (coachBone != null)
            {
                coachStartRotations[bone] = coachBone.localRotation;
            }

            if (userBone != null)
            {
                userStartRotations[bone] = userBone.localRotation;
            }
        }

        IsTracking = coachStartRotations.Count > 0 && userStartRotations.Count > 0;
        Sample();
    }

    public void Sample()
    {
        if (!IsTracking || coachAnimator == null || userAnimator == null)
        {
            return;
        }

        foreach (HumanBodyBones bone in trackedBones)
        {
            if (coachStartRotations.TryGetValue(bone, out Quaternion coachStart))
            {
                Transform coachBone = coachAnimator.GetBoneTransform(bone);
                if (coachBone != null)
                {
                    MaximumCoachMotionDegrees = Mathf.Max(
                        MaximumCoachMotionDegrees,
                        Quaternion.Angle(coachStart, coachBone.localRotation));
                }
            }

            if (userStartRotations.TryGetValue(bone, out Quaternion userStart))
            {
                Transform userBone = userAnimator.GetBoneTransform(bone);
                if (userBone != null)
                {
                    MaximumUserMotionDegrees = Mathf.Max(
                        MaximumUserMotionDegrees,
                        Quaternion.Angle(userStart, userBone.localRotation));
                }
            }
        }
    }

    public void Stop()
    {
        Sample();
        IsTracking = false;
    }
}
