using System;
using UnityEngine;

/// <summary>
/// 一帧 Humanoid 姿势，直接对应 UnityEngine.HumanPose。
/// muscle space 与具体骨架无关，因此同一份数据可驱动男女不同 avatar。
/// </summary>
[Serializable]
public class CoachMotionFrame
{
    /// <summary>HumanPose.muscles 的固定长度（HumanTrait.MuscleCount）</summary>
    public static int MuscleCount => HumanTrait.MuscleCount;

    /// <summary>相对该段起点的时间（秒）</summary>
    public float time;

    /// <summary>HumanPose.bodyPosition</summary>
    public Vector3 bodyPosition;

    /// <summary>HumanPose.bodyRotation</summary>
    public Quaternion bodyRotation;

    /// <summary>HumanPose.muscles</summary>
    public float[] muscles = new float[MuscleCount];

    public static CoachMotionFrame FromHumanPose(float time, HumanPose pose)
    {
        var frame = new CoachMotionFrame
        {
            time = time,
            bodyPosition = pose.bodyPosition,
            bodyRotation = pose.bodyRotation,
            muscles = new float[MuscleCount]
        };

        int count = Mathf.Min(MuscleCount, pose.muscles?.Length ?? 0);
        for (int i = 0; i < count; i++)
        {
            frame.muscles[i] = pose.muscles[i];
        }

        return frame;
    }

    public void ApplyTo(ref HumanPose pose)
    {
        pose.bodyPosition = bodyPosition;
        pose.bodyRotation = bodyRotation;

        if (pose.muscles == null || pose.muscles.Length != MuscleCount)
        {
            pose.muscles = new float[MuscleCount];
        }

        Array.Copy(muscles, pose.muscles, MuscleCount);
    }
}
