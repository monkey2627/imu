using System;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class RecordedMotionClip
{
    public int formatVersion = 2;
    public float fps = 30f;
    public float duration;
    public bool completed;
    public List<RecordedMotionFrame> frames = new List<RecordedMotionFrame>();
}

[Serializable]
public class RecordedMotionFrame
{
    public float time;
    // Kept as "bones" for backward compatibility with existing recordings.
    public List<RecordedBonePose> bones = new List<RecordedBonePose>();
    public List<RecordedBonePose> coachBones = new List<RecordedBonePose>();
}

[Serializable]
public class RecordedBonePose
{
    public string bone;
    public Vector3 localPosition;
    public Quaternion localRotation;
}
