using System;
using UnityEngine;

[Serializable]
public sealed class RhythmPoseDiagnostic
{
    public string target,difficulty,reason;
    public bool sessionActive,paused,freshInput,tracked,postureMatched,endpointAvailable;
    public int validSegmentCount;
    public float poseScore,anatomicalScore,poseThreshold,progress,requiredExtension,endpointQuality,thighAngleDegrees,shinAngleDegrees;
}

public enum RhythmBodyTarget
{
    LeftHand,
    RightHand,
    LeftFoot,
    RightFoot,
    Waist
}

public enum RhythmJudgement
{
    None,
    Perfect,
    Great,
    Good,
    Miss
}

[Serializable]
public sealed class RhythmNoteVariantPool
{
    public RhythmBodyTarget target;
    public GameObject[] prefabs = new GameObject[0];
    public Material[] materials = new Material[0];
}

public readonly struct RhythmNoteResult
{
    public RhythmNoteResult(
        RhythmBodyTarget target,
        RhythmJudgement judgement,
        float poseScore,
        float timingError,
        Vector3 localHitPosition)
    {
        Target = target;
        Judgement = judgement;
        PoseScore = poseScore;
        TimingError = timingError;
        LocalHitPosition = localHitPosition;
    }

    public RhythmBodyTarget Target { get; }
    public RhythmJudgement Judgement { get; }
    public bool IsHit =>
        Judgement == RhythmJudgement.Perfect ||
        Judgement == RhythmJudgement.Great ||
        Judgement == RhythmJudgement.Good;
    public float PoseScore { get; }
    public float TimingError { get; }
    public Vector3 LocalHitPosition { get; }
}
