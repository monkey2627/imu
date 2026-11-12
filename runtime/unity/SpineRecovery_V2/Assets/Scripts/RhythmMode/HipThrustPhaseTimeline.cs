using System;
using UnityEngine;

/// <summary>Frame boundaries measured on the recorded coach, not on the seven-second artwork demo.</summary>
[CreateAssetMenu(menuName = "Training/Hip Thrust Phase Timeline")]
public sealed class HipThrustPhaseTimeline : ScriptableObject
{
    [Serializable]
    public struct Cycle
    {
        public int liftStartFrame, holdStartFrame, lowerStartFrame, lowerEndFrame;
        public float lowHeight, peakHeight;
    }
    public AnimationClip sourceClip;
    public int sourceStartFrame = 180;
    public int repeatStartFrame = 30, repeatEndFrame = 765;
    public Cycle[] cycles;
    // Frame -> measured hip-to-knee height. This also drives the supplied 13-frame illustration.
    public AnimationCurve recordedHeight;

    public struct Sample
    {
        public int phase, cycle;
        public float progress, poseHeight, holdSeconds;
        public Vector3 completedSeconds;
    }

    public Sample Evaluate(float sourceFrame, bool hasPreviousLoop = false)
    {
        var result = new Sample { phase = -1, cycle = -1, completedSeconds = Vector3.one * 999f };
        if (cycles == null || sourceClip == null) return result;
        float fps = sourceClip.frameRate;
        for (int i = 0; i < cycles.Length; i++)
        {
            Cycle c = cycles[i];
            Vector3 ends = new Vector3(c.holdStartFrame, c.lowerStartFrame, c.lowerEndFrame);
            for (int p = 0; p < 3; p++)
            {
                float age = sourceFrame - ends[p];
                if (age < 0 && hasPreviousLoop) age += repeatEndFrame - repeatStartFrame;
                if (age >= 0) result.completedSeconds[p] = Mathf.Min(result.completedSeconds[p], age / fps);
            }
            if (sourceFrame < c.liftStartFrame || sourceFrame >= c.lowerEndFrame) continue;
            result.cycle = i;
            int begin, end;
            if (sourceFrame < c.holdStartFrame) { result.phase = 0; begin = c.liftStartFrame; end = c.holdStartFrame; }
            else if (sourceFrame < c.lowerStartFrame) { result.phase = 1; begin = c.holdStartFrame; end = c.lowerStartFrame; }
            else { result.phase = 2; begin = c.lowerStartFrame; end = c.lowerEndFrame; }
            result.progress = Mathf.Clamp01((sourceFrame - begin) / (end - begin));
            result.poseHeight = result.phase == 1 ? 1f : Mathf.InverseLerp(c.lowHeight, c.peakHeight, recordedHeight.Evaluate(sourceFrame));
            result.holdSeconds = (sourceFrame - c.holdStartFrame) / fps;
        }
        return result;
    }

    public void Validate()
    {
        if (sourceClip == null || cycles == null || cycles.Length == 0 || recordedHeight == null || recordedHeight.length < 2)
            throw new InvalidOperationException("The recorded bridge timeline requires a clip, at least one cycle and its height curve.");
        int previousEnd = repeatStartFrame;
        foreach (Cycle c in cycles)
        {
            if (!(previousEnd <= c.liftStartFrame && c.liftStartFrame < c.holdStartFrame &&
                c.holdStartFrame < c.lowerStartFrame && c.lowerStartFrame < c.lowerEndFrame &&
                c.lowerEndFrame <= repeatEndFrame && c.peakHeight > c.lowHeight))
                throw new InvalidOperationException("Invalid recorded bridge phase boundaries.");
            previousEnd = c.lowerEndFrame;
        }
        if (repeatStartFrame < 0 || repeatEndFrame > Mathf.RoundToInt(sourceClip.length * sourceClip.frameRate))
            throw new InvalidOperationException("Repeat interval exceeds the coach clip.");
    }
}
