using System;
using System.Collections.Generic;
using UnityEngine;

public sealed class ReplayPosePair
{
    public ReplayActorPose User { get; }
    public ReplayActorPose Coach { get; }

    public ReplayPosePair(ReplayActorPose user, ReplayActorPose coach)
    {
        User = user;
        Coach = coach;
    }
}

public sealed class ReplayTimeline
{
    private readonly IReadOnlyList<ReplayFrame> frames;
    private int cachedInterval;

    public double Duration { get; }

    public ReplayTimeline(IReadOnlyList<ReplayFrame> frames)
    {
        this.frames = frames ?? throw new ArgumentNullException(nameof(frames));
        if (frames.Count == 0)
        {
            throw new ArgumentException("Replay timeline needs at least one frame.", nameof(frames));
        }

        Duration = frames[frames.Count - 1].timestamp;
    }

    public ReplayPosePair Evaluate(double time)
    {
        if (frames.Count == 1 || time <= frames[0].timestamp)
        {
            return FromExactFrame(frames[0]);
        }

        if (time >= Duration)
        {
            return FromExactFrame(frames[frames.Count - 1]);
        }

        int interval = FindInterval(time);
        ReplayFrame from = frames[interval];
        ReplayFrame to = frames[interval + 1];
        double span = to.timestamp - from.timestamp;
        float amount = span <= 0d
            ? 0f
            : Mathf.Clamp01((float)((time - from.timestamp) / span));
        return new ReplayPosePair(
            ReplayActorPose.Interpolate(from.user, to.user, amount),
            ReplayActorPose.Interpolate(from.coach, to.coach, amount));
    }

    private int FindInterval(double time)
    {
        int lastInterval = frames.Count - 2;
        int index = Mathf.Clamp(cachedInterval, 0, lastInterval);
        if (frames[index].timestamp <= time &&
            time <= frames[index + 1].timestamp)
        {
            return index;
        }

        int low = 0;
        int high = lastInterval;
        while (low <= high)
        {
            int middle = low + (high - low) / 2;
            if (frames[middle].timestamp <= time &&
                time <= frames[middle + 1].timestamp)
            {
                cachedInterval = middle;
                return middle;
            }

            if (time < frames[middle].timestamp)
            {
                high = middle - 1;
            }
            else
            {
                low = middle + 1;
            }
        }

        cachedInterval = Mathf.Clamp(low, 0, lastInterval);
        return cachedInterval;
    }

    private static ReplayPosePair FromExactFrame(ReplayFrame frame)
    {
        return new ReplayPosePair(frame.user.Clone(), frame.coach.Clone());
    }
}
