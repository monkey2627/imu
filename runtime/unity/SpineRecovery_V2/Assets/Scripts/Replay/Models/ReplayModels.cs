using System;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public sealed class ReplayManifest
{
    public const int CurrentFormatVersion = 3;
    public const string RootBodyPoseCoordinateSystem = "root-body-v1";
    public const string DefaultPayloadFile = "poses.bin";

    public int formatVersion = CurrentFormatVersion;
    public string sessionId;
    public string createdAtUtc;
    public string actionId;
    public string actionPackageId;
    public string actionPackageVersion;
    public string sourceScene;
    public string returnPracticeScene;
    public string returnGamingScene;
    public string avatarId;
    public string coachAvatarId;
    public string gender;
    public string rigId;
    public string coordinateSystemVersion = "session-origin-v1";
    public float sampleRate = 30f;
    public int frameCount;
    public double duration;
    public bool completed;
    public string payloadFile = DefaultPayloadFile;
    public string payloadSha256;
    public bool legacy;
    public string compatibilityNotice;

    public ReplayManifest Clone()
    {
        return (ReplayManifest)MemberwiseClone();
    }
}

[Serializable]
public sealed class ReplayActorPose
{
    public Vector3 bodyPosition;
    public Quaternion bodyRotation = Quaternion.identity;
    public float[] muscles = Array.Empty<float>();
    public Vector3 presentationRootPosition;
    public Quaternion presentationRootRotation = Quaternion.identity;
    public bool trackingValid = true;
    public float trackingConfidence = 1f;

    public ReplayActorPose Clone()
    {
        return new ReplayActorPose
        {
            bodyPosition = bodyPosition,
            bodyRotation = bodyRotation,
            muscles = muscles == null ? null : (float[])muscles.Clone(),
            presentationRootPosition = presentationRootPosition,
            presentationRootRotation = presentationRootRotation,
            trackingValid = trackingValid,
            trackingConfidence = trackingConfidence
        };
    }

    public static ReplayActorPose Interpolate(
        ReplayActorPose from,
        ReplayActorPose to,
        float amount)
    {
        if (from == null || to == null)
        {
            throw new ArgumentNullException(from == null ? nameof(from) : nameof(to));
        }

        if (from.muscles == null || to.muscles == null ||
            from.muscles.Length != to.muscles.Length)
        {
            throw new InvalidOperationException("Replay muscle arrays are incompatible.");
        }

        float t = Mathf.Clamp01(amount);
        var result = new ReplayActorPose
        {
            bodyPosition = Vector3.Lerp(from.bodyPosition, to.bodyPosition, t),
            bodyRotation = Quaternion.Slerp(from.bodyRotation, to.bodyRotation, t),
            presentationRootPosition = Vector3.Lerp(
                from.presentationRootPosition,
                to.presentationRootPosition,
                t),
            presentationRootRotation = Quaternion.Slerp(
                from.presentationRootRotation,
                to.presentationRootRotation,
                t),
            trackingValid = t < 0.5f ? from.trackingValid : to.trackingValid,
            trackingConfidence = Mathf.Lerp(
                from.trackingConfidence,
                to.trackingConfidence,
                t),
            muscles = new float[from.muscles.Length]
        };

        for (int i = 0; i < result.muscles.Length; i++)
        {
            result.muscles[i] = Mathf.Lerp(from.muscles[i], to.muscles[i], t);
        }

        return result;
    }
}

[Serializable]
public sealed class ReplayFrame
{
    public double timestamp;
    public ReplayActorPose user = new ReplayActorPose();
    public ReplayActorPose coach = new ReplayActorPose();

    public ReplayFrame Clone()
    {
        return new ReplayFrame
        {
            timestamp = timestamp,
            user = user?.Clone(),
            coach = coach?.Clone()
        };
    }
}

public sealed class ReplaySession
{
    public ReplayManifest Manifest { get; }
    public IReadOnlyList<ReplayFrame> Frames { get; }

    public ReplaySession(ReplayManifest manifest, IReadOnlyList<ReplayFrame> frames)
    {
        Manifest = manifest ?? throw new ArgumentNullException(nameof(manifest));
        Frames = frames ?? throw new ArgumentNullException(nameof(frames));
    }
}

public enum ReplayLoadStatus
{
    Success,
    Unavailable,
    Corrupt,
    UnsupportedVersion,
    Incomplete
}

public sealed class ReplayLoadResult
{
    public ReplayLoadStatus Status { get; }
    public ReplaySession Session { get; }
    public string Error { get; }
    public bool IsSuccess => Status == ReplayLoadStatus.Success && Session != null;

    private ReplayLoadResult(
        ReplayLoadStatus status,
        ReplaySession session,
        string error)
    {
        Status = status;
        Session = session;
        Error = error;
    }

    public static ReplayLoadResult Success(ReplaySession session) =>
        new ReplayLoadResult(ReplayLoadStatus.Success, session, null);

    public static ReplayLoadResult Failure(ReplayLoadStatus status, string error) =>
        new ReplayLoadResult(status, null, error);
}

public static class ReplayDataValidator
{
    public const double TimestampTolerance = 0.000001d;
    // Unity 2022.3 humanoid poses contain 95 muscle channels. Persistence and
    // validation also run on worker threads, where HumanTrait.MuscleCount is
    // not legal to call, so the V3 wire format keeps the value explicitly.
    public const int HumanMuscleCount = 95;

    public static bool TryValidate(
        ReplayManifest manifest,
        IReadOnlyList<ReplayFrame> frames,
        bool requireCompleted,
        out string error)
    {
        if (manifest == null)
        {
            error = "Replay manifest is missing.";
            return false;
        }

        if (manifest.formatVersion != ReplayManifest.CurrentFormatVersion)
        {
            error = $"Replay format {manifest.formatVersion} is not supported.";
            return false;
        }

        if (string.IsNullOrWhiteSpace(manifest.sessionId) ||
            manifest.sessionId.IndexOfAny(System.IO.Path.GetInvalidFileNameChars()) >= 0)
        {
            error = "Replay sessionId is missing or unsafe.";
            return false;
        }

        if (requireCompleted && !manifest.completed)
        {
            error = "Replay session is incomplete.";
            return false;
        }

        if (frames == null || frames.Count == 0)
        {
            error = "Replay has no pose frames.";
            return false;
        }

        if (manifest.frameCount != frames.Count)
        {
            error = $"Manifest frameCount {manifest.frameCount} does not match payload {frames.Count}.";
            return false;
        }

        if (Math.Abs(frames[0].timestamp) > TimestampTolerance)
        {
            error = "Replay first frame timestamp must be zero.";
            return false;
        }

        double previous = -1d;
        for (int i = 0; i < frames.Count; i++)
        {
            ReplayFrame frame = frames[i];
            if (frame == null || !IsFinite(frame.timestamp))
            {
                error = $"Replay frame {i} timestamp is invalid.";
                return false;
            }

            if (i > 0 && frame.timestamp <= previous)
            {
                error = $"Replay frame timestamps are not strictly monotonic at {i}.";
                return false;
            }

            if (!TryValidateActor(frame.user, i, "user", out error) ||
                !TryValidateActor(frame.coach, i, "coach", out error))
            {
                return false;
            }

            previous = frame.timestamp;
        }

        double expectedDuration = frames[frames.Count - 1].timestamp;
        if (!IsFinite(manifest.duration) ||
            Math.Abs(manifest.duration - expectedDuration) > 0.001d)
        {
            error = "Replay duration does not match the final frame.";
            return false;
        }

        if (!IsFinite(manifest.sampleRate) || manifest.sampleRate <= 0f)
        {
            error = "Replay sample rate is invalid.";
            return false;
        }

        error = null;
        return true;
    }

    public static Quaternion NormalizeFinite(Quaternion value)
    {
        if (!IsFinite(value.x) || !IsFinite(value.y) ||
            !IsFinite(value.z) || !IsFinite(value.w))
        {
            throw new InvalidOperationException("Replay quaternion contains NaN or Infinity.");
        }

        float magnitude = Mathf.Sqrt(
            value.x * value.x + value.y * value.y +
            value.z * value.z + value.w * value.w);
        if (magnitude < 0.000001f)
        {
            throw new InvalidOperationException("Replay quaternion has zero magnitude.");
        }

        float inverse = 1f / magnitude;
        return new Quaternion(
            value.x * inverse,
            value.y * inverse,
            value.z * inverse,
            value.w * inverse);
    }

    public static bool IsFinite(float value) =>
        !float.IsNaN(value) && !float.IsInfinity(value);

    public static bool IsFinite(double value) =>
        !double.IsNaN(value) && !double.IsInfinity(value);

    private static bool TryValidateActor(
        ReplayActorPose actor,
        int frameIndex,
        string actorName,
        out string error)
    {
        if (actor == null)
        {
            error = $"Replay frame {frameIndex} has no {actorName} pose.";
            return false;
        }

        if (actor.muscles == null || actor.muscles.Length != HumanMuscleCount)
        {
            error = $"Replay frame {frameIndex} {actorName} muscle count is invalid.";
            return false;
        }

        if (!IsFinite(actor.bodyPosition) ||
            !IsFinite(actor.presentationRootPosition) ||
            !IsFinite(actor.trackingConfidence))
        {
            error = $"Replay frame {frameIndex} {actorName} position/tracking data is invalid.";
            return false;
        }

        try
        {
            actor.bodyRotation = NormalizeFinite(actor.bodyRotation);
            actor.presentationRootRotation = NormalizeFinite(actor.presentationRootRotation);
        }
        catch (InvalidOperationException exception)
        {
            error = $"Replay frame {frameIndex} {actorName}: {exception.Message}";
            return false;
        }

        for (int i = 0; i < actor.muscles.Length; i++)
        {
            if (!IsFinite(actor.muscles[i]))
            {
                error = $"Replay frame {frameIndex} {actorName} muscle {i} is invalid.";
                return false;
            }
        }

        error = null;
        return true;
    }

    private static bool IsFinite(Vector3 value) =>
        IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);
}
