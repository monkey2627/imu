using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Backward-compatible scene adapter. Existing practice scenes keep their
/// serialized MotionRecorder references while all recording is delegated to
/// the versioned ReplayRecorder pipeline.
/// </summary>
[DefaultExecutionOrder(9999)]
public sealed class MotionRecorder : MonoBehaviour
{
    [SerializeField] private Animator userAnimator;
    [SerializeField] private Animator coachAnimator;
    [SerializeField] private Transform sessionOrigin;
    [SerializeField, Min(1f)] private float sampleRate = 30f;

    private ReplayRecorder replayRecorder;

    public Animator UserAnimator => userAnimator;
    public Animator CoachAnimator => coachAnimator;
    public ReplayRecorder Recorder => replayRecorder;
    public ReplayRecorderState State => replayRecorder != null
        ? replayRecorder.State
        : ReplayRecorderState.Idle;
    public string CurrentSessionId => replayRecorder?.Manifest?.sessionId;

    private void Awake()
    {
        EnsureRecorder();
    }

    public void BeginOverwrite()
    {
        TryBeginRecording(out _);
    }

    public bool TryBeginRecording(out string error)
    {
        EnsureRecorder();
        ReplayManifest manifest = ReplayContextResolver.CaptureCurrent(
            userAnimator,
            coachAnimator,
            sampleRate);
        bool started = replayRecorder.TryBegin(
            manifest,
            new ReplayRepository(),
            out error);
        if (!started)
        {
            Debug.LogError($"Motion recording could not start: {error}", this);
        }

        return started;
    }

    public void FinishAndSave(bool completed)
    {
        EnsureRecorder();
        if (completed)
        {
            replayRecorder.RequestFinish();
        }
        else
        {
            replayRecorder.Cancel();
        }
    }

    public void RequestFinish()
    {
        EnsureRecorder();
        replayRecorder.RequestFinish();
    }

    public void CancelRecording()
    {
        EnsureRecorder();
        replayRecorder.Cancel();
    }

    // Kept for existing regression tests and legacy diagnostics. Replay V3
    // stores its configured sample rate in the manifest, while this helper
    // still reports the actually observed cadence for old V2 clips.
    private static float CalculateActualSampleRate(
        int frameCount,
        float duration,
        float fallback)
    {
        return frameCount > 1 && duration > 0.0001f
            ? (frameCount - 1) / duration
            : Mathf.Max(1f, fallback);
    }

    private void EnsureRecorder()
    {
        if (replayRecorder == null)
        {
            replayRecorder = GetComponent<ReplayRecorder>();
        }

        if (replayRecorder == null)
        {
            replayRecorder = gameObject.AddComponent<ReplayRecorder>();
        }

        if (sessionOrigin == null)
        {
            sessionOrigin = FindSharedPresentationRoot(
                userAnimator != null ? userAnimator.transform : null,
                coachAnimator != null ? coachAnimator.transform : null);
            if (sessionOrigin == null)
            {
                GameObject originObject = GameObject.Find("ReplaySessionOrigin");
                if (originObject == null)
                {
                    originObject = new GameObject("ReplaySessionOrigin");
                    originObject.transform.SetPositionAndRotation(
                        CalculateFallbackSessionOrigin(
                            userAnimator != null ? userAnimator.transform : null,
                            coachAnimator != null ? coachAnimator.transform : null),
                        Quaternion.identity);
                }

                sessionOrigin = originObject.transform;
            }
        }

        replayRecorder.Configure(
            userAnimator,
            coachAnimator,
            sessionOrigin,
            userAnimator != null ? userAnimator.transform : null,
            coachAnimator != null ? coachAnimator.transform : null,
            sampleRate);
    }

    public static Transform FindSharedPresentationRoot(
        Transform userRoot,
        Transform coachRoot)
    {
        if (userRoot == null || coachRoot == null)
        {
            return null;
        }

        var userAncestors = new HashSet<Transform>();
        for (Transform current = userRoot.parent;
             current != null;
             current = current.parent)
        {
            userAncestors.Add(current);
        }

        for (Transform current = coachRoot.parent;
             current != null;
             current = current.parent)
        {
            if (userAncestors.Contains(current))
            {
                return current;
            }
        }

        return null;
    }

    public static Vector3 CalculateFallbackSessionOrigin(
        Transform userRoot,
        Transform coachRoot)
    {
        if (userRoot == null && coachRoot == null)
        {
            return Vector3.zero;
        }

        Vector3 userPosition = userRoot != null
            ? userRoot.position
            : coachRoot.position;
        Vector3 coachPosition = coachRoot != null
            ? coachRoot.position
            : userRoot.position;
        Vector3 midpoint = (userPosition + coachPosition) * 0.5f;
        return new Vector3(midpoint.x, 0f, midpoint.z);
    }
}
