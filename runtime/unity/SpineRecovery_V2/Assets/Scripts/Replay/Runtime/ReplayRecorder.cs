using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

public enum ReplayRecorderState
{
    Idle,
    Preparing,
    Recording,
    FinishRequested,
    Committing,
    Committed,
    Cancelled,
    Faulted
}

public sealed class ReplayRecordingBuffer
{
    private const double MinimumTimestampStep = 0.000001d;

    private readonly List<ReplayFrame> frames = new List<ReplayFrame>();
    private double startedAt;
    private double nextSampleAt;
    private double sampleInterval;

    public ReplayRecorderState State { get; private set; } = ReplayRecorderState.Idle;
    public ReplayManifest Manifest { get; private set; }
    public IReadOnlyList<ReplayFrame> Frames => frames;
    public string Error { get; private set; }

    public void Begin(ReplayManifest manifest, float sampleRate)
    {
        if (State != ReplayRecorderState.Idle &&
            State != ReplayRecorderState.Committed &&
            State != ReplayRecorderState.Cancelled &&
            State != ReplayRecorderState.Faulted)
        {
            throw new InvalidOperationException($"Cannot begin a replay while recorder is {State}.");
        }

        Manifest = manifest?.Clone() ?? throw new ArgumentNullException(nameof(manifest));
        Manifest.formatVersion = ReplayManifest.CurrentFormatVersion;
        Manifest.sampleRate = Mathf.Max(1f, sampleRate);
        Manifest.completed = false;
        Manifest.frameCount = 0;
        Manifest.duration = 0d;
        Manifest.payloadFile = ReplayManifest.DefaultPayloadFile;
        Manifest.payloadSha256 = string.Empty;
        frames.Clear();
        sampleInterval = 1d / Manifest.sampleRate;
        nextSampleAt = sampleInterval;
        Error = null;
        State = ReplayRecorderState.Preparing;
    }

    public void CapturePreparedFirstFrame(double monotonicNow, ReplayFrame frame)
    {
        if (State != ReplayRecorderState.Preparing)
        {
            throw new InvalidOperationException("The recorder is not preparing its first frame.");
        }

        ReplayFrame first = RequireFrame(frame).Clone();
        first.timestamp = 0d;
        frames.Add(first);
        startedAt = monotonicNow;
        nextSampleAt = sampleInterval;
        State = ReplayRecorderState.Recording;
    }

    public bool ShouldCaptureSample(double monotonicNow)
    {
        return State == ReplayRecorderState.Recording &&
            monotonicNow - startedAt + ReplayDataValidator.TimestampTolerance >= nextSampleAt;
    }

    public void CaptureSample(double monotonicNow, ReplayFrame frame)
    {
        if (State != ReplayRecorderState.Recording)
        {
            throw new InvalidOperationException("The recorder is not recording.");
        }

        double elapsed = Math.Max(0d, monotonicNow - startedAt);
        AddStrictlyMonotonicFrame(frame, elapsed);
        nextSampleAt += sampleInterval;
        if (nextSampleAt <= elapsed)
        {
            // Skip missed intervals after a hitch; never synthesize duplicate
            // catch-up frames in the same rendered frame.
            nextSampleAt = elapsed + sampleInterval;
        }
    }

    public void RequestFinish()
    {
        if (State == ReplayRecorderState.Preparing ||
            State == ReplayRecorderState.Recording)
        {
            State = ReplayRecorderState.FinishRequested;
        }
    }

    public void CaptureFinalFrame(double monotonicNow, ReplayFrame frame)
    {
        if (State != ReplayRecorderState.FinishRequested)
        {
            throw new InvalidOperationException("A replay finish was not requested.");
        }

        if (frames.Count == 0)
        {
            ReplayFrame first = RequireFrame(frame).Clone();
            first.timestamp = 0d;
            frames.Add(first);
            startedAt = monotonicNow;
        }

        double elapsed = Math.Max(0d, monotonicNow - startedAt);
        AddStrictlyMonotonicFrame(frame, elapsed);
        Manifest.completed = true;
        Manifest.frameCount = frames.Count;
        Manifest.duration = frames[frames.Count - 1].timestamp;
        State = ReplayRecorderState.Committing;
    }

    public void MarkCommitted(ReplayManifest manifest)
    {
        Manifest = manifest ?? throw new ArgumentNullException(nameof(manifest));
        State = ReplayRecorderState.Committed;
    }

    public void MarkCancelled()
    {
        if (State == ReplayRecorderState.Committed)
        {
            return;
        }

        if (Manifest != null)
        {
            Manifest.completed = false;
            Manifest.frameCount = frames.Count;
            Manifest.duration = frames.Count > 0
                ? frames[frames.Count - 1].timestamp
                : 0d;
        }

        State = ReplayRecorderState.Cancelled;
    }

    public void MarkFaulted(string error)
    {
        Error = string.IsNullOrWhiteSpace(error) ? "Unknown replay recorder error." : error;
        State = ReplayRecorderState.Faulted;
    }

    private void AddStrictlyMonotonicFrame(ReplayFrame frame, double timestamp)
    {
        ReplayFrame copy = RequireFrame(frame).Clone();
        double previous = frames.Count == 0 ? -1d : frames[frames.Count - 1].timestamp;
        copy.timestamp = previous < 0d
            ? 0d
            : Math.Max(timestamp, previous + MinimumTimestampStep);
        frames.Add(copy);
    }

    private static ReplayFrame RequireFrame(ReplayFrame frame) =>
        frame ?? throw new ArgumentNullException(nameof(frame));
}

[DefaultExecutionOrder(10000)]
public sealed class ReplayRecorder : MonoBehaviour
{
    [SerializeField] private Animator userAnimator;
    [SerializeField] private Animator coachAnimator;
    [SerializeField] private Transform userPresentationRoot;
    [SerializeField] private Transform coachPresentationRoot;
    [SerializeField] private Transform sessionOrigin;
    [SerializeField, Min(1f)] private float sampleRate = 30f;

    private readonly ReplayRecordingBuffer buffer = new ReplayRecordingBuffer();
    private ReplayRepository repository;
    private HumanPoseHandler userPoseHandler;
    private HumanPoseHandler coachPoseHandler;
    private HumanPose userHumanPose;
    private HumanPose coachHumanPose;
    private Task<ReplayManifest> commitTask;
    private bool partialSaveStarted;

    public event Action<ReplayRecorderState> StateChanged;
    public event Action<ReplayManifest> Committed;
    public event Action<string> Faulted;

    public ReplayRecorderState State => buffer.State;
    public ReplayManifest Manifest => buffer.Manifest;
    public IReadOnlyList<ReplayFrame> Frames => buffer.Frames;
    public string Error => buffer.Error;

    public void Configure(
        Animator user,
        Animator coach,
        Transform origin,
        Transform userRoot = null,
        Transform coachRoot = null,
        float requestedSampleRate = 30f)
    {
        userAnimator = user;
        coachAnimator = coach;
        sessionOrigin = origin;
        userPresentationRoot = userRoot != null ? userRoot : user?.transform;
        coachPresentationRoot = coachRoot != null ? coachRoot : coach?.transform;
        sampleRate = Mathf.Max(1f, requestedSampleRate);
    }

    public bool TryBegin(ReplayManifest manifest, ReplayRepository replayRepository, out string error)
    {
        if (!TryCreatePoseHandler(userAnimator, "user", out userPoseHandler, out error) ||
            !TryCreatePoseHandler(coachAnimator, "coach", out coachPoseHandler, out error))
        {
            DisposePoseHandlers();
            buffer.MarkFaulted(error);
            NotifyFault(error);
            return false;
        }

        if (userPresentationRoot == null || coachPresentationRoot == null)
        {
            error = "ReplayRecorder needs both presentation roots.";
            DisposePoseHandlers();
            buffer.MarkFaulted(error);
            NotifyFault(error);
            return false;
        }

        repository = replayRepository ?? new ReplayRepository();
        try
        {
            buffer.Begin(manifest, sampleRate);
        }
        catch (Exception exception)
        {
            error = exception.Message;
            DisposePoseHandlers();
            buffer.MarkFaulted(error);
            NotifyFault(error);
            return false;
        }

        commitTask = null;
        partialSaveStarted = false;
        ReplaySessionContext.SetPending(buffer.Manifest);
        NotifyState();
        error = null;
        return true;
    }

    public void RequestFinish()
    {
        ReplayRecorderState before = State;
        buffer.RequestFinish();
        if (State != before)
        {
            NotifyState();
        }
    }

    public void Cancel()
    {
        if (State == ReplayRecorderState.Idle ||
            State == ReplayRecorderState.Committed ||
            State == ReplayRecorderState.Cancelled ||
            State == ReplayRecorderState.Faulted)
        {
            return;
        }

        buffer.MarkCancelled();
        ReplaySessionContext.ClearIfMatches(buffer.Manifest?.sessionId);
        NotifyState();
        SavePartialInBackground();
        DisposePoseHandlers();
    }

    private void Update()
    {
        if (State != ReplayRecorderState.Committing ||
            commitTask == null || !commitTask.IsCompleted)
        {
            return;
        }

        if (commitTask.IsFaulted)
        {
            string error = commitTask.Exception?.GetBaseException().Message ??
                "Replay commit failed.";
            buffer.MarkFaulted(error);
            DisposePoseHandlers();
            NotifyFault(error);
            return;
        }

        if (commitTask.IsCanceled)
        {
            const string error = "Replay commit was cancelled.";
            buffer.MarkFaulted(error);
            DisposePoseHandlers();
            NotifyFault(error);
            return;
        }

        ReplayManifest committedManifest = commitTask.Result;
        buffer.MarkCommitted(committedManifest);
        ReplaySessionContext.SetCommitted(committedManifest);
        DisposePoseHandlers();
        NotifyState();
        Committed?.Invoke(committedManifest);
    }

    private void LateUpdate()
    {
        try
        {
            double now = Time.realtimeSinceStartupAsDouble;
            if (State == ReplayRecorderState.Preparing)
            {
                buffer.CapturePreparedFirstFrame(now, CaptureFrame());
                NotifyState();
                return;
            }

            if (State == ReplayRecorderState.Recording && buffer.ShouldCaptureSample(now))
            {
                buffer.CaptureSample(now, CaptureFrame());
                return;
            }

            if (State == ReplayRecorderState.FinishRequested)
            {
                // This is deliberately performed in LateUpdate: all tracking,
                // coach animation and scoring updates for the rendered frame
                // have completed before the terminal pose is sampled.
                buffer.CaptureFinalFrame(now, CaptureFrame());
                NotifyState();
                commitTask = repository.CommitCompletedAsync(
                    buffer.Manifest,
                    buffer.Frames);
            }
        }
        catch (Exception exception)
        {
            buffer.MarkFaulted(exception.Message);
            DisposePoseHandlers();
            NotifyFault(exception.Message);
        }
    }

    private ReplayFrame CaptureFrame()
    {
        userPoseHandler.GetHumanPose(ref userHumanPose);
        coachPoseHandler.GetHumanPose(ref coachHumanPose);
        return new ReplayFrame
        {
            user = BuildActorPose(
                userHumanPose,
                userAnimator,
                userPresentationRoot,
                userAnimator.gameObject.activeInHierarchy),
            coach = BuildActorPose(
                coachHumanPose,
                coachAnimator,
                coachPresentationRoot,
                coachAnimator.gameObject.activeInHierarchy)
        };
    }

    private ReplayActorPose BuildActorPose(
        HumanPose pose,
        Animator animator,
        Transform presentationRoot,
        bool trackingValid)
    {
        float humanScale = GetHumanScale(animator);
        Vector3 bodyWorldPosition = pose.bodyPosition * humanScale;
        Vector3 bodyLocalPosition = presentationRoot.InverseTransformPoint(bodyWorldPosition);
        Quaternion bodyLocalRotation = Quaternion.Inverse(presentationRoot.rotation)
            * pose.bodyRotation;
        Vector3 rootPosition = sessionOrigin != null
            ? sessionOrigin.InverseTransformPoint(presentationRoot.position)
            : presentationRoot.position;
        Quaternion rootRotation = sessionOrigin != null
            ? Quaternion.Inverse(sessionOrigin.rotation) * presentationRoot.rotation
            : presentationRoot.rotation;

        return new ReplayActorPose
        {
            // GetHumanPose returns a world-space, avatar-scale-normalized body
            // position and a world-space body rotation. SetHumanPose expects
            // both values relative to the target humanoid root, so persist the
            // root-relative normalized representation instead of leaking the
            // Practice scene's world coordinates into Playback.
            bodyPosition = bodyLocalPosition / humanScale,
            bodyRotation = ReplayDataValidator.NormalizeFinite(bodyLocalRotation),
            muscles = pose.muscles == null
                ? Array.Empty<float>()
                : (float[])pose.muscles.Clone(),
            presentationRootPosition = rootPosition,
            presentationRootRotation = ReplayDataValidator.NormalizeFinite(rootRotation),
            trackingValid = trackingValid,
            trackingConfidence = trackingValid ? 1f : 0f
        };
    }

    private static float GetHumanScale(Animator animator)
    {
        return animator == null || animator.humanScale <= 0.0001f
            ? 1.0f
            : animator.humanScale;
    }

    private void SavePartialInBackground()
    {
        if (partialSaveStarted || repository == null ||
            buffer.Manifest == null || buffer.Frames.Count == 0)
        {
            return;
        }

        partialSaveStarted = true;
        ReplayManifest manifest = buffer.Manifest.Clone();
        var frames = new List<ReplayFrame>(buffer.Frames.Count);
        for (int i = 0; i < buffer.Frames.Count; i++)
        {
            frames.Add(buffer.Frames[i].Clone());
        }

        Task.Run(() => repository.SaveIncomplete(manifest, frames)).ContinueWith(task =>
        {
            if (task.IsFaulted)
            {
                Debug.LogError(
                    $"Replay partial save failed: {task.Exception?.GetBaseException().Message}",
                    this);
            }
        }, TaskScheduler.FromCurrentSynchronizationContext());
    }

    private static bool TryCreatePoseHandler(
        Animator animator,
        string actorName,
        out HumanPoseHandler handler,
        out string error)
    {
        handler = null;
        if (animator == null)
        {
            error = $"Replay {actorName} Animator is missing.";
            return false;
        }

        if (animator.avatar == null || !animator.avatar.isValid ||
            !animator.avatar.isHuman || !animator.isHuman)
        {
            error = $"Replay {actorName} Avatar must be a valid Humanoid.";
            return false;
        }

        handler = new HumanPoseHandler(animator.avatar, animator.transform);
        error = null;
        return true;
    }

    private void NotifyState() => StateChanged?.Invoke(State);

    private void NotifyFault(string error)
    {
        ReplaySessionContext.ClearIfMatches(buffer.Manifest?.sessionId);
        NotifyState();
        Faulted?.Invoke(error);
        Debug.LogError($"ReplayRecorder faulted: {error}", this);
    }

    private void DisposePoseHandlers()
    {
        userPoseHandler?.Dispose();
        coachPoseHandler?.Dispose();
        userPoseHandler = null;
        coachPoseHandler = null;
    }

    private void OnDisable()
    {
        if (State == ReplayRecorderState.Preparing ||
            State == ReplayRecorderState.Recording ||
            State == ReplayRecorderState.FinishRequested)
        {
            Cancel();
        }
    }

    private void OnApplicationQuit()
    {
        if (State == ReplayRecorderState.Preparing ||
            State == ReplayRecorderState.Recording ||
            State == ReplayRecorderState.FinishRequested)
        {
            Cancel();
        }
    }

    private void OnDestroy()
    {
        DisposePoseHandlers();
    }
}
