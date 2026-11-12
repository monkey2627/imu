using System;
using UnityEngine;

public enum ReplayPlayerState
{
    Unavailable,
    Loading,
    WaitingForPlacement,
    Ready,
    Playing,
    Paused,
    Seeking,
    Ended,
    Error
}

[DefaultExecutionOrder(5000)]
public sealed class ReplayPlayer : MonoBehaviour
{
    private ReplaySession session;
    private ReplayTimeline timeline;
    private HumanPoseHandler userPoseHandler;
    private HumanPoseHandler coachPoseHandler;
    private Animator userAnimator;
    private Animator coachAnimator;
    private Transform userPresentationRoot;
    private Transform coachPresentationRoot;
    private Transform replayRoot;
    private bool placementReady;
    private bool resumeAfterSeek;
    private bool useRecordedRootBodyPose;
    private Vector3 fallbackUserBodyPosition;
    private Quaternion fallbackUserBodyRotation = Quaternion.identity;
    private Vector3 fallbackCoachBodyPosition;
    private Quaternion fallbackCoachBodyRotation = Quaternion.identity;
    private double currentTime;
    private string error;

    public event Action<ReplayPlayerState> StateChanged;
    public event Action<double> TimeChanged;

    public ReplayPlayerState State { get; private set; } = ReplayPlayerState.Unavailable;
    public ReplaySession Session => session;
    public ReplayManifest Manifest => session?.Manifest;
    public double CurrentTime => currentTime;
    public double Duration => timeline?.Duration ?? 0d;
    public string Error => error;
    public bool HasReplay => timeline != null;
    public float NormalizedProgress => Duration <= 0d
        ? 0f
        : Mathf.Clamp01((float)(currentTime / Duration));
    public bool ControlsEnabled => HasReplay && placementReady &&
        State != ReplayPlayerState.Error &&
        State != ReplayPlayerState.Unavailable &&
        State != ReplayPlayerState.Loading &&
        State != ReplayPlayerState.WaitingForPlacement;

    public void SetLoading()
    {
        error = null;
        SetState(ReplayPlayerState.Loading);
    }

    public bool TryLoad(
        ReplaySession replaySession,
        Animator replayUserAnimator,
        Animator replayCoachAnimator,
        Transform userRoot,
        Transform coachRoot,
        Transform spatialReplayRoot,
        out string loadError)
    {
        loadError = null;
        DisposePoseHandlers();
        session = replaySession;
        timeline = null;
        placementReady = false;
        currentTime = 0d;

        if (replaySession == null ||
            !ReplayDataValidator.TryValidate(
                replaySession.Manifest,
                replaySession.Frames,
                requireCompleted: true,
                out loadError))
        {
            Fail(string.IsNullOrWhiteSpace(loadError)
                ? "Replay session is invalid."
                : loadError);
            return false;
        }

        if (!TryCreateHandler(
                replayUserAnimator,
                "user",
                out userPoseHandler,
                out loadError) ||
            !TryCreateHandler(
                replayCoachAnimator,
                "coach",
                out coachPoseHandler,
                out loadError))
        {
            DisposePoseHandlers();
            Fail(loadError);
            return false;
        }

        if (userRoot == null || coachRoot == null || spatialReplayRoot == null)
        {
            loadError = "Replay presentation roots are missing.";
            DisposePoseHandlers();
            Fail(loadError);
            return false;
        }

        userAnimator = replayUserAnimator;
        coachAnimator = replayCoachAnimator;
        userPresentationRoot = userRoot;
        coachPresentationRoot = coachRoot;
        replayRoot = spatialReplayRoot;
        timeline = new ReplayTimeline(replaySession.Frames);
        useRecordedRootBodyPose = string.Equals(
            replaySession.Manifest.coordinateSystemVersion,
            ReplayManifest.RootBodyPoseCoordinateSystem,
            System.StringComparison.Ordinal);
        if (!useRecordedRootBodyPose)
        {
            Debug.LogWarning(
                "[ReplayPlayer] Legacy world-space body pose detected; " +
                "Playback will keep each avatar's local body baseline and apply muscles only.",
                this);
        }

        userAnimator.applyRootMotion = false;
        userAnimator.enabled = false;
        // Disable both Animator state machines so the old PlayBack coach clip
        // cannot compete with the recorded Practice session. ReplayPlayer is
        // the only writer and applies the user and coach samples from the same
        // frame on the same clock.
        coachAnimator.applyRootMotion = false;
        coachAnimator.enabled = false;

        CaptureFallbackBodyPose(
            userPoseHandler,
            userAnimator,
            userPresentationRoot,
            out fallbackUserBodyPosition,
            out fallbackUserBodyRotation);
        CaptureFallbackBodyPose(
            coachPoseHandler,
            coachAnimator,
            coachPresentationRoot,
            out fallbackCoachBodyPosition,
            out fallbackCoachBodyRotation);

        ApplyAt(0d);
        SetState(ReplayPlayerState.WaitingForPlacement);
        loadError = null;
        return true;
    }

    public void MarkPlacementReady()
    {
        if (!HasReplay || State == ReplayPlayerState.Error)
        {
            return;
        }

        placementReady = true;
        currentTime = 0d;
        ApplyAt(0d);
        SetState(ReplayPlayerState.Ready);
    }

    public void Play()
    {
        if (!ControlsEnabled)
        {
            return;
        }

        if (State == ReplayPlayerState.Ended || currentTime >= Duration)
        {
            currentTime = 0d;
            ApplyAt(currentTime);
        }

        SetState(ReplayPlayerState.Playing);
    }

    public void Pause()
    {
        if (State == ReplayPlayerState.Playing)
        {
            SetState(ReplayPlayerState.Paused);
        }
    }

    public void Restart()
    {
        if (!ControlsEnabled)
        {
            return;
        }

        currentTime = 0d;
        ApplyAt(0d);
        SetState(ReplayPlayerState.Paused);
    }

    public void TogglePlayPause()
    {
        if (State == ReplayPlayerState.Playing)
        {
            Pause();
        }
        else
        {
            Play();
        }
    }

    public void SeekNormalized(float normalized)
    {
        SeekSeconds(Mathf.Clamp01(normalized) * Duration);
    }

    public void SeekRelative(float seconds)
    {
        SeekSeconds(currentTime + seconds);
    }

    public void SeekSeconds(double seconds)
    {
        if (!ControlsEnabled && State != ReplayPlayerState.Seeking)
        {
            return;
        }

        currentTime = Math.Max(0d, Math.Min(Duration, seconds));
        ApplyAt(currentTime);
        if (State != ReplayPlayerState.Seeking)
        {
            SetState(currentTime >= Duration
                ? ReplayPlayerState.Ended
                : ReplayPlayerState.Paused);
        }
    }

    public void BeginSeek()
    {
        if (!ControlsEnabled)
        {
            return;
        }

        resumeAfterSeek = State == ReplayPlayerState.Playing;
        SetState(ReplayPlayerState.Seeking);
    }

    public void EndSeek()
    {
        if (State != ReplayPlayerState.Seeking)
        {
            return;
        }

        if (currentTime >= Duration)
        {
            SetState(ReplayPlayerState.Ended);
        }
        else
        {
            SetState(resumeAfterSeek
                ? ReplayPlayerState.Playing
                : ReplayPlayerState.Paused);
        }
    }

    public void Advance(double deltaTime)
    {
        if (State != ReplayPlayerState.Playing || deltaTime <= 0d)
        {
            return;
        }

        currentTime = Math.Min(Duration, currentTime + deltaTime);
        ApplyAt(currentTime);
        if (currentTime >= Duration)
        {
            SetState(ReplayPlayerState.Ended);
        }
    }

    public void SetUnavailable(string reason)
    {
        error = reason;
        SetState(ReplayPlayerState.Unavailable);
    }

    public void Fail(string reason)
    {
        error = string.IsNullOrWhiteSpace(reason)
            ? "Replay failed for an unknown reason."
            : reason;
        SetState(ReplayPlayerState.Error);
        Debug.LogError($"ReplayPlayer error: {error}", this);
    }

    private void Update()
    {
        Advance(Time.unscaledDeltaTime);
    }

    private void ApplyAt(double time)
    {
        if (timeline == null)
        {
            return;
        }

        ReplayPosePair pose = timeline.Evaluate(time);
        ApplyActorPose(
            pose.User,
            userPoseHandler,
            userPresentationRoot,
            useRecordedRootBodyPose,
            fallbackUserBodyPosition,
            fallbackUserBodyRotation);
        ApplyActorPose(
            pose.Coach,
            coachPoseHandler,
            coachPresentationRoot,
            useRecordedRootBodyPose,
            fallbackCoachBodyPosition,
            fallbackCoachBodyRotation);
        TimeChanged?.Invoke(currentTime);
    }

    private static void ApplyActorPose(
        ReplayActorPose actor,
        HumanPoseHandler poseHandler,
        Transform presentationRoot,
        bool useRecordedRootBodyPose,
        Vector3 fallbackBodyPosition,
        Quaternion fallbackBodyRotation)
    {
        // The PlayBack scene owns presentation position, facing and scale.
        // HumanPose owns the recorded humanoid pose for each actor. Keeping
        // those responsibilities separate prevents recorded Practice root
        // coordinates from rotating, enlarging or relocating the page.
        Vector3 authoredPosition = presentationRoot.position;
        Quaternion authoredRotation = presentationRoot.rotation;
        Vector3 authoredScale = presentationRoot.localScale;

        var humanPose = new HumanPose
        {
            bodyPosition = useRecordedRootBodyPose
                ? actor.bodyPosition
                : fallbackBodyPosition,
            bodyRotation = useRecordedRootBodyPose
                ? actor.bodyRotation
                : fallbackBodyRotation,
            muscles = (float[])actor.muscles.Clone()
        };
        poseHandler.SetHumanPose(ref humanPose);
        presentationRoot.SetPositionAndRotation(
            authoredPosition,
            authoredRotation);
        presentationRoot.localScale = authoredScale;
    }

    private static void CaptureFallbackBodyPose(
        HumanPoseHandler poseHandler,
        Animator animator,
        Transform presentationRoot,
        out Vector3 bodyPosition,
        out Quaternion bodyRotation)
    {
        var currentPose = new HumanPose
        {
            muscles = new float[HumanTrait.MuscleCount]
        };
        poseHandler.GetHumanPose(ref currentPose);

        float humanScale = animator == null || animator.humanScale <= 0.0001f
            ? 1.0f
            : animator.humanScale;
        bodyPosition = presentationRoot.InverseTransformPoint(
                currentPose.bodyPosition * humanScale)
            / humanScale;
        bodyRotation = Quaternion.Inverse(presentationRoot.rotation)
            * currentPose.bodyRotation;
    }

    private static bool TryCreateHandler(
        Animator animator,
        string actorName,
        out HumanPoseHandler handler,
        out string handlerError)
    {
        handler = null;
        if (animator == null || animator.avatar == null ||
            !animator.avatar.isValid || !animator.avatar.isHuman || !animator.isHuman)
        {
            handlerError = $"Replay {actorName} Avatar is missing or not Humanoid-compatible.";
            return false;
        }

        handler = new HumanPoseHandler(animator.avatar, animator.transform);
        handlerError = null;
        return true;
    }

    private void SetState(ReplayPlayerState newState)
    {
        if (State == newState)
        {
            return;
        }

        State = newState;
        StateChanged?.Invoke(State);
    }

    private void DisposePoseHandlers()
    {
        userPoseHandler?.Dispose();
        coachPoseHandler?.Dispose();
        userPoseHandler = null;
        coachPoseHandler = null;
    }

    private void OnDestroy()
    {
        DisposePoseHandlers();
    }

}
