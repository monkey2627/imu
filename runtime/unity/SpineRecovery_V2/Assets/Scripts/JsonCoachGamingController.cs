using System;
using System.Collections;
using UnityEngine;

public sealed class JsonCoachGamingController : MonoBehaviour
{
    public const float DefaultCheckpointHoldSeconds = 0.2f;

    public event Action ActionCompleted;
    public event Action ScoringSessionStarted;
    public event Action<float> ScoringProgressChanged;
    public event Action<JsonGamingCheckpointInfo> ScoringCheckpointStarted;
    public event Action<JsonGamingCheckpointInfo> ScoringCheckpointEnded;
    public event Action ScoringSessionCompleted;

    [SerializeField] private Animator coachAnimator;
    [SerializeField] private float checkpointHoldSeconds = DefaultCheckpointHoldSeconds;

    private CoachMotionPackage package;
    private KeyframeCoachMotionBackend backend;
    private Coroutine playbackRoutine;

    public Animator CoachAnimator => coachAnimator;
    public int ScoringUnitCount => BuildScoringUnitCount(package);

    public static int BuildScoringUnitCount(CoachMotionPackage motionPackage)
    {
        if (motionPackage?.segments == null)
        {
            return 0;
        }

        int unitsPerSet = 0;
        foreach (CoachMotionSegment segment in motionPackage.segments)
        {
            unitsPerSet += Mathf.Max(1, segment?.repeatCount ?? 1);
        }

        return Mathf.Max(1, motionPackage.setCount) * unitsPerSet;
    }

    public void Initialize(Animator animator, CoachMotionPackage motionPackage)
    {
        coachAnimator = animator;
        package = motionPackage;
        backend?.Dispose();
        backend = new KeyframeCoachMotionBackend(coachAnimator, package, true);
        backend.ReturnIdle();
    }

    public void PlayAction()
    {
        if (playbackRoutine != null)
        {
            StopCoroutine(playbackRoutine);
        }
        playbackRoutine = StartCoroutine(PlayPackage());
    }

    public void TogglePause()
    {
        if (Time.timeScale > 0f)
        {
            Time.timeScale = 0f;
            backend?.Freeze();
        }
        else
        {
            Time.timeScale = 1f;
            backend?.Resume();
        }
    }

    public void ReturnIdle()
    {
        if (playbackRoutine != null)
        {
            StopCoroutine(playbackRoutine);
            playbackRoutine = null;
        }
        backend?.ReturnIdle();
    }

    private IEnumerator PlayPackage()
    {
        if (backend == null || !backend.IsReady || ScoringUnitCount == 0)
        {
            Debug.LogError("JsonCoachGamingController needs a valid cached motion package.", this);
            yield break;
        }

        ScoringSessionStarted?.Invoke();
        ScoringProgressChanged?.Invoke(0f);
        float holdSeconds = Mathf.Max(0.01f, checkpointHoldSeconds);

        int scoringUnitIndex = 0;
        int scoringUnitCount = ScoringUnitCount;
        int setCount = Mathf.Max(1, package.setCount);
        for (int setIndex = 0; setIndex < setCount; setIndex++)
        {
            for (int segmentIndex = 0; segmentIndex < package.segments.Length; segmentIndex++)
            {
                CoachMotionSegment segment = package.segments[segmentIndex];
                int repeatCount = Mathf.Max(1, segment.repeatCount);
                for (int repeatIndex = 0; repeatIndex < repeatCount; repeatIndex++)
                {
                    var checkpoint = new JsonGamingCheckpointInfo(
                        string.IsNullOrEmpty(segment.label) ? segment.sourceStateName : segment.label,
                        scoringUnitIndex,
                        scoringUnitCount,
                        setIndex,
                        setCount,
                        segmentIndex,
                        package.segments.Length,
                        repeatIndex,
                        repeatCount,
                        holdSeconds);
                    ScoringCheckpointStarted?.Invoke(checkpoint);
                    backend.PlaySegment(segmentIndex);
                    while (!backend.HasReachedSegmentEnd())
                    {
                        backend.Tick(Time.deltaTime);
                        yield return null;
                    }

                    backend.SnapToSegmentEnd();
                    backend.Freeze();
                    yield return new WaitForSeconds(holdSeconds);
                    ScoringCheckpointEnded?.Invoke(checkpoint);
                    scoringUnitIndex++;
                    ScoringProgressChanged?.Invoke((float)scoringUnitIndex / scoringUnitCount);
                }
            }
        }

        ScoringSessionCompleted?.Invoke();
        ActionCompleted?.Invoke();
        playbackRoutine = null;
    }

    private void OnDestroy()
    {
        Time.timeScale = 1f;
        backend?.Dispose();
    }
}
