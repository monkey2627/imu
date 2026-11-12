using System.Collections;
using UnityEngine;

public class CoachActionController : MonoBehaviour
{
    public event System.Action ActionCompleted;
    public event System.Action ScoringSessionStarted;
    public event System.Action<float> ScoringProgressChanged;
    public event System.Action<SegmentPlaybackInfo> SegmentPlaybackStarted;
    public event System.Action<ScoringCheckpointInfo> ScoringCheckpointStarted;
    public event System.Action<ScoringCheckpointInfo> ScoringCheckpointEnded;
    public event System.Action ScoringSessionCompleted;

    public string StartSegmentStateName => startSegmentState;
    public Animator CoachAnimator => coachAnimator;
    // Read-only presentation state; the sequence and its existing hold coroutine remain authoritative.
    public bool IsPlayingAction => hasStarted;
    public bool IsActionPaused => isPaused;
    public bool IsHoldingPose => isDifficultyHold;
    public float HoldElapsedSeconds { get; private set; }
    public int CurrentTotalScoredSegments { get; private set; } = 1;
    public double[] AssessmentDurations { get; private set; } = new double[0];
    public float SegmentSeconds(string state,float speed)
    {
        if(coachAnimator!=null && coachAnimator.runtimeAnimatorController!=null)
            foreach(var clip in coachAnimator.runtimeAnimatorController.animationClips)
                if(clip!=null && clip.name==state)return clip.length/Mathf.Max(.05f,speed);
        return 0;
    }

    public struct ScoringCheckpointInfo
    {
        public string stateName;
        public int loopIndex;
        public int loopCount;
        public int pairIndex;
        public bool isUp;
        public float holdSeconds;

        public ScoringCheckpointInfo(
            string stateName,
            int loopIndex,
            int loopCount,
            int pairIndex,
            bool isUp,
            float holdSeconds)
        {
            this.stateName = stateName;
            this.loopIndex = loopIndex;
            this.loopCount = loopCount;
            this.pairIndex = pairIndex;
            this.isUp = isUp;
            this.holdSeconds = holdSeconds;
        }
    }

    public struct SegmentPlaybackInfo
    {
        public string stateName;
        public float playbackSpeed;

        public SegmentPlaybackInfo(string stateName, float playbackSpeed)
        {
            this.stateName = stateName;
            this.playbackSpeed = playbackSpeed;
        }
    }

    public enum TrainingDifficulty
    {
        Simple,
        Standard,
        Hard
    }

    [System.Serializable]
    private class DifficultyProfile
    {
        [Min(0f)] public float holdSeconds = 1f;
        [Min(0.05f)] public float upSpeed = 1f;
        [Min(0.05f)] public float downSpeed = 1f;
        [Min(1)] public int loopCount = 3;
    }

    [SerializeField] private Animator coachAnimator;
    [SerializeField] private DeadBugGamingSideFeedback sideFeedback;
    [SerializeField] private string actionStateName = "Deadbug1";
    [SerializeField] private bool returnIdleWhenActionCompletes = true;
    [SerializeField] private bool playRandomIdleVariations = true;
    [SerializeField] private string[] idleStateNames = { "Idle" };
    [SerializeField] private float idleMinDuration = 4f;
    [SerializeField] private float idleMaxDuration = 8f;
    [SerializeField] private float idleTransitionDuration = 0.25f;
    [SerializeField] private int animatorLayer = 0;
    [SerializeField] private bool pauseAnimatorUntilActionStarts = true;

    [Header("Segmented Dead Bug")]
    [SerializeField] private bool useSegmentedDeadBugSequence = true;
    [SerializeField] private DifficultyProfile simpleDifficulty = new DifficultyProfile
    {
        holdSeconds = 0f,
        upSpeed = 1.25f,
        downSpeed = 1.25f,
        loopCount = 1
    };
    [SerializeField] private DifficultyProfile standardDifficulty = new DifficultyProfile
    {
        holdSeconds = 0.1f,
        upSpeed = 1f,
        downSpeed = 1f,
        loopCount = 2
    };
    [SerializeField] private DifficultyProfile hardDifficulty = new DifficultyProfile
    {
        holdSeconds = 0.2f,
        upSpeed = 0.75f,
        downSpeed = 0.75f,
        loopCount = 3
    };
    [SerializeField] private string startSegmentState = "deadbug_start";
    [SerializeField] private string endSegmentState = "deadbug_end";
    [SerializeField] private string[] upSegmentStates =
    {
        "deadbug_1_up",
        "deadbug_2_up",
        "deadbug_3_up",
        "deadbug_4_up"
    };
    [SerializeField] private string[] downSegmentStates =
    {
        "deadbug_1_down",
        "deadbug_2_down",
        "deadbug_3_down",
        "deadbug_4_down"
    };

    private bool isPaused = false;
    private bool hasStarted = false;
    private string currentActionStateName;
    private Coroutine actionRoutine;
    private Coroutine idleRoutine;
    private Coroutine actionDiagnosticsRoutine;
    private TrainingDifficulty selectedDifficulty = TrainingDifficulty.Standard;
    public string SelectedDifficultyKey => selectedDifficulty==TrainingDifficulty.Simple?"simple":selectedDifficulty==TrainingDifficulty.Hard?"hard":"standard";
    private float activePlaybackSpeed = 1f;
    private bool isDifficultyHold;

    private void Awake()
    {
        ResolveCoachAnimator();

        if (sideFeedback == null)
        {
            sideFeedback = FindObjectOfType<DeadBugGamingSideFeedback>();
        }

        if (coachAnimator != null)
        {
            // The coach may be outside the active camera while the start
            // menu is visible. Its pose must still advance before it is moved
            // over the user at the end of the countdown.
            coachAnimator.cullingMode = AnimatorCullingMode.AlwaysAnimate;

            // DbPreview uses the action as its default state. Keep the coach on
            // the first frame until the countdown calls PlayAction(), which
            // restores the speed to 1 before playing the requested state.
            coachAnimator.speed = 0f;
            coachAnimator.enabled = false;
        }

        currentActionStateName = actionStateName;
    }

    private void OnEnable()
    {
        StartIdleVariations();
    }

    private void Update()
    {
        if (pauseAnimatorUntilActionStarts && !hasStarted && coachAnimator != null)
        {
            coachAnimator.speed = 0f;
        }
    }

    private void OnDisable()
    {
        StopIdleVariations();
        StopActionRoutine();
    }

    public void PlayAction1()
    {
        PlayAction(actionStateName);
    }

    public void PlayAction(string stateName)
    {
        ResolveCoachAnimator();

        if (coachAnimator == null)
        {
            Debug.LogError("CoachActionController needs a coach Animator.");
            return;
        }

        if (string.IsNullOrEmpty(stateName))
        {
            Debug.LogError("CoachActionController needs an action state name.");
            return;
        }

        if (useSegmentedDeadBugSequence && HasSegmentedDeadBugStates())
        {
            StartSegmentedDeadBug();
            return;
        }

        currentActionStateName = stateName;
        coachAnimator.enabled = true;
        coachAnimator.cullingMode = AnimatorCullingMode.AlwaysAnimate;
        coachAnimator.speed = 1f;
        activePlaybackSpeed = 1f;
        isPaused = false;
        hasStarted = true;
        StopIdleVariations();

        // The coach can be reparented under a GroundRoot and its VRM bind pose can
        // be regenerated at edit time. Rebind here so the existing controller is
        // initialized against the current hierarchy before starting the action.
        coachAnimator.Rebind();
        coachAnimator.Update(0f);

        if (sideFeedback != null)
        {
            sideFeedback.BeginFeedback();
        }

        if (!HasAnimatorState(currentActionStateName))
        {
            Debug.LogWarning($"Animator state '{currentActionStateName}' was not found on layer {animatorLayer}.", this);
        }

        coachAnimator.Play(currentActionStateName, animatorLayer, 0f);
        coachAnimator.Update(0f);

        AnimatorStateInfo startedState = coachAnimator.GetCurrentAnimatorStateInfo(animatorLayer);
        Debug.Log(
            $"[CoachAction] Play '{currentActionStateName}', controller=" +
            $"{coachAnimator.runtimeAnimatorController?.name ?? "None"}, " +
            $"isRequestedState={startedState.IsName(currentActionStateName)}, " +
            $"stateHash={startedState.shortNameHash}, length={startedState.length:0.###}, " +
            $"normalizedTime={startedState.normalizedTime:0.###}, speed={coachAnimator.speed}.",
            this);

        if (actionDiagnosticsRoutine != null)
        {
            StopCoroutine(actionDiagnosticsRoutine);
        }
        actionDiagnosticsRoutine = StartCoroutine(LogActionProgress());

        StopActionRoutine();

        if (returnIdleWhenActionCompletes)
        {
            actionRoutine = StartCoroutine(ReturnIdleAfterActionCompletes());
        }
    }

    public void SetActionStateName(string stateName)
    {
        if (!string.IsNullOrEmpty(stateName))
        {
            actionStateName = stateName;
            currentActionStateName = stateName;
        }
    }

    public void SetTrainingDifficulty(TrainingDifficulty difficulty)
    {
        selectedDifficulty = difficulty;
    }

    public void SetCurrentDifficultyHoldSeconds(float holdSeconds)
    {
        GetSelectedDifficultyProfile().holdSeconds = Mathf.Max(0f, holdSeconds);
    }

    public void ReturnIdle()
    {
        if (coachAnimator == null)
        {
            return;
        }

        coachAnimator.speed = 1f;
        activePlaybackSpeed = 1f;
        isPaused = false;
        isDifficultyHold = false;
        hasStarted = false;

        StopActionRoutine();

        if (sideFeedback != null)
        {
            sideFeedback.EndFeedback();
        }

        PlayIdleStateImmediately();
        StartIdleVariations();
    }

    public void TogglePause()
    {
        if (coachAnimator == null || !hasStarted || isDifficultyHold)
        {
            return;
        }

        isPaused = !isPaused;
        coachAnimator.speed = isPaused ? 0f : activePlaybackSpeed;
    }

    public void StopAtCurrentPose()
    {
        hasStarted = false;
        isPaused = false;
        isDifficultyHold = false;
        activePlaybackSpeed = 0f;
        if (coachAnimator != null)
        {
            coachAnimator.speed = 0f;
        }
    }

    public void CancelTraining()
    {
        StopActionRoutine();
        StopIdleVariations();
        if (actionDiagnosticsRoutine != null) StopCoroutine(actionDiagnosticsRoutine);
        actionDiagnosticsRoutine = null;
        StopAtCurrentPose();
        if (sideFeedback != null) sideFeedback.EndFeedback();
    }

    private IEnumerator ReturnIdleAfterActionCompletes()
    {
        yield return null;

        while (hasStarted)
        {
            AnimatorStateInfo stateInfo = coachAnimator.GetCurrentAnimatorStateInfo(0);
            bool isActionState = stateInfo.IsName(currentActionStateName);
            bool completed = isActionState && stateInfo.normalizedTime >= 1f && !coachAnimator.IsInTransition(0);

            if (completed)
            {
                Debug.Log(
                    $"[CoachAction] '{currentActionStateName}' completed at " +
                    $"normalizedTime={stateInfo.normalizedTime:0.###}, length={stateInfo.length:0.###}; returning to idle.",
                    this);
                actionRoutine = null;
                ReturnIdle();
                ActionCompleted?.Invoke();
                yield break;
            }

            yield return null;
        }

        actionRoutine = null;
    }

    private IEnumerator LogActionProgress()
    {
        yield return new WaitForSeconds(0.5f);

        if (coachAnimator != null)
        {
            AnimatorStateInfo stateInfo = coachAnimator.GetCurrentAnimatorStateInfo(animatorLayer);
            Debug.Log(
                $"[CoachAction] After 0.5s: requested='{currentActionStateName}', " +
                $"isRequestedState={stateInfo.IsName(currentActionStateName)}, " +
                $"stateHash={stateInfo.shortNameHash}, length={stateInfo.length:0.###}, " +
                $"normalizedTime={stateInfo.normalizedTime:0.###}, speed={coachAnimator.speed}.",
                this);
        }

        actionDiagnosticsRoutine = null;
    }

    private void StartIdleVariations()
    {
        if (!playRandomIdleVariations || hasStarted || coachAnimator == null || idleRoutine != null)
        {
            return;
        }

        idleRoutine = StartCoroutine(PlayIdleVariations());
    }

    private void StopIdleVariations()
    {
        if (idleRoutine == null)
        {
            return;
        }

        StopCoroutine(idleRoutine);
        idleRoutine = null;
    }

    private IEnumerator PlayIdleVariations()
    {
        PlayIdleStateImmediately();

        while (!hasStarted)
        {
            float minDuration = Mathf.Max(0.1f, idleMinDuration);
            float maxDuration = Mathf.Max(minDuration, idleMaxDuration);
            yield return new WaitForSeconds(Random.Range(minDuration, maxDuration));

            if (!hasStarted)
            {
                CrossFadeRandomIdleState();
            }
        }

        idleRoutine = null;
    }

    private void PlayIdleStateImmediately()
    {
        string idleStateName = GetFirstAvailableIdleState();
        if (string.IsNullOrEmpty(idleStateName))
        {
            return;
        }

        coachAnimator.Play(idleStateName, animatorLayer, 0f);
    }

    private void CrossFadeRandomIdleState()
    {
        string idleStateName = GetRandomAvailableIdleState();
        if (string.IsNullOrEmpty(idleStateName))
        {
            return;
        }

        coachAnimator.CrossFadeInFixedTime(idleStateName, idleTransitionDuration, animatorLayer, 0f);
    }

    private string GetFirstAvailableIdleState()
    {
        if (idleStateNames == null || idleStateNames.Length == 0)
        {
            return HasAnimatorState("Idle") ? "Idle" : null;
        }

        foreach (string idleStateName in idleStateNames)
        {
            if (HasAnimatorState(idleStateName))
            {
                return idleStateName;
            }
        }

        return HasAnimatorState("Idle") ? "Idle" : null;
    }

    private string GetRandomAvailableIdleState()
    {
        if (idleStateNames == null || idleStateNames.Length == 0)
        {
            return GetFirstAvailableIdleState();
        }

        int startIndex = Random.Range(0, idleStateNames.Length);
        for (int i = 0; i < idleStateNames.Length; i++)
        {
            string idleStateName = idleStateNames[(startIndex + i) % idleStateNames.Length];
            if (HasAnimatorState(idleStateName))
            {
                return idleStateName;
            }
        }

        return GetFirstAvailableIdleState();
    }

    private void StartSegmentedDeadBug()
    {
        StopIdleVariations();
        StopActionRoutine();

        coachAnimator.enabled = true;
        coachAnimator.cullingMode = AnimatorCullingMode.AlwaysAnimate;
        coachAnimator.speed = 1f;
        activePlaybackSpeed = 1f;
        isPaused = false;
        isDifficultyHold = false;
        hasStarted = true;

        coachAnimator.Rebind();
        coachAnimator.Update(0f);

        if (sideFeedback != null)
        {
            sideFeedback.BeginFeedback();
        }

        DifficultyProfile profile = GetSelectedDifficultyProfile();
        Debug.Log(
            $"[CoachAction] Start segmented Dead Bug: difficulty={selectedDifficulty}, " +
            $"hold={profile.holdSeconds:0.##}s, upSpeed={profile.upSpeed:0.##}, " +
            $"downSpeed={profile.downSpeed:0.##}, loops={profile.loopCount}.",
            this);

        actionRoutine = StartCoroutine(PlaySegmentedDeadBug(profile));
    }

    private IEnumerator PlaySegmentedDeadBug(DifficultyProfile profile)
    {
        // deadbug_start is only the coach moving from standing to lying down.
        // Do not open the scoring session until that transition has finished.
        yield return PlaySegmentToEnd(startSegmentState, 1f);

        int pairCount = Mathf.Min(upSegmentStates.Length, downSegmentStates.Length);
        int safeLoopCount = Mathf.Max(1, profile.loopCount);
        int totalScoredSegments = Mathf.Max(1, safeLoopCount * pairCount * 2);
        CurrentTotalScoredSegments = totalScoredSegments;
        AssessmentDurations=new double[safeLoopCount*pairCount];
        for(int i=0;i<AssessmentDurations.Length;i++)
            AssessmentDurations[i]=SegmentSeconds(upSegmentStates[i%pairCount],profile.upSpeed)+
                SegmentSeconds(downSegmentStates[i%pairCount],profile.downSpeed);
        int completedScoredSegments = 0;

        ScoringSessionStarted?.Invoke();
        ScoringProgressChanged?.Invoke(0f);

        for (int loop = 0; loop < safeLoopCount && hasStarted; loop++)
        {
            for (int pair = 0; pair < pairCount && hasStarted; pair++)
            {
                yield return PlaySegmentToEnd(
                    upSegmentStates[pair],
                    profile.upSpeed,
                    (float)completedScoredSegments / totalScoredSegments,
                    1f / totalScoredSegments);
                completedScoredSegments++;
                ScoringCheckpointInfo upCheckpoint = new ScoringCheckpointInfo(
                    upSegmentStates[pair], loop, safeLoopCount, pair, true, profile.holdSeconds);
                ScoringCheckpointStarted?.Invoke(upCheckpoint);
                if (profile.holdSeconds > 0f) yield return HoldCurrentPose(profile.holdSeconds);
                ScoringCheckpointEnded?.Invoke(upCheckpoint);

                yield return PlaySegmentToEnd(
                    downSegmentStates[pair],
                    profile.downSpeed,
                    (float)completedScoredSegments / totalScoredSegments,
                    1f / totalScoredSegments);
                completedScoredSegments++;
                ScoringCheckpointInfo downCheckpoint = new ScoringCheckpointInfo(
                    downSegmentStates[pair], loop, safeLoopCount, pair, false, profile.holdSeconds);
                ScoringCheckpointStarted?.Invoke(downCheckpoint);
                if (profile.holdSeconds > 0f) yield return HoldCurrentPose(profile.holdSeconds);
                ScoringCheckpointEnded?.Invoke(downCheckpoint);
            }
        }

        if (hasStarted)
        {
            // Scoring ends while the coach is still lying down. The return from
            // lying to standing (deadbug_end) is deliberately excluded too.
            ScoringProgressChanged?.Invoke(1f);
            ScoringSessionCompleted?.Invoke();
            yield return PlaySegmentToEnd(endSegmentState, 1f);
        }

        if (coachAnimator != null)
        {
            coachAnimator.speed = 0f;
        }

        isDifficultyHold = false;
        hasStarted = false;
        actionRoutine = null;

        if (sideFeedback != null)
        {
            sideFeedback.EndFeedback();
        }

        Debug.Log("[CoachAction] Segmented Dead Bug complete.", this);
        ActionCompleted?.Invoke();
    }

    private IEnumerator PlaySegmentToEnd(
        string stateName,
        float playbackSpeed,
        float scoringProgressStart = -1f,
        float scoringProgressSpan = 0f)
    {
        if (coachAnimator == null || string.IsNullOrEmpty(stateName))
        {
            yield break;
        }

        activePlaybackSpeed = Mathf.Max(0.05f, playbackSpeed);
        isDifficultyHold = false;
        SegmentPlaybackStarted?.Invoke(new SegmentPlaybackInfo(stateName, activePlaybackSpeed));
        coachAnimator.speed = activePlaybackSpeed;
        coachAnimator.Play(stateName, animatorLayer, 0f);
        coachAnimator.Update(0f);

        bool reportScoringProgress = scoringProgressStart >= 0f;
        if (reportScoringProgress)
        {
            ScoringProgressChanged?.Invoke(Mathf.Clamp01(scoringProgressStart));
        }

        while (hasStarted)
        {
            AnimatorStateInfo stateInfo = coachAnimator.GetCurrentAnimatorStateInfo(animatorLayer);
            if (reportScoringProgress && stateInfo.IsName(stateName))
            {
                float stateProgress = Mathf.Clamp01(stateInfo.normalizedTime);
                ScoringProgressChanged?.Invoke(Mathf.Clamp01(
                    scoringProgressStart + scoringProgressSpan * stateProgress));
            }

            if (stateInfo.IsName(stateName) &&
                stateInfo.normalizedTime >= 1f &&
                !coachAnimator.IsInTransition(animatorLayer))
            {
                coachAnimator.Play(stateName, animatorLayer, 1f);
                coachAnimator.Update(0f);
                if (reportScoringProgress)
                {
                    ScoringProgressChanged?.Invoke(Mathf.Clamp01(
                        scoringProgressStart + scoringProgressSpan));
                }
                yield break;
            }

            yield return null;
        }
    }

    private IEnumerator HoldCurrentPose(float holdSeconds)
    {
        float safeHoldSeconds = Mathf.Max(0f, holdSeconds);
        if (safeHoldSeconds <= 0f || !hasStarted)
        {
            yield break;
        }

        isDifficultyHold = true;
        coachAnimator.speed = 0f;

        float elapsed = 0f;
        HoldElapsedSeconds = 0f;
        while (elapsed < safeHoldSeconds && hasStarted)
        {
            elapsed += Time.deltaTime;
            HoldElapsedSeconds = elapsed;
            yield return null;
        }

        isDifficultyHold = false;
    }

    private bool HasSegmentedDeadBugStates()
    {
        if (!HasAnimatorState(startSegmentState) || !HasAnimatorState(endSegmentState) ||
            upSegmentStates == null || downSegmentStates == null ||
            upSegmentStates.Length == 0 || upSegmentStates.Length != downSegmentStates.Length)
        {
            return false;
        }

        for (int index = 0; index < upSegmentStates.Length; index++)
        {
            if (!HasAnimatorState(upSegmentStates[index]) || !HasAnimatorState(downSegmentStates[index]))
            {
                return false;
            }
        }

        return true;
    }

    private DifficultyProfile GetSelectedDifficultyProfile()
    {
        // Difficulty is a scoring choice. Preserve the scene's original standard
        // speed, repetition count and phase timing for all three choices.
        return standardDifficulty;
    }

    private bool HasAnimatorState(string stateName)
    {
        return !string.IsNullOrEmpty(stateName) &&
            coachAnimator != null &&
            coachAnimator.HasState(animatorLayer, Animator.StringToHash(stateName));
    }

    private void ResolveCoachAnimator()
    {
        if (coachAnimator != null)
        {
            return;
        }

        Animator[] animators = FindObjectsOfType<Animator>(true);
        foreach (Animator animator in animators)
        {
            if (animator != null &&
                string.Equals(animator.gameObject.name, "coach", System.StringComparison.OrdinalIgnoreCase))
            {
                coachAnimator = animator;
                return;
            }
        }

        Debug.LogError("CoachActionController could not find an Animator on the 'coach' object.", this);
    }

    private void StopActionRoutine()
    {
        if (actionRoutine == null)
        {
            return;
        }

        StopCoroutine(actionRoutine);
        actionRoutine = null;
    }
}
