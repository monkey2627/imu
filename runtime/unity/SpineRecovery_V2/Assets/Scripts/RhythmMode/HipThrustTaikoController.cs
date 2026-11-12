using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

/// <summary>HipThrust-only transport. Coach, preview and taiko chart sample one animation clock.</summary>
[DefaultExecutionOrder(100)]
public sealed class HipThrustTaikoController : MonoBehaviour
{
    [Serializable]
    public struct HipBeat
    {
        public int upFrame;
        public int holdEndFrame;
        public int downFrame;
    }

    [Header("Source animation and sampled chart (frames relative to the imported clip)")]
    [SerializeField] private AnimationClip sourceClip;
    [Tooltip("Optional recorded three-phase chart. Null retains the original taiko timing.")]
    [SerializeField] private HipThrustPhaseTimeline phaseTimeline;
    [Header("Optional new-rule scenes (unset preserves the original game)")]
    [SerializeField] private HipThrustRuleScoring ruleScoring;
    [SerializeField] private HipThrustPrescriptionDose hold30Dose;
    private HipThrustHold30Timeline hold30Timeline;
    [SerializeField] private HipBeat[] sourceBeats;
    [SerializeField] private int repeatStartFrame = 408;
    [SerializeField] private int repeatEndFrame = 1320;
    [SerializeField] private string routineStateName = "hipthrust_routine";
    [SerializeField] private string actionDisplayName = "臀桥";
    [SerializeField] private string restTransitionStateName;
    [SerializeField] private string hipThrustStartStateName = "hipthrust_start";
    [SerializeField] private RuntimeAnimatorController hipThrustAnimatorController;

    [Header("Existing training flow")]
    [SerializeField] private Animator coachAnimator;
    [SerializeField] private Animator coachPreviewAnimator;
    [SerializeField] private DeadBugGamingPoseScorer poseScorer;
    [SerializeField] private RhythmSongClock songClock;
    [SerializeField] private DeadBugStartMenuController startMenu;
    [SerializeField] private GameObject startMenuRoot;
    [SerializeField] private GameObject endMenuRoot;
    [SerializeField] private TMP_Text repetitionsText;
    [SerializeField] private TMP_Text scoreLabel;
    [SerializeField] private TMP_Text difficultyLabel;
    [SerializeField] private TMP_Text phaseLabel;
    [SerializeField] private TMP_Text resultLabel;

    [Header("Existing taiko artwork")]
    [SerializeField] private Canvas gamingCanvas;
    [SerializeField] private RectTransform canvasVisualRoot;
    [SerializeField] private Image backgroundImage;
    [SerializeField] private Image trackImage;
    [SerializeField] private Image judgementImage;
    [SerializeField] private Image sunlightImage;
    [SerializeField] private RectTransform noteSpawnAnchor;
    [SerializeField] private RectTransform noteClipArea;
    [SerializeField] private Sprite hipUpSprite;
    [SerializeField] private Sprite hipDownSprite;
    [SerializeField] private Sprite sliderSprite;
    [Min(0.1f)] [SerializeField] private float approachDuration = 2.2f;
    [Min(40f)] [SerializeField] private float uiNoteSize = 104f;
    [Min(0.02f)] [SerializeField] private float judgementWindow = 0.15f;
    [Range(0.01f, 0.5f)] [SerializeField] private float hipHeightTolerance = 0.16f;

    private sealed class Beat
    {
        public double upTime, holdEndTime, downTime;
        public RectTransform root;
        public Image up, hold, down;
        public bool spawned, upResolved, downResolved, holdResolved;
        public bool upHit, downHit;
        public float heldSeconds, sampledSeconds;
    }
    private readonly List<Beat> beats = new List<Beat>();
    private double elapsed;
    private int passes = 1;
    private bool running, paused;
    private int score, hitCount, missCount;
    private int consecutiveHits;
    private AvatarMotionSource motionSource;
    private float lastSampleTime;
    private HipThrustHitParticles hitParticles;
    private float nextHoldSpark;

    public bool IsRunning => running;
    public bool IsPaused => paused;
    public double Elapsed => elapsed;
    public float SourceTime => SourceTimeAt(elapsed);
    public double Duration => hold30Timeline != null ? hold30Timeline.Duration : sourceClip.length + (passes - 1) * LoopDuration;
    public int BeatCount => beats.Count;
    public int HitCount => hitCount;
    public int MissCount => missCount;
    public int Score => ruleScoring != null ? Mathf.FloorToInt(ruleScoring.LiveScore+.0001f) : score;
    public HipThrustRuleScoring RuleScoring => ruleScoring;
    public HipThrustPrescriptionDose PrescriptionDose => hold30Dose;
    public HipThrustHold30Timeline Hold30Timeline => hold30Timeline;
    public bool CanStart => hold30Dose == null || hold30Dose.Total > 0;
    public int CompletedDemonstrations { get; private set; }
    public float PresentationPhaseDuration(HipThrustPhaseTimeline.Sample sample)
    {
        if (sample.phase < 0 || sample.cycle < 0) return 1;
        if (hold30Timeline != null)
        {
            var rep = hold30Timeline.Repetitions[Mathf.Min(CompletedDemonstrations, hold30Timeline.Repetitions.Length-1)];
            return (float)(sample.phase==0 ? rep.top-rep.start : sample.phase==1 ? rep.lower-rep.top : rep.end-rep.lower);
        }
        var c = phaseTimeline.cycles[sample.cycle];
        return (sample.phase==0 ? c.holdStartFrame-c.liftStartFrame : sample.phase==1 ? c.lowerStartFrame-c.holdStartFrame : c.lowerEndFrame-c.lowerStartFrame)/FrameRate;
    }
    public int ConsecutiveHits => consecutiveHits;
    public HipThrustPhaseTimeline PhaseTimeline => phaseTimeline;
    public Animator CoachAnimator => coachAnimator;
    public string RoutineStateName => routineStateName;
    public string ActionDisplayName => actionDisplayName;
    public Animator PreviewAnimator => coachPreviewAnimator;
    public event Action Completed;
    public GameTrainingAssessment Assessment { get; set; }
    public HipThrustHold30Timeline.Repetition[] AssessmentRepetitions()
    {
        if(hold30Timeline!=null)return hold30Timeline.Repetitions;
        var result=new List<HipThrustHold30Timeline.Repetition>();
        if(phaseTimeline!=null)
        for(int pass=0;pass<passes;pass++)for(int i=0;i<phaseTimeline.cycles.Length;i++)
        {
            var c=phaseTimeline.cycles[i];double offset=pass*LoopDuration;
            result.Add(new HipThrustHold30Timeline.Repetition(i,c.holdStartFrame,
                c.liftStartFrame/(double)FrameRate+offset,c.holdStartFrame/(double)FrameRate+offset,
                c.lowerEndFrame/(double)FrameRate+offset,(c.lowerStartFrame-c.holdStartFrame)/(double)FrameRate));
        }
        return result.ToArray();
    }
    public event Action<bool> HitResolved;
    private float FrameRate => sourceClip.frameRate;
    private double LoopStart => (phaseTimeline != null ? phaseTimeline.repeatStartFrame : repeatStartFrame) / (double)FrameRate;
    private double LoopEnd => (phaseTimeline != null ? phaseTimeline.repeatEndFrame : repeatEndFrame) / (double)FrameRate;
    private double LoopDuration => LoopEnd - LoopStart;
    private Vector2 JudgePosition => judgementImage.rectTransform.anchoredPosition;
    private float PixelsPerSecond => Mathf.Abs(noteSpawnAnchor.anchoredPosition.x - JudgePosition.x) / approachDuration;

    private void Awake()
    {
        if (phaseTimeline != null)
        {
            phaseTimeline.Validate();
            sourceClip = phaseTimeline.sourceClip;
        }
        if (sourceClip == null || sourceBeats == null || sourceBeats.Length == 0 ||
            coachAnimator == null || coachPreviewAnimator == null || gamingCanvas == null)
        {
            Debug.LogError("HipThrust requires its source animation, beat map, coach and preview references.", this);
            enabled = false;
            return;
        }
        coachAnimator.runtimeAnimatorController = hipThrustAnimatorController;
        coachPreviewAnimator.runtimeAnimatorController = hipThrustAnimatorController;
        // Runtime-only configuration; never mutate shared sprite or animation assets in edit mode.
        foreach (Image image in canvasVisualRoot.GetComponentsInChildren<Image>(true)) image.raycastTarget = false;
        if (sunlightImage != null) sunlightImage.gameObject.SetActive(false);
        hitParticles = new GameObject("HipThrustHitParticles", typeof(RectTransform)).AddComponent<HipThrustHitParticles>();
        hitParticles.transform.SetParent(canvasVisualRoot, false);
        hitParticles.rectTransform.anchorMin = Vector2.zero;
        hitParticles.rectTransform.anchorMax = Vector2.one;
        hitParticles.rectTransform.offsetMin = hitParticles.rectTransform.offsetMax = Vector2.zero;
        hitParticles.raycastTarget = false;
        SetPhase("准备动作");
        UpdateScore();
    }

    private void Update()
    {
        if (!running)
        {
            // The original start menu and countdown explicitly start this state. Calibration's
            // frozen first pose cannot trigger a game, and no hidden menu is auto-clicked.
            if (gamingCanvas.gameObject.activeInHierarchy && !startMenuRoot.activeSelf &&
                coachAnimator.enabled && coachAnimator.speed > 0f &&
                coachAnimator.GetCurrentAnimatorStateInfo(0).IsName(hipThrustStartStateName)) StartMode();
            return;
        }
        if (startMenuRoot.activeSelf || !gamingCanvas.gameObject.activeInHierarchy)
        {
            StopMode();
            return;
        }
        if (!paused) elapsed = Math.Min(Duration, elapsed + Time.deltaTime);
        RenderTimeline();
    }

    private void LateUpdate()
    {
        if (!running || paused) return;
        if (ruleScoring != null) ruleScoring.Tick(Time.deltaTime);
        if (Assessment != null) Assessment.ObserveBridge(Time.deltaTime);
        UpdateJudgements();
        if (elapsed >= Duration) CompleteSession();
    }

    public void StartMode()
    {
        if (!CanStart) { StopAndShowMenu(); return; }
        ClearNotes();
        int cycleCount = phaseTimeline != null ? phaseTimeline.cycles.Length : sourceBeats.Length;
        int repetitions = cycleCount;
        if (repetitionsText != null && int.TryParse(repetitionsText.text, out int requested)) repetitions = requested;
        passes = Mathf.Clamp(Mathf.CeilToInt(repetitions / (float)cycleCount), 1, 3);
        hold30Timeline = hold30Dose != null ? new HipThrustHold30Timeline(phaseTimeline, hold30Dose.Total, hold30Dose.preserveRecordedTiming, hold30Dose.skipRecordedPlateauAfterHold, hold30Dose.preserveSourceGaps) : null;
        CompletedDemonstrations = 0;
        if (ruleScoring != null) ruleScoring.Begin(startMenu.SelectedDifficultyKey);
        elapsed = 0d;
        lastSampleTime = 0f;
        score = hitCount = missCount = consecutiveHits = 0;
        paused = false;
        running = true;
        if (endMenuRoot != null) endMenuRoot.SetActive(false);
        BuildChart();
        if (Assessment != null) Assessment.BeginBridge();
        if (difficultyLabel != null) difficultyLabel.text = GameWorkoutPlan.DifficultyName(startMenu.SelectedDifficultyKey);
        if (songClock != null) songClock.StartMusicImmediately();
        UpdateScore();
        RenderTimeline();
    }

    private void BuildChart()
    {
        if (hold30Timeline != null)
        {
            foreach (var rep in hold30Timeline.Repetitions)
                beats.Add(new Beat { upTime=rep.top, holdEndTime=rep.lower, downTime=rep.end });
            return;
        }
        if (phaseTimeline != null)
        {
            for (int pass = 0; pass < passes; pass++)
            foreach (var cycle in phaseTimeline.cycles)
                beats.Add(new Beat { upTime = cycle.holdStartFrame / (double)FrameRate + pass * LoopDuration,
                    holdEndTime = cycle.lowerStartFrame / (double)FrameRate + pass * LoopDuration,
                    downTime = cycle.lowerEndFrame / (double)FrameRate + pass * LoopDuration });
            return;
        }
        for (int pass = 0; pass < passes; pass++)
        foreach (HipBeat source in sourceBeats)
            beats.Add(new Beat {
                upTime = source.upFrame / (double)FrameRate + pass * LoopDuration,
                holdEndTime = source.holdEndFrame / (double)FrameRate + pass * LoopDuration,
                downTime = source.downFrame / (double)FrameRate + pass * LoopDuration
            });
    }

    public float SourceTimeAt(double time)
    {
        if (hold30Timeline != null)
        { hold30Timeline.Evaluate(time, out float frame, out _, out _); return frame / FrameRate; }
        double lastLoopEnd = LoopEnd + (passes - 1) * LoopDuration;
        if (time < LoopStart) return (float)Math.Max(0d, time);
        if (time < lastLoopEnd) return (float)(LoopStart + (time - LoopStart) % LoopDuration);
        return Mathf.Clamp((float)(time - (passes - 1) * LoopDuration), 0f, sourceClip.length);
    }

    private void RenderTimeline()
    {
        if (hold30Timeline != null) { hold30Timeline.Evaluate(elapsed, out _, out _, out int count); CompletedDemonstrations=count; }
        float normalized = SourceTimeAt(elapsed) / sourceClip.length;
        if(hold30Timeline!=null && !string.IsNullOrEmpty(restTransitionStateName) && hold30Timeline.TryGetWrapProgress(elapsed,out float wrap))
        {SampleCoach(coachAnimator,wrap,restTransitionStateName);SampleCoach(coachPreviewAnimator,wrap,restTransitionStateName);}
        else {SampleCoach(coachAnimator, normalized);SampleCoach(coachPreviewAnimator, normalized);}
        // The ring presentation reads this same clock after coach sampling. It owns its own UI.
        if (phaseTimeline != null) return;
        string phase = elapsed < LoopStart ? "准备动作" : "准备抬起";
        if (elapsed > beats[beats.Count - 1].downTime) phase = "完成动作";
        foreach (Beat beat in beats)
        {
            if (!beat.spawned && elapsed >= beat.upTime - approachDuration) CreateGroup(beat);
            if (beat.root != null)
            {
                // The red head latches to the judge while the blue endpoint keeps approaching.
                float redX = Mathf.Max(0f, (float)(beat.upTime - elapsed) * PixelsPerSecond);
                float blueX = Mathf.Max(0f, (float)(beat.downTime - elapsed) * PixelsPerSecond);
                Vector2 position = JudgePosition + Vector2.right * redX;
                beat.root.anchoredPosition = position - noteClipArea.anchoredPosition;
                beat.down.rectTransform.anchoredPosition = new Vector2(blueX - redX, 0f);
                bool visible = elapsed < beat.downTime;
                beat.up.gameObject.SetActive(visible);
                beat.down.gameObject.SetActive(visible);
                // Keep the solid ribbon outside the judgement ring, even as it contracts.
                float tailStart = Mathf.Max(redX, judgementImage.rectTransform.rect.width * 0.5f);
                float tailWidth = Mathf.Max(0f, blueX - tailStart);
                SetRect(beat.hold.rectTransform, new Vector2(tailWidth, 38f),
                    new Vector2(tailStart - redX + tailWidth * 0.5f, 0f));
                beat.hold.gameObject.SetActive(visible && tailWidth > 0f);
                if (elapsed > beat.downTime + judgementWindow + 0.2d)
                {
                    Destroy(beat.root.gameObject);
                    beat.root = null;
                }
            }
            if (elapsed >= beat.upTime && elapsed < beat.holdEndTime) phase = "保持臀部高度";
            else if (elapsed >= beat.holdEndTime && elapsed < beat.downTime) phase = "臀部放下";
            else if (elapsed >= beat.upTime - 1d && elapsed < beat.upTime) phase = "臀部抬起";
        }
        SetPhase(paused ? "已暂停" : phase);
    }

    private void SampleCoach(Animator animator, float normalized, string stateName=null)
    {
        animator.enabled = true;
        animator.cullingMode = AnimatorCullingMode.AlwaysAnimate;
        animator.speed = 0f;
        animator.Play(stateName??routineStateName, 0, normalized);
        animator.Update(0f);
    }

    private void CreateGroup(Beat beat)
    {
        beat.spawned = true;
        beat.root = new GameObject("HipThrust_RedHoldBlue", typeof(RectTransform)).GetComponent<RectTransform>();
        beat.root.SetParent(noteClipArea, false);
        float downX = (float)(beat.downTime - beat.upTime) * PixelsPerSecond;
        float holdWidth = downX;
        SetRect(beat.root, new Vector2(downX + uiNoteSize, uiNoteSize), Vector2.zero);
        // Connect both note centers; the timed pose prompts still follow the actual animation.
        beat.hold = MakeImage("BlueHold", sliderSprite, beat.root);
        beat.hold.preserveAspect = false;
        beat.hold.color = new Color(0.35f, 0.85f, 1f, 0.95f);
        SetRect(beat.hold.rectTransform, new Vector2(holdWidth, 38f), new Vector2(holdWidth / 2f, 0f));
        beat.up = MakeImage("Red_HipsUp", hipUpSprite, beat.root);
        beat.down = MakeImage("Blue_HipsDown", hipDownSprite, beat.root);
        SetRect(beat.up.rectTransform, Vector2.one * uiNoteSize, Vector2.zero);
        SetRect(beat.down.rectTransform, Vector2.one * uiNoteSize, new Vector2(downX, 0f));
        beat.up.transform.SetAsLastSibling();
    }

    private void UpdateJudgements()
    {
        bool poseMatches = PoseMatchesCoach();
        foreach (Beat beat in beats)
        {
            if (Math.Abs(elapsed - beat.upTime) <= judgementWindow) beat.upHit |= poseMatches;
            if (Math.Abs(elapsed - beat.downTime) <= judgementWindow) beat.downHit |= poseMatches;
            if (!beat.upResolved && elapsed >= beat.upTime + judgementWindow)
            {
                beat.upResolved = true;
                ResolveHit(beat.upHit, 1000, beat);
            }
            float overlap = Mathf.Max(0f, Mathf.Min((float)elapsed, (float)beat.holdEndTime) - Mathf.Max(lastSampleTime, (float)beat.upTime));
            if (overlap > 0f)
            {
                beat.sampledSeconds += overlap;
                if (poseMatches) beat.heldSeconds += overlap;
            }
            if (phaseTimeline == null && poseMatches && beat.upHit && elapsed >= beat.upTime && elapsed < beat.holdEndTime && Time.time >= nextHoldSpark)
            {
                TailEndpoints(beat, out Vector2 from, out Vector2 to);
                hitParticles.Trail(from, to);
                nextHoldSpark = Time.time + 0.06f;
            }
            if (!beat.holdResolved && elapsed >= beat.holdEndTime)
            {
                beat.holdResolved = true;
                if (ruleScoring == null && beat.sampledSeconds > 0f && beat.heldSeconds / beat.sampledSeconds >= 0.8f) score += 500;
            }
            if (!beat.downResolved && elapsed >= beat.downTime + judgementWindow)
            {
                beat.downResolved = true;
                ResolveHit(beat.downHit, 1000, beat);
            }
        }
        lastSampleTime = (float)elapsed;
        UpdateScore();
    }

    private bool PoseMatchesCoach()
    {
        if (ruleScoring != null && ruleScoring.sideBridge != null)
            return ruleScoring.MatchesSideBridgeCoach();
        if(ruleScoring!=null)return ruleScoring.MatchesHipCoach();
        if (poseScorer == null || !poseScorer.IsCalibrated || poseScorer.UserAnimator == null) return false;
        if (motionSource == null) motionSource = FindObjectOfType<AvatarMotionSource>();
        if (motionSource == null || !motionSource.HasFreshFrame) return false;
        Animator user = poseScorer.UserAnimator;
        // Hip-to-knee height normalized by femur length distinguishes raised and lowered hips,
        // even when the incoming avatar only supplies bone rotations and has a different stature.
        if (!TryHipHeight(user, out float userHeight) || !TryHipHeight(coachAnimator, out float coachHeight)) return false;
        return Mathf.Abs(userHeight - coachHeight) <= hipHeightTolerance;
    }

    private static bool TryHipHeight(Animator animator, out float height)
    {
        height = 0f;
        if (animator == null || !animator.isHuman) return false;
        Transform hips = animator.GetBoneTransform(HumanBodyBones.Hips);
        Transform left = animator.GetBoneTransform(HumanBodyBones.LeftLowerLeg);
        Transform right = animator.GetBoneTransform(HumanBodyBones.RightLowerLeg);
        Transform thigh = animator.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
        if (hips == null || left == null || right == null || thigh == null) return false;
        float length = Vector3.Distance(thigh.position, left.position);
        if (length < 0.0001f) return false;
        height = Vector3.Dot(hips.position - (left.position + right.position) / 2f, Vector3.up) / length;
        return true;
    }

    private void TailEndpoints(Beat beat, out Vector2 from, out Vector2 to)
    {
        float left = JudgePosition.x + (float)(beat.upTime - elapsed) * PixelsPerSecond;
        float right = JudgePosition.x + (float)(beat.downTime - elapsed) * PixelsPerSecond;
        // Keep particles on the visible lane rather than over the score panel.
        from = new Vector2(Mathf.Max(JudgePosition.x, left), JudgePosition.y);
        to = new Vector2(Mathf.Max(from.x, Mathf.Min(noteClipArea.anchoredPosition.x + noteClipArea.rect.xMax, right)), JudgePosition.y);
    }

    private void ResolveHit(bool hit, int points, Beat beat)
    {
        if (hit)
        {
            hitCount++; if (ruleScoring == null) score += points;
            consecutiveHits++;
        }
        else { missCount++; consecutiveHits = 0; }
        // Emit at the real endpoint judgement, before the recorded-artwork early return.
        // The hold bonus is not a HIT and deliberately does not emit feedback.
        HitResolved?.Invoke(hit);
        if (phaseTimeline != null) return;
        if (hit)
        {
            TailEndpoints(beat, out Vector2 from, out Vector2 to);
            hitParticles.Burst(JudgePosition, from, to);
            if (beat.hold != null) beat.hold.color = new Color(0.65f, 1f, 1f);
        }
        TMP_Text feedback = MakeText(hit ? "HIT" : "MISS", canvasVisualRoot, 42f);
        feedback.color = hit ? new Color(1f, 0.88f, 0.3f) : Color.white;
        SetRect(feedback.rectTransform, new Vector2(180, 65), JudgePosition + new Vector2(0, 92));
        Destroy(feedback.gameObject, 0.65f);
    }

    public void TogglePause()
    {
        if (!running) return;
        paused = !paused;
        if(paused&&ruleScoring!=null)ruleScoring.BreakContinuity();
        if (hitParticles != null) hitParticles.Paused = paused;
        // The animation and all chart positions freeze together. Resume never restarts the chart.
        SetPhase(paused ? "已暂停" : "继续训练");
        AudioSource music = songClock != null ? songClock.GetComponent<AudioSource>() : null;
        if (music != null) { if (paused) music.Pause(); else music.UnPause(); }
    }

    public void StopAndShowMenu()
    {
        StopMode();
        if (startMenu != null) startMenu.ShowMenu();
    }

    public void StopMode()
    {
        running = paused = false;
        coachAnimator.speed = coachPreviewAnimator.speed = 0f;
        ClearNotes();
        if (songClock != null) songClock.StopClock();
        if (sunlightImage != null) sunlightImage.gameObject.SetActive(false);
    }

    private void CompleteSession()
    {
        if (Assessment != null) Assessment.Finish();
        running = paused = false;
        if (songClock != null) songClock.StopClock();
        if (resultLabel != null) resultLabel.text = ruleScoring != null ? ruleScoring.Result(beats.Count) : $"训练完成  {beats.Count} 次臀桥\n得分 {score}   HIT {hitCount}   MISS {missCount}";
        ClearNotes();
        gamingCanvas.gameObject.SetActive(false);
        if (endMenuRoot != null) endMenuRoot.SetActive(true);
        Completed?.Invoke();
    }

    private void OnDisable()
    {
        if (coachAnimator != null && coachPreviewAnimator != null) StopMode();
    }

    private void ClearNotes()
    {
        if (hitParticles != null) hitParticles.Clear();
        nextHoldSpark = 0f;
        foreach (Beat beat in beats) if (beat.root != null) Destroy(beat.root.gameObject);
        beats.Clear();
    }
    private void SetPhase(string value) { if (phaseLabel != null) phaseLabel.text = value; }
    private void UpdateScore() { if (scoreLabel != null) scoreLabel.text = Score.ToString(); }
    private static Image MakeImage(string name, Sprite sprite, Transform parent)
    {
        Image image = new GameObject(name, typeof(RectTransform), typeof(Image)).GetComponent<Image>();
        image.transform.SetParent(parent, false);
        image.sprite = sprite;
        image.preserveAspect = true;
        image.raycastTarget = false;
        return image;
    }
    private static TMP_Text MakeText(string value, Transform parent, float size)
    {
        var text = new GameObject("HipThrust_" + value, typeof(RectTransform)).AddComponent<TextMeshProUGUI>();
        text.transform.SetParent(parent, false);
        text.text = value;
        text.fontSize = size;
        text.alignment = TextAlignmentOptions.Center;
        text.raycastTarget = false;
        return text;
    }
    private static void SetRect(RectTransform rect, Vector2 size, Vector2 position)
    {
        rect.anchorMin = rect.anchorMax = rect.pivot = Vector2.one * 0.5f;
        rect.sizeDelta = size;
        rect.anchoredPosition = position;
    }
}
