using System.Collections;
using TMPro;
using UnityEngine;

public sealed class GamingGuideController : MonoBehaviour
{
    [Header("Guide UI")]
    [SerializeField] private GameObject startCanvas;
    [SerializeField] private GameObject endCanvas;
    [SerializeField] private GameObject gamingCanvas;
    [SerializeField] private GameObject countdownRoot;
    [SerializeField] private TMP_Text countdownText;
    [Min(0)] [SerializeField] private int countdownSeconds = 3;

    [Header("Guide Actors")]
    [SerializeField] private CoachActionController coachController;
    [SerializeField] private Animator coachAnimator;
    [SerializeField] private Transform guideModelRoot;
    [SerializeField] private Animator guideModelAnimator;

    [Header("Rhythm Mode")]
    [SerializeField] private RhythmModeController rhythmMode;
    [SerializeField] private RhythmSongClock songClock;

    [Header("Voice Guidance")]
    [SerializeField] private AudioSource guideVoiceSource;
    [SerializeField] private AudioClip introVoice;
    [SerializeField] private AudioClip ringLayoutVoice;
    [SerializeField] private AudioClip hitVoice;
    [SerializeField] private AudioClip stillMissVoice;
    [SerializeField] private AudioClip directionMissVoice;
    [SerializeField] private AudioClip completeVoice;
    [Range(0f, 1f)] [SerializeField] private float voiceVolume = 1f;

    [Header("View Placement")]
    [Min(0.5f)] [SerializeField] private float cameraDistance = 2.2f;
    [SerializeField] private float horizontalOffset = -0.9f;
    [SerializeField] private float verticalOffset = -0.65f;
    [Min(0.05f)] [SerializeField] private float modelScale = 0.8f;
    [SerializeField] private Vector3 modelEulerAngles = new Vector3(0f, 180f, 0f);

    [Header("Demonstration")]
    [Min(0f)] [SerializeField] private float modelDelaySeconds = 0.5f;
    [Min(0f)] [SerializeField] private float judgementHoldSeconds = 1.2f;
    [SerializeField] private string firstHitPairPrefix = "deadbug_1";
    [SerializeField] private string secondHitPairPrefix = "deadbug_2";
    [SerializeField] private string stillMissPairPrefix = "deadbug_3";
    [SerializeField] private string oppositeMissPairPrefix = "deadbug_4";
    [SerializeField] private string oppositeModelPairPrefix = "deadbug_3";
    [Min(1)] [SerializeField] private int expectedMissNoteCount = 4;
    [Min(0.1f)] [SerializeField] private float completionTimeoutSeconds = 1.5f;

    private bool missDemonstrationStarted;
    private bool guideCompleted;
    private int missNoteCount;

    private void Awake()
    {
        SetActive(startCanvas, false);
        SetActive(endCanvas, false);
        SetActive(gamingCanvas, true);
        SetActive(countdownRoot, false);

        if (guideVoiceSource == null)
        {
            guideVoiceSource = gameObject.AddComponent<AudioSource>();
        }
        guideVoiceSource.playOnAwake = false;
        guideVoiceSource.loop = false;
        guideVoiceSource.spatialBlend = 0f;
        guideVoiceSource.volume = voiceVolume;

        if (coachAnimator != null && guideModelAnimator != null)
        {
            guideModelAnimator.runtimeAnimatorController = coachAnimator.runtimeAnimatorController;
            guideModelAnimator.applyRootMotion = false;
            guideModelAnimator.cullingMode = AnimatorCullingMode.AlwaysAnimate;
            guideModelAnimator.enabled = true;
            guideModelAnimator.Rebind();
            guideModelAnimator.Update(0f);
            guideModelAnimator.Play("deadbug_start", 0, 0f);
            guideModelAnimator.Update(0f);
            guideModelAnimator.speed = 0f;
        }

        if (coachController != null)
        {
            coachController.SetTrainingDifficulty(CoachActionController.TrainingDifficulty.Simple);
            coachController.SetCurrentDifficultyHoldSeconds(judgementHoldSeconds);
        }
    }

    private void OnEnable()
    {
        if (coachController != null)
        {
            coachController.SegmentPlaybackStarted += HandleCoachSegmentStarted;
            coachController.ScoringCheckpointEnded += HandleScoringCheckpointEnded;
        }

        if (rhythmMode != null)
        {
            rhythmMode.NoteResolved += HandleNoteResolved;
        }
    }

    private void Start()
    {
        StartCoroutine(RunGuide());
    }

    private void OnDisable()
    {
        if (coachController != null)
        {
            coachController.SegmentPlaybackStarted -= HandleCoachSegmentStarted;
            coachController.ScoringCheckpointEnded -= HandleScoringCheckpointEnded;
        }

        if (rhythmMode != null)
        {
            rhythmMode.NoteResolved -= HandleNoteResolved;
        }
    }

    private IEnumerator RunGuide()
    {
        yield return null;
        PlaceGuideModelInView();

        if (songClock != null)
        {
            songClock.StartMusicImmediately();
        }
        PlayGuideVoice(introVoice);

        SetActive(countdownRoot, true);
        for (int second = Mathf.Max(0, countdownSeconds); second > 0; second--)
        {
            if (countdownText != null)
            {
                countdownText.text = second.ToString();
            }
            yield return new WaitForSeconds(1f);
        }
        SetActive(countdownRoot, false);

        rhythmMode?.ShowTutorialVisuals();
        yield return null;
        if (ringLayoutVoice != null)
        {
            PlayGuideVoice(ringLayoutVoice);
            yield return new WaitForSecondsRealtime(ringLayoutVoice.length);
        }

        if (coachController != null)
        {
            coachController.PlayAction("animation_pose");
        }
    }

    private void PlaceGuideModelInView()
    {
        if (guideModelRoot == null)
        {
            return;
        }

        Camera targetCamera = Camera.main;
        if (targetCamera == null)
        {
            targetCamera = FindObjectOfType<Camera>();
        }
        if (targetCamera == null)
        {
            return;
        }

        guideModelRoot.SetParent(targetCamera.transform, false);
        guideModelRoot.localPosition = new Vector3(horizontalOffset, verticalOffset, cameraDistance);
        guideModelRoot.localRotation = Quaternion.Euler(modelEulerAngles);
        guideModelRoot.localScale = Vector3.one * modelScale;
        guideModelRoot.gameObject.SetActive(true);
    }

    private void HandleCoachSegmentStarted(CoachActionController.SegmentPlaybackInfo info)
    {
        if (guideCompleted || guideModelAnimator == null)
        {
            return;
        }

        string guideState = GetGuideState(info.stateName);
        bool isHitPair = !string.IsNullOrEmpty(info.stateName) &&
            (info.stateName.StartsWith(firstHitPairPrefix + "_", System.StringComparison.Ordinal) ||
             info.stateName.StartsWith(secondHitPairPrefix + "_", System.StringComparison.Ordinal));
        bool isMissPair = !string.IsNullOrEmpty(info.stateName) &&
            (info.stateName.StartsWith(stillMissPairPrefix + "_", System.StringComparison.Ordinal) ||
             info.stateName.StartsWith(oppositeMissPairPrefix + "_", System.StringComparison.Ordinal));

        if (isHitPair)
        {
            rhythmMode?.SetTutorialJudgementOverride(RhythmJudgement.Perfect);
        }
        else if (isMissPair)
        {
            missDemonstrationStarted = true;
            rhythmMode?.SetTutorialJudgementOverride(RhythmJudgement.Miss);
        }

        if (info.stateName == firstHitPairPrefix + "_up")
        {
            PlayGuideVoice(hitVoice);
        }
        else if (info.stateName == stillMissPairPrefix + "_up")
        {
            PlayGuideVoice(stillMissVoice);
        }
        else if (info.stateName == oppositeMissPairPrefix + "_up")
        {
            PlayGuideVoice(directionMissVoice);
        }
        if (!string.IsNullOrEmpty(guideState))
        {
            StartCoroutine(PlayGuideStateAfterDelay(guideState, info.playbackSpeed));
        }
    }

    private string GetGuideState(string coachState)
    {
        if (string.IsNullOrEmpty(coachState))
        {
            return null;
        }

        if (coachState == "deadbug_start")
        {
            return coachState;
        }

        if (coachState.StartsWith(firstHitPairPrefix + "_", System.StringComparison.Ordinal) ||
            coachState.StartsWith(secondHitPairPrefix + "_", System.StringComparison.Ordinal))
        {
            return coachState;
        }

        if (coachState.StartsWith(stillMissPairPrefix + "_", System.StringComparison.Ordinal))
        {
            return null;
        }

        if (coachState.StartsWith(oppositeMissPairPrefix + "_", System.StringComparison.Ordinal))
        {
            return oppositeModelPairPrefix + coachState.Substring(oppositeMissPairPrefix.Length);
        }

        return null;
    }

    private IEnumerator PlayGuideStateAfterDelay(string stateName, float playbackSpeed)
    {
        yield return new WaitForSeconds(Mathf.Max(0f, modelDelaySeconds));
        if (guideCompleted || guideModelAnimator == null)
        {
            yield break;
        }

        guideModelAnimator.speed = Mathf.Max(0.05f, playbackSpeed);
        guideModelAnimator.Play(stateName, 0, 0f);
        guideModelAnimator.Update(0f);
    }

    private void HandleScoringCheckpointEnded(CoachActionController.ScoringCheckpointInfo info)
    {
        if (guideCompleted || info.pairIndex != 3 || info.isUp)
        {
            return;
        }

        coachController.StopAtCurrentPose();
        StartCoroutine(CompleteAfterMissOrTimeout());
    }

    private void HandleNoteResolved(RhythmNoteResult result)
    {
        Debug.Log(
            $"[GamingGuide] {result.Target} resolved as {result.Judgement} " +
            $"(score={result.PoseScore:0.0}).",
            this);

        if (missDemonstrationStarted && result.Judgement == RhythmJudgement.Miss)
        {
            missNoteCount++;
        }
    }

    private IEnumerator CompleteAfterMissOrTimeout()
    {
        float elapsed = 0f;
        while (missNoteCount < expectedMissNoteCount && elapsed < completionTimeoutSeconds)
        {
            elapsed += Time.deltaTime;
            yield return null;
        }

        yield return new WaitForSeconds(0.2f);
        CompleteGuide();
    }

    private void CompleteGuide()
    {
        if (guideCompleted)
        {
            return;
        }

        guideCompleted = true;
        Debug.Log("[GamingGuide] Demonstration complete.", this);
        PlayGuideVoice(completeVoice);
        if (rhythmMode != null)
        {
            rhythmMode.StopRhythmMode();
        }
        if (coachController != null)
        {
            coachController.ReturnIdle();
        }
        if (guideModelAnimator != null)
        {
            guideModelAnimator.speed = 0f;
        }

        if (GamingGuideNavigation.HasReturnScene)
        {
            StartCoroutine(ReturnToOriginAfterVoice());
        }
    }

    private IEnumerator ReturnToOriginAfterVoice()
    {
        float delay = completeVoice != null ? completeVoice.length : 0f;
        if (delay > 0f)
        {
            yield return new WaitForSecondsRealtime(delay);
        }

        GamingGuideNavigation.ReturnToOrigin();
    }

    private static void SetActive(GameObject target, bool active)
    {
        if (target != null && target.activeSelf != active)
        {
            target.SetActive(active);
        }
    }

    private void PlayGuideVoice(AudioClip clip)
    {
        if (guideVoiceSource == null || clip == null)
        {
            return;
        }

        guideVoiceSource.Stop();
        guideVoiceSource.clip = clip;
        guideVoiceSource.volume = voiceVolume;
        guideVoiceSource.Play();
    }
}
