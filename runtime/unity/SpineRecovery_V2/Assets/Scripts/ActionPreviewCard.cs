using UnityEngine;

public class ActionPreviewCard : MonoBehaviour
{
    [SerializeField] private string actionName;
    [SerializeField] private string targetSceneName;
    [SerializeField] private RectTransform hitArea;
    [SerializeField] private Animator previewAnimator;
    [SerializeField] private GameObject selectionHighlight;
    [SerializeField] private string previewStateName;
    [SerializeField] private bool rewindWhenDeselected = true;

    [Header("Gaming Coach Sync")]
    [SerializeField] private bool mirrorCoachSequenceInGaming = true;
    [SerializeField] private CoachActionController coachActionController;
    [SerializeField] private string segmentedStartStateName = "deadbug_start";

    private bool isSelected;
    private string mirroredStateName;

    public string ActionName => actionName;
    public string TargetSceneName => targetSceneName;
    public RectTransform HitArea => hitArea != null ? hitArea : transform as RectTransform;
    public bool IsSelected => isSelected;

    private void Awake()
    {
        if (hitArea == null)
        {
            hitArea = transform as RectTransform;
        }

        ResolvePreviewAnimator();
        ResolveCoachController();
        SubscribeToCoach();
        SetSelected(false);
    }

    private void OnEnable()
    {
        ResolveCoachController();
        SubscribeToCoach();
    }

    private void OnDisable()
    {
        UnsubscribeFromCoach();
    }

    private void OnDestroy()
    {
        UnsubscribeFromCoach();
    }

    public void PlayPreview()
    {
        ResolvePreviewAnimator();

        if (previewAnimator == null)
        {
            Debug.LogWarning($"ActionPreviewCard '{name}' could not find its preview Animator.", this);
            return;
        }

        previewAnimator.enabled = true;
        previewAnimator.cullingMode = AnimatorCullingMode.AlwaysAnimate;

        if (ShouldMirrorCoach())
        {
            SubscribeToCoach();
            PrepareForCoachSequence();
            return;
        }

        previewAnimator.speed = 1f;
        PlayPreviewFromStart();
    }

    public void PauseAndRewindPreview()
    {
        ResolvePreviewAnimator();

        if (previewAnimator == null)
        {
            return;
        }

        previewAnimator.speed = 0f;
        if (ShouldMirrorCoach() && HasPreviewState(segmentedStartStateName))
        {
            previewAnimator.Play(segmentedStartStateName, 0, 0f);
        }
        else
        {
            PlayPreviewFromStart();
        }
        previewAnimator.Update(0f);
    }

    public void SetSelected(bool selected)
    {
        isSelected = selected;

        if (selectionHighlight != null)
        {
            selectionHighlight.SetActive(selected);
        }

        if (previewAnimator == null)
        {
            return;
        }

        previewAnimator.speed = selected ? 1f : 0f;

        if (!selected && rewindWhenDeselected)
        {
            PlayPreviewFromStart();
            previewAnimator.Update(0f);
        }
        else if (selected && !string.IsNullOrEmpty(previewStateName))
        {
            PlayPreviewFromStart();
        }
    }

    public void RestartSelectedPreview()
    {
        if (!isSelected || previewAnimator == null)
        {
            return;
        }

        previewAnimator.speed = 1f;
        PlayPreviewFromStart();
    }

    private void PlayPreviewFromStart()
    {
        if (previewAnimator == null || string.IsNullOrEmpty(previewStateName))
        {
            return;
        }

        previewAnimator.Play(previewStateName, 0, 0f);
    }

    private void ResolvePreviewAnimator()
    {
        if (previewAnimator != null)
        {
            return;
        }

        previewAnimator = GetComponentInChildren<Animator>(true);
        if (previewAnimator != null || string.IsNullOrWhiteSpace(actionName))
        {
            return;
        }

        string expectedPreviewRootName = actionName.Replace(" ", string.Empty) + "Preview";
        Transform[] sceneTransforms = FindObjectsOfType<Transform>(true);
        foreach (Transform sceneTransform in sceneTransforms)
        {
            string candidateName = sceneTransform.name.Replace(" ", string.Empty);
            if (!string.Equals(candidateName, expectedPreviewRootName, System.StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            previewAnimator = sceneTransform.GetComponentInChildren<Animator>(true);
            if (previewAnimator != null)
            {
                break;
            }
        }
    }

    private void ResolveCoachController()
    {
        if (!mirrorCoachSequenceInGaming || coachActionController != null)
        {
            return;
        }

        coachActionController = FindObjectOfType<CoachActionController>(true);
    }

    private bool ShouldMirrorCoach()
    {
        return mirrorCoachSequenceInGaming && coachActionController != null;
    }

    private void SubscribeToCoach()
    {
        if (!ShouldMirrorCoach())
        {
            return;
        }

        coachActionController.ScoringSessionStarted -= HandleCoachSessionStarted;
        coachActionController.SegmentPlaybackStarted -= HandleCoachSegmentStarted;
        coachActionController.ScoringCheckpointStarted -= HandleCoachCheckpointStarted;
        coachActionController.ScoringSessionCompleted -= HandleCoachSessionCompleted;

        coachActionController.ScoringSessionStarted += HandleCoachSessionStarted;
        coachActionController.SegmentPlaybackStarted += HandleCoachSegmentStarted;
        coachActionController.ScoringCheckpointStarted += HandleCoachCheckpointStarted;
        coachActionController.ScoringSessionCompleted += HandleCoachSessionCompleted;
    }

    private void UnsubscribeFromCoach()
    {
        if (coachActionController == null)
        {
            return;
        }

        coachActionController.ScoringSessionStarted -= HandleCoachSessionStarted;
        coachActionController.SegmentPlaybackStarted -= HandleCoachSegmentStarted;
        coachActionController.ScoringCheckpointStarted -= HandleCoachCheckpointStarted;
        coachActionController.ScoringSessionCompleted -= HandleCoachSessionCompleted;
    }

    private void HandleCoachSessionStarted()
    {
        PrepareForCoachSequence();
    }

    private void HandleCoachSegmentStarted(CoachActionController.SegmentPlaybackInfo info)
    {
        ResolvePreviewAnimator();
        if (previewAnimator == null || string.IsNullOrEmpty(info.stateName))
        {
            return;
        }

        if (!HasPreviewState(info.stateName))
        {
            Debug.LogWarning(
                $"ActionPreviewCard '{name}' cannot mirror missing state '{info.stateName}'.",
                this);
            return;
        }

        mirroredStateName = info.stateName;
        previewAnimator.enabled = true;
        previewAnimator.cullingMode = AnimatorCullingMode.AlwaysAnimate;
        previewAnimator.speed = Mathf.Max(0.05f, info.playbackSpeed);
        previewAnimator.Play(mirroredStateName, 0, 0f);
        previewAnimator.Update(0f);
    }

    private void HandleCoachCheckpointStarted(CoachActionController.ScoringCheckpointInfo info)
    {
        if (previewAnimator == null || !string.Equals(mirroredStateName, info.stateName))
        {
            return;
        }

        // Snap to the exact same segment-end pose as the coach, then freeze for
        // the difficulty-specific hold. The next segment event resumes playback.
        previewAnimator.Play(info.stateName, 0, 1f);
        previewAnimator.Update(0f);
        previewAnimator.speed = 0f;
    }

    private void HandleCoachSessionCompleted()
    {
        if (previewAnimator != null)
        {
            previewAnimator.speed = 0f;
        }
    }

    private void PrepareForCoachSequence()
    {
        ResolvePreviewAnimator();
        if (previewAnimator == null)
        {
            return;
        }

        previewAnimator.enabled = true;
        previewAnimator.cullingMode = AnimatorCullingMode.AlwaysAnimate;
        previewAnimator.Rebind();
        previewAnimator.Update(0f);
        previewAnimator.speed = 0f;

        if (HasPreviewState(segmentedStartStateName))
        {
            mirroredStateName = segmentedStartStateName;
            previewAnimator.Play(segmentedStartStateName, 0, 0f);
            previewAnimator.Update(0f);
        }
    }

    private bool HasPreviewState(string stateName)
    {
        return previewAnimator != null &&
            !string.IsNullOrEmpty(stateName) &&
            previewAnimator.HasState(0, Animator.StringToHash(stateName));
    }
}
