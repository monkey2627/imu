using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class DeadBugStartMenuController : MonoBehaviour
{
    public event Action TrainingStarted;
    public Func<bool> TrainingStartAllowed { get; set; }
    public string SelectedDifficultyKey => selectedDifficulty == Difficulty.Simple ? "simple" : selectedDifficulty == Difficulty.Hard ? "hard" : "standard";
    public bool IsTrainingMenuVisible => menuRoot != null && menuRoot.activeSelf;
    public const float PracticeControlTopY = 100f;
    public const float PracticeControlVerticalSpacing = 100f;

    private enum Difficulty
    {
        Simple,
        Standard,
        Hard
    }

    [Header("Menu")]
    [SerializeField] private GameObject menuRoot;
    [SerializeField] private GameObject endMenuRoot;
    [SerializeField] private Button startButton;
    [Tooltip("Only hide the start menu when Start Training is clicked. Use this in Practice so its existing threshold-selection flow remains in control.")]
    [SerializeField] private bool hideMenuOnlyOnStart;
    [Tooltip("Model animators whose renderers stay hidden while the start menu is visible.")]
    [SerializeField] private Animator[] modelsHiddenWhileMenuVisible;
    [SerializeField] private GameObject gamingCanvasRoot;
    [SerializeField] private Button simpleDifficultyButton;
    [SerializeField] private Button standardDifficultyButton;
    [SerializeField] private Button hardDifficultyButton;
    [SerializeField] private Sprite selectedDifficultySprite;
    [SerializeField] private Sprite unselectedDifficultySprite;
    [SerializeField] private TMP_Text timesText;
    [SerializeField] private int simpleTimes = 4;
    [SerializeField] private int standardTimes = 8;
    [SerializeField] private int hardTimes = 12;
    [Header("Modal Layering")]
    [Tooltip("World-space content to hide while the start or completion menu is open. This keeps the menu as the visual top layer even when models are closer to the camera.")]
    [SerializeField] private GameObject[] occludedWorldRoots;
    [Tooltip("Hide every other scene Canvas and Renderer while a Practice confirmation/completion menu is visible.")]
    [SerializeField] private bool hideAllPracticeVisualsWhileMenuVisible = true;
    [Tooltip("Canvases that must remain visible for menu interaction, such as the gaze cursor.")]
    [SerializeField] private Canvas[] canvasesVisibleWhileMenuOpen;

    [Header("Training")]
    [SerializeField] private CoachActionController coachActionController;
    [SerializeField] private SegmentedCoachController practiceCoachController;
    [SerializeField] private DeadBugGamingPoseScorer poseScorer;
    [SerializeField] private DeadBugGamingSideFeedback sideFeedback;
    [SerializeField] private GameObject actionMenu;
    [Tooltip("The four-button Practice control panel. It is resolved by name in legacy scenes when not assigned.")]
    [SerializeField] private GameObject practiceActionMenu;
    [SerializeField] private DeadBugCountdownController countdownController;
    [SerializeField] private RhythmSongClock rhythmSongClock;

    [Header("Difficulty Actions")]
    [SerializeField] private string simpleActionStateName = "basic1";
    [SerializeField] private string standardActionStateName = "Deadbug1";
    [SerializeField] private string hardActionStateName = "basic3";
    [SerializeField] private Difficulty defaultDifficulty = Difficulty.Standard;

    private Difficulty selectedDifficulty;
    private ScorePopup pendingPracticeScorePopup;
    private VoicePromptManager completionVoicePromptManager;
    private bool practiceCompletionGateActive;
    private bool practiceCompletionPresentationReady;
    private bool practiceCompletionUserUpright;
    private readonly Dictionary<GameObject, bool> occludedRootActiveStates =
        new Dictionary<GameObject, bool>();
    private readonly Dictionary<Renderer, bool> menuHiddenRendererStates = new Dictionary<Renderer, bool>();
    private readonly Dictionary<Renderer, bool> modalHiddenRendererStates =
        new Dictionary<Renderer, bool>();
    private readonly Dictionary<Canvas, bool> modalHiddenCanvasStates =
        new Dictionary<Canvas, bool>();
    private TMP_Text calibrationStatusText;
    private string originalStartButtonText;
    private bool originalStartButtonTextActive;
    private bool calibrationStatusCached;

    private bool IsPracticeCoachMode =>
        hideMenuOnlyOnStart || practiceCoachController != null;

    private void Awake()
    {
        if (menuRoot == null)
        {
            menuRoot = gameObject;
        }

        ResolvePracticeCoachController();
        ConfigurePracticeActionMenu();
        ResolveTimesText();
        ResolveGamingCanvasRoot();
        ResolveEndMenuRoot();
        SpineFlowTrainingSession.PracticeStartConfirmationReset +=
            HandlePracticeStartConfirmationReset;
        InitializePresentation(
            SpineFlowTrainingSession.ShouldShowPracticeStartConfirmation);

        if (endMenuRoot != null)
        {
            endMenuRoot.SetActive(false);
        }

        if (actionMenu == null)
        {
            actionMenu = gamingCanvasRoot;
        }

        if (actionMenu != null)
        {
            actionMenu.SetActive(false);
        }

        if (countdownController == null)
        {
            countdownController = FindObjectOfType<DeadBugCountdownController>(true);
        }

        if (rhythmSongClock == null)
        {
            rhythmSongClock = FindObjectOfType<RhythmSongClock>(true);
        }

        if (coachActionController == null)
        {
            coachActionController = FindObjectOfType<CoachActionController>();
        }

        if (coachActionController != null)
        {
            coachActionController.ActionCompleted += ShowEndMenu;
        }

        ResolvePracticeCoachController();

        if (practiceCoachController != null)
        {
            practiceCoachController.OnAllComplete += ShowEndMenu;
        }

        if (sideFeedback == null)
        {
            sideFeedback = FindObjectOfType<DeadBugGamingSideFeedback>();
        }

        if (startButton == null)
        {
            startButton = FindStartButton();
        }

        if (startButton != null)
        {
            startButton.onClick.AddListener(StartTraining);
        }
        else
        {
            Debug.LogWarning("DeadBugStartMenuController needs a Start Button reference.", this);
        }

        if (poseScorer == null)
        {
            poseScorer = FindObjectOfType<DeadBugGamingPoseScorer>(true);
        }

        if (poseScorer != null)
        {
            poseScorer.CalibrationStateChanged += HandleCalibrationStateChanged;
            poseScorer.CalibrationProgressChanged += HandleCalibrationProgressChanged;
            CacheCalibrationStatusText();
            ApplyCalibrationGate();
        }

        ResolveDifficultyButtons();
        RegisterDifficultyButton(simpleDifficultyButton, Difficulty.Simple);
        RegisterDifficultyButton(standardDifficultyButton, Difficulty.Standard);
        RegisterDifficultyButton(hardDifficultyButton, Difficulty.Hard);
        SelectDifficulty(defaultDifficulty);
    }

    private void ResolveGamingCanvasRoot()
    {
        if (gamingCanvasRoot == null || gamingCanvasRoot.name != "GamingCanvas")
        {
            Transform[] sceneTransforms = FindObjectsOfType<Transform>(true);
            foreach (Transform sceneTransform in sceneTransforms)
            {
                if (sceneTransform.name == "GamingCanvas")
                {
                    gamingCanvasRoot = sceneTransform.gameObject;
                    break;
                }
            }
        }
    }

    private void ResolveEndMenuRoot()
    {
        if (endMenuRoot != null)
        {
            return;
        }

        Transform[] sceneTransforms = FindObjectsOfType<Transform>(true);
        foreach (Transform sceneTransform in sceneTransforms)
        {
            if (sceneTransform.name == "EndMenuCanvas")
            {
                endMenuRoot = sceneTransform.gameObject;
                break;
            }
        }
    }

    private void ResolvePracticeActionMenu()
    {
        if (practiceActionMenu != null)
        {
            return;
        }

        Transform[] sceneTransforms = FindObjectsOfType<Transform>(true);
        foreach (Transform sceneTransform in sceneTransforms)
        {
            if (sceneTransform.gameObject.scene == gameObject.scene &&
                sceneTransform.name == "ActionMenu")
            {
                practiceActionMenu = sceneTransform.gameObject;
                break;
            }
        }
    }

    private void ConfigurePracticeActionMenu()
    {
        if (!IsPracticeCoachMode)
        {
            return;
        }

        ResolvePracticeActionMenu();
        if (practiceActionMenu == null)
        {
            return;
        }

        foreach (Transform child in practiceActionMenu.transform)
        {
            if (child.name == "Pause" || child.name == "Back")
            {
                child.gameObject.SetActive(false);
                continue;
            }

            int columnIndex;
            switch (child.name)
            {
                case "Start":
                    columnIndex = 0;
                    break;
                case "Game":
                    columnIndex = 1;
                    break;
                case "PlayBack":
                    columnIndex = 2;
                    break;
                case "Home":
                    columnIndex = 3;
                    break;
                default:
                    continue;
            }

            child.gameObject.SetActive(true);
            if (child is RectTransform rectTransform)
            {
                rectTransform.anchoredPosition = new Vector2(
                    0f,
                    PracticeControlTopY -
                    columnIndex * PracticeControlVerticalSpacing);
            }
        }
    }

    private void OnDestroy()
    {
        CancelPendingPracticeCompletion();
        RestoreOccludedWorldRoots();
        RestoreModelsHiddenByMenu();
        RestoreSceneVisualsHiddenByMenu();
        SpineFlowTrainingSession.PracticeStartConfirmationReset -=
            HandlePracticeStartConfirmationReset;

        if (coachActionController != null)
        {
            coachActionController.ActionCompleted -= ShowEndMenu;
        }

        if (practiceCoachController != null)
        {
            practiceCoachController.OnAllComplete -= ShowEndMenu;
        }

        if (startButton != null)
        {
            startButton.onClick.RemoveListener(StartTraining);
        }

        if (poseScorer != null)
        {
            poseScorer.CalibrationStateChanged -= HandleCalibrationStateChanged;
            poseScorer.CalibrationProgressChanged -= HandleCalibrationProgressChanged;
        }

        UnregisterDifficultyButton(simpleDifficultyButton, Difficulty.Simple);
        UnregisterDifficultyButton(standardDifficultyButton, Difficulty.Standard);
        UnregisterDifficultyButton(hardDifficultyButton, Difficulty.Hard);
    }

    private void Start()
    {
        if (!IsPracticeCoachMode &&
            GamingGuideNavigation.ConsumeCalibrationRestart(gameObject.scene.name))
        {
            StartCoroutine(RestartCalibrationAfterGuide());
        }
    }

    private void Update()
    {
        // The final score popup and a spoken completion cue can finish in
        // either order. Recheck after the voice becomes idle so neither is
        // cut off by the controls-only presentation.
        if (practiceCompletionGateActive &&
            practiceCompletionPresentationReady &&
            practiceCompletionUserUpright)
        {
            TryShowGatedPracticeCompletionControls();
        }
    }

    private IEnumerator RestartCalibrationAfterGuide()
    {
        // Let the returned scene finish enabling its avatar and VMC receiver
        // before collecting the first standing calibration sample.
        yield return null;

        if (poseScorer == null)
        {
            poseScorer = FindObjectOfType<DeadBugGamingPoseScorer>(true);
        }

        if (poseScorer == null)
        {
            Debug.LogError(
                "Cannot restart standing calibration after the gaming guide: " +
                "DeadBugGamingPoseScorer was not found.",
                this);
            yield break;
        }

        poseScorer.BeginCalibration();
        ApplyCalibrationGate();
    }

    private void InitializePresentation(bool shouldShowPracticeStartConfirmation)
    {
        if (!IsPracticeCoachMode || shouldShowPracticeStartConfirmation)
        {
            ShowMenu();
            return;
        }

        // Returning from Replay, Game, or another scene keeps the one-time
        // mobile confirmation acknowledged. Re-enter the same controls-only
        // state that follows that confirmation instead of revealing the
        // avatars underneath the four Practice controls.
        ShowPracticeSelectionControlsOnly();
    }

    public void StartTraining()
    {
        if (TrainingStartAllowed != null && !TrainingStartAllowed()) return;
        if (!IsPracticeCoachMode && poseScorer != null && !poseScorer.CanStartTraining)
        {
            ApplyCalibrationGate();
            if (!poseScorer.IsCalibrating)
            {
                poseScorer.BeginCalibration();
            }
            return;
        }

        string selectedActionStateName = GetSelectedActionStateName();
        TrainingStarted?.Invoke();
        if (IsPracticeCoachMode)
        {
            SpineFlowTrainingSession.AcknowledgePracticeStartConfirmation();
        }

        if (!IsPracticeCoachMode)
        {
            SpineFlowTrainingSession.BeginGameTraining(
                selectedDifficulty.ToString().ToLowerInvariant());
        }

        if (startButton != null)
        {
            startButton.interactable = false;
        }

        if (rhythmSongClock != null)
        {
            rhythmSongClock.StartMusicImmediately();
        }

        if (endMenuRoot != null)
        {
            endMenuRoot.SetActive(false);
        }

        if (menuRoot != null)
        {
            menuRoot.SetActive(false);
        }

        if (hideMenuOnlyOnStart)
        {
            // Confirmation only opens the control phase. Keep the avatars and
            // all non-control presentation hidden until Play actually starts
            // the segmented coach sequence.
            ShowPracticeSelectionControlsOnly();
            return;
        }

        RestoreModelsHiddenByMenu();
        RestoreSceneVisualsHiddenByMenu();
        RestoreOccludedWorldRoots();
        DisableRetiredDifficultySelectionCanvas();

        if (gamingCanvasRoot != null)
        {
            gamingCanvasRoot.SetActive(true);
        }

        if (sideFeedback != null)
        {
            sideFeedback.HideUserModelForTraining();
        }

        if (actionMenu != null)
        {
            actionMenu.SetActive(true);
        }

        if (countdownController != null)
        {
            countdownController.BeginCountdown(selectedActionStateName);
        }
        else if (coachActionController != null)
        {
            coachActionController.PlayAction(selectedActionStateName);
        }
    }

    public void ShowMenu()
    {
        RefreshTrainingVolumeText();

        if (menuRoot != null)
        {
            menuRoot.SetActive(true);
        }

        HideWorldContentForMenu();
        if (IsPracticeCoachMode)
        {
            SetPracticeActionMenuCanvasVisible(false);
        }

        if (gamingCanvasRoot != null)
        {
            gamingCanvasRoot.SetActive(false);
        }

        if (sideFeedback != null)
        {
            sideFeedback.RestoreUserModelVisibility();
        }

        if (startButton != null)
        {
            startButton.interactable =
                IsPracticeCoachMode || poseScorer == null || poseScorer.CanStartTraining;
        }
        ApplyCalibrationGate();
    }

    private void HideWorldContentForMenu()
    {
        if (IsPracticeCoachMode)
        {
            HideOccludedWorldRoots();
            HideSceneVisualsExceptMenus();
            return;
        }

        HideModelsBehindMenu();
    }

    private void HandlePracticeStartConfirmationReset()
    {
        if (IsPracticeCoachMode)
        {
            ShowMenu();
        }
    }

    private void HideSceneVisualsExceptMenus()
    {
        if (!hideAllPracticeVisualsWhileMenuVisible)
        {
            return;
        }

        foreach (Renderer sceneRenderer in FindObjectsOfType<Renderer>(true))
        {
            if (sceneRenderer == null ||
                sceneRenderer.gameObject.scene != gameObject.scene ||
                IsPartOfMenu(sceneRenderer.transform) ||
                modalHiddenRendererStates.ContainsKey(sceneRenderer))
            {
                continue;
            }

            modalHiddenRendererStates.Add(sceneRenderer, sceneRenderer.enabled);
            sceneRenderer.enabled = false;
        }

        foreach (Canvas sceneCanvas in FindObjectsOfType<Canvas>(true))
        {
            if (sceneCanvas == null ||
                sceneCanvas.gameObject.scene != gameObject.scene ||
                IsPartOfMenu(sceneCanvas.transform) ||
                IsCanvasAllowedWhileMenuOpen(sceneCanvas) ||
                modalHiddenCanvasStates.ContainsKey(sceneCanvas))
            {
                continue;
            }

            modalHiddenCanvasStates.Add(sceneCanvas, sceneCanvas.enabled);
            sceneCanvas.enabled = false;
        }
    }

    private bool IsPartOfMenu(Transform candidate)
    {
        return IsTransformUnderRoot(candidate, menuRoot) ||
            IsTransformUnderRoot(candidate, endMenuRoot);
    }

    private static bool IsTransformUnderRoot(Transform candidate, GameObject root)
    {
        if (candidate == null || root == null)
        {
            return false;
        }

        Transform rootTransform = root.transform;
        return candidate == rootTransform || candidate.IsChildOf(rootTransform);
    }

    private bool IsCanvasAllowedWhileMenuOpen(Canvas candidate)
    {
        if (canvasesVisibleWhileMenuOpen != null)
        {
            foreach (Canvas allowedCanvas in canvasesVisibleWhileMenuOpen)
            {
                if (candidate == allowedCanvas)
                {
                    return true;
                }
            }
        }

        // Keep legacy scenes usable before their explicit gaze-canvas reference
        // is serialized.
        return candidate.name == "GazedotCanvas";
    }

    private void RestoreSceneVisualsHiddenByMenu()
    {
        foreach (KeyValuePair<Renderer, bool> rendererState in modalHiddenRendererStates)
        {
            if (rendererState.Key != null)
            {
                rendererState.Key.enabled = rendererState.Value;
            }
        }

        modalHiddenRendererStates.Clear();

        foreach (KeyValuePair<Canvas, bool> canvasState in modalHiddenCanvasStates)
        {
            if (canvasState.Key != null)
            {
                canvasState.Key.enabled = canvasState.Value;
            }
        }

        modalHiddenCanvasStates.Clear();
    }

    private void HideModelsBehindMenu()
    {
        if (modelsHiddenWhileMenuVisible == null)
        {
            return;
        }

        Renderer[] sceneRenderers = FindObjectsOfType<Renderer>(true);
        foreach (Animator modelAnimator in modelsHiddenWhileMenuVisible)
        {
            if (modelAnimator == null)
            {
                continue;
            }

            foreach (Renderer modelRenderer in sceneRenderers)
            {
                if (!RendererBelongsToAnimator(modelRenderer, modelAnimator))
                {
                    continue;
                }

                if (!menuHiddenRendererStates.ContainsKey(modelRenderer))
                {
                    menuHiddenRendererStates.Add(modelRenderer, modelRenderer.enabled);
                }

                modelRenderer.enabled = false;
            }
        }
    }

    private static bool RendererBelongsToAnimator(Renderer modelRenderer, Animator modelAnimator)
    {
        if (modelRenderer == null || modelAnimator == null)
        {
            return false;
        }

        Transform animatorRoot = modelAnimator.transform;
        Transform rendererTransform = modelRenderer.transform;
        if (rendererTransform == animatorRoot || rendererTransform.IsChildOf(animatorRoot))
        {
            return true;
        }

        Animator ownerAnimator = modelRenderer.GetComponentInParent<Animator>();
        if (ownerAnimator == modelAnimator)
        {
            return true;
        }

        SkinnedMeshRenderer skinnedRenderer = modelRenderer as SkinnedMeshRenderer;
        if (skinnedRenderer == null)
        {
            return false;
        }

        Transform rootBone = skinnedRenderer.rootBone;
        if (rootBone != null &&
            (rootBone == animatorRoot || rootBone.IsChildOf(animatorRoot)))
        {
            return true;
        }

        Transform[] bones = skinnedRenderer.bones;
        foreach (Transform bone in bones)
        {
            if (bone != null && (bone == animatorRoot || bone.IsChildOf(animatorRoot)))
            {
                return true;
            }
        }

        return false;
    }

    private void RestoreModelsHiddenByMenu()
    {
        foreach (KeyValuePair<Renderer, bool> rendererState in menuHiddenRendererStates)
        {
            if (rendererState.Key != null)
            {
                rendererState.Key.enabled = rendererState.Value;
            }
        }

        menuHiddenRendererStates.Clear();
    }

    private void ShowPracticeSelectionControlsOnly()
    {
        if (menuRoot != null)
        {
            menuRoot.SetActive(false);
        }

        if (endMenuRoot != null)
        {
            endMenuRoot.SetActive(false);
        }

        // This method can run either immediately after the confirmation menu
        // or on a fresh Practice scene load. Establish the complete modal
        // state here so both paths hide the same non-control presentation.
        HideWorldContentForMenu();
        SetPracticeActionMenuCanvasVisible(true);
    }

    public void BeginPracticePlaybackPresentation()
    {
        ResetPracticeCompletionGate();
        RestoreModelsHiddenByMenu();
        RestoreSceneVisualsHiddenByMenu();
        RestoreOccludedWorldRoots();
        DisableRetiredDifficultySelectionCanvas();

        // Keep the GameObject and SegmentedCoachController alive. Only the
        // world-space Canvas is hidden so it cannot overlap the two avatars or
        // receive gaze interaction during formal playback.
        SetPracticeActionMenuCanvasVisible(false);
    }

    private void DisableRetiredDifficultySelectionCanvas()
    {
        foreach (Canvas sceneCanvas in FindObjectsOfType<Canvas>(true))
        {
            if (sceneCanvas == null ||
                sceneCanvas.gameObject.scene != gameObject.scene ||
                sceneCanvas.name != "DifficultySelectionCanvas")
            {
                continue;
            }

            sceneCanvas.enabled = false;
            sceneCanvas.gameObject.SetActive(false);
        }
    }

    private void ShowEndMenu()
    {
        // FinishScoring raises coach completion synchronously, immediately after
        // starting the final score popup. Keep the avatars and score visible
        // until that popup has completed; otherwise the completion controls hide
        // the whole practice presentation in the same frame.
        if (IsPracticeCoachMode)
        {
            BeginPracticeCompletionPostureGate();
            ScorePopup scorePopup = FindObjectOfType<ScorePopup>(true);
            if (scorePopup != null && scorePopup.IsShowing)
            {
                CancelPendingPracticeCompletion();
                pendingPracticeScorePopup = scorePopup;
                pendingPracticeScorePopup.DisplayFinished +=
                    HandleFinalPracticeScoreFinished;
                return;
            }

            MarkPracticeCompletionPresentationReady();
            return;
        }

        if (endMenuRoot == null)
        {
            ResolveEndMenuRoot();
        }

        if (endMenuRoot == null)
        {
            Debug.LogWarning("DeadBugStartMenuController could not find EndMenuCanvas.", this);
            return;
        }

        if (gamingCanvasRoot != null)
        {
            gamingCanvasRoot.SetActive(false);
        }

        HideWorldContentForMenu();

        Transform startMenuTransform = menuRoot != null ? menuRoot.transform : transform;
        Transform endMenuTransform = endMenuRoot.transform;
        endMenuTransform.position = startMenuTransform.position;
        endMenuTransform.rotation = startMenuTransform.rotation;
        endMenuTransform.localScale = startMenuTransform.localScale;

        Quaternion childRotation = IsPracticeCoachMode
            ? Quaternion.identity
            : Quaternion.Euler(0f, 180f, 0f);
        for (int childIndex = 0; childIndex < endMenuTransform.childCount; childIndex++)
        {
            endMenuTransform.GetChild(childIndex).localRotation = childRotation;
        }
        Canvas startCanvas = startMenuTransform.GetComponent<Canvas>();
        Canvas endCanvas = endMenuTransform.GetComponent<Canvas>();
        if (startCanvas != null && endCanvas != null)
        {
            endCanvas.renderMode = startCanvas.renderMode;
            endCanvas.worldCamera = startCanvas.worldCamera;
            endCanvas.planeDistance = startCanvas.planeDistance;
            endCanvas.sortingLayerID = startCanvas.sortingLayerID;
            endCanvas.sortingOrder = startCanvas.sortingOrder;
        }

        endMenuRoot.SetActive(true);

    }

    private void HandleFinalPracticeScoreFinished()
    {
        CancelPendingPracticeCompletion();
        MarkPracticeCompletionPresentationReady();
    }

    private void CancelPendingPracticeCompletion()
    {
        if (pendingPracticeScorePopup == null)
        {
            return;
        }

        pendingPracticeScorePopup.DisplayFinished -=
            HandleFinalPracticeScoreFinished;
        pendingPracticeScorePopup = null;
    }

    public bool ShowPracticeCompletionControls()
    {
        ResolvePracticeActionMenu();
        if (practiceActionMenu == null)
        {
            Debug.LogWarning(
                "DeadBugStartMenuController could not find the Practice ActionMenu.",
                this);
            return false;
        }

        // Completion returns to the same controls-only presentation used after
        // the start confirmation. This also clears stale score/feedback UI.
        HideWorldContentForMenu();

        if (endMenuRoot != null)
        {
            endMenuRoot.SetActive(false);
        }

        practiceActionMenu.SetActive(true);
        SetPracticeActionMenuCanvasVisible(true);
        ResetPracticeCompletionGate();

        return true;
    }

    public void BeginPracticeCompletionPostureGate()
    {
        if (!IsPracticeCoachMode || practiceCompletionGateActive)
        {
            return;
        }

        practiceCompletionGateActive = true;
        practiceCompletionPresentationReady = false;
        practiceCompletionUserUpright = false;
        SetPracticeActionMenuCanvasVisible(false);
    }

    public bool NotifyPracticeCompletionUserUpright()
    {
        if (!practiceCompletionGateActive)
        {
            return false;
        }

        practiceCompletionUserUpright = true;
        return TryShowGatedPracticeCompletionControls();
    }

    private void MarkPracticeCompletionPresentationReady()
    {
        BeginPracticeCompletionPostureGate();
        practiceCompletionPresentationReady = true;
        TryShowGatedPracticeCompletionControls();
    }

    private bool TryShowGatedPracticeCompletionControls()
    {
        return practiceCompletionGateActive &&
            practiceCompletionPresentationReady &&
            practiceCompletionUserUpright &&
            IsPracticeCompletionVoiceFinished() &&
            ShowPracticeCompletionControls();
    }

    private bool IsPracticeCompletionVoiceFinished()
    {
        ResolveCompletionVoicePromptManager();
        return completionVoicePromptManager == null ||
            !completionVoicePromptManager.IsBusy;
    }

    private void ResolveCompletionVoicePromptManager()
    {
        if (completionVoicePromptManager != null &&
            completionVoicePromptManager.gameObject.scene == gameObject.scene)
        {
            return;
        }

        completionVoicePromptManager = null;
        foreach (VoicePromptManager candidate in
            FindObjectsOfType<VoicePromptManager>(true))
        {
            if (candidate.gameObject.scene == gameObject.scene)
            {
                completionVoicePromptManager = candidate;
                return;
            }
        }
    }

    private void ResetPracticeCompletionGate()
    {
        practiceCompletionGateActive = false;
        practiceCompletionPresentationReady = false;
        practiceCompletionUserUpright = false;
    }

    public bool IsPracticeCompletionPostureGateActive =>
        practiceCompletionGateActive;

    private void SetPracticeActionMenuCanvasVisible(bool visible)
    {
        ResolvePracticeActionMenu();
        if (practiceActionMenu == null)
        {
            return;
        }

        practiceActionMenu.SetActive(true);
        Canvas[] actionCanvases =
            practiceActionMenu.GetComponentsInChildren<Canvas>(true);
        foreach (Canvas actionCanvas in actionCanvases)
        {
            actionCanvas.enabled = visible;
        }
    }

    public void SelectSimpleDifficulty()
    {
        SelectDifficulty(Difficulty.Simple);
    }

    public void SelectStandardDifficulty()
    {
        SelectDifficulty(Difficulty.Standard);
    }

    public void SelectHardDifficulty()
    {
        SelectDifficulty(Difficulty.Hard);
    }

    private void SelectDifficulty(Difficulty difficulty)
    {
        selectedDifficulty = difficulty;

        if (coachActionController != null)
        {
            coachActionController.SetActionStateName(GetSelectedActionStateName());
            coachActionController.SetTrainingDifficulty(GetCoachTrainingDifficulty(difficulty));
        }

        SetButtonSelected(simpleDifficultyButton, difficulty == Difficulty.Simple);
        SetButtonSelected(standardDifficultyButton, difficulty == Difficulty.Standard);
        SetButtonSelected(hardDifficultyButton, difficulty == Difficulty.Hard);
        UpdateTimesText();
    }

    private string GetSelectedActionStateName()
    {
        if (!IsPracticeCoachMode) return standardActionStateName;
        switch (selectedDifficulty)
        {
            case Difficulty.Simple:
                return simpleActionStateName;
            case Difficulty.Hard:
                return hardActionStateName;
            case Difficulty.Standard:
            default:
                return standardActionStateName;
        }
    }

    private CoachActionController.TrainingDifficulty GetCoachTrainingDifficulty(Difficulty difficulty)
    {
        switch (difficulty)
        {
            case Difficulty.Simple:
                return CoachActionController.TrainingDifficulty.Simple;
            case Difficulty.Hard:
                return CoachActionController.TrainingDifficulty.Hard;
            case Difficulty.Standard:
            default:
                return CoachActionController.TrainingDifficulty.Standard;
        }
    }

    private void ResolveDifficultyButtons()
    {
        Button[] buttons = GetComponentsInChildren<Button>(true);
        foreach (Button button in buttons)
        {
            string buttonName = button.name.Trim();
            if (simpleDifficultyButton == null && buttonName.Contains("Simple"))
            {
                simpleDifficultyButton = button;
            }
            else if (standardDifficultyButton == null &&
                (buttonName.Contains("Standard") || buttonName.Contains("Idle") || buttonName.Contains("Normal")))
            {
                standardDifficultyButton = button;
            }
            else if (hardDifficultyButton == null && buttonName.Contains("Hard"))
            {
                hardDifficultyButton = button;
            }
        }
    }

    private void RegisterDifficultyButton(Button button, Difficulty difficulty)
    {
        if (button == null)
        {
            return;
        }

        switch (difficulty)
        {
            case Difficulty.Simple:
                button.onClick.AddListener(SelectSimpleDifficulty);
                break;
            case Difficulty.Standard:
                button.onClick.AddListener(SelectStandardDifficulty);
                break;
            case Difficulty.Hard:
                button.onClick.AddListener(SelectHardDifficulty);
                break;
        }
    }

    private void UnregisterDifficultyButton(Button button, Difficulty difficulty)
    {
        if (button == null)
        {
            return;
        }

        switch (difficulty)
        {
            case Difficulty.Simple:
                button.onClick.RemoveListener(SelectSimpleDifficulty);
                break;
            case Difficulty.Standard:
                button.onClick.RemoveListener(SelectStandardDifficulty);
                break;
            case Difficulty.Hard:
                button.onClick.RemoveListener(SelectHardDifficulty);
                break;
        }
    }

    private void SetButtonSelected(Button button, bool selected)
    {
        if (button == null)
        {
            return;
        }

        if (selectedDifficultySprite != null)
        {
            Image buttonImage = button.targetGraphic as Image;
            if (buttonImage == null)
            {
                buttonImage = button.GetComponent<Image>();
            }

            if (buttonImage != null)
            {
                Sprite targetSprite = selected ? selectedDifficultySprite : unselectedDifficultySprite;
                if (targetSprite != null)
                {
                    buttonImage.sprite = targetSprite;
                }
            }

            // The selected state is represented by its blue sprite, so keep all
            // buttons interactable and avoid Unity's disabled-color tint.
            button.interactable = true;
        }
        else
        {
            button.interactable = !selected;
        }
    }

    private void UpdateTimesText()
    {
        RefreshTrainingVolumeText();
    }

    private void ResolvePracticeCoachController()
    {
        if (practiceCoachController == null)
        {
            practiceCoachController = FindObjectOfType<SegmentedCoachController>(true);
        }
    }

    private void ResolveTimesText()
    {
        if (timesText != null)
        {
            return;
        }

        Transform searchRoot = menuRoot != null ? menuRoot.transform : transform;
        foreach (Transform candidate in
                 searchRoot.GetComponentsInChildren<Transform>(true))
        {
            if (candidate.name != "Times")
            {
                continue;
            }

            timesText = candidate.GetComponentInChildren<TMP_Text>(true);
            break;
        }
    }

    private void RefreshTrainingVolumeText()
    {
        ResolvePracticeCoachController();
        ResolveTimesText();
        if (timesText == null)
        {
            return;
        }

        int fallbackSets = IsPracticeCoachMode && practiceCoachController != null
            ? practiceCoachController.TotalSets
            : 1;
        int fallbackRepetitions =
            IsPracticeCoachMode && practiceCoachController != null
                ? practiceCoachController.RepetitionsPerSet
                : GetSelectedFallbackRepetitions();
        SpineFlowTrainingVolume volume =
            SpineFlowTrainingSession.GetPlannedTrainingVolume(
                IsPracticeCoachMode ? "coach" : "game",
                fallbackSets,
                fallbackRepetitions);

        // A phone-authored session explicitly presents its planned set count.
        // Standalone practice preserves the legacy "times" meaning and shows
        // the configured repetitions per set (the scene fallback is 2 x 4).
        timesText.text = SpineFlowTrainingSession.HasMobileSession
            ? volume.sets.ToString()
            : volume.repetitionsPerSet.ToString();
    }

    private int GetSelectedFallbackRepetitions()
    {
        if (!IsPracticeCoachMode) return Mathf.Max(1, standardTimes);
        switch (selectedDifficulty)
        {
            case Difficulty.Simple:
                return Mathf.Max(1, simpleTimes);
            case Difficulty.Hard:
                return Mathf.Max(1, hardTimes);
            case Difficulty.Standard:
            default:
                return Mathf.Max(1, standardTimes);
        }
    }

    public TMP_Text TimesText => timesText;

    private void HideOccludedWorldRoots()
    {
        ResolvePracticeVisualRoot();
        if (occludedWorldRoots == null)
        {
            return;
        }

        foreach (GameObject worldRoot in occludedWorldRoots)
        {
            if (worldRoot == null || occludedRootActiveStates.ContainsKey(worldRoot))
            {
                continue;
            }

            occludedRootActiveStates.Add(worldRoot, worldRoot.activeSelf);
            worldRoot.SetActive(false);
        }
    }

    private void ResolvePracticeVisualRoot()
    {
        if (!IsPracticeCoachMode)
        {
            return;
        }

        Transform[] sceneTransforms = FindObjectsOfType<Transform>(true);
        foreach (Transform sceneTransform in sceneTransforms)
        {
            if (sceneTransform.gameObject.scene != gameObject.scene ||
                sceneTransform.name != "FollowViewRoot")
            {
                continue;
            }

            GameObject visualRoot = sceneTransform.gameObject;
            if (occludedWorldRoots != null)
            {
                foreach (GameObject configuredRoot in occludedWorldRoots)
                {
                    if (configuredRoot == visualRoot)
                    {
                        return;
                    }
                }
            }

            int configuredCount =
                occludedWorldRoots != null ? occludedWorldRoots.Length : 0;
            GameObject[] expandedRoots = new GameObject[configuredCount + 1];
            if (configuredCount > 0)
            {
                System.Array.Copy(
                    occludedWorldRoots,
                    expandedRoots,
                    configuredCount);
            }

            expandedRoots[configuredCount] = visualRoot;
            occludedWorldRoots = expandedRoots;
            return;
        }
    }

    private void RestoreOccludedWorldRoots()
    {
        foreach (KeyValuePair<GameObject, bool> rootState in occludedRootActiveStates)
        {
            if (rootState.Key != null)
            {
                rootState.Key.SetActive(rootState.Value);
            }
        }

        occludedRootActiveStates.Clear();
    }

    private void HandleCalibrationStateChanged(bool isCalibrated)
    {
        if (IsPracticeCoachMode)
        {
            return;
        }

        ApplyCalibrationGate();
        if (isCalibrated)
        {
            RestoreStartButtonText();
        }
    }

    private void HandleCalibrationProgressChanged(float progress, string message)
    {
        if (IsPracticeCoachMode)
        {
            return;
        }

        if (poseScorer != null && poseScorer.CanStartTraining)
        {
            RestoreStartButtonText();
            ApplyCalibrationGate();
            return;
        }

        if (startButton != null)
        {
            startButton.interactable = false;
        }

        string status = string.IsNullOrEmpty(message)
            ? (poseScorer != null && poseScorer.GetComponent<SupineGameCalibration>() != null
                ? poseScorer.CalibrationInstruction : "请自然站立")
            : message;
        ShowCalibrationStatus(status);
    }

    private void ApplyCalibrationGate()
    {
        if (IsPracticeCoachMode || startButton == null || poseScorer == null)
        {
            return;
        }

        bool ready = poseScorer.CanStartTraining;
        startButton.interactable = ready;
        if (ready)
        {
            RestoreStartButtonText();
        }
        else if (!poseScorer.IsCalibrating)
        {
            ShowCalibrationStatus(poseScorer.CalibrationInstruction);
        }
    }

    private void CacheCalibrationStatusText()
    {
        if (calibrationStatusCached || startButton == null)
        {
            return;
        }

        calibrationStatusText = startButton.GetComponentInChildren<TMP_Text>(true);
        if (calibrationStatusText == null)
        {
            return;
        }

        originalStartButtonText = calibrationStatusText.text;
        originalStartButtonTextActive = calibrationStatusText.gameObject.activeSelf;
        calibrationStatusText.enableAutoSizing = true;
        calibrationStatusText.fontSizeMin = 12f;
        calibrationStatusText.fontSizeMax = 24f;
        calibrationStatusText.raycastTarget = false;
        calibrationStatusCached = true;
    }

    private void ShowCalibrationStatus(string message)
    {
        CacheCalibrationStatusText();
        if (calibrationStatusText == null)
        {
            return;
        }

        calibrationStatusText.text = message;
        calibrationStatusText.gameObject.SetActive(true);
    }

    private void RestoreStartButtonText()
    {
        if (!calibrationStatusCached || calibrationStatusText == null)
        {
            return;
        }

        calibrationStatusText.text = originalStartButtonText;
        calibrationStatusText.gameObject.SetActive(originalStartButtonTextActive);
    }

    private Button FindStartButton()
    {
        Button[] buttons = GetComponentsInChildren<Button>(true);
        foreach (Button button in buttons)
        {
            string buttonName = button.name.Trim();
            if (buttonName.Contains("Start") ||
                buttonName.Contains("\u5f00\u59cb") ||
                buttonName.Contains("Practice") ||
                buttonName.Contains("Training"))
            {
                return button;
            }
        }

        return null;
    }
}
