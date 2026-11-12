using System.Text;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

// Installed only in the three new visual games. The panel survives menu visibility changes.
public sealed class GameWorkoutScene : MonoBehaviour
{
    public string actionId;
    [Tooltip("Keep workout transitions within the three recorded-coach scenes.")]
    public bool useRecordedCoachAnimations;
    public string hipThrustSceneOverride;
    public bool hideWorkoutToggle;
    public DeadBugStartMenuController menu;
    public DeadBugGamingPoseScorer scorer;
    public CoachActionController coach;
    public HipThrustTaikoController taiko;
    public RhythmModeController rhythm;
    public DeadBugCountdownController countdown;
    public Canvas gameCanvas;
    public TMP_FontAsset font;
    public DeadBugStartMenuController Menu => menu;
    public DeadBugGamingPoseScorer Scorer => scorer;
    public bool IsPlaying { get; private set; }
    private GameWorkoutSession session;
    private GameObject panel, uiRoot;
    private TMP_Text summary, details, status;
    private Button start, stop;
    private ScrollRect scroll;

    private void Awake()
    {
        if (GameTestMenu.EnabledForBuild && System.Array.IndexOf(GameTestMenu.Scenes, gameObject.scene.name) >= 0)
            gameObject.AddComponent<GameTestMenu>().Initialize(this);
        var assessment=gameObject.AddComponent<GameTrainingAssessment>();
        assessment.Initialize(this);
        session = GameWorkoutSession.Instance;
        menu.TrainingStartAllowed = () => session.AllowsStart(this) && (taiko == null || taiko.CanStart);
        menu.TrainingStarted += OnStarted;
        if (taiko != null) taiko.Completed += OnCompleted;
        else coach.ActionCompleted += OnCompleted;
    }

    private void Start()
    {
        BuildUI();
        session.Changed += Refresh;
        session.Attach(this);
    }

    private void OnDestroy()
    {
        if (menu != null) { menu.TrainingStarted -= OnStarted; menu.TrainingStartAllowed = null; }
        if (taiko != null) taiko.Completed -= OnCompleted;
        else if (coach != null) coach.ActionCompleted -= OnCompleted;
        if (session != null) { session.Changed -= Refresh; session.Detach(this); }
        if (uiRoot != null) Destroy(uiRoot);
    }

    private void OnStarted() { IsPlaying = true; session.OnSetStarted(this); Refresh(); }
    private void OnCompleted() { IsPlaying = false; session.OnSetCompleted(this); Refresh(); }
    private void Update()
    {
        if (IsPlaying && menu.IsTrainingMenuVisible) { IsPlaying = false; Refresh(); }
    }

    public void PrepareMenu(string difficulty)
    {
        menu.ShowMenu();
        if (difficulty == "simple") menu.SelectSimpleDifficulty();
        else if (difficulty == "hard") menu.SelectHardDifficulty();
        else menu.SelectStandardDifficulty();
    }

    public bool PrepareDose(GameWorkoutAction action, out string error)
    {
        error = null;
        if (!GameWorkoutPlan.RequiresExactReps(action.action)) return true;
        if (taiko == null || taiko.PrescriptionDose == null) { error = "动作的处方次数入口缺失。"; return false; }
        // The session executor already iterates sets. Each loaded run must receive only this set's reps.
        return taiko.PrescriptionDose.SetDose(1, action.reps, out error);
    }

    public void StopTraining()
    {
        IsPlaying = false;
        if (countdown != null) countdown.CancelCountdown();
        coach.CancelTraining();
        scorer.CancelScoringSession();
        if (taiko != null) taiko.StopMode();
        if (rhythm != null) rhythm.StopRhythmMode();
        menu.ShowMenu();
        Refresh();
    }

    public void TogglePanel() { panel.SetActive(!panel.activeSelf); Refresh(); }

    private void BuildUI()
    {
        uiRoot = new GameObject("Workout UI", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
        Canvas canvas = uiRoot.GetComponent<Canvas>();
        canvas.renderMode = gameCanvas.renderMode;
        canvas.worldCamera = gameCanvas.worldCamera;
        canvas.sortingOrder = 1100;
        canvas.planeDistance = gameCanvas.planeDistance;
        var scaler = uiRoot.GetComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920, 1080);
        scaler.matchWidthOrHeight = 0.5f;
        if (canvas.renderMode == RenderMode.WorldSpace)
        {
            uiRoot.transform.SetParent(gameCanvas.transform.parent, false);
            uiRoot.transform.SetPositionAndRotation(gameCanvas.transform.position, gameCanvas.transform.rotation);
            uiRoot.transform.localScale = gameCanvas.transform.localScale;
            ((RectTransform)uiRoot.transform).sizeDelta = ((RectTransform)gameCanvas.transform).rect.size;
        }
        var toggle = ButtonAt("动作串", uiRoot.transform, new Vector2(24, -24), new Vector2(210, 100), TogglePanel);
        toggle.gameObject.SetActive(!hideWorkoutToggle);
        summary = toggle.GetComponentInChildren<TMP_Text>();
        summary.fontSize = 25;
        panel = Rect("Workout Panel", uiRoot.transform, new Vector2(24, -135), new Vector2(440, 650)).gameObject;
        panel.AddComponent<Image>().color = new Color(0.025f, 0.04f, 0.1f, 1f);
        Text("动作串", panel.transform, new Vector2(20, -15), new Vector2(350, 40), 30);
        ButtonAt("×", panel.transform, new Vector2(375, -10), new Vector2(50, 45), TogglePanel);
        status = Text("", panel.transform, new Vector2(20, -65), new Vector2(400, 65), 23);
        var viewport = Rect("列表", panel.transform, new Vector2(20, -140), new Vector2(390, 410));
        viewport.gameObject.AddComponent<Image>().color = new Color(0, 0, 0, 0.01f);
        viewport.gameObject.AddComponent<RectMask2D>();
        scroll = viewport.gameObject.AddComponent<ScrollRect>();
        scroll.horizontal = false;
        scroll.viewport = viewport;
        details = Text("", viewport, Vector2.zero, new Vector2(385, 410), 24);
        details.alignment = TextAlignmentOptions.TopLeft;
        details.raycastTarget = true;
        scroll.content = details.rectTransform;
        var bar = Rect("滚动条", panel.transform, new Vector2(414, -140), new Vector2(8, 410));
        bar.gameObject.AddComponent<Image>().color = new Color(1, 1, 1, 0.1f);
        var handle = Rect("Handle", bar, Vector2.zero, new Vector2(8, 100));
        var handleImage = handle.gameObject.AddComponent<Image>();
        handleImage.color = new Color(0.5f, 0.8f, 0.9f);
        var scrollbar = bar.gameObject.AddComponent<Scrollbar>();
        scrollbar.handleRect = handle;
        scrollbar.targetGraphic = handleImage;
        scrollbar.direction = Scrollbar.Direction.BottomToTop;
        scroll.verticalScrollbar = scrollbar;
        scroll.verticalScrollbarVisibility = ScrollRect.ScrollbarVisibility.AutoHide;
        start = ButtonAt("开始动作串", panel.transform, new Vector2(20, -575), new Vector2(190, 55), () =>
        {
            if (!session.StartPlan(out string error)) status.text = error;
            else panel.SetActive(false);
        });
        stop = ButtonAt("停止动作串", panel.transform, new Vector2(230, -575), new Vector2(190, 55), session.Cancel);
        panel.SetActive(false);
    }

    private void Refresh()
    {
        if (summary == null) return;
        var plan = session.Plan;
        string current = GameWorkoutPlan.ActionName(actionId);
        if (plan != null && !session.IsComplete) current = GameWorkoutPlan.ActionName(plan.actions[session.ActionIndex].action);
        summary.text = $"动作串  >\n{(session.IsComplete ? "全部完成" : current)}";
        status.text = session.Status;
        start.interactable = plan != null && !session.IsRunning && !IsPlaying;
        stop.interactable = session.IsRunning;
        if (plan == null) details.text = $"当前动作：{current}\n\n待完成：未设置\n\n已完成：无\n\n等待传入动作、难度及组数参数";
        else
        {
            var text = new StringBuilder();
            text.AppendLine("<color=#7FE4FF>当前动作</color>");
            if (session.IsComplete) text.AppendLine("全部完成");
            else text.AppendLine(Describe(plan.actions[session.ActionIndex]) + $"\n已完成 {session.CompletedSets} 组");
            text.AppendLine("\n<color=#FFE18A>待完成</color>");
            if (session.ActionIndex + 1 >= plan.actions.Length) text.AppendLine("无后续动作");
            for (int i = session.ActionIndex + 1; i < plan.actions.Length; i++) text.AppendLine($"{i + 1}. {Describe(plan.actions[i])}");
            text.AppendLine("\n<color=#94EDAE>已完成</color>");
            if (session.ActionIndex == 0) text.AppendLine("无");
            for (int i = 0; i < session.ActionIndex; i++) text.AppendLine($"已完成 {Describe(plan.actions[i])}");
            details.text = text.ToString();
        }
        details.rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, Mathf.Max(410, details.GetPreferredValues(details.text, 385, 0).y + 20));
    }

    private static string Describe(GameWorkoutAction item) => $"{GameWorkoutPlan.ActionName(item.action)} · {GameWorkoutPlan.DifficultyName(item.difficulty)} · {item.sets} 组" + (GameWorkoutPlan.RequiresExactReps(item.action) ? $" × {item.reps} 次" : "");

    private static RectTransform Rect(string name, Transform parent, Vector2 position, Vector2 size)
    {
        var rect = new GameObject(name, typeof(RectTransform)).GetComponent<RectTransform>();
        rect.SetParent(parent, false);
        rect.anchorMin = rect.anchorMax = rect.pivot = new Vector2(0, 1);
        rect.anchoredPosition = position;
        rect.sizeDelta = size;
        return rect;
    }

    private TMP_Text Text(string text, Transform parent, Vector2 position, Vector2 size, float fontSize)
    {
        var label = Rect("Label", parent, position, size).gameObject.AddComponent<TextMeshProUGUI>();
        label.font = font;
        label.text = text;
        label.fontSize = fontSize;
        label.color = Color.white;
        label.alignment = TextAlignmentOptions.MidlineLeft;
        label.raycastTarget = false;
        return label;
    }

    private Button ButtonAt(string text, Transform parent, Vector2 position, Vector2 size, UnityEngine.Events.UnityAction action)
    {
        var rect = Rect(text, parent, position, size);
        var image = rect.gameObject.AddComponent<Image>();
        image.color = new Color(0.08f, 0.22f, 0.34f, 0.96f);
        var button = rect.gameObject.AddComponent<Button>();
        button.targetGraphic = image;
        button.onClick.AddListener(action);
        var label = Text(text, rect, new Vector2(10, -5), size - new Vector2(20, 10), 25);
        label.alignment = TextAlignmentOptions.Center;
        return button;
    }
}
