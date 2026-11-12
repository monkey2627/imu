using TMPro;
using UnityEngine;
using UnityEngine.UI;

/// <summary>The delivered artwork observes the same sampled animation clock as the coach and judgement.</summary>
[DefaultExecutionOrder(300)]
public sealed class HipThrustRingBridgeView : MonoBehaviour
{
    public HipThrustTaikoController transport;
    public GameWorkoutScene entry;
    public RingBridge.RingBridgeChart chart;
    public Camera presentationCamera;
    public GameObject artwork;
    public Canvas gamingCanvas;
    public Text score, combo;
    public Material displayMaterial;
    public TMP_FontAsset font;
    public RenderTexture coachPreview;
    public HipThrustPhaseTimeline.Sample CurrentSample { get; private set; }
    public RenderTexture Composition => composition;
    public bool IsPreviewVisible => previewPanel != null && previewPanel.activeInHierarchy;
    public bool IsCountdownVisible => countdownPanel != null && countdownPanel.gameObject.activeInHierarchy;
    public string CoachPhaseText => coachPhase != null ? coachPhase.text : "";

    RenderTexture composition;
    Canvas display;
    RawImage screen;
    RectTransform pauseButton, menuButton, pausePanelRect, previewPanelRect, countdownPanel, phaseProgress;
    GameObject pausePanel, previewPanel;
    GameObject endMenu;
    TMP_Text countdownSource, countdownNumber, coachPhase, coachTiming;
    TMP_Text ruleHint;
    RectTransform assessmentPanel,holdProgress;
    TMP_Text qualityMetric,holdMetric,repetitionMetric;
    readonly Image[] phaseBadges = new Image[3];
    readonly TMP_Text[] phaseBadgeLabels = new TMP_Text[3];
    static readonly string[] PhaseNames = { "抬起", "保持", "放下" };
    static readonly Color[] PhaseColors = { new Color(1,.49f,.67f), new Color(1,.84f,.39f), new Color(.73f,.61f,1) };
    SpriteRenderer background;
    RectTransform hudRect;
    Canvas hud;
    int captureWidth, captureHeight;

    void Awake()
    {
        chart.autoPlay = false;
        background = artwork.transform.Find("Background/rb_bg").GetComponent<SpriteRenderer>();
        hudRect = (RectTransform)artwork.transform.Find("HUD");
        hud = hudRect.GetComponent<Canvas>();
        // The old skin hid every Graphic, including the countdown. Observe that
        // controller's actual text/visibility; do not introduce a second timer.
        foreach (var text in gamingCanvas.GetComponentsInChildren<TMP_Text>(true))
            if (text.name == "CountDown") countdownSource = text;
        foreach (var canvas in FindObjectsOfType<Canvas>(true))
            if (canvas.gameObject.scene == gameObject.scene && canvas.name == "EndMenuCanvas") endMenu = canvas.gameObject;
        EnsureDisplay();
        SetVisible(false);
    }

    void LateUpdate()
    {
        EnsureDisplay();
        bool menuVisible = entry.menu.IsTrainingMenuVisible;
        // The existing result button calls StopAndShowMenu, which activates the
        // start menu without dismissing the higher-sorting result canvas.
        if (menuVisible && endMenu != null && endMenu.activeSelf) endMenu.SetActive(false);
        bool visible = gamingCanvas.gameObject.activeInHierarchy && !menuVisible;
        SetVisible(visible);
        if (!visible) return;
        RenderAtSourceFrame(transport.IsRunning ? transport.SourceTime * transport.PhaseTimeline.sourceClip.frameRate : 0);
        score.text = transport.Score.ToString("N0", System.Globalization.CultureInfo.InvariantCulture);
        if(transport.RuleScoring!=null&&transport.RuleScoring.sideBridge!=null&&!transport.IsRunning)score.text="0";
        combo.text = transport.RuleScoring != null ?
            (transport.RuleScoring.sideBridge!=null?"HOLD / 100":"LAST " + transport.RuleScoring.LastRoundBest.ToString("0")) : transport.ConsecutiveHits + " COMBO";
        // The mobile bootstrap converts canvases on scene load. This HUD belongs
        // to the artwork camera's render texture, so retain its world-space scale.
        hud.renderMode = RenderMode.WorldSpace;
        hud.worldCamera = presentationCamera;
        hudRect.sizeDelta = new Vector2(844,390);
        hudRect.anchoredPosition3D = Vector3.zero;
        hudRect.localPosition = Vector3.zero;
        hudRect.localScale = Vector3.one * .01f;
        pausePanel.SetActive(transport.IsPaused && !GameTestMenu.EnabledForBuild);
        UpdateCoachPresentation();
        FitControls();
    }

    void SetVisible(bool visible)
    {
        artwork.SetActive(visible);
        display.gameObject.SetActive(visible);
        presentationCamera.enabled = visible;
    }

    void UpdateCoachPresentation()
    {
        bool counting = countdownSource != null && countdownSource.gameObject.activeInHierarchy;
        countdownPanel.gameObject.SetActive(counting);
        if (counting) countdownNumber.text = countdownSource.text;
        pauseButton.GetComponent<Button>().interactable = transport.IsRunning;
        int phase = transport.IsRunning ? CurrentSample.phase : -1;
        coachPhase.text = counting ? "准备开始" : phase < 0 ? "准备动作" : PhaseNames[phase];
        coachPhase.color = phase < 0 ? Color.white : PhaseColors[phase];
        if (transport.IsPaused) coachPhase.text += " · 已暂停";
        coachTiming.text = "跟随教练完成动作";
        if (phase >= 0)
        {
            var cycle = transport.PhaseTimeline.cycles[CurrentSample.cycle];
            float frames = phase == 0 ? cycle.holdStartFrame-cycle.liftStartFrame :
                phase == 1 ? cycle.lowerStartFrame-cycle.holdStartFrame : cycle.lowerEndFrame-cycle.lowerStartFrame;
            float remaining = frames * (1-CurrentSample.progress) / transport.PhaseTimeline.sourceClip.frameRate;
            if (transport.Hold30Timeline != null) transport.Hold30Timeline.Evaluate(transport.Elapsed,out _,out remaining,out _);
            coachTiming.text = "剩余 " + remaining.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + " 秒";
        }
        for (int i=0;i<phaseBadges.Length;i++)
        {
            phaseBadges[i].color = i == phase ? PhaseColors[i] : new Color(.2f,.21f,.36f);
            phaseBadgeLabels[i].color = i == phase ? new Color(.08f,.07f,.18f) : new Color(.8f,.8f,.9f);
        }
        phaseProgress.anchorMax = new Vector2(phase < 0 ? 0 : CurrentSample.progress, 1);
        phaseProgress.GetComponent<Image>().color = phase < 0 ? Color.white : PhaseColors[phase];
        if (ruleHint != null)
        {
            ruleHint.text = transport.RuleScoring.sideBridge!=null&&!transport.IsRunning?"跟随倒计时准备动作":transport.RuleScoring.Status;
            if (transport.PrescriptionDose != null && transport.RuleScoring.sideBridge==null)
                ruleHint.text = $"示范 {transport.CompletedDemonstrations}/{transport.PrescriptionDose.Total} 次 · " + ruleHint.text;
        }
        if(assessmentPanel!=null)
        {
            var side=transport.RuleScoring.sideBridge;var rule=side.Rule;
            bool started=transport.IsRunning;
            bool ready=started&&transport.RuleScoring.InputFresh&&transport.RuleScoring.calibration!=null&&transport.RuleScoring.calibration.IsCalibrated;
            qualityMetric.text=ready?$"动作质量 {side.TrackingQuality:0} / 100":"动作质量 — / 100";
            holdMetric.text=started?$"{(rule.RequiresContinuousHold?"连续有效":"有效保持")} {rule.DisplayHoldSeconds:0.0} / {rule.TargetHoldSeconds:0.0} 秒":"有效保持 · 准备开始计时";
            repetitionMetric.text=$"合格 {(started?rule.QualifiedRounds:0)}/{transport.PrescriptionDose.Total} 次 · 示范 {(started?transport.CompletedDemonstrations:0)}/{transport.PrescriptionDose.Total} 次";
            holdProgress.anchorMax=new Vector2(started&&rule.TargetHoldSeconds>0?Mathf.Clamp01(rule.DisplayHoldSeconds/rule.TargetHoldSeconds):0,1);
        }
    }

    public void RenderAtSourceFrame(float frame)
    {
        bool repeated = transport.IsRunning && transport.Elapsed >= transport.PhaseTimeline.repeatEndFrame / (double)transport.PhaseTimeline.sourceClip.frameRate;
        CurrentSample = transport.PhaseTimeline.Evaluate(frame, repeated);
        if (transport.IsRunning && transport.Hold30Timeline != null)
            CurrentSample = transport.Hold30Timeline.Evaluate(transport.Elapsed,out _,out _,out _);
        chart.RenderRecorded(CurrentSample);
    }

    public void SetCaptureResolution(int width, int height)
    {
        captureWidth = width; captureHeight = height;
        EnsureDisplay();
    }

    void EnsureDisplay()
    {
        if (display == null)
        {
            var root = new GameObject("RingBridge Display", typeof(Canvas), typeof(GraphicRaycaster));
            display = root.GetComponent<Canvas>();
            display.renderMode = RenderMode.ScreenSpaceOverlay; display.sortingOrder = -90;
            screen = new GameObject("Artwork", typeof(RectTransform), typeof(RawImage)).GetComponent<RawImage>();
            screen.transform.SetParent(root.transform, false); screen.raycastTarget = false; screen.material = displayMaterial;
            screen.rectTransform.anchorMin = Vector2.zero; screen.rectTransform.anchorMax = Vector2.one;
            screen.rectTransform.offsetMin = screen.rectTransform.offsetMax = Vector2.zero;
            pauseButton = Button(root.transform, "", new Color(0,0,0,0), () => transport.TogglePause());
            pauseButton.name = "Pause";
            menuButton = Button(root.transform, "返回菜单", new Color(.12f,.13f,.3f,.9f), () => entry.StopTraining());
            menuButton.name = "Return to Menu";
            pausePanelRect = Panel(root.transform, "Pause Menu", new Color(.06f,.07f,.2f,.97f));
            pausePanel = pausePanelRect.gameObject;
            var resume = Button(pausePanelRect, "继续训练", new Color(.25f,.26f,.5f), () => transport.TogglePause());
            Layout(resume, new Vector2(0, 44), new Vector2(220, 42));
            var workout = Button(pausePanelRect, "动作串", new Color(.18f,.20f,.4f), () => entry.TogglePanel());
            Layout(workout, Vector2.zero, new Vector2(220, 36));
            var exit = Button(pausePanelRect, "返回菜单", new Color(.18f,.20f,.4f), () => entry.StopTraining());
            Layout(exit, new Vector2(0, -44), new Vector2(220, 36));
            previewPanelRect = Panel(root.transform, "Recorded Coach", new Color(.055f,.055f,.16f,.96f));
            previewPanel = previewPanelRect.gameObject;
            Label(previewPanelRect, "教练示范", new Vector2(0,77), new Vector2(210,22), 14);
            var preview = new GameObject("Coach", typeof(RectTransform), typeof(RawImage)).GetComponent<RawImage>();
            preview.transform.SetParent(previewPanelRect, false); preview.texture = coachPreview; preview.raycastTarget = false;
            // Match the render texture's aspect: hands/feet must never be stretched.
            float aspect = coachPreview != null ? (float)coachPreview.width/coachPreview.height : 16f/9f;
            Vector2 size = new Vector2(Mathf.Min(212,112*aspect),Mathf.Min(112,212/aspect));
            Layout(preview.rectTransform, new Vector2(0,10), size);
            for(int i=0;i<3;i++)
            {
                var badge = Panel(previewPanelRect, PhaseNames[i], PhaseColors[i]);
                Layout(badge,new Vector2((i-1)*67,-53),new Vector2(60,20));
                phaseBadges[i]=badge.GetComponent<Image>();
                phaseBadgeLabels[i]=Label(badge,PhaseNames[i],Vector2.zero,new Vector2(60,20),11);
            }
            var progressTrack=Panel(previewPanelRect,"Phase Progress",new Color(.2f,.21f,.36f));
            Layout(progressTrack,new Vector2(0,-71),new Vector2(194,3));
            phaseProgress=Panel(progressTrack,"Fill",Color.white);
            phaseProgress.anchorMin=Vector2.zero; phaseProgress.anchorMax=Vector2.one;
            phaseProgress.offsetMin=phaseProgress.offsetMax=Vector2.zero;
            coachPhase=Label(root.transform,"准备动作",Vector2.zero,new Vector2(220,25),18);
            coachPhase.name="Coach Phase";
            coachTiming=Label(root.transform,"跟随教练完成动作",Vector2.zero,new Vector2(220,20),12);
            coachTiming.name="Coach Timing";
            if (transport.RuleScoring != null)
                ruleHint=Label(root.transform,"保持位计分 · 满分 100",Vector2.zero,new Vector2(240,20),11);
            if(transport.RuleScoring!=null&&transport.RuleScoring.sideBridge!=null)
            {
                assessmentPanel=Panel(root.transform,"Measured Side Bridge",new Color(.055f,.055f,.16f,.96f));
                qualityMetric=Label(assessmentPanel,"动作质量 — / 100",new Vector2(0,25),new Vector2(210,20),14);
                holdMetric=Label(assessmentPanel,"有效保持",new Vector2(0,4),new Vector2(210,18),12);
                repetitionMetric=Label(assessmentPanel,"合格次数",new Vector2(0,-28),new Vector2(218,18),11);
                var track=Panel(assessmentPanel,"Measured Hold Progress",new Color(.2f,.21f,.36f));
                Layout(track,new Vector2(0,-11),new Vector2(194,4));
                holdProgress=Panel(track,"Fill",PhaseColors[1]);
                holdProgress.anchorMin=holdProgress.anchorMax=Vector2.zero;holdProgress.offsetMin=holdProgress.offsetMax=Vector2.zero;
            }
            countdownPanel=Panel(root.transform,"Start Countdown",new Color(.06f,.07f,.2f,.96f));
            Label(countdownPanel,"准备开始",new Vector2(0,37),new Vector2(210,24),17);
            countdownNumber=Label(countdownPanel,"",new Vector2(0,-9),new Vector2(210,75),58);
            // Pause stays above the presentation, including the permanently visible coach.
            pausePanelRect.SetAsLastSibling();
            previewPanel.SetActive(true); pausePanel.SetActive(false); countdownPanel.gameObject.SetActive(false);
        }
        int width = captureWidth > 0 ? captureWidth : Mathf.Max(1, Screen.width);
        var scaler = display.GetComponent<CanvasScaler>();
        if (scaler != null) scaler.enabled = false;
        display.scaleFactor = 1;
        int height = captureHeight > 0 ? captureHeight : Mathf.Max(1, Screen.height);
        if (composition == null || composition.width != width || composition.height != height)
        {
            ReleaseTexture();
            composition = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
            composition.name = "RingBridge sRGB composition"; composition.Create();
            screen.texture = composition; presentationCamera.targetTexture = composition;
        }
        presentationCamera.aspect = (float)width / height;
        presentationCamera.orthographicSize = Mathf.Max(1.95f, 4.22f / presentationCamera.aspect);
        float scale = Mathf.Max(presentationCamera.orthographicSize * 2f / 3.9f,
            presentationCamera.orthographicSize * 2f * presentationCamera.aspect / 8.44f);
        if (background != null) background.transform.localScale = Vector3.one * scale;
    }

    void FitControls()
    {
        float height = presentationCamera.orthographicSize * 2f;
        float width = height * presentationCamera.aspect;
        float scale = Screen.height / (height * 100f);
        Place(pauseButton, new Vector2(40,30), new Vector2(44,44));
        Place(menuButton, new Vector2(65,369), new Vector2(94,28));
        Place(previewPanelRect, new Vector2(720,217), new Vector2(224,185));
        if(assessmentPanel!=null)Place(assessmentPanel,new Vector2(720,78),new Vector2(224,82));
        Place(coachPhase.rectTransform,new Vector2(720,329),new Vector2(220,25));
        Place(coachTiming.rectTransform,new Vector2(720,353),new Vector2(220,20));
        if(ruleHint!=null) Place(ruleHint.rectTransform,new Vector2(714,376),new Vector2(250,20));
        Place(countdownPanel,new Vector2(422,215),new Vector2(220,128));
        Place(pausePanelRect, new Vector2(422,195), new Vector2(280,170));
        void Place(RectTransform rect, Vector2 point, Vector2 size)
        {
            rect.anchorMin = rect.anchorMax = new Vector2(.5f + (point.x-422)*.01f/width, .5f-(point.y-195)*.01f/height);
            rect.anchoredPosition = Vector2.zero;
            rect.sizeDelta = size;
            rect.localScale = Vector3.one * scale;
        }
    }

    TMP_Text Label(Transform parent,string text,Vector2 position,Vector2 size,float fontSize)
    {
        var label=new GameObject(text,typeof(RectTransform)).AddComponent<TextMeshProUGUI>();
        label.transform.SetParent(parent,false); label.font=font; label.text=text; label.fontSize=fontSize;
        label.alignment=TextAlignmentOptions.Center; label.raycastTarget=false;
        Layout(label.rectTransform,position,size); return label;
    }
    RectTransform Panel(Transform parent, string name, Color color)
    {
        var image = new GameObject(name, typeof(RectTransform), typeof(Image)).GetComponent<Image>();
        image.transform.SetParent(parent, false); image.color = color;
        return image.rectTransform;
    }
    RectTransform Button(Transform parent, string title, Color color, UnityEngine.Events.UnityAction action)
    {
        var rect = Panel(parent, title, color);
        var button = rect.gameObject.AddComponent<Button>(); button.targetGraphic = rect.GetComponent<Image>(); button.onClick.AddListener(action);
        var text = new GameObject("Label", typeof(RectTransform)).AddComponent<TextMeshProUGUI>();
        text.transform.SetParent(rect, false); text.font = font; text.text = title; text.fontSize = 18;
        text.alignment = TextAlignmentOptions.Center; text.raycastTarget = false;
        text.rectTransform.anchorMin = Vector2.zero; text.rectTransform.anchorMax = Vector2.one;
        text.rectTransform.offsetMin = text.rectTransform.offsetMax = Vector2.zero;
        return rect;
    }
    static void Layout(RectTransform rect, Vector2 position, Vector2 size)
    {
        rect.anchorMin = rect.anchorMax = rect.pivot = Vector2.one*.5f;
        rect.anchoredPosition = position; rect.sizeDelta = size;
    }
    void OnDisable()
    {
        if (display != null) display.gameObject.SetActive(false);
        if (presentationCamera != null) presentationCamera.enabled = false;
    }
    void OnDestroy() { ReleaseTexture(); if (display != null) Destroy(display.gameObject); }
    void ReleaseTexture()
    {
        if (composition == null) return;
        if (presentationCamera != null) presentationCamera.targetTexture = null;
        composition.Release(); Destroy(composition); composition = null;
    }
}
