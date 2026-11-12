using TMPro;
using UnityEngine;
using UnityEngine.UI;

/// <summary>The hip-thrust demonstration window, observing the existing segmented coach.</summary>
[DefaultExecutionOrder(150)]
public sealed class RingSkinCCoachView : MonoBehaviour
{
    public GameWorkoutScene entry;
    public Animator coachAnimator;
    public Animator previewAnimator;
    public Camera previewCamera;
    public ActionPreviewCard legacyCard;
    public TMP_Text countdown;
    public GameObject endMenu;

    public bool IsVisible => display != null && display.gameObject.activeInHierarchy;
    public int Phase { get; private set; } = -1;
    public float PhaseProgress { get; private set; }
    public string PhaseText => phaseLabel != null ? phaseLabel.text : "";
    public RectTransform PanelRect => panel;

    static readonly string[] Names = { "伸展", "收回" };
    static readonly Color[] Colors = { new Color(1,.49f,.67f), new Color(.73f,.61f,1) };
    Canvas display;
    RectTransform panel, progress;
    TMP_Text phaseLabel, timingLabel;
    readonly Image[] badges = new Image[2];
    readonly TMP_Text[] badgeLabels = new TMP_Text[2];
    RenderTexture texture, originalTexture;
    string segment;

    void Awake()
    {
        // Retain the card and its animation/menu listeners; replace only its old square graphics.
        foreach (var graphic in legacyCard.GetComponentsInChildren<Graphic>(true))
            graphic.enabled = false;
        originalTexture = previewCamera.targetTexture;
        texture = new RenderTexture(840,360,24,RenderTextureFormat.ARGB32) {
            name = "RingSkinC Coach Preview", antiAliasing = 2
        };
        texture.Create();
        previewCamera.targetTexture = texture;
        BuildWindow();
        display.gameObject.SetActive(false);
    }

    void OnEnable()
    {
        entry.coach.SegmentPlaybackStarted += OnSegment;
    }

    void OnDisable()
    {
        entry.coach.SegmentPlaybackStarted -= OnSegment;
        if (display != null) display.gameObject.SetActive(false);
    }

    void OnSegment(CoachActionController.SegmentPlaybackInfo info) => segment = info.stateName;

    void LateUpdate()
    {
        bool visible = entry.gameCanvas.gameObject.activeInHierarchy && !entry.menu.IsTrainingMenuVisible &&
            (endMenu == null || !endMenu.activeInHierarchy);
        display.gameObject.SetActive(visible);
        if (!visible) return;

        // Sample the real coach's current state/time, including pause and endpoint holds.
        // No second playback clock or bone rotation correction is introduced.
        if (entry.coach.IsPlayingAction && coachAnimator.enabled)
        {
            var state = coachAnimator.GetCurrentAnimatorStateInfo(0);
            previewAnimator.enabled = true;
            previewAnimator.speed = 0f;
            previewAnimator.Play(state.fullPathHash, 0, Mathf.Clamp01(state.normalizedTime));
            previewAnimator.Update(0f);
        }
        UpdatePhase();
        FitWindow();
    }

    void UpdatePhase()
    {
        Phase = -1;
        PhaseProgress = 0f;
        float remaining = 0;
        string label = "准备动作";
        bool running = entry.coach.IsPlayingAction;
        if (countdown != null && countdown.gameObject.activeInHierarchy) label = "准备开始";
        else if (running && !string.IsNullOrEmpty(segment))
        {
            if (segment.EndsWith("_up")) Phase = 0;
            else if (segment.EndsWith("_down")) Phase = 1;
            else if (segment.EndsWith("_end")) label = "结束动作";
            if (Phase >= 0)
            {
                var state = coachAnimator.GetCurrentAnimatorStateInfo(0);
                PhaseProgress = Mathf.Clamp01(state.normalizedTime);
                remaining = (1 - PhaseProgress) * state.length / Mathf.Max(.05f, coachAnimator.speed);
                label = Names[Phase];
            }
        }
        bool paused = running && (entry.coach.IsActionPaused || Time.timeScale == 0);
        phaseLabel.text = label + (paused ? " · 已暂停" : "");
        phaseLabel.color = Phase >= 0 ? Colors[Phase] : Color.white;
        timingLabel.text = paused ? "继续后跟随教练完成动作" : Phase >= 0 ? $"剩余 {remaining:0.0} 秒" : "跟随教练完成动作";
        for (int i=0; i<Names.Length; i++)
        {
            badges[i].color = i == Phase ? Colors[i] : new Color(.2f,.21f,.36f);
            badgeLabels[i].color = i == Phase ? new Color(.08f,.07f,.18f) : new Color(.8f,.8f,.9f);
        }
        progress.anchorMax = new Vector2(PhaseProgress,1);
        progress.GetComponent<Image>().color = Phase >= 0 ? Colors[Phase] : Color.white;
    }

    void BuildWindow()
    {
        var root = new GameObject("RingSkinC Coach Display", typeof(Canvas));
        display = root.GetComponent<Canvas>();
        display.renderMode = RenderMode.ScreenSpaceOverlay;
        display.sortingOrder = 900;
        panel = Panel(root.transform,"Recorded Coach",new Color(.055f,.055f,.16f,.96f));
        Label(panel,"教练示范",new Vector2(0,77),new Vector2(210,22),14);
        var preview = new GameObject("Coach",typeof(RectTransform),typeof(RawImage)).GetComponent<RawImage>();
        preview.transform.SetParent(panel,false);
        preview.texture = texture;
        preview.raycastTarget = false;
        float aspect = (float)texture.width / texture.height;
        Layout(preview.rectTransform,new Vector2(0,10),new Vector2(Mathf.Min(212,112*aspect),Mathf.Min(112,212/aspect)));
        for (int i=0; i<Names.Length; i++)
        {
            var badge = Panel(panel,Names[i],Colors[i]);
            Layout(badge,new Vector2((i-.5f)*100,-53),new Vector2(94,20));
            badges[i] = badge.GetComponent<Image>();
            badgeLabels[i] = Label(badge,Names[i],Vector2.zero,new Vector2(94,20),11);
        }
        var track = Panel(panel,"Phase Progress",new Color(.2f,.21f,.36f));
        Layout(track,new Vector2(0,-71),new Vector2(194,3));
        progress = Panel(track,"Fill",Color.white);
        progress.anchorMin = Vector2.zero; progress.anchorMax = Vector2.one;
        progress.offsetMin = progress.offsetMax = Vector2.zero;
        phaseLabel = Label(root.transform,"准备动作",Vector2.zero,new Vector2(220,25),18);
        phaseLabel.name = "Coach Phase";
        timingLabel = Label(root.transform,"跟随教练完成动作",Vector2.zero,new Vector2(220,20),12);
        timingLabel.name = "Coach Timing";
    }

    void FitWindow()
    {
        // The same 844 x 390 design coordinates as the hip-thrust demonstration panel.
        // Preserve aspect at 4:3, 16:9, wide glasses displays and after resizing.
        var scaler = display.GetComponent<CanvasScaler>();
        if (scaler != null) scaler.enabled = false;
        display.scaleFactor = 1;
        float scale = Mathf.Min(Screen.width / 844f, Screen.height / 390f);
        Place(panel,new Vector2(720,217),new Vector2(224,185),scale);
        Place(phaseLabel.rectTransform,new Vector2(720,329),new Vector2(220,25),scale);
        Place(timingLabel.rectTransform,new Vector2(720,353),new Vector2(220,20),scale);
    }

    static void Place(RectTransform rect,Vector2 point,Vector2 size,float scale)
    {
        Layout(rect,new Vector2(point.x-422,195-point.y)*scale,size);
        rect.localScale = Vector3.one*scale;
    }
    static void Layout(RectTransform rect,Vector2 point,Vector2 size)
    {
        rect.anchorMin = rect.anchorMax = rect.pivot = Vector2.one*.5f;
        rect.anchoredPosition = point; rect.sizeDelta = size;
    }
    TMP_Text Label(Transform parent,string value,Vector2 point,Vector2 size,float fontSize)
    {
        var text = new GameObject(value,typeof(RectTransform)).AddComponent<TextMeshProUGUI>();
        text.transform.SetParent(parent,false); text.font = entry.font; text.text = value;
        text.fontSize = fontSize; text.alignment = TextAlignmentOptions.Center; text.raycastTarget = false;
        Layout(text.rectTransform,point,size); return text;
    }
    static RectTransform Panel(Transform parent,string name,Color color)
    {
        var image = new GameObject(name,typeof(RectTransform),typeof(Image)).GetComponent<Image>();
        image.transform.SetParent(parent,false); image.color = color; image.raycastTarget = false;
        return image.rectTransform;
    }
    void OnDestroy()
    {
        if (previewCamera != null) previewCamera.targetTexture = originalTexture;
        if (texture != null) { texture.Release(); Destroy(texture); }
        if (display != null) Destroy(display.gameObject);
    }
}
