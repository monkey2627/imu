using System;
using System.Net;
using System.Net.Sockets;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

/// <summary>Test-build navigation. Switching deliberately cancels the current set.</summary>
public sealed class GameTestMenu : MonoBehaviour
{
    public static readonly string[] Scenes = {
        "BirdDog_RingSkinC_Recorded", "DeadBug_RingSkinC_Recorded",
        "HipThrust_RingBridge_NewScoring", "HipThrust_RingBridge_Hold30",
        "SideBridge_RingBridge_Dynamic", "SideBridge_RingBridge_Hold30"
    };
    public static readonly string[] Titles = {
        "鸟狗式", "死虫式", "臀桥 · 动态", "臀桥 · 保持30秒",
        "侧桥 · 动态", "侧桥 · 保持30秒"
    };
    public static bool EnabledForBuild
    {
        get {
#if UNITY_EDITOR || SPINE_TEST_BUILD
            return true;
#else
            return false;
#endif
        }
    }
    public GameObject Panel => panel;
    public Button[] GameButtons { get; private set; }
    public bool IsPaused => entry != null && (entry.taiko != null ? entry.taiko.IsPaused : entry.coach.IsActionPaused);
    GameWorkoutScene entry;
    GameObject root, panel, menuToggle;
    RectTransform safeArea;
    Button resume;
    TMP_Text heading, connection;
    bool opened, loading;
    bool ownsPause;
    float previousTimeScale;
    bool previousAudioPause;
    float nextConnectionRefresh;
    string addresses;

    public void Initialize(GameWorkoutScene value)
    {
        entry = value;
        var calibration = entry.scorer.GetComponent<SupineGameCalibration>();
        if (calibration != null) calibration.skipCalibrationForPreview = false;
    }
    void Start()
    {
        root = new GameObject("Test Game Navigation", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
        var canvas = root.GetComponent<Canvas>(); canvas.renderMode = RenderMode.ScreenSpaceOverlay; canvas.sortingOrder = 1500;
        var scaler = root.GetComponent<CanvasScaler>(); scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920, 1080); scaler.matchWidthOrHeight = 1;
        safeArea = Box("Safe Area", root.transform, Vector2.zero, Vector2.zero, Color.clear);
        safeArea.GetComponent<Image>().raycastTarget = false;
        menuToggle = MakeButton("动作选择", safeArea, new Vector2(-160, -70), new Vector2(250, 76), Open).gameObject;
        var toggleRect = (RectTransform)menuToggle.transform; toggleRect.anchorMin = toggleRect.anchorMax = new Vector2(1, 1);
        var shade = Box("Action Selection", root.transform, Vector2.zero, Vector2.zero, new Color(.025f, .025f, .09f, .94f));
        shade.anchorMin = Vector2.zero; shade.anchorMax = Vector2.one; shade.offsetMin = shade.offsetMax = Vector2.zero;
        panel = shade.gameObject;
        var card = Box("Card", shade, Vector2.zero, new Vector2(1280, 840), new Color(.12f, .105f, .25f));
        heading = Label("已暂停 · 选择动作", card, new Vector2(0, 340), new Vector2(1160, 75), 46);
        Label("切换动作将结束当前训练，并进入所选游戏的开始界面", card, new Vector2(0, 266), new Vector2(1180, 60), 27);
        GameButtons = new Button[Scenes.Length];
        for (int i = 0; i < Scenes.Length; i++) {
            int index = i;
            GameButtons[i] = MakeButton(Titles[i] + (Scenes[i] == gameObject.scene.name ? "  · 当前" : ""), card,
                new Vector2(i % 2 == 0 ? -297 : 297, 150 - i / 2 * 140), new Vector2(560, 112), () => SelectGame(index));
            GameButtons[i].name = "Select " + Scenes[i];
        }
        connection = Label("", card, new Vector2(0, -238), new Vector2(1180, 66), 24);
        connection.color = new Color(.68f, .81f, .92f);
        resume = MakeButton("继续训练", card, new Vector2(-297, -342), new Vector2(560, 80), Close);
        MakeButton("返回开始界面", card, new Vector2(297, -342), new Vector2(560, 80), ReturnToMenu);
        addresses = LocalAddresses();
        panel.SetActive(false);
        Screen.sleepTimeout = SleepTimeout.NeverSleep;
    }
    void Update()
    {
        if (root == null || entry == null) return;
        Rect area = Screen.safeArea;
        safeArea.anchorMin = new Vector2(area.xMin / Screen.width, area.yMin / Screen.height);
        safeArea.anchorMax = new Vector2(area.xMax / Screen.width, area.yMax / Screen.height);
        safeArea.offsetMin = safeArea.offsetMax = Vector2.zero;
        if (Input.GetKeyDown(KeyCode.Escape)) { if (panel.activeSelf) Close(); else Open(); }
        SynchronizePause();
        bool visible = opened || IsPaused;
        panel.SetActive(visible);
        menuToggle.SetActive(entry.menu.IsTrainingMenuVisible && !visible);
        if (!visible) return;
        heading.text = loading ? "正在切换动作…" : IsPaused ? "已暂停 · 选择动作" : "选择测试动作";
        resume.GetComponentInChildren<TMP_Text>().text = IsPaused ? "继续训练" : "关闭";
        if (Time.unscaledTime >= nextConnectionRefresh) {
            nextConnectionRefresh = Time.unscaledTime + .5f;
            var motion = FindObjectOfType<AvatarMotionSource>();
            connection.text = (motion != null && motion.HasFreshFrame ? "已收到动作数据" : "等待动作数据") +
                "  ·  本机 " + addresses + ":39539";
        }
    }
    public void Open()
    {
        if (loading) return;
        if (!entry.menu.IsTrainingMenuVisible && !IsPaused) TogglePause();
        // Countdown cannot be paused by the transport; return to a clean menu.
        if (!entry.menu.IsTrainingMenuVisible && !IsPaused) entry.StopTraining();
        opened = true;
        SynchronizePause();
    }
    public void Close() { if (loading) return; opened = false; if (IsPaused) TogglePause(); SynchronizePause(); }
    void TogglePause() { if (entry.taiko != null) entry.taiko.TogglePause(); else entry.coach.TogglePause(); }
    public void ReturnToMenu()
    {
        if (loading) return;
        GameWorkoutSession.Instance.Cancel();
        entry.StopTraining(); opened = false; RestorePause();
    }
    public void SelectGame(int index)
    {
        if (loading || index < 0 || index >= Scenes.Length) return;
        if (!Application.CanStreamedLevelBeLoaded(Scenes[index])) {
            connection.text = "此动作未包含在当前安装包中"; nextConnectionRefresh = Time.unscaledTime + 5; return;
        }
        loading = true;
        foreach (var button in GameButtons) button.interactable = false;
        GameWorkoutSession.Instance.Cancel();
        entry.StopTraining();
        RestorePause();
        foreach (var haptics in FindObjectsOfType<CoachHapticFeedbackController>()) haptics.StopAllHaptics();
        Time.timeScale = 1;
        SceneManager.LoadSceneAsync(Scenes[index], LoadSceneMode.Single);
    }
    void OnApplicationPause(bool paused) { if (paused && entry != null && !entry.menu.IsTrainingMenuVisible && !IsPaused) Open(); }
    void OnDestroy() { RestorePause(); if (root != null) Destroy(root); }
    void SynchronizePause()
    {
        if (!IsPaused) { RestorePause(); return; }
        if (ownsPause) return;
        previousTimeScale = Time.timeScale; previousAudioPause = AudioListener.pause;
        ownsPause = true; Time.timeScale = 0; AudioListener.pause = true;
    }
    void RestorePause()
    {
        if (!ownsPause) return;
        Time.timeScale = previousTimeScale; AudioListener.pause = previousAudioPause; ownsPause = false;
    }
    static string LocalAddresses()
    {
        try {
            var values = Array.FindAll(Dns.GetHostAddresses(Dns.GetHostName()), a => a.AddressFamily == AddressFamily.InterNetwork && !IPAddress.IsLoopback(a));
            return values.Length > 0 ? string.Join(" / ", Array.ConvertAll(values, a => a.ToString())) : "未连接局域网";
        } catch { return "未连接局域网"; }
    }
    RectTransform Box(string name, Transform parent, Vector2 position, Vector2 size, Color color)
    {
        var image = new GameObject(name, typeof(RectTransform), typeof(Image)).GetComponent<Image>();
        image.transform.SetParent(parent, false); image.color = color;
        var rect = image.rectTransform; rect.anchorMin = rect.anchorMax = rect.pivot = new Vector2(.5f, .5f);
        rect.anchoredPosition = position; rect.sizeDelta = size; return rect;
    }
    TMP_Text Label(string value, Transform parent, Vector2 position, Vector2 size, float fontSize)
    {
        var text = new GameObject("Label", typeof(RectTransform), typeof(TextMeshProUGUI)).GetComponent<TMP_Text>();
        text.transform.SetParent(parent, false); text.font = entry.font; text.text = value; text.fontSize = fontSize;
        text.alignment = TextAlignmentOptions.Center; text.raycastTarget = false;
        text.rectTransform.anchoredPosition = position; text.rectTransform.sizeDelta = size; return text;
    }
    Button MakeButton(string title, Transform parent, Vector2 position, Vector2 size, UnityEngine.Events.UnityAction action)
    {
        var rect = Box(title, parent, position, size, new Color(.27f, .23f, .46f));
        var button = rect.gameObject.AddComponent<Button>(); button.targetGraphic = rect.GetComponent<Image>(); button.onClick.AddListener(action);
        Label(title, rect, Vector2.zero, size - new Vector2(24, 12), 32); return button;
    }
}
