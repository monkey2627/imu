using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public sealed class MobilePoseCalibrationController : MonoBehaviour
{
    public enum CalibrationMode { SceneDefault, BirdDog, DeadBug, SideBridge, HipThrust, Crunch, LyingCat }
    // PoseHold: a known lying-game hold solved in Python (KNOWN_HOLDS in pose_calibration.py, docs/09).
    public enum StepKind { Standing, ArmsForward, StartPose, Prompt, BirdDogBase, BirdDogTransition, BirdDogLeftReach, BirdDogRightReach, PoseHold }

    [Serializable]
    public sealed class Step
    {
        public StepKind kind;
        [TextArea] public string instruction;
        [Min(0f)] public float seconds = 5f;
        public bool captureParallelShanks;
        public string poseId;
    }

    [SerializeField] private MobilePoseWebSocketSource mobilePose;
    [SerializeField] private List<Step> steps = new List<Step>();
    [SerializeField] private CalibrationMode selectedMode = CalibrationMode.SceneDefault;
    [Tooltip("Optional action-specific start pose ID. Empty uses the action's default pose.")]
    [SerializeField] private string startPoseId = "";

    public bool IsCalibrating { get; private set; }
    public bool IsComplete { get; private set; }
    public float RemainingSeconds { get; private set; }
    public string CurrentInstruction { get; private set; }
    private string armCalibrationResult;
    private float resultUntil;
    private bool armPriorSent;
    private string sentAction;    // action last announced to Python (actionSelect); null after a disconnect
    private bool restarting;      // waiting for a (re)started Python bridge before this calibration

    private void Awake()
    {
        if (mobilePose == null) mobilePose = GetComponent<MobilePoseWebSocketSource>();
        if (steps.Count == 0) steps = CreateDefaultSteps(gameObject.scene.name);
        if (selectedMode == CalibrationMode.SceneDefault) selectedMode = ModeForScene(gameObject.scene.name);
    }

    /// <summary>Python resets every action-level state itself at each calibration start (run.py
    /// reset_action_state), so the running bridge is reused.  It is restarted only when it reports that
    /// its code changed after it started (codeStale), so a calibration never runs old code.</summary>
    public void BeginCalibration()
    {
        if (IsCalibrating || restarting) return;
        MobilePosePythonLauncher launcher = mobilePose != null ? mobilePose.GetComponent<MobilePosePythonLauncher>() : null;
        if (launcher == null) launcher = FindObjectOfType<MobilePosePythonLauncher>(true);
        MobilePosePythonLauncher.ConsumeFreshProcess();
        if (launcher != null && mobilePose != null && !mobilePose.IsReplaying && mobilePose.PythonCodeStale)
        {
            StopAllCoroutines();
            StartCoroutine(RestartBridgeThenCalibrate(launcher));
            return;
        }
        if (mobilePose != null && !mobilePose.IsReplaying)
        {
            StopAllCoroutines();
            StartCoroutine(WaitForBridgeThenCalibrate());
            return;
        }
        BeginCalibrationOnCurrentBridge();
    }

    /// <summary>The bridge Unity launched at Play: it may still be loading the model or waiting for SlimeVR.</summary>
    private IEnumerator WaitForBridgeThenCalibrate()
    {
        restarting = true;
        IsCalibrating = true;
        IsComplete = false;
        armCalibrationResult = null;
        RemainingSeconds = 0f;
        float deadline = Time.realtimeSinceStartup + MobilePosePythonLauncher.RestartConnectTimeoutSeconds;
        while (!MobilePosePythonLauncher.IsBridgeReady(mobilePose))
        {
            if (Time.realtimeSinceStartup > deadline)
            {
                restarting = false;
                IsCalibrating = false;
                CurrentInstruction = "Python 未就绪：一直没有收到姿态数据（SlimeVR 是否已连接、追踪器是否已绑定？）";
                yield break;
            }
            CurrentInstruction = mobilePose.IsConnected
                ? "Python 已启动，正在等待 SlimeVR 数据…"
                : "正在等待 Python 启动（加载模型），请稍候…";
            yield return new WaitForSecondsRealtime(0.2f);
        }
        restarting = false;
        IsCalibrating = false;
        BeginCalibrationOnCurrentBridge();
    }

    private IEnumerator RestartBridgeThenCalibrate(MobilePosePythonLauncher launcher)
    {
        restarting = true;
        IsCalibrating = true;
        IsComplete = false;
        armCalibrationResult = null;
        RemainingSeconds = 0f;
        yield return launcher.RestartBridge(mobilePose, message => CurrentInstruction = message);
        restarting = false;
        IsCalibrating = false;
        if (!string.IsNullOrEmpty(launcher.RestartError))
        {
            CurrentInstruction = "重启 Python 失败：" + launcher.RestartError;
            yield break;
        }
        armPriorSent = false;         // the new bridge needs the per-scene pose options again
        sentAction = null;
        BeginCalibrationOnCurrentBridge();
    }

    private void BeginCalibrationOnCurrentBridge()
    {
        if (IsCalibrating) return;
        // A new button press must never consume the previous action's terminal
        // exerciseCalibration frame.
        if (mobilePose != null) mobilePose.ClearExerciseCalibrationState();
        if (selectedMode != CalibrationMode.SceneDefault && selectedMode != CalibrationMode.BirdDog)
        {
            if (mobilePose == null || !mobilePose.IsConnected || mobilePose.IsReplaying)
            {
                CurrentInstruction = "Connect Python to live SlimeVR input before starting this calibration.";
                return;
            }
            string action = ExerciseAction(selectedMode);
            IsComplete = false;
            IsCalibrating = true;
            CurrentInstruction = "Starting " + ModeDescription(selectedMode);
            string requestedStartPose = string.IsNullOrWhiteSpace(startPoseId)
                ? DefaultStartPose(selectedMode)
                : startPoseId.Trim();
            mobilePose.SendCalibrationCommand(
                "{\"command\":\"exerciseCalibrationStart\",\"action\":\"" + action +
                "\",\"startPose\":\"" + EscapeJson(requestedStartPose) + "\"}");
            return;
        }
        if (selectedMode == CalibrationMode.BirdDog) steps = CreateDefaultSteps(SceneToken(selectedMode));
        StopAllCoroutines();
        StartCoroutine(RunSteps());
    }

    private static string ExerciseAction(CalibrationMode mode)
    {
        switch (mode)
        {
            case CalibrationMode.DeadBug: return "deadbug";
            case CalibrationMode.SideBridge: return "sidebridge_left";
            case CalibrationMode.HipThrust: return "hipthrust";
            case CalibrationMode.Crunch: return "crunch";
            case CalibrationMode.LyingCat: return "lyingcat";
            default: return string.Empty;
        }
    }

    /// <summary>Action id Python routes on (run.py actionSelect), including bird dog and the scene default.</summary>
    private string SelectedAction()
    {
        CalibrationMode mode = selectedMode == CalibrationMode.SceneDefault ? ModeForScene(gameObject.scene.name) : selectedMode;
        return mode == CalibrationMode.BirdDog ? "birddog" : ExerciseAction(mode);
    }

    /// <summary>Tell Python which action is selected; it clears the previous action's state on a change.</summary>
    private void AnnounceAction()
    {
        if (mobilePose == null || !mobilePose.IsConnected || mobilePose.IsReplaying || IsCalibrating) return;
        string action = SelectedAction();
        if (action == sentAction) return;
        mobilePose.SendCalibrationCommand("{\"command\":\"actionSelect\",\"action\":\"" + EscapeJson(action) + "\"}");
        sentAction = action;
    }

    private static string DefaultStartPose(CalibrationMode mode)
    {
        switch (mode)
        {
            case CalibrationMode.DeadBug: return "tabletop";
            case CalibrationMode.SideBridge: return "side_rest";
            case CalibrationMode.HipThrust: return "supine";
            case CalibrationMode.Crunch: return "supine";
            case CalibrationMode.LyingCat: return "quadruped";
            default: return string.Empty;
        }
    }

    private static string EscapeJson(string value)
    {
        return (value ?? string.Empty).Replace("\\", "\\\\").Replace("\"", "\\\"");
    }

    private static CalibrationMode ModeForScene(string sceneName)
    {
        string lower = (sceneName ?? string.Empty).ToLowerInvariant();
        if (lower.Contains("birddog")) return CalibrationMode.BirdDog;
        if (lower.Contains("deadbug")) return CalibrationMode.DeadBug;
        if (lower.Contains("sidebridge")) return CalibrationMode.SideBridge;
        if (lower.Contains("hipthrust")) return CalibrationMode.HipThrust;
        if (lower.Contains("crunch")) return CalibrationMode.Crunch;
        if (lower.Contains("lyingcat") || lower.Contains("cat")) return CalibrationMode.LyingCat;
        return CalibrationMode.SceneDefault;
    }

    private static string SceneToken(CalibrationMode mode)
    {
        switch (mode)
        {
            case CalibrationMode.BirdDog: return "BirdDog";
            case CalibrationMode.DeadBug: return "DeadBug";
            case CalibrationMode.SideBridge: return "SideBridge";
            case CalibrationMode.HipThrust: return "HipThrust";
            case CalibrationMode.Crunch: return "Crunch";
            case CalibrationMode.LyingCat: return "LyingCat";
            default: return string.Empty;
        }
    }

    private static string ModeDescription(CalibrationMode mode)
    {
        // Keep the on-screen summary aligned with the one-step action-start flow.
        if (mode == CalibrationMode.BirdDog) return "鸟狗式：只保持四点跪姿起始位置。";
        if (mode == CalibrationMode.DeadBug) return "死虫：只保持桌面式起始位置。";
        if (mode == CalibrationMode.SideBridge) return "侧桥：只保持对应侧卧起始位置。";
        if (mode == CalibrationMode.HipThrust) return "臀桥：只保持屈膝仰卧起始位置。";
        if (mode == CalibrationMode.Crunch) return "卷腹：只保持仰卧屈膝、双手抱头的起始位置。";
        if (mode == CalibrationMode.LyingCat) return "卧位猫：只保持四点跪姿起始位置。";
        switch (mode)
        {
            case CalibrationMode.BirdDog: return "鸟狗：四点跪姿 + 左右对侧伸展；保持原有鸟狗校准流程。";
            case CalibrationMode.DeadBug: return "死虫：仰卧起始、桌面腿、左右对侧伸展；手和腿按动作姿势校准。";
            case CalibrationMode.SideBridge: return "侧桥：仰卧基准 + 侧向支撑姿势；校准支撑前臂、腰侧和腿部方向。";
            case CalibrationMode.HipThrust: return "臀桥：仰卧伸腿基准 + 屈膝臀桥姿势；校准腿部弯曲和躯干方向。";
            case CalibrationMode.Crunch: return "卷腹：仰卧手臂放置 + 卷腹姿势；校准手臂和躯干屈曲方向。";
            case CalibrationMode.LyingCat: return "卧位猫：四点跪姿中立位；手和小腿保持固定，建立支点参考，动作时关注腰部上下变化。";
            default: return "按当前场景的默认校准步骤执行。";
        }
    }

    private IEnumerator RunSteps()
    {
        IsCalibrating = true;
        IsComplete = false;
        if (mobilePose == null)
        {
            CurrentInstruction = "MobilePose input is missing from this scene.";
            IsCalibrating = false;
            yield break;
        }
        CurrentInstruction = "Waiting for MobilePose / Python connection";
        while (mobilePose != null && !mobilePose.IsConnected)
            yield return null;

        // Naming the action lets Python switch to its model before the recording header is written.
        bool birdDog = steps.Exists(s => s != null && s.kind == StepKind.BirdDogBase);
        if (!mobilePose.IsReplaying)
            mobilePose.SendCalibrationCommand(birdDog
                ? "{\"command\":\"calibrationSessionStart\",\"name\":\"birddog\"}"
                : "{\"command\":\"calibrationSessionStart\"}");

        foreach (Step step in steps)
        {
            if (step == null) continue;
            CurrentInstruction = step.instruction;
            RemainingSeconds = Mathf.Max(0f, step.seconds);

            if ((step.kind == StepKind.Standing || step.kind == StepKind.BirdDogBase) &&
                mobilePose != null && mobilePose.IsYawResetEnabled && !mobilePose.IsReplaying)
                mobilePose.SendCalibrationCommand("{\"command\":\"yawReset\"}");

            if (step.kind == StepKind.BirdDogBase && mobilePose != null &&
                mobilePose.IsPoseCalibrationEnabled && !mobilePose.IsReplaying)
                mobilePose.SendCalibrationCommand("{\"command\":\"birdDogStartCalibrate\"}");

            while (RemainingSeconds > 0f)
            {
                RemainingSeconds = Mathf.Max(0f, RemainingSeconds - Time.unscaledDeltaTime);
                yield return null;
            }

            if (mobilePose != null && !mobilePose.IsReplaying)
            {
                if (step.kind == StepKind.Standing && mobilePose.IsPoseCalibrationEnabled)
                    mobilePose.SendCalibrationCommand("{\"command\":\"standCalibrate\"}");
                else if (step.kind == StepKind.ArmsForward && mobilePose.IsPoseCalibrationEnabled)
                {
                    mobilePose.SendCalibrationCommand("{\"command\":\"armsForwardCalibrate\"}");
                    resultUntil = Time.unscaledTime + 3.5f;
                }
                else if (step.kind == StepKind.BirdDogBase && mobilePose.IsPoseCalibrationEnabled)
                {
                    mobilePose.SendCalibrationCommand("{\"command\":\"birdDogBaseCalibrate\"}");
                    resultUntil = Time.unscaledTime + 4f;
                }
                else if (step.kind == StepKind.BirdDogLeftReach && mobilePose.IsPoseCalibrationEnabled)
                {
                    mobilePose.SendCalibrationCommand("{\"command\":\"birdDogLeftReachCalibrate\"}");
                    resultUntil = Time.unscaledTime + 4f;
                }
                else if (step.kind == StepKind.BirdDogRightReach && mobilePose.IsPoseCalibrationEnabled)
                {
                    mobilePose.SendCalibrationCommand("{\"command\":\"birdDogRightReachCalibrate\"}");
                    resultUntil = Time.unscaledTime + 4f;
                }
                else if (step.kind == StepKind.PoseHold && mobilePose.IsPoseCalibrationEnabled &&
                         !string.IsNullOrEmpty(step.poseId))
                {
                    // Python searches the whole step for the latest still (and level) half second.
                    mobilePose.SendCalibrationCommand(
                        "{\"command\":\"poseHoldCalibrate\",\"pose\":\"" + step.poseId + "\",\"seconds\":" +
                        Mathf.Max(1f, step.seconds).ToString("0.##", System.Globalization.CultureInfo.InvariantCulture) + "}");
                    resultUntil = Time.unscaledTime + 4f;
                }
                if (step.kind == StepKind.StartPose &&
                    step.captureParallelShanks && mobilePose.IsPoseCalibrationEnabled)
                {
                    bool captured = mobilePose.CaptureParallelShankCorrection();
                    CurrentInstruction = captured
                        ? "起始姿勢校正完成，請保持準備，完成後再開始遊戲。"
                        : "尚未收到預測姿勢，請確認 IMU 與 Python 已連線後重新校準。";
                    resultUntil = Time.unscaledTime + 4f;
                }
            }
        }

        if (mobilePose != null && !mobilePose.IsReplaying)
            mobilePose.SendCalibrationCommand("{\"command\":\"calibrationSessionEnd\"}");

        RemainingSeconds = 0f;
        CurrentInstruction = "Calibration complete. Start the game when ready.";
        IsCalibrating = false;
        IsComplete = true;
    }

    private void Update()
    {
        // No automatic calibration on Play: it starts only from the "开始校准" / "重新校准" button.
        if (IsCalibrating && !restarting && selectedMode != CalibrationMode.BirdDog && mobilePose != null)
        {
            string controlError = mobilePose.LatestControlMessage ?? string.Empty;
            if (controlError.IndexOf("exerciseCalibrationStart failed:", StringComparison.Ordinal) >= 0)
            {
                IsCalibrating = false;
                IsComplete = false;
                CurrentInstruction = controlError;
                resultUntil = Time.unscaledTime + 8f;
                return;
            }
            MobilePoseWebSocketSource.ExerciseCalibrationFrame state = mobilePose.LatestExerciseCalibration;
            if (state != null && state.action == ExerciseAction(selectedMode))
            {
                RemainingSeconds = state.remaining;
                CurrentInstruction = "第 " + state.step + " / " + state.total + " 步 · " +
                    (state.preparing ? "准备，摆好姿势" : "保持不动") + "\n" + state.instruction;
                if (!state.active && (state.outcome == "complete" || state.outcome == "failed" || state.outcome == "cancelled"))
                {
                    IsCalibrating = false;
                    IsComplete = state.outcome == "complete";
                    if (state.results != null && state.results.Length > 0)
                        armCalibrationResult = string.Join("\n", state.results);
                    resultUntil = Time.unscaledTime + 8f;
                }
            }
        }
        if (mobilePose != null)
        {
            // The straight-arm prior (Python arm_prior.py) is wrong for the side-bridge support arm,
            // whose level forward forearm comes with a bent elbow; send the per-scene choice on connect.
            if (!mobilePose.IsConnected)
            {
                armPriorSent = false;
                sentAction = null;
            }
            else
                AnnounceAction();
            if (mobilePose.IsConnected && !armPriorSent)
            {
                bool sideBridge = gameObject.scene.name.ToLowerInvariant().Contains("sidebridge");
                mobilePose.SendCalibrationCommand(
                    "{\"command\":\"setPoseOptions\",\"armPrior\":" + (sideBridge ? "false" : "true") + "}");
                armPriorSent = true;
            }
        }
        if (mobilePose == null || Time.unscaledTime >= resultUntil) return;
        string message = mobilePose.LatestControlMessage;
        if (message.StartsWith("arm calibration:") || message.StartsWith("bird dog calibration:") ||
            message.StartsWith("pose hold calibration") || message.StartsWith("start pose fit"))
            armCalibrationResult = message;
    }

    private GUIStyle panelStyle, countdownStyle, gridStyle, buttonStyle, titleStyle;
    private float styleScale = -1f;

    /// <summary>Readable from a mat on the floor: sizes follow the screen height (1080p = 1).</summary>
    private void EnsureStyles()
    {
        float scale = Mathf.Max(0.7f, Screen.height / 1080f);
        if (panelStyle != null && Mathf.Approximately(scale, styleScale)) return;
        styleScale = scale;
        var background = new Texture2D(1, 1);
        background.SetPixel(0, 0, new Color(0f, 0f, 0f, 0.78f));
        background.Apply();
        panelStyle = new GUIStyle(GUI.skin.box)
        {
            fontSize = Mathf.RoundToInt(40 * scale), wordWrap = true, alignment = TextAnchor.UpperCenter,
            padding = new RectOffset(24, 24, 18, 18)
        };
        panelStyle.normal.background = background;
        panelStyle.normal.textColor = Color.white;
        countdownStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = Mathf.RoundToInt(110 * scale), fontStyle = FontStyle.Bold, alignment = TextAnchor.MiddleCenter
        };
        countdownStyle.normal.textColor = new Color(1f, 0.85f, 0.2f);
        gridStyle = new GUIStyle(GUI.skin.button) { fontSize = Mathf.RoundToInt(30 * scale) };
        buttonStyle = new GUIStyle(GUI.skin.button) { fontSize = Mathf.RoundToInt(34 * scale), fontStyle = FontStyle.Bold };
        titleStyle = new GUIStyle(GUI.skin.label) { fontSize = Mathf.RoundToInt(30 * scale), alignment = TextAnchor.MiddleCenter };
        titleStyle.normal.textColor = Color.white;
    }

    private void OnGUI()
    {
        EnsureStyles();
        float scale = styleScale;
        float width = Mathf.Min(Screen.width - 40f, 1500f * scale);
        float left = (Screen.width - width) / 2f;
        string text = string.IsNullOrEmpty(CurrentInstruction)
            ? "校准与游戏分开进行，准备好后开始。"
            : CurrentInstruction;
        if (!string.IsNullOrEmpty(armCalibrationResult) && Time.unscaledTime < resultUntil)
        {
            text += "\n" + armCalibrationResult;
            if (armCalibrationResult.Contains("CHECK FAILED"))
                text += "\n姿势与预期不符，校准可能不准，建议完成后点“重新校准”。";
        }
        if (mobilePose != null && mobilePose.PythonCodeStale)
            text = "⚠ Python 仍在运行旧代码（代码已更新）。点击“开始校准”或“重新校准”会自动重启 Python。\n" + text;
        float top = 12f * scale;
        if (!IsCalibrating)
        {
            GUI.Label(new Rect(left, top, width, 44f * scale), "选择校准动作", titleStyle);
            top += 46f * scale;
            string[] names = { "场景默认", "鸟狗", "死虫", "侧桥", "臀桥", "卷腹", "卧位猫" };
            int mode = GUI.SelectionGrid(new Rect(left, top, width, 60f * scale), (int)selectedMode, names, names.Length,
                gridStyle);
            if (mode != (int)selectedMode)
            {
                selectedMode = (CalibrationMode)mode;
                steps = selectedMode == CalibrationMode.SceneDefault
                    ? CreateDefaultSteps(gameObject.scene.name)
                    : CreateDefaultSteps(SceneToken(selectedMode));
                // A new action is not calibrated: show "开始校准", drop the old result and let Python
                // clear the previous action (actionSelect is sent from Update while connected).
                IsComplete = false;
                armCalibrationResult = null;
                if (mobilePose != null) mobilePose.ClearExerciseCalibrationState();
            }
            top += 72f * scale;
            text = ModeDescription(selectedMode) + "\n" + text;
        }
        float countdownWidth = IsCalibrating ? 220f * scale : 0f;
        float textWidth = width - countdownWidth;
        float height = Mathf.Max(panelStyle.CalcHeight(new GUIContent(text), textWidth), 150f * scale);
        GUI.Box(new Rect(left, top, textWidth, height), text, panelStyle);
        if (IsCalibrating)
        {
            GUI.Box(new Rect(left + textWidth, top, countdownWidth, height), GUIContent.none, panelStyle);
            GUI.Label(new Rect(left + textWidth, top, countdownWidth, height),
                Mathf.CeilToInt(RemainingSeconds).ToString(), countdownStyle);
        }
        var buttonRect = new Rect((Screen.width - 340f * scale) / 2f, top + height + 14f * scale, 340f * scale, 72f * scale);
        if (!IsCalibrating && !IsComplete && GUI.Button(buttonRect, "开始校准", buttonStyle))
            BeginCalibration();
        if (IsComplete && GUI.Button(buttonRect, "重新校准", buttonStyle))
            BeginCalibration();
    }

    private static Step PoseHold(string poseId, string instruction, float seconds) =>
        new Step { kind = StepKind.PoseHold, poseId = poseId, instruction = instruction, seconds = seconds };

    private static List<Step> CreateDefaultSteps(string sceneName)
    {
        string lower = (sceneName ?? string.Empty).ToLowerInvariant();

        // Runtime calibration is now a single action-specific start-pose hold.
        // Keep the legacy multi-step definitions below for old scenes, but make
        // the current UI present the same simple-start flow as Python.
        if (lower.Contains("birddog"))
            return new List<Step> { new Step { kind = StepKind.BirdDogBase,
                instruction = "保持鸟狗式起始姿势：双手撑地、手掌平放、手指朝前，双膝着地，背部平稳，保持不动。", seconds = 8f } };
        if (lower.Contains("deadbug"))
            return new List<Step> { PoseHold("deadbug_tabletop",
                "保持死虫式起始姿势：仰卧，双臂向上伸直，双腿抬起，大腿垂直、小腿水平，保持不动。", 8f) };
        if (lower.Contains("hipthrust"))
            return new List<Step> { PoseHold("hipthrust_start",
                "保持臀桥起始姿势：仰卧屈膝，双脚踩地，双臂放在身体两侧，保持不动。", 8f) };
        if (lower.Contains("sidebridge"))
        {
            string sidePose = lower.Contains("right") ? "sidebridge_right_start" : "sidebridge_left_start";
            return new List<Step> { PoseHold(sidePose,
                "保持侧桥起始姿势：侧卧，前臂支撑，身体保持稳定，保持不动。", 8f) };
        }
        if (lower.Contains("crunch"))
            // No poseHoldCalibrate: Python's "supine_arms_side" hold would turn hands-behind-head forearms
            // toward the feet.  The crunch start pose is fitted by exerciseCalibrationStart (start_pose_fit.py).
            return new List<Step> { PoseHold("",
                "保持卷腹起始姿势：仰卧屈膝、双脚踩地；双手抱在头后、手肘向两侧打开，保持不动。（双臂放体侧会校准失败）", 8f) };
        if (lower.Contains("lyingcat") || lower.Contains("cat"))
            return new List<Step> { PoseHold("cat_neutral",
                "保持跪姿猫式起始姿势：双手撑地、双膝着地，背部平稳，保持不动。", 8f) };

        if (lower.Contains("birddog"))
        {
            return new List<Step>
            {
                new Step
                {
                    kind = StepKind.BirdDogBase,
                    instruction = "四点跪姿：双手撑地、双膝着地；双前臂彼此平行，双小腿彼此平行，保持稳定。",
                    seconds = 10f
                },
                new Step
                {
                    kind = StepKind.BirdDogTransition,
                    instruction = "转换：平稳抬起左手、右腿，进入伸展姿势；缓慢连续移动。",
                    seconds = 6f
                },
                new Step
                {
                    kind = StepKind.BirdDogLeftReach,
                    instruction = "保持四点跪姿，左手向前伸；右腿抬高并伸直，尽量与躯干同高，保持身体稳定。",
                    seconds = 8f
                },
                new Step
                {
                    kind = StepKind.BirdDogTransition,
                    instruction = "转换：平稳回到四点跪姿，再换边；缓慢连续移动。",
                    seconds = 6f
                },
                new Step
                {
                    kind = StepKind.BirdDogRightReach,
                    instruction = "保持四点跪姿，右手向前伸；左腿抬高并伸直，尽量与躯干同高，保持身体稳定。",
                    seconds = 8f
                }
            };
        }

        // Lying games: holds taken from the action itself or its set-up, no standing up (docs/09).
        if (lower.Contains("deadbug"))
        {
            return new List<Step>
            {
                PoseHold("supine_arms_side", "仰卧，屈膝脚踩地，双臂伸直放在身体两侧，掌心向下，保持不动。", 6f),
                PoseHold("deadbug_tabletop", "双腿抬起：大腿垂直、小腿与地面平行；双臂伸直指向天花板，保持不动。", 8f),
                PoseHold("deadbug_reach_left", "左手向头顶方向伸展（手臂贴近地面方向），右腿伸直，保持不动。", 6f),
                new Step { kind = StepKind.Prompt, instruction = "回到双臂朝天、小腿平行的姿势。", seconds = 3f },
                PoseHold("deadbug_reach_right", "右手向头顶方向伸展（手臂贴近地面方向），左腿伸直，保持不动。", 6f),
                new Step { kind = StepKind.Prompt, instruction = "回到起始姿势，准备开始游戏。", seconds = 2f }
            };
        }
        if (lower.Contains("hipthrust"))
        {
            return new List<Step>
            {
                PoseHold("supine_legs_straight", "仰卧，双腿伸直并拢，双臂伸直放在身体两侧，掌心向下，保持不动。", 6f),
                PoseHold("hipthrust_start", "屈膝，双脚踩地与髋同宽，双臂保持在身体两侧，保持不动。", 6f)
            };
        }
        if (lower.Contains("sidebridge"))
        {
            return new List<Step>
            {
                PoseHold("supine_legs_straight", "仰卧，双腿伸直并拢，双臂伸直放在身体两侧，掌心向下，保持不动。", 6f),
                PoseHold("sidebridge_left_start",
                    "向左侧卧：左前臂撑地、肘在肩正下方、前臂指向身体前方；右手放在右侧大腿上；屈膝，髋部先不抬起，保持不动。", 8f)
            };
        }

        bool parallelShanks = lower.Contains("birddog") || lower.Contains("deadbug");
        string startPose = lower.Contains("birddog")
            ? "擺好鳥狗四點跪姿，雙小腿保持平行。"
            : lower.Contains("deadbug")
                ? "仰臥並屈膝抬起雙腿，擺好死蟲起始姿勢。"
                : lower.Contains("sidebridge")
                    ? "擺好側橋起始姿勢。"
                    : lower.Contains("hipthrust")
                        ? "躺好並擺出臀橋起始姿勢。"
                        : "擺好本遊戲的起始姿勢。";

        return new List<Step>
        {
            new Step { kind = StepKind.Standing, instruction = "站直，雙腳朝前，雙手自然下垂並保持不動。", seconds = 5f },
            new Step { kind = StepKind.ArmsForward, instruction = "雙臂向前平舉至肩高，手臂伸直、掌心向下並保持不動。", seconds = 5f },
            new Step { kind = StepKind.StartPose, instruction = startPose, seconds = 10f, captureParallelShanks = parallelShanks }
        };
    }
}
