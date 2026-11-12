using System;
using System.IO;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// Receives MobilePose's 24-joint SMPL stream and retargets it onto a Humanoid
/// Animator so the existing game scoring reads the live predicted pose.
/// </summary>
[DisallowMultipleComponent]
public sealed class MobilePoseWebSocketSource : MonoBehaviour
{
    [SerializeField] private Animator animator;
    [SerializeField] private string host = "127.0.0.1";
    [SerializeField, Min(1)] private int port = 21200;
    [SerializeField] private bool applyReceivedPose = true;
    [SerializeField] private bool retargetThroughGlobalBindPose = true;
    [SerializeField, Min(0)] private int displayBufferFrames = 4;
    [SerializeField] private bool showDiagnostics = true;
    [SerializeField] private string selectedModel = "waist_mobileposer_amass";
    [Tooltip("Recording to replay: file name under realUse/recordings or an absolute path; empty = newest.")]
    [SerializeField] private string replayPath = "";
    [Tooltip("Start/stop an input recording with every scored game session (file named after the scene).")]
    [SerializeField] private bool autoRecordSessions = true;
    [Tooltip("Move the avatar up/down so its lowest bone touches the floor. Off disables the grounding component.")]
    [SerializeField] private bool groundAvatar;
    // Reset the waist heading before scene-specific calibration so limb directions share a stable frame.
    [Tooltip("Ask Python to reset yaw before the scene-specific calibration phase.")]
    [SerializeField] private bool autoYawReset = true;
    [Tooltip("Enable sensor-based calibration steps.")]
    [SerializeField] private bool poseCalibration = true;
    [Tooltip("Use the measured orientation for the sensed bones (forearms, shanks, waist) in the output.")]
    [SerializeField] private bool anchorSensedBones;

    private const int JointCount = 24;
    private static readonly HumanBodyBones[] JointToBone =
    {
        HumanBodyBones.Hips, HumanBodyBones.LeftUpperLeg, HumanBodyBones.RightUpperLeg,
        HumanBodyBones.Spine, HumanBodyBones.LeftLowerLeg, HumanBodyBones.RightLowerLeg,
        HumanBodyBones.Chest, HumanBodyBones.LeftFoot, HumanBodyBones.RightFoot,
        HumanBodyBones.UpperChest, HumanBodyBones.LeftToes, HumanBodyBones.RightToes,
        HumanBodyBones.Neck, HumanBodyBones.LeftShoulder, HumanBodyBones.RightShoulder,
        HumanBodyBones.Head, HumanBodyBones.LeftUpperArm, HumanBodyBones.RightUpperArm,
        HumanBodyBones.LeftLowerArm, HumanBodyBones.RightLowerArm, HumanBodyBones.LeftHand,
        HumanBodyBones.RightHand, HumanBodyBones.LastBone, HumanBodyBones.LastBone
    };
    private static readonly int[] JointParents =
    {
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
        9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
    };

    private readonly Transform[] bones = new Transform[JointCount];
    private readonly Quaternion[] restWorldRotations = new Quaternion[JointCount];
    private readonly Quaternion[] predictedWorldRotations = new Quaternion[JointCount];
    private Quaternion leftShankOffset = Quaternion.identity;
    private Quaternion rightShankOffset = Quaternion.identity;
    private bool shankCorrectionCaptured;
    private readonly object sync = new object();
    private readonly System.Collections.Generic.List<PoseSample> poseBuffer = new System.Collections.Generic.List<PoseSample>(16);
    private CancellationTokenSource cancellation;
    private float[] latestJoints;
    private int latestFrame = -1;
    private int latestModelFrame = -1;
    private int latestInferEvery = 2;
    private long latestPoseReceivedTimestamp;
    private float playbackModelFrame = float.NegativeInfinity;
    private string status = "Waiting for MobilePose";
    private AvatarMotionSource motionSource;
    private int reportedFrame = -1;
    private float lastPacketRealtime;
    private ClientWebSocket activeSocket;
    private readonly SemaphoreSlim sendLock = new SemaphoreSlim(1, 1);
    private string serverModel = "not reported";
    private string inputSource = "";
    private string replayFile = "";
    private string recordingFile = "";
    private float recordingSeconds;
    private string controlMessage = "";
    private float armResultUntil;
    private AutoGroundRootFromAvatar avatarGrounding;

    // Panel choices survive scene reloads ("play again") within one run; otherwise every reload would
    // silently send the scene's default model to Python.
    private static string rememberedModel;
    private static string rememberedReplayPath;
    private static bool? rememberedAutoRecord;
    private static bool? rememberedGrounding;
    private static bool? rememberedYawReset;
    private static bool? rememberedPoseCalibration;
    private static bool? rememberedAnchor;

    // Sensor label -> SMPL joint of the bone it measures (same order Python uses).

    private static readonly string[] ModelIds =
    {
        "waist_mobileposer_amass", "waist_mobileposer_deadbug",
        "waist_mobileposer_ours_all10", "waist_mobileposer_ours_deadbug",
        "waist_streamposer_amass", "waist_streamposer_deadbug",
        "waist_streamposer_ours_all10", "waist_streamposer_ours_deadbug",
        "waist_line14_amass", "waist_line14_deadbug",
        "waist_line14_ours_all10", "waist_line14_ours_deadbug",
        "waist_mobileposer_ours_offsetfree", "waist_streamposer_ours_offsetfree",
        "waist_line14_ours_offsetfree",
        "waist_imudino_fiveimu", "waist_imudino_stream_wristankle",
        "waist_imudino_stream_wristankle_limbmix"
    };
    private static readonly string[] ModelLabels =
    {
        "MobilePoser / AMASS", "MobilePoser / Deadbug",
        "MobilePoser / Ours 10-action", "MobilePoser / Deadbug tuned",
        "StreamPoser / AMASS", "StreamPoser / Deadbug",
        "StreamPoser / Ours 10-action", "StreamPoser / Deadbug tuned",
        "Line14 / AMASS", "Line14 / Deadbug",
        "Line14 / Ours 10-action", "Line14 / Deadbug tuned",
        "MobilePoser / Ours offset-free", "StreamPoser / Ours offset-free",
        "Line14 / Ours offset-free",
        "IMUDINO / Five IMU (AMASS + BirdDog)", "IMUDINO Stream / wrist+ankle",
        "IMUDINO Stream / wrist+ankle limb-mix"
    };

    public bool IsConnected { get; private set; }
    public int LatestFrame { get { lock (sync) return latestFrame; } }
    public string Status { get { lock (sync) return status; } }
    public string LatestControlMessage { get { lock (sync) return controlMessage; } }

    private void Awake()
    {
        if (rememberedModel != null) selectedModel = rememberedModel;
        if (rememberedReplayPath != null) replayPath = rememberedReplayPath;
        if (rememberedAutoRecord.HasValue) autoRecordSessions = rememberedAutoRecord.Value;
        if (rememberedGrounding.HasValue) groundAvatar = rememberedGrounding.Value;
        if (rememberedYawReset.HasValue) autoYawReset = rememberedYawReset.Value;
        if (rememberedPoseCalibration.HasValue) poseCalibration = rememberedPoseCalibration.Value;
        if (rememberedAnchor.HasValue) anchorSensedBones = rememberedAnchor.Value;
        if (animator == null)
        {
            DeadBugGamingPoseScorer scorer = FindObjectOfType<DeadBugGamingPoseScorer>(true);
            animator = scorer != null ? scorer.UserAnimator : null;
        }
        CacheBones();
        ResolveGrounding();
        motionSource = GetComponent<AvatarMotionSource>();
        if (motionSource == null) motionSource = gameObject.AddComponent<AvatarMotionSource>();
        if (animator != null) motionSource.Configure(animator);
        motionSource.SetAvailable(false);
    }

    private void Start()
    {
    }

    private void ResolveGrounding()
    {
        AutoGroundRootFromAvatar[] components = FindObjectsOfType<AutoGroundRootFromAvatar>(true);
        foreach (AutoGroundRootFromAvatar component in components)
        {
            if (animator == null || component.transform == animator.transform ||
                component.transform.IsChildOf(animator.transform) || animator.transform.IsChildOf(component.transform))
            {
                avatarGrounding = component;
                break;
            }
        }
        if (avatarGrounding != null) avatarGrounding.enabled = groundAvatar;
    }

    public Animator UserAnimator => animator;
    public void ConfigureRuntime(Animator target)
    {
        if (target == null || animator == target) return;
        animator = target;
        CacheBones();
        if (motionSource == null) motionSource = GetComponent<AvatarMotionSource>();
        if (motionSource == null) motionSource = gameObject.AddComponent<AvatarMotionSource>();
        motionSource.Configure(animator);
        ResolveGrounding();
    }
    public bool AutoRecordSessions => autoRecordSessions;
    public bool IsReplaying => Replaying();
    public bool IsPoseCalibrationEnabled => poseCalibration;
    public bool IsYawResetEnabled => autoYawReset;
    public void SendCalibrationCommand(string json) => SendControl(json);
    public void SendSessionCommand(string json) => SendControl(json);
    public void SendSessionCommands(params string[] json) => SendControls(json);
    public string CurrentSceneName => SceneManager.GetActiveScene().name;
    public void SetGroundingEnabled(bool enabled)
    {
        groundAvatar = enabled;
        rememberedGrounding = enabled;
        if (avatarGrounding == null) ResolveGrounding();
        if (avatarGrounding != null) avatarGrounding.enabled = enabled;
    }
    public void ShowCalibrationResultForSeconds(float seconds) => armResultUntil = Time.unscaledTime + seconds;

    public bool CaptureParallelShankCorrection()
    {
        if (LatestFrame < 0 || bones[4] == null || bones[5] == null) return false;
        Quaternion mean = Quaternion.Slerp(predictedWorldRotations[4], predictedWorldRotations[5], 0.5f);
        leftShankOffset = mean * Quaternion.Inverse(predictedWorldRotations[4]);
        rightShankOffset = mean * Quaternion.Inverse(predictedWorldRotations[5]);
        shankCorrectionCaptured = true;
        return true;
    }

    private void OnDestroy()
    {
    }

    private bool Replaying()
    {
        lock (sync) return inputSource == "replay";
    }

    /*
    // The set begins with the start countdown: record from here so the standing calibration window
    // is in the file, and yaw-reset with 1 s left so that window is already after the reset.
    private void Update()
    {
        bool counting = false;
        if (counting && !countdownWasRunning)
        {
            yawResetThisCountdown = false;
            if (autoRecordSessions && !Replaying() && poseCalibration)
            {
                SendControls("{\"command\":\"stopRecording\"}",
                    "{\"command\":\"startRecording\",\"name\":\"" + EscapeJson(SceneManager.GetActiveScene().name) + "\"}");
                autoRecording = true;
                recordingThisSet = true;
            }
        }
        if (counting && autoYawReset && !yawResetThisCountdown && !Replaying())
        {
            yawResetThisCountdown = true;
            SendControl("{\"command\":\"yawReset\"}");
        }
        countdownWasRunning = counting;
    }

    private void ApplyStartPoseHold()
    {
        if (coachController != null)
        {
        }
    }

    /// <summary>Large instruction banner for the two moments whose pose the calibration relies on.</summary>
    private void DrawCalibrationBanner()
    {
        string text = null;
        if (false)
            text = autoYawReset
                ? ""
                : "";
        else if (false)
            text = null;
        else if (Time.unscaledTime < armResultUntil)
        {
            string message;
            lock (sync) message = controlMessage;
            if (message.StartsWith("arm calibration:"))
                text = (message.Contains("not level") || message.Contains("moving") || message.Contains("rejected")
                    ? "前臂未完全校准（手臂要伸平、停稳）\n" : "前臂已校准\n") + message.Substring(16).Trim();
        }
        if (false)
            text = null;
        if (text == null) return;
        if (bannerStyle == null)
        {
            bannerStyle = new GUIStyle(GUI.skin.box) { fontSize = 34, alignment = TextAnchor.MiddleCenter, wordWrap = true };
            bannerStyle.normal.textColor = Color.white;
        }
        float width = Mathf.Min(Screen.width - 40f, 900f);
        GUI.Box(new Rect((Screen.width - width) / 2f, 40f, width, 130f), text, bannerStyle);
    }

    */

    private string PoseOptionsMessage()
    {
        return "{\"command\":\"setPoseOptions\",\"calibration\":" + (poseCalibration ? "true" : "false") +
               ",\"anchor\":" + (anchorSensedBones ? "true" : "false") + "}";
    }

    private void OnEnable()
    {
        cancellation = new CancellationTokenSource();
        _ = RunConnectionLoopAsync(cancellation.Token);
    }

    private void OnDisable()
    {
        cancellation?.Cancel();
        cancellation?.Dispose();
        cancellation = null;
        activeSocket = null;
        IsConnected = false;
        motionSource?.SetAvailable(false);
    }

    private bool CacheBones()
    {
        if (animator == null || !animator.isHuman)
        {
            SetStatus("User Animator missing or not Humanoid");
            Debug.LogError("MobilePose requires the Humanoid Animator used by the game scorer.", this);
            return false;
        }

        int mappedBoneCount = 0;
        for (int i = 0; i < JointCount; i++)
        {
            if (JointToBone[i] == HumanBodyBones.LastBone) continue;
            bones[i] = animator.GetBoneTransform(JointToBone[i]);
            if (bones[i] == null)
            {
                // Humanoid rigs may legitimately omit optional bones such as
                // UpperChest, shoulders, or toes. Keep caching later joints so
                // one missing torso bone cannot disable both arms.
                Debug.LogWarning("MobilePose Humanoid rig is missing optional bone " +
                    JointToBone[i] + "; remaining mapped bones will still be driven.", this);
                continue;
            }
            restWorldRotations[i] = bones[i].rotation;
            mappedBoneCount++;
        }
        return mappedBoneCount > 0;
    }

    private void LateUpdate()
    {
        float[] joints;
        lock (sync)
        {
            // Window IMUDINO needs a deep buffer; the stateful stream models emit continuous
            // poses, so only cover one output interval plus one frame of packet jitter.
            int delay = serverModel == null || !serverModel.StartsWith("waist_imudino") ? 0
                : serverModel.StartsWith("waist_imudino_stream_") ? latestInferEvery + 1
                : Mathf.Max(displayBufferFrames, latestInferEvery + 1);
            double elapsed = latestPoseReceivedTimestamp == 0 ? 0.0
                : (double)(System.Diagnostics.Stopwatch.GetTimestamp() - latestPoseReceivedTimestamp)
                    / System.Diagnostics.Stopwatch.Frequency;
            float target = (float)(latestModelFrame - delay + elapsed * 60.0);
            // Packet jitter must never rewind playback into a previous pose.
            playbackModelFrame = Mathf.Max(playbackModelFrame, Mathf.Min(latestModelFrame, target));
            joints = delay == 0 ? latestJoints : BufferedPoseAt(playbackModelFrame);
        }
        if (!applyReceivedPose || joints == null || joints.Length != JointCount * 4) return;

        for (int i = 0; i < JointCount; i++)
        {
            int offset = i * 4;
            Quaternion local = new Quaternion(joints[offset], joints[offset + 1],
                joints[offset + 2], joints[offset + 3]);
            float norm = local.x * local.x + local.y * local.y + local.z * local.z + local.w * local.w;
            if (float.IsNaN(norm) || float.IsInfinity(norm) || norm < 0.000001f)
            {
                predictedWorldRotations[i] = JointParents[i] < 0
                    ? Quaternion.identity : predictedWorldRotations[JointParents[i]];
                continue;
            }

            local.Normalize();
            int parent = JointParents[i];
            predictedWorldRotations[i] = parent < 0
                ? local : predictedWorldRotations[parent] * local;
        }

        if (shankCorrectionCaptured)
        {
            predictedWorldRotations[4] = leftShankOffset * predictedWorldRotations[4];
            predictedWorldRotations[7] = leftShankOffset * predictedWorldRotations[7];
            predictedWorldRotations[10] = leftShankOffset * predictedWorldRotations[10];
            predictedWorldRotations[5] = rightShankOffset * predictedWorldRotations[5];
            predictedWorldRotations[8] = rightShankOffset * predictedWorldRotations[8];
            predictedWorldRotations[11] = rightShankOffset * predictedWorldRotations[11];
        }

        for (int i = 0; i < JointCount; i++)
        {
            if (bones[i] == null) continue;
            bones[i].rotation = retargetThroughGlobalBindPose
                ? predictedWorldRotations[i] * restWorldRotations[i]
                : i == 0 ? predictedWorldRotations[i]
                : bones[i].parent.rotation * predictedWorldRotations[i];
        }
        if (latestFrame != reportedFrame)
        {
            reportedFrame = latestFrame;
            lastPacketRealtime = Time.unscaledTime;
            motionSource?.ReportFrame(latestFrame, JointCount);
        }
        else if (Time.unscaledTime - lastPacketRealtime > 0.25f)
        {
            motionSource?.SetAvailable(false);
        }
    }

    private float[] BufferedPoseAt(float targetFrame)
    {
        if (poseBuffer.Count == 0) return latestJoints;
        if (poseBuffer.Count == 1 || targetFrame <= poseBuffer[0].modelFrame) return poseBuffer[0].joints;

        for (int i = 1; i < poseBuffer.Count; i++)
        {
            PoseSample right = poseBuffer[i];
            if (targetFrame > right.modelFrame) continue;
            PoseSample left = poseBuffer[i - 1];
            float span = right.modelFrame - left.modelFrame;
            if (span <= 0f) return right.joints;
            float amount = Mathf.Clamp01((targetFrame - left.modelFrame) / span);
            var result = new float[JointCount * 4];
            for (int joint = 0; joint < JointCount; joint++)
            {
                int offset = joint * 4;
                Quaternion a = new Quaternion(left.joints[offset], left.joints[offset + 1],
                    left.joints[offset + 2], left.joints[offset + 3]);
                Quaternion b = new Quaternion(right.joints[offset], right.joints[offset + 1],
                    right.joints[offset + 2], right.joints[offset + 3]);
                Quaternion q = Quaternion.Slerp(a, b, amount);
                result[offset] = q.x;
                result[offset + 1] = q.y;
                result[offset + 2] = q.z;
                result[offset + 3] = q.w;
            }
            return result;
        }
        return poseBuffer[poseBuffer.Count - 1].joints;
    }

    private void OnGUI()
    {
        if (!showDiagnostics) return;
        GUI.Box(new Rect(12f, 12f, 390f, 650f), GUIContent.none);
        GUILayout.BeginArea(new Rect(22f, 18f, 370f, 640f));
        GUILayout.Label("实时动作模型  |  " + Status);
        GUILayout.Label("Frame " + LatestFrame + "  |  Python: " + serverModel);
        GUILayout.Space(5f);
        for (int row = 0; row < (ModelIds.Length + 1) / 2; row++)
        {
            GUILayout.BeginHorizontal();
            for (int column = 0; column < 2; column++)
            {
                int index = row * 2 + column;
                if (index >= ModelIds.Length) break;
                string label = (selectedModel == ModelIds[index] ? "[x] " : "") + ModelLabels[index];
                if (GUILayout.Button(label, GUILayout.Height(31f))) SelectModel(ModelIds[index]);
            }
            GUILayout.EndHorizontal();
        }
        DrawInputControls();
        DrawGameModeControls();
        GUILayout.EndArea();
    }

    private void DrawInputControls()
    {
        string source, replaying, recording, message;
        float seconds;
        lock (sync)
        {
            source = inputSource;
            replaying = replayFile;
            recording = recordingFile;
            seconds = recordingSeconds;
            message = controlMessage;
        }
        bool isReplay = source == "replay";
        bool isRecording = !string.IsNullOrEmpty(recording);

        GUILayout.Space(6f);
        GUILayout.Label(isReplay ? "输入: 回放 " + replaying : "输入: 实时 IMU");
        GUILayout.BeginHorizontal();
        GUI.enabled = IsConnected && !isReplay;
        if (GUILayout.Button(isRecording ? $"■ 停止录制 ({seconds:0.0}s)" : "● 开始录制", GUILayout.Height(31f)))
            SendControl(isRecording ? "{\"command\":\"stopRecording\"}" : "{\"command\":\"startRecording\"}");
        GUI.enabled = IsConnected;
        if (GUILayout.Button(isReplay ? "↺ 重新回放" : "▶ 回放", GUILayout.Height(31f)))
            SendControl("{\"command\":\"startReplay\",\"path\":\"" + EscapeJson(replayPath) + "\"}");
        GUI.enabled = IsConnected && isReplay;
        if (GUILayout.Button("回到实时", GUILayout.Height(31f)))
            SendControl("{\"command\":\"stopReplay\"}");
        GUI.enabled = true;
        GUILayout.EndHorizontal();
        GUILayout.BeginHorizontal();
        GUILayout.Label("回放文件", GUILayout.Width(56f));
        replayPath = GUILayout.TextField(replayPath ?? "");
        rememberedReplayPath = replayPath;
        GUILayout.EndHorizontal();
        GUILayout.Label(string.IsNullOrEmpty(replayPath) ? "（留空 = 最新一次录制）" : "（文件名或绝对路径）");
        if (!string.IsNullOrEmpty(message)) GUILayout.Label(message);
    }

    private void DrawGameModeControls()
    {
        GUILayout.Space(4f);
        autoRecordSessions = GUILayout.Toggle(autoRecordSessions, " 每局自动录制输入");
        rememberedAutoRecord = autoRecordSessions;
        bool grounded = GUILayout.Toggle(groundAvatar, " 人物贴地");
        if (grounded != groundAvatar) SetGroundingEnabled(grounded);
        autoYawReset = GUILayout.Toggle(autoYawReset, " 每组开始时自动 yaw reset（站直面向屏幕）");
        rememberedYawReset = autoYawReset;
        bool calibrate = GUILayout.Toggle(poseCalibration, "启用传感器校准步骤");
        bool anchor = GUILayout.Toggle(anchorSensedBones, " 传感器骨骼直接用测量");
        if (calibrate != poseCalibration || anchor != anchorSensedBones)
        {
            poseCalibration = calibrate;
            anchorSensedBones = anchor;
            SendControl(PoseOptionsMessage());
        }
        rememberedPoseCalibration = poseCalibration;
        rememberedAnchor = anchorSensedBones;
        if (avatarGrounding != null)
        {
            bool enabled = GUILayout.Toggle(groundAvatar, "人物贴地");
            if (enabled != groundAvatar) SetGroundingEnabled(enabled);
        }
        if (Time.unscaledTime < armResultUntil)
        {
            string result;
            lock (sync) result = controlMessage;
            if (result.StartsWith("arm calibration:")) GUILayout.Label(result);
        }
    }

    private static string EscapeJson(string value)
    {
        return (value ?? "").Trim().Replace("\\", "\\\\").Replace("\"", "\\\"");
    }

    private async Task RunConnectionLoopAsync(CancellationToken token)
    {
        string url = $"ws://{host}:{port}";
        while (!token.IsCancellationRequested)
        {
            try
            {
                using (var socket = new ClientWebSocket())
                {
                    await socket.ConnectAsync(new Uri(url), token).ConfigureAwait(false);
                    activeSocket = socket;
                    IsConnected = true;
                    SetStatus("Connected to " + url);
                    await SendModelSelectionAsync(socket, selectedModel, token).ConfigureAwait(false);
                    await SendTextAsync(socket, PoseOptionsMessage(), token).ConfigureAwait(false);
                    await ReceiveLoopAsync(socket, token).ConfigureAwait(false);
                    activeSocket = null;
                }
                IsConnected = false;
                motionSource?.SetAvailable(false);
            }
            catch (OperationCanceledException) when (token.IsCancellationRequested) { return; }
            catch (Exception exception)
            {
                IsConnected = false;
                motionSource?.SetAvailable(false);
                SetStatus("Connection failed: " + exception.Message);
            }

            try { await Task.Delay(1000, token).ConfigureAwait(false); }
            catch (OperationCanceledException) { return; }
        }
    }

    private async Task ReceiveLoopAsync(ClientWebSocket socket, CancellationToken token)
    {
        byte[] buffer = new byte[65536];
        while (!token.IsCancellationRequested && socket.State == WebSocketState.Open)
        {
            using (var message = new MemoryStream())
            {
                WebSocketReceiveResult result;
                do
                {
                    result = await socket.ReceiveAsync(new ArraySegment<byte>(buffer), token).ConfigureAwait(false);
                    if (result.MessageType == WebSocketMessageType.Close) return;
                    message.Write(buffer, 0, result.Count);
                } while (!result.EndOfMessage);

                if (result.MessageType != WebSocketMessageType.Text) continue;
                ProcessMessage(Encoding.UTF8.GetString(message.ToArray()));
            }
        }
    }

    private void ProcessMessage(string json)
    {
        try
        {
            PoseFrame frame = JsonUtility.FromJson<PoseFrame>(json);
            if (frame == null || frame.joints == null || frame.joints.Length != JointCount * 4) return;
            lock (sync)
            {
                if (frame.frame <= latestFrame
                    || (!string.IsNullOrEmpty(frame.model) && frame.model != serverModel)
                    || (frame.inputSource ?? "") != inputSource
                    || (frame.replayFile ?? "") != replayFile)
                {
                    poseBuffer.Clear();
                    playbackModelFrame = float.NegativeInfinity;
                }
                latestJoints = frame.joints;
                latestFrame = frame.frame;
                latestModelFrame = frame.modelFps > 0 ? frame.modelOutputRepresentsFrame : frame.frame;
                latestInferEvery = Mathf.Max(1, frame.inferEvery);
                latestPoseReceivedTimestamp = System.Diagnostics.Stopwatch.GetTimestamp();
                if (poseBuffer.Count == 0 || latestModelFrame > poseBuffer[poseBuffer.Count - 1].modelFrame)
                {
                    poseBuffer.Add(new PoseSample(latestModelFrame, frame.joints));
                    if (poseBuffer.Count > 32) poseBuffer.RemoveAt(0);
                }
                if (!string.IsNullOrEmpty(frame.model))
                {
                    serverModel = frame.model;
                    if (Array.IndexOf(ModelIds, frame.model) >= 0)
                        selectedModel = rememberedModel = frame.model;
                }
                inputSource = frame.inputSource ?? "";
                replayFile = frame.replayFile ?? "";
                recordingFile = frame.recordingFile ?? "";
                recordingSeconds = frame.recordingSeconds;
                controlMessage = frame.controlMessage ?? "";
                status = "Receiving MobilePose";
            }
        }
        catch (Exception exception)
        {
            SetStatus("Invalid pose frame: " + exception.Message);
        }
    }

    private void SetStatus(string value)
    {
        lock (sync) status = value;
    }

    [Serializable]
    private sealed class PoseFrame
    {
        public int frame;
        public int modelOutputRepresentsFrame;
        public int modelFps;
        public int inferEvery;
        public float[] joints;
        public string model;
        public string inputSource;
        public string replayFile;
        public string recordingFile;
        public float recordingSeconds;
        public string controlMessage;
    }

    private sealed class PoseSample
    {
        public readonly int modelFrame;
        public readonly float[] joints;

        public PoseSample(int modelFrame, float[] joints)
        {
            this.modelFrame = modelFrame;
            this.joints = joints;
        }
    }

    private void SelectModel(string modelId)
    {
        selectedModel = modelId;
        rememberedModel = modelId;
        ClientWebSocket socket = activeSocket;
        if (socket != null && cancellation != null)
            _ = SendModelSelectionAsync(socket, modelId, cancellation.Token);
    }

    private void SendControl(string json)
    {
        ClientWebSocket socket = activeSocket;
        if (socket != null && cancellation != null)
            _ = SendTextAsync(socket, json, cancellation.Token);
    }

    private void SendControls(params string[] messages)
    {
        ClientWebSocket socket = activeSocket;
        if (socket != null && cancellation != null)
            _ = SendInOrderAsync(socket, messages, cancellation.Token);
    }

    private async Task SendInOrderAsync(ClientWebSocket socket, string[] messages, CancellationToken token)
    {
        foreach (string message in messages)
            await SendTextAsync(socket, message, token).ConfigureAwait(false);
    }

    private Task SendModelSelectionAsync(ClientWebSocket socket, string modelId, CancellationToken token)
    {
        return SendTextAsync(socket, "{\"selectModel\":\"" + modelId + "\"}", token);
    }

    private async Task SendTextAsync(ClientWebSocket socket, string json, CancellationToken token)
    {
        if (socket == null || socket.State != WebSocketState.Open) return;
        byte[] payload = Encoding.UTF8.GetBytes(json);
        await sendLock.WaitAsync(token).ConfigureAwait(false);
        try
        {
            if (socket.State == WebSocketState.Open)
                await socket.SendAsync(new ArraySegment<byte>(payload), WebSocketMessageType.Text, true, token)
                    .ConfigureAwait(false);
        }
        catch (OperationCanceledException) { }
        catch (WebSocketException exception) { SetStatus("Send to Python failed: " + exception.Message); }
        finally { sendLock.Release(); }
    }
}
