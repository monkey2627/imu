using System.Reflection;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// Installs the EVMC4U receiver only for the playable game scenes.
/// The receiver drives the existing user Animator and reports fresh VMC
/// packets through the project's neutral AvatarMotionSource boundary.
/// </summary>
[DefaultExecutionOrder(-1000)]
public sealed class VmcGameModeBootstrap : MonoBehaviour
{
    private const int VmcInputPort = 39539;
    private const string PreviewCameraResource = "MobilePose User Preview Camera";
    private const string PreviewCanvasResource = "MobilePose User Preview";
    private bool mobilePoseSetupPending;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void RegisterSceneHook()
    {
        SceneManager.sceneLoaded -= InstallForScene;
        SceneManager.sceneLoaded += InstallForScene;
    }

    private static void InstallForScene(Scene scene, LoadSceneMode mode)
    {
        Animator userAnimator = FindUserAnimator();
        bool hasRhythmMode = FindObjectOfType<RhythmModeController>(true) != null;
        bool hasRecordedGameScorer = FindObjectOfType<DeadBugGamingPoseScorer>(true) != null;
        bool useMobilePose = hasRecordedGameScorer || scene.name.Contains("RingBridge");
        VmcGameModeBootstrap existingBootstrap =
            FindObjectOfType<VmcGameModeBootstrap>(true);
        if (!hasRhythmMode && !hasRecordedGameScorer)
        {
            return;
        }

        if (existingBootstrap != null)
        {
            // A scene-serialized receiver can turn into a missing script if its
            // .meta GUID changes; without this the avatar silently gets no pose.
            if (useMobilePose &&
                FindObjectOfType<MobilePoseWebSocketSource>(true) == null)
            {
                Debug.LogWarning("MobilePoseWebSocketSource missing from scene; adding it at runtime.");
                existingBootstrap.gameObject.AddComponent<MobilePoseWebSocketSource>();
                if (FindObjectOfType<MobilePoseUserDisplay>(true) == null)
                {
                    existingBootstrap.gameObject.AddComponent<MobilePoseUserDisplay>();
                }
            }
            if (useMobilePose)
            {
                existingBootstrap.mobilePoseSetupPending = true;
                EnsureMobilePoseModules(existingBootstrap.gameObject);
                if (userAnimator != null) existingBootstrap.TrySetupMobilePosePreview(userAnimator);
            }
            EnsureHitHaptics(existingBootstrap.gameObject);
            return;
        }

        GameObject root = new GameObject("VMC Game Mode Receiver");
        root.AddComponent<VmcGameModeBootstrap>();
        if (useMobilePose)
        {
            VmcGameModeBootstrap bootstrap = root.GetComponent<VmcGameModeBootstrap>();
            bootstrap.mobilePoseSetupPending = true;
            MobilePoseWebSocketSource mobilePose = root.AddComponent<MobilePoseWebSocketSource>();
            if (userAnimator != null) mobilePose.ConfigureRuntime(userAnimator);
            EnsureMobilePoseModules(root);
            if (userAnimator != null) bootstrap.TrySetupMobilePosePreview(userAnimator);
            // The receiver resolves the scorer's user Animator in Awake. Keep
            // this branch isolated from the legacy OSC/VMC input path.
            mobilePose.name = "MobilePose WebSocket Receiver";
        }
        EnsureHitHaptics(root);
    }

    private void Update()
    {
        if (!mobilePoseSetupPending) return;
        Animator userAnimator = FindUserAnimator();
        if (userAnimator == null) return;

        MobilePoseWebSocketSource mobilePose = GetComponent<MobilePoseWebSocketSource>();
        if (mobilePose == null) mobilePose = gameObject.AddComponent<MobilePoseWebSocketSource>();
        mobilePose.ConfigureRuntime(userAnimator);
        EnsureMobilePoseModules(gameObject);
        TrySetupMobilePosePreview(userAnimator);
        EnsureHitHaptics(gameObject);
        mobilePoseSetupPending = false;
    }

    private void TrySetupMobilePosePreview(Animator userAnimator)
    {
        if (userAnimator == null) return;
        MobilePoseWebSocketSource mobilePose = GetComponent<MobilePoseWebSocketSource>();
        if (mobilePose != null) mobilePose.ConfigureRuntime(userAnimator);
        EnsureMobilePosePreview(userAnimator);
    }

    private static void EnsureMobilePosePreview(Animator userAnimator)
    {
        Camera previewCamera = FindNamedObject("MobilePose User Preview Camera")?.GetComponent<Camera>();
        if (previewCamera == null)
        {
            GameObject prefab = Resources.Load<GameObject>(PreviewCameraResource);
            if (prefab != null)
            {
                GameObject instance = Instantiate(prefab);
                instance.name = "MobilePose User Preview Camera";
                previewCamera = instance.GetComponent<Camera>();
            }
            else
            {
                Debug.LogError($"Missing Resources prefab '{PreviewCameraResource}'.");
            }
        }

        Canvas previewCanvas = FindNamedObject("MobilePose User Preview")?.GetComponent<Canvas>();
        if (previewCanvas == null)
        {
            GameObject prefab = Resources.Load<GameObject>(PreviewCanvasResource);
            if (prefab != null)
            {
                GameObject instance = Instantiate(prefab);
                instance.name = "MobilePose User Preview";
                previewCanvas = instance.GetComponent<Canvas>();
            }
            else
            {
                Debug.LogError($"Missing Resources prefab '{PreviewCanvasResource}'.");
            }
        }

        if (previewCamera != null)
        {
            HipThrustPreviewFraming framing = previewCamera.GetComponent<HipThrustPreviewFraming>();
            if (framing != null)
            {
                framing.avatar = userAnimator;
                framing.previewCamera = previewCamera;
            }
        }

        MobilePoseUserDisplay display = FindObjectOfType<MobilePoseUserDisplay>(true);
        if (display == null)
        {
            GameObject root = FindObjectOfType<VmcGameModeBootstrap>(true).gameObject;
            display = root.AddComponent<MobilePoseUserDisplay>();
        }
        display.Configure(userAnimator);
    }

    private static GameObject FindNamedObject(string objectName)
    {
        foreach (Transform candidate in FindObjectsOfType<Transform>(true))
        {
            if (candidate.name == objectName) return candidate.gameObject;
        }
        return null;
    }

    private static void EnsureMobilePoseModules(GameObject root)
    {
        if (FindObjectOfType<MobilePoseCalibrationController>(true) == null)
            root.AddComponent<MobilePoseCalibrationController>();
        if (FindObjectOfType<MobilePoseSessionRecorder>(true) == null)
            root.AddComponent<MobilePoseSessionRecorder>();
    }

    private static void EnsureHitHaptics(GameObject sharedRoot)
    {
        if (FindObjectOfType<CoachHapticFeedbackController>(true) == null)
        {
            sharedRoot.AddComponent<CoachHapticFeedbackController>();
        }

        RhythmModeController rhythmController =
            FindObjectOfType<RhythmModeController>(true);
        if (rhythmController != null &&
            FindObjectOfType<RhythmHitHapticAdapter>(true) == null)
        {
            rhythmController.gameObject.AddComponent<RhythmHitHapticAdapter>();
        }
    }

    private void Awake()
    {
        Animator userAnimator = FindUserAnimator();
        if (userAnimator == null)
        {
            return;
        }

        if (FindObjectOfType<DeadBugGamingPoseScorer>(true) != null)
        {
            // Recorded coaching scenes use the Python MobilePose stream, even
            // when their rhythm presentation also has a RhythmModeController.
            // Do not start a second OSC receiver that could overwrite the same
            // Animator.
            return;
        }

        EVMC4U.ExternalReceiver receiver =
            gameObject.AddComponent<EVMC4U.ExternalReceiver>();
        uOSC.uOscServer server = gameObject.AddComponent<uOSC.uOscServer>();
        AvatarMotionSource source = gameObject.AddComponent<AvatarMotionSource>();
        VmcGameModeSourceBridge bridge =
            gameObject.AddComponent<VmcGameModeSourceBridge>();

        server.port = VmcInputPort;
        server.autoStart = true;
        receiver.Model = userAnimator.gameObject;
        receiver.RootPositionSynchronize = false;
        receiver.RootRotationSynchronize = false;
        receiver.RootScaleOffsetSynchronize = false;
        // VMC positions use the sender's skeleton scale. Applying them to a
        // different VRM stretches the avatar, so this receiver is rotation-only.
        receiver.BonePositionSynchronize = false;
        receiver.BonePositionFilterEnable = false;
        receiver.BoneRotationFilterEnable = false;
        receiver.CutBonesEnable = true;
        receiver.CutBoneHips = false;
        receiver.CutBoneSpine = false;
        receiver.CutBoneChest = false;
        receiver.CutBoneUpperChest = false;
        receiver.CutBoneLeftUpperLeg = false;
        receiver.CutBoneLeftLowerLeg = false;
        receiver.CutBoneLeftFoot = false;
        receiver.CutBoneLeftToes = false;
        receiver.CutBoneRightUpperLeg = false;
        receiver.CutBoneRightLowerLeg = false;
        receiver.CutBoneRightFoot = false;
        receiver.CutBoneRightToes = false;
        receiver.CutBoneNeck = true;
        receiver.CutBoneHead = true;
        receiver.CutBoneLeftEye = true;
        receiver.CutBoneRightEye = true;
        receiver.CutBoneJaw = true;
        source.Configure(userAnimator);
        bridge.Configure(receiver, source);
    }

    private static Animator FindUserAnimator()
    {
        PoseScorer scorer = FindObjectOfType<PoseScorer>(true);
        if (scorer != null)
        {
            FieldInfo field = typeof(PoseScorer).GetField(
                "userAnimator",
                BindingFlags.Instance | BindingFlags.NonPublic);
            Animator animator = field?.GetValue(scorer) as Animator;
            if (animator != null)
            {
                return animator;
            }
        }

        DeadBugGamingPoseScorer deadBugScorer =
            FindObjectOfType<DeadBugGamingPoseScorer>(true);
        return deadBugScorer != null ? deadBugScorer.UserAnimator : null;
    }
}

/// <summary>Feeds the existing scoring freshness gate from EVMC4U packets.</summary>
public sealed class VmcGameModeSourceBridge : MonoBehaviour
{
    private EVMC4U.ExternalReceiver receiver;
    private AvatarMotionSource source;

    public void Configure(
        EVMC4U.ExternalReceiver externalReceiver,
        AvatarMotionSource motionSource)
    {
        receiver = externalReceiver;
        source = motionSource;
    }

    private void Update()
    {
        if (receiver == null || source == null)
        {
            return;
        }

        source.SetAvailable(!receiver.Freeze && receiver.GetAvailable() > 0);
        if(receiver.Freeze)return;
        if (receiver.LastBonePacketCounterInFrame > 0)
        {
            source.ReportFrame(
                receiver.GetRemoteTime(),
                receiver.LastBonePacketCounterInFrame);
        }
    }
}
