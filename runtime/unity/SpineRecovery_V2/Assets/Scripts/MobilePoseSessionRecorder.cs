using System.Globalization;
using System.Text;
using UnityEngine;

[DisallowMultipleComponent]
public sealed class MobilePoseSessionRecorder : MonoBehaviour
{
    [SerializeField] private MobilePoseWebSocketSource mobilePose;
    [SerializeField] private CoachActionController coachController;
    [SerializeField] private DeadBugGamingPoseScorer deadBugScorer;
    [SerializeField] private HipThrustTaikoController hipThrustController;
    [SerializeField, Min(1f)] private float sampleRate = 30f;

    private HumanPoseHandler coachHandler;
    private HumanPoseHandler userHandler;
    private HumanPose coachPose;
    private readonly Transform[] userBones = new Transform[24];
    private readonly Quaternion[] restWorld = new Quaternion[24];
    private readonly Quaternion[] globals = new Quaternion[24];
    private readonly Quaternion[] savedLocal = new Quaternion[24];
    private static readonly int[] parents = { -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21 };
    private float nextSample;
    private bool recording;

    private void Awake()
    {
        if (mobilePose == null) mobilePose = GetComponent<MobilePoseWebSocketSource>();
        if (coachController == null) coachController = FindObjectOfType<CoachActionController>(true);
        if (deadBugScorer == null) deadBugScorer = FindObjectOfType<DeadBugGamingPoseScorer>(true);
        if (hipThrustController == null) hipThrustController = FindObjectOfType<HipThrustTaikoController>(true);
        CacheUserBones();
    }

    private void OnEnable()
    {
        if (coachController == null) coachController = FindObjectOfType<CoachActionController>(true);
        if (coachController != null)
        {
            coachController.ScoringSessionStarted += BeginSession;
            coachController.ScoringSessionCompleted += EndSession;
        }
        if (hipThrustController == null) hipThrustController = FindObjectOfType<HipThrustTaikoController>(true);
        if (hipThrustController != null) hipThrustController.Completed += EndSession;
    }

    private void OnDisable()
    {
        if (coachController != null)
        {
            coachController.ScoringSessionStarted -= BeginSession;
            coachController.ScoringSessionCompleted -= EndSession;
        }
        if (hipThrustController != null) hipThrustController.Completed -= EndSession;
        if (recording) EndSession();
        coachHandler?.Dispose();
        userHandler?.Dispose();
        coachHandler = null;
        userHandler = null;
    }

    private void BeginSession()
    {
        if (mobilePose == null || !mobilePose.AutoRecordSessions || mobilePose.IsReplaying) return;
        mobilePose.SendSessionCommands(
            "{\"command\":\"stopRecording\"}",
            "{\"command\":\"startRecording\",\"name\":\"" +
            EscapeJson(mobilePose.CurrentSceneName) + "\"}");
        recording = true;
        nextSample = 0f;
    }

    private void EndSession()
    {
        if (!recording) return;
        recording = false;
        mobilePose?.SendSessionCommand("{\"command\":\"stopRecording\"}");
    }

    private void LateUpdate()
    {
        if (hipThrustController != null && hipThrustController.IsRunning && !recording)
            BeginSession();
        if (!recording || Time.unscaledTime < nextSample) return;
        nextSample = Time.unscaledTime + 1f / sampleRate;
        Animator coach = deadBugScorer != null
            ? deadBugScorer.CoachAnimator
            : hipThrustController != null ? hipThrustController.CoachAnimator
            : coachController != null ? coachController.CoachAnimator : null;
        Animator user = deadBugScorer != null ? deadBugScorer.UserAnimator : mobilePose.UserAnimator;
        if (coach == null || user == null || !coach.isHuman || !user.isHuman) return;
        if (coachHandler == null) coachHandler = new HumanPoseHandler(coach.avatar, coach.transform);
        if (userHandler == null) userHandler = new HumanPoseHandler(user.avatar, user.transform);
        if (userBones[0] == null) CacheUserBones();
        for (int i = 0; i < 24; i++) if (userBones[i] != null) savedLocal[i] = userBones[i].localRotation;
        Vector3 savedHips = userBones[0] != null ? userBones[0].localPosition : Vector3.zero;
        coachHandler.GetHumanPose(ref coachPose);
        try
        {
            userHandler.SetHumanPose(ref coachPose);
            StringBuilder json = new StringBuilder("{\"command\":\"coachPose\",\"q\":[", 1200);
            for (int i = 0; i < 24; i++)
            {
                if (i > 0) json.Append(',');
                globals[i] = userBones[i] != null
                    ? userBones[i].rotation * Quaternion.Inverse(restWorld[i])
                    : parents[i] >= 0 ? globals[parents[i]] : Quaternion.identity;
                AppendQuaternion(json, globals[i]);
            }
            mobilePose.SendSessionCommand(json.Append("]}").ToString());
        }
        finally
        {
            for (int i = 0; i < 24; i++) if (userBones[i] != null) userBones[i].localRotation = savedLocal[i];
            if (userBones[0] != null) userBones[0].localPosition = savedHips;
        }
    }

    private void CacheUserBones()
    {
        Animator user = deadBugScorer != null ? deadBugScorer.UserAnimator : mobilePose != null ? mobilePose.UserAnimator : null;
        if (user == null || !user.isHuman) return;
        for (int i = 0; i < 24; i++)
        {
            HumanBodyBones bone = MobilePoseJointBone(i);
            if (bone == HumanBodyBones.LastBone) continue;
            userBones[i] = user.GetBoneTransform(bone);
            if (userBones[i] != null) restWorld[i] = userBones[i].rotation;
        }
    }

    private static HumanBodyBones MobilePoseJointBone(int index)
    {
        switch (index)
        {
            case 0: return HumanBodyBones.Hips;
            case 1: return HumanBodyBones.LeftUpperLeg;
            case 2: return HumanBodyBones.RightUpperLeg;
            case 3: return HumanBodyBones.Spine;
            case 4: return HumanBodyBones.LeftLowerLeg;
            case 5: return HumanBodyBones.RightLowerLeg;
            case 6: return HumanBodyBones.Chest;
            case 7: return HumanBodyBones.LeftFoot;
            case 8: return HumanBodyBones.RightFoot;
            case 9: return HumanBodyBones.UpperChest;
            case 10: return HumanBodyBones.LeftToes;
            case 11: return HumanBodyBones.RightToes;
            case 12: return HumanBodyBones.Neck;
            case 13: return HumanBodyBones.LeftShoulder;
            case 14: return HumanBodyBones.RightShoulder;
            case 15: return HumanBodyBones.Head;
            case 16: return HumanBodyBones.LeftUpperArm;
            case 17: return HumanBodyBones.RightUpperArm;
            case 18: return HumanBodyBones.LeftLowerArm;
            case 19: return HumanBodyBones.RightLowerArm;
            case 20: return HumanBodyBones.LeftHand;
            case 21: return HumanBodyBones.RightHand;
            case 22: return HumanBodyBones.LastBone;
            default: return HumanBodyBones.LastBone;
        }
    }

    private static void AppendQuaternion(StringBuilder json, Quaternion q)
    {
        json.Append('[').Append(q.x.ToString("0.#####", CultureInfo.InvariantCulture)).Append(',')
            .Append(q.y.ToString("0.#####", CultureInfo.InvariantCulture)).Append(',')
            .Append(q.z.ToString("0.#####", CultureInfo.InvariantCulture)).Append(',')
            .Append(q.w.ToString("0.#####", CultureInfo.InvariantCulture)).Append(']');
    }

    private static string EscapeJson(string value) => (value ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"");
}
