using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public sealed class MobilePoseCalibrationController : MonoBehaviour
{
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

    public bool IsCalibrating { get; private set; }
    public bool IsComplete { get; private set; }
    public float RemainingSeconds { get; private set; }
    public string CurrentInstruction { get; private set; }
    private string armCalibrationResult;
    private float resultUntil;
    private bool armPriorSent;

    private void Awake()
    {
        if (mobilePose == null) mobilePose = GetComponent<MobilePoseWebSocketSource>();
        if (steps.Count == 0) steps = CreateDefaultSteps(gameObject.scene.name);
    }

    public void BeginCalibration()
    {
        if (IsCalibrating) return;
        StopAllCoroutines();
        StartCoroutine(RunSteps());
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

        if (!mobilePose.IsReplaying)
            mobilePose.SendCalibrationCommand("{\"command\":\"calibrationSessionStart\"}");

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
        if (mobilePose != null)
        {
            // The straight-arm prior (Python arm_prior.py) is wrong for the side-bridge support arm,
            // whose level forward forearm comes with a bent elbow; send the per-scene choice on connect.
            if (!mobilePose.IsConnected)
                armPriorSent = false;
            else if (!armPriorSent)
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
            message.StartsWith("pose hold calibration"))
            armCalibrationResult = message;
    }

    private void OnGUI()
    {
        float width = Mathf.Min(Screen.width - 40f, 900f);
        string text = string.IsNullOrEmpty(CurrentInstruction)
            ? "Calibration is separate from the game. Start it when ready."
            : CurrentInstruction;
        if (IsCalibrating) text += "\n" + Mathf.CeilToInt(RemainingSeconds) + " s";
        if (!string.IsNullOrEmpty(armCalibrationResult) && Time.unscaledTime < resultUntil)
        {
            text += "\n" + armCalibrationResult;
            if (armCalibrationResult.Contains("CHECK FAILED"))
                text += "\n姿势与预期不符，校准可能不准，建议完成后点“重新校准”。";
        }
        GUI.Box(new Rect((Screen.width - width) / 2f, 30f, width, 150f), text);
        if (!IsCalibrating && !IsComplete && GUI.Button(
                new Rect((Screen.width - 220f) / 2f, 190f, 220f, 48f),
                "Start calibration"))
            BeginCalibration();
        if (IsComplete && GUI.Button(new Rect((Screen.width - 220f) / 2f, 190f, 220f, 48f), "重新校准"))
            BeginCalibration();
    }

    private static Step PoseHold(string poseId, string instruction, float seconds) =>
        new Step { kind = StepKind.PoseHold, poseId = poseId, instruction = instruction, seconds = seconds };

    private static List<Step> CreateDefaultSteps(string sceneName)
    {
        string lower = (sceneName ?? string.Empty).ToLowerInvariant();
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
