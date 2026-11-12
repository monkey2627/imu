using System;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

// Editor-only synthetic tracking and completion events exercise integration without changing production input.
[InitializeOnLoad]
public static class GameWorkoutVerification
{
    const string Flag = "GameWorkoutVerification.Active";
    const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;
    static GameWorkoutScene entry;
    static SupineGameCalibration userPose;
    static AvatarMotionSource source;
    static int phase, completed, lastAction = -1, lastSet = -1;
    static double started, deadline;
    static string report;
    static GameWorkoutVerification() { if (SessionState.GetBool(Flag, false)) EditorApplication.update += Tick; }

    [MenuItem("Tools/Game Workout/Verify Calibration and Sequence")]
    public static void Run()
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode) throw new Exception("Exit Play first.");
        EditorSceneManager.OpenScene("Assets/Scenes/HipThrust.unity");
        SessionState.SetBool(Flag, true);
        EditorApplication.update -= Tick;
        EditorApplication.update += Tick;
        EditorApplication.isPlaying = true;
    }

    static void Require(bool condition, string message) { if (!condition) throw new Exception(message); }
    static void Pose(float seconds)
    {
        userPose.referenceSeconds = seconds;
        Require(userPose.Prepare(entry.scorer.UserAnimator), "Cannot sample test pose");
    }

    static void Tick()
    {
        if (!EditorApplication.isPlaying || EditorApplication.isCompiling) return;
        try
        {
            if (deadline == 0) { deadline = EditorApplication.timeSinceStartup + 180; report = "Synthetic tracking/completion integration test.\n"; }
            if (EditorApplication.timeSinceStartup > deadline) throw new Exception("Timeout phase " + phase);
            var current = UnityEngine.Object.FindObjectOfType<GameWorkoutScene>();
            if (current == null) return;
            if (entry != current)
            {
                entry = current;
                entry.scorer.GetComponent<SupineGameCalibration>().skipCalibrationForPreview = false;
                entry.scorer.BeginCalibration();
                source = UnityEngine.Object.FindObjectOfType<AvatarMotionSource>();
                if (source == null) return;
                var bridge = UnityEngine.Object.FindObjectOfType<VmcGameModeSourceBridge>();
                if (bridge != null) bridge.enabled = false;
                var receiver = UnityEngine.Object.FindObjectOfType<EVMC4U.ExternalReceiver>();
                if (receiver != null) receiver.enabled = false;
                userPose = new GameObject("Editor only test pose").AddComponent<SupineGameCalibration>();
                userPose.referenceClip = entry.scorer.GetComponent<SupineGameCalibration>().referenceClip;
                Pose(phase == 0 ? 0 : 6.8f);
                source.SetAvailable(true);
                if (phase == 0) started = EditorApplication.timeSinceStartup;
            }
            userPose.SampleReference();
            var session = GameWorkoutSession.Instance;
            if (phase == 0 && EditorApplication.timeSinceStartup - started > 2)
            {
                Require(!SupineGameCalibration.IsLying(entry.scorer.UserAnimator) && !entry.scorer.IsCalibrated, "Standing pose passed supine calibration");
                report += "PASS: standing pose rejected.\n";
                Pose(6.8f);
                source.SetAvailable(false);
                started = EditorApplication.timeSinceStartup;
                phase = 1;
            }
            else if (phase == 1 && EditorApplication.timeSinceStartup - started > 2)
            {
                Require(SupineGameCalibration.IsLying(entry.scorer.UserAnimator) && !entry.scorer.IsCalibrated, "Disconnected tracker passed calibration");
                source.SetAvailable(true);
                report += "PASS: lying without tracking rejected.\n";
                phase = 2;
            }
            else if (phase == 2 && entry.scorer.IsCalibrated)
            {
                report += "PASS: stable lying pose calibrated.\n";
                Require(!session.SetPlanJson("{\"actions\":[{\"action\":\"unknown\",\"difficulty\":\"simple\",\"sets\":1}]}", out _), "Invalid action accepted");
                Require(!session.SetPlanJson("{\"actions\":[{\"action\":\"hipthrust\",\"difficulty\":\"easy\",\"sets\":0}]}", out _), "Invalid difficulty/sets accepted");
                var plan = new GameWorkoutPlan { actions = new[] {
                    new GameWorkoutAction { action = "hipthrust", difficulty = "simple", sets = 2 },
                    new GameWorkoutAction { action = "birddog", difficulty = "standard", sets = 1 },
                    new GameWorkoutAction { action = "deadbug", difficulty = "hard", sets = 1 }
                } };
                Require(session.SetPlan(plan, out string error), error);
                plan.actions[0].sets = 99;
                Require(session.Plan.actions[0].sets == 2, "Caller can mutate running plan");
                entry.TogglePanel();
                started = EditorApplication.timeSinceStartup;
                phase = 22;
            }
            else if (phase == 22 && EditorApplication.timeSinceStartup - started > 2)
            {
                Canvas.ForceUpdateCanvases();
                ScreenCapture.CaptureScreenshot("output/workout/sequence-panel.png");
                started = EditorApplication.timeSinceStartup;
                phase = 23;
            }
            else if (phase == 23 && EditorApplication.timeSinceStartup - started > 2)
            {
                Require(session.StartPlan(out string error), error);
                Require(!session.SetPlan(session.Plan, out _), "Running plan replacement accepted");
                entry.menu.StartTraining();
                Require(!entry.IsPlaying, "Manual start bypassed sequence transition");
                phase = 3;
            }
            else if (phase == 3)
            {
                if (session.IsComplete)
                {
                    Require(completed == 4 && session.ActionIndex == 3, "Incorrect total set count");
                    report += "PASS: two hip sets, bird dog and dead bug transitions; all three difficulties; plan completion.\n";
                    entry.TogglePanel();
                    started = EditorApplication.timeSinceStartup;
                    phase = 32;
                }
                else if (session.IsRunning && (session.ActionIndex != lastAction || session.CompletedSets != lastSet) && entry.IsPlaying)
                {
                    bool ready = entry.taiko != null ? entry.taiko.IsRunning : entry.rhythm.IsRunning;
                    if (!ready) return;
                    Require(entry.scorer.IsCalibrated, "Sequence bypassed calibration");
                    int action = session.ActionIndex, set = session.CompletedSets;
                    if (entry.taiko != null)
                    {
                        Require(entry.taiko.BeatCount == 4, "Hip difficulty not applied");
                        typeof(HipThrustTaikoController).GetField("elapsed", Private).SetValue(entry.taiko, entry.taiko.Duration);
                        typeof(HipThrustTaikoController).GetMethod("LateUpdate", Private).Invoke(entry.taiko, null);
                    }
                    else
                    {
                        string difficulty = typeof(CoachActionController).GetField("selectedDifficulty", Private).GetValue(entry.coach).ToString();
                        Require(difficulty == (action == 1 ? "Standard" : "Hard"), "Ring difficulty not applied");
                        entry.coach.CancelTraining();
                        entry.scorer.CancelScoringSession();
                        entry.rhythm.StopRhythmMode();
                        ((Action)typeof(CoachActionController).GetField("ActionCompleted", Private).GetValue(entry.coach))?.Invoke();
                    }
                    completed++;
                    lastAction = action;
                    lastSet = set;
                    int newAction = session.ActionIndex, newSet = session.CompletedSets;
                    session.OnSetCompleted(entry);
                    Require(newAction == session.ActionIndex && newSet == session.CompletedSets, "Duplicate completion counted twice");
                    report += $"PASS: action {action}, set {set + 1}; duplicate completion ignored.\n";
                }
            }
            else if (phase == 32 && EditorApplication.timeSinceStartup - started > 2)
            {
                Canvas.ForceUpdateCanvases();
                ScreenCapture.CaptureScreenshot("output/workout/sequence-complete.png");
                started = EditorApplication.timeSinceStartup;
                phase = 33;
            }
            else if (phase == 33 && EditorApplication.timeSinceStartup - started > 2)
            {
                Require(session.StartPlan(out string error), error);
                phase = 4;
            }
            else if (phase == 4 && entry.IsPlaying)
            {
                session.Cancel();
                started = EditorApplication.timeSinceStartup;
                phase = 5;
            }
            else if (phase == 5 && EditorApplication.timeSinceStartup - started > 7)
            {
                Require(!session.IsRunning && !entry.IsPlaying && !entry.taiko.IsRunning && entry.menu.IsTrainingMenuVisible, "Cancel leaked countdown/start");
                report += "PASS: restart and cancellation during countdown; no delayed start.\n";
                File.WriteAllText("output/workout/verification.txt", report);
                Finish();
            }
        }
        catch (Exception ex)
        {
            Directory.CreateDirectory("output/workout");
            File.WriteAllText("output/workout/verification.txt", "FAIL phase " + phase + ": " + ex + "\n" + report);
            Debug.LogException(ex);
            Finish();
        }
    }

    static void Finish()
    {
        SessionState.SetBool(Flag, false);
        EditorApplication.update -= Tick;
        EditorApplication.isPlaying = false;
    }
}
