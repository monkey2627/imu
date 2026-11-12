using System;
using System.IO;
using System.Linq;
using TMPro;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class GameWorkoutTools
{
    [MenuItem("Tools/Game Workout/Load Example Parameters (Play Mode)")]
    public static void LoadExample()
    {
        if (!EditorApplication.isPlaying) throw new InvalidOperationException("Enter Play mode first.");
        var session = GameWorkoutSession.Instance;
        string json = File.ReadAllText("output/workout/example-plan.json");
        if (!session.SetPlanJson(json, out string error)) throw new InvalidOperationException(error);
        UnityEngine.Object.FindObjectOfType<GameWorkoutScene>().TogglePanel();
        double captureAt = EditorApplication.timeSinceStartup + 2;
        void CapturePreview()
        {
            if (EditorApplication.timeSinceStartup < captureAt) return;
            EditorApplication.update -= CapturePreview;
            if (!EditorApplication.isPlaying) return;
            Directory.CreateDirectory("output/workout");
            ScreenCapture.CaptureScreenshot("output/workout/sequence-panel-final.png", 2);
        }
        EditorApplication.update += CapturePreview;
    }
    [MenuItem("Tools/Game Workout/Install in New Visual Games")]
    public static void Install()
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode) throw new InvalidOperationException("Exit Play mode first.");
        var current = UnityEngine.SceneManagement.SceneManager.GetActiveScene();
        if (current.isDirty) throw new InvalidOperationException("Save the current scene before installing.");
        string previous = current.path;
        Directory.CreateDirectory("output/workout");
        var clip = AssetDatabase.LoadAllAssetsAtPath("Assets/Movements/hipthrust.fbx").OfType<AnimationClip>().Single(c => c.name == "animation_pose");
        var font = AssetDatabase.LoadAssetAtPath<TMP_FontAsset>("Assets/Fonts/SourceHanSans/SourceHanSansSC-Heavy SDF.asset");
        foreach (string action in new[] { "deadbug", "birddog", "hipthrust" })
        {
            string path = "Assets/Scenes/" + GameWorkoutPlan.SceneFor(action) + ".unity";
            string backup = "output/workout/" + GameWorkoutPlan.SceneFor(action) + ".before.unity";
            if (!File.Exists(backup)) File.Copy(path, backup);
            var scene = EditorSceneManager.OpenScene(path);
            T Find<T>() where T : Component => scene.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<T>(true)).FirstOrDefault();
            var scorer = Find<DeadBugGamingPoseScorer>();
            var calibration = scorer.GetComponent<SupineGameCalibration>() ?? scorer.gameObject.AddComponent<SupineGameCalibration>();
            calibration.referenceClip = clip;
            calibration.referenceSeconds = 6.8f;
            var entry = Find<GameWorkoutScene>() ?? new GameObject("Game Workout").AddComponent<GameWorkoutScene>();
            entry.actionId = action;
            entry.scorer = scorer;
            entry.menu = Find<DeadBugStartMenuController>();
            entry.coach = Find<CoachActionController>();
            entry.taiko = Find<HipThrustTaikoController>();
            entry.rhythm = Find<RhythmModeController>();
            entry.countdown = Find<DeadBugCountdownController>();
            entry.gameCanvas = scene.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<Canvas>(true)).Single(c => c.name == "GamingCanvas");
            entry.font = font;
            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene);
        }
        EditorSceneManager.OpenScene(previous);
        File.WriteAllText("output/workout/install.txt", "Installed supine calibration and workout entry in three new visual games. Classic scenes untouched.");
        Debug.Log("GAME_WORKOUT_INSTALLED");
    }
}
