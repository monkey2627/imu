using System;
using System.Linq;
using TMPro;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class RingSkinCCoachBuilder
{
    public static readonly string[] Actions = { "BirdDog", "DeadBug" };

    [MenuItem("Tools/Ring Skin C/Install Hip Thrust Coach Window")]
    public static void Install()
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode) throw new InvalidOperationException("Exit Play first.");
        var setup = EditorSceneManager.GetSceneManagerSetup();
        if (setup.Any(s => s.isLoaded && UnityEngine.SceneManagement.SceneManager.GetSceneByPath(s.path).isDirty))
            throw new InvalidOperationException("Save current scenes first.");
        try
        {
            const string materialPath = "Assets/RingSkinC/CoachDisplayComposite.mat";
            var material = AssetDatabase.LoadAssetAtPath<Material>(materialPath);
            if (material == null)
            {
                material = new Material(Shader.Find("RingSkinC/DisplayGammaComposite")) { renderQueue = 3000 };
                AssetDatabase.CreateAsset(material,materialPath);
            }
            foreach (string action in Actions)
            {
                var scene = EditorSceneManager.OpenScene(RecordedCoachSceneTools.Recorded(action));
                var all = scene.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<Transform>(true)).ToArray();
                T Find<T>() where T: Component => all.Select(t=>t.GetComponent<T>()).Where(c=>c!=null).Single();
                var entry = Find<GameWorkoutScene>();
                var sequence = new SerializedObject(entry.coach);
                foreach (string difficulty in new[] { "simpleDifficulty", "standardDifficulty", "hardDifficulty" })
                    sequence.FindProperty(difficulty).FindPropertyRelative("holdSeconds").floatValue = 0f;
                sequence.ApplyModifiedPropertiesWithoutUndo();
                var map = JsonUtility.FromJson<RecordedCoachSceneTools.PhaseMap>(System.IO.File.ReadAllText(
                    RecordedCoachSceneTools.Root + action + ".phases.json"));
                var rhythm = new SerializedObject(entry.rhythm);
                var targets = rhythm.FindProperty("recordedSegmentTargets");
                targets.arraySize = map.firstTargets.Length;
                for (int i=0; i<targets.arraySize; i++)
                {
                    var target = targets.GetArrayElementAtIndex(i);
                    target.FindPropertyRelative("stateName").stringValue = action.ToLowerInvariant()+"_"+(i+1)+"_up";
                    target.FindPropertyRelative("first").enumValueIndex = map.firstTargets[i];
                    target.FindPropertyRelative("second").enumValueIndex = map.secondTargets[i];
                }
                rhythm.ApplyModifiedPropertiesWithoutUndo();
                var card = Find<ActionPreviewCard>();
                var view = all.Select(t=>t.GetComponent<RingSkinCCoachView>()).FirstOrDefault(c=>c!=null);
                if (view == null) view = new GameObject("Recorded Coach Window").AddComponent<RingSkinCCoachView>();
                view.entry = entry; view.legacyCard = card;
                view.coachAnimator = (Animator)new SerializedObject(entry.coach).FindProperty("coachAnimator").objectReferenceValue;
                view.previewAnimator = (Animator)new SerializedObject(card).FindProperty("previewAnimator").objectReferenceValue;
                view.previewCamera = all.Single(t=>t.name==action+"PreviewCamera").GetComponent<Camera>();
                view.countdown = (TMP_Text)new SerializedObject(entry.countdown).FindProperty("countdownText").objectReferenceValue;
                view.endMenu = all.Single(t=>t.name=="EndMenuCanvas").gameObject;
                var camera = view.previewCamera;
                camera.clearFlags = CameraClearFlags.SolidColor; camera.backgroundColor = Color.clear;
                camera.allowHDR = false; camera.nearClipPlane = .01f; camera.farClipPlane = 30;
                var framing = camera.GetComponent<HipThrustPreviewFraming>() ?? camera.gameObject.AddComponent<HipThrustPreviewFraming>();
                framing.avatar = view.previewAnimator; framing.previewCamera = camera; framing.framingPadding = .5f;
                // UI and artwork must share the UI render queue so canvas sorting can put the coach above the sky.
                Find<RhythmSkinCVisuals>().displayMaterial = material;
                EditorSceneManager.MarkSceneDirty(scene); EditorSceneManager.SaveScene(scene);
                Debug.Log("COACH_WINDOW_INSTALLED " + scene.path);
            }
            AssetDatabase.SaveAssets();
        }
        finally
        {
            // A fresh batch editor starts in an unnamed scene, which cannot be restored by path.
            if (setup.Length > 0 && setup.All(s => !string.IsNullOrEmpty(s.path) && System.IO.File.Exists(s.path)))
                EditorSceneManager.RestoreSceneManagerSetup(setup);
        }
    }
}
