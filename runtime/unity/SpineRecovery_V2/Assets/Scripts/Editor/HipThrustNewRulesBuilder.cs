using System;
using System.IO;
using System.Linq;
using TMPro;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class HipThrustNewRulesBuilder
{
    public const string NewScore = "Assets/Scenes/HipThrust_RingBridge_NewScoring.unity";
    public const string Hold30 = "Assets/Scenes/HipThrust_RingBridge_Hold30.unity";
    public const string PreviewDose = "Assets/RingBridge/HipThrustHold30_PreviewPrescription.json";
    [MenuItem("Tools/Ring Bridge/Create Two New Scoring Scenes")]
    public static void Build()
    {
        if(EditorApplication.isPlayingOrWillChangePlaymode) throw new InvalidOperationException("Exit Play first.");
        foreach(string path in new[]{NewScore,Hold30})
        {
            if(File.Exists(path)) throw new IOException("Scene already exists: "+path);
            if(!AssetDatabase.CopyAsset(RingBridgeGameBuilder.ScenePath,path)) throw new IOException("Scene copy failed.");
            var scene=EditorSceneManager.OpenScene(path);
            T Find<T>() where T:Component => scene.GetRootGameObjects().SelectMany(r=>r.GetComponentsInChildren<T>(true)).Single();
            var game=Find<HipThrustTaikoController>(); var entry=Find<GameWorkoutScene>();
            var rule=game.gameObject.AddComponent<HipThrustRuleScoring>();
            rule.transport=game; rule.calibration=entry.scorer;
            var data=new SerializedObject(game); data.FindProperty("ruleScoring").objectReferenceValue=rule;
            var description=scene.GetRootGameObjects().SelectMany(r=>r.GetComponentsInChildren<TMP_Text>(true)).FirstOrDefault(t=>t.name=="Description");
            entry.actionId=path==Hold30?"hipthrust_hold30":"hipthrust_newscore";
            // Explicit action keys choose variants; a later legacy hipthrust action stays legacy.
            entry.hipThrustSceneOverride="";
            if(path==Hold30)
            {
                var dose=game.gameObject.AddComponent<HipThrustPrescriptionDose>();
                dose.transport=game; dose.menu=entry.menu; dose.menuDescription=description;
                dose.previewPrescription=AssetDatabase.LoadAssetAtPath<TextAsset>(PreviewDose);
                if(dose.previewPrescription==null) throw new IOException("Preview prescription missing.");
                data.FindProperty("hold30Dose").objectReferenceValue=dose;
                if(description!=null) description.text="演示处方 1 组 × 4 次 · 保持 30 秒";
            }
            else if(description!=null) description.text="新评分 · 抬起进入保持位后计分 · 满分 100";
            data.ApplyModifiedPropertiesWithoutUndo();
            var card=new SerializedObject(Find<ActionPreviewCard>());
            card.FindProperty("targetSceneName").stringValue=Path.GetFileNameWithoutExtension(path); card.ApplyModifiedPropertiesWithoutUndo();
            EditorSceneManager.MarkSceneDirty(scene); EditorSceneManager.SaveScene(scene);
        }
        var scenes=EditorBuildSettings.scenes.ToList();
        foreach(string path in new[]{NewScore,Hold30}) if(!scenes.Any(s=>s.path==path)) scenes.Add(new EditorBuildSettingsScene(path,true));
        EditorBuildSettings.scenes=scenes.ToArray(); AssetDatabase.SaveAssets();
    }
}
