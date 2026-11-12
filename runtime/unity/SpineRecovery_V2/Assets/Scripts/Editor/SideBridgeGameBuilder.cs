using System;
using System.IO;
using System.Linq;
using TMPro;
using UnityEditor;
using UnityEditor.Animations;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class SideBridgeGameBuilder
{
    public const string Root="Assets/Movements/RecordedSideBridge/";
    public const string Dynamic="Assets/Scenes/SideBridge_RingBridge_Dynamic.unity";
    public const string Hold30="Assets/Scenes/SideBridge_RingBridge_Hold30.unity";
    public const string TimelinePath=Root+"SideBridgeTimeline.asset";
    [MenuItem("Tools/Ring Bridge/Create Left Side Bridge Games")]
    public static void Build()
    {
        if(EditorApplication.isPlayingOrWillChangePlaymode)throw new InvalidOperationException("Exit Play first.");
        if(File.Exists(Dynamic)||File.Exists(Hold30))throw new IOException("Side bridge scenes already exist.");
        AssetDatabase.ImportAsset(Root+"SideBridge.coachrecording",ImportAssetOptions.ForceSynchronousImport);
        var clips=AssetDatabase.LoadAllAssetsAtPath(Root+"SideBridge.coachrecording").OfType<AnimationClip>().ToDictionary(c=>c.name);
        if(clips.Count!=9||clips.Values.Any(c=>!c.humanMotion))throw new Exception("Side bridge HumanPose clips missing.");
        var controller=AnimatorController.CreateAnimatorControllerAtPath(Root+"SideBridge_Recorded.controller");
        var machine=controller.layers[0].stateMachine;
        foreach(var clip in clips.Values){var state=machine.AddState(clip.name);state.motion=clip;if(clip.name=="sidebridge_start")machine.defaultState=state;}
        machine.AddState("Idle").motion=clips["sidebridge_start"];
        var frames=JsonUtility.FromJson<CoachMotionPackage>(File.ReadAllText(Root+"SideBridge.coachrecording")).segments[0].frames;
        var wrap=RecordedCoachMotionImporter.Bake("sidebridge_wrap",new[]{frames[1459],frames[435]},0,1,1/HipThrustHold30Timeline.WrapSeconds,0,false);
        AssetDatabase.CreateAsset(wrap,Root+"SideBridge_Wrap.anim");machine.AddState("sidebridge_wrap").motion=wrap;
        AssetDatabase.SaveAssets();
        HipThrustPhaseTimeline timeline=null;
        foreach(string path in new[]{Dynamic,Hold30})
        {
            if(!AssetDatabase.CopyAsset(HipThrustNewRulesBuilder.NewScore,path))throw new IOException("Scene copy failed.");
            var scene=EditorSceneManager.OpenScene(path);
            T Find<T>() where T:Component=>scene.GetRootGameObjects().SelectMany(r=>r.GetComponentsInChildren<T>(true)).Single();
            var game=Find<HipThrustTaikoController>();var entry=Find<GameWorkoutScene>();var data=new SerializedObject(game);
            var coach=(Animator)data.FindProperty("coachAnimator").objectReferenceValue;
            var preview=(Animator)data.FindProperty("coachPreviewAnimator").objectReferenceValue;
            coach.runtimeAnimatorController=preview.runtimeAnimatorController=controller;
            PrefabUtility.RecordPrefabInstancePropertyModifications(coach);PrefabUtility.RecordPrefabInstancePropertyModifications(preview);
            if(timeline==null){timeline=BakeTimeline(coach,clips["sidebridge_routine"]);AssetDatabase.CreateAsset(timeline,TimelinePath);}
            data.FindProperty("sourceClip").objectReferenceValue=clips["sidebridge_routine"];
            data.FindProperty("phaseTimeline").objectReferenceValue=timeline;
            data.FindProperty("hipThrustAnimatorController").objectReferenceValue=controller;
            data.FindProperty("routineStateName").stringValue="sidebridge_routine";
            data.FindProperty("hipThrustStartStateName").stringValue="sidebridge_start";
            data.FindProperty("actionDisplayName").stringValue="左侧桥";
            data.FindProperty("restTransitionStateName").stringValue="sidebridge_wrap";
            var sourceBeats=data.FindProperty("sourceBeats");sourceBeats.arraySize=timeline.cycles.Length;
            for(int i=0;i<timeline.cycles.Length;i++)
            {var b=sourceBeats.GetArrayElementAtIndex(i);b.FindPropertyRelative("upFrame").intValue=timeline.cycles[i].holdStartFrame;b.FindPropertyRelative("holdEndFrame").intValue=timeline.cycles[i].lowerStartFrame;b.FindPropertyRelative("downFrame").intValue=timeline.cycles[i].lowerEndFrame;}
            var side=game.gameObject.AddComponent<SideBridgePoseScoring>();
            Find<HipThrustRuleScoring>().sideBridge=side;
            var dose=game.gameObject.AddComponent<HipThrustPrescriptionDose>();
            dose.transport=game;dose.menu=entry.menu;dose.prescriptionActionId="side_bridge";dose.actionDisplayName="左侧桥";
            dose.preserveRecordedTiming=path==Dynamic;dose.skipRecordedPlateauAfterHold=path==Hold30;
            dose.preserveSourceGaps=true;
            dose.previewPrescription=AssetDatabase.LoadAssetAtPath<TextAsset>(Root+"SideBridge_PreviewPrescription.json");
            dose.menuDescription=scene.GetRootGameObjects().SelectMany(r=>r.GetComponentsInChildren<TMP_Text>(true)).First(t=>t.name=="Description");
            data.FindProperty("hold30Dose").objectReferenceValue=dose;data.ApplyModifiedPropertiesWithoutUndo();
            entry.actionId=path==Dynamic?"sidebridge":"sidebridge_hold30";entry.hipThrustSceneOverride="";entry.useRecordedCoachAnimations=true;
            SetString(entry.coach,"actionStateName","sidebridge_start");
            foreach(string key in new[]{"simpleActionStateName","standardActionStateName","hardActionStateName"})SetString(entry.menu,key,"sidebridge_start");
            SetString(entry.countdown,"autoStartActionStateName","sidebridge_start");
            var card=Find<ActionPreviewCard>();SetString(card,"actionName","左侧桥");SetString(card,"previewStateName","sidebridge_start");SetString(card,"segmentedStartStateName","sidebridge_start");SetString(card,"targetSceneName",Path.GetFileNameWithoutExtension(path));
            var calibration=entry.scorer.GetComponent<SupineGameCalibration>();calibration.leftSideBridge=true;calibration.referenceClip=clips["sidebridge_routine"];calibration.referenceSeconds=timeline.cycles[0].liftStartFrame/30f;
            foreach(var text in scene.GetRootGameObjects().SelectMany(r=>r.GetComponentsInChildren<TMP_Text>(true)))
                text.text=text.text.Replace("Hip Thrust 臀桥",path==Dynamic?"左侧桥 · 动态":"左侧桥 · 保持30秒").Replace("臀桥","左侧桥");
            dose.menuDescription.text="演示处方 1 组 × 3 次 · "+(path==Dynamic?"录制节奏":"保持 30 秒");
            var song=new SerializedObject(Find<RhythmSongClock>());song.FindProperty("song").objectReferenceValue=AssetDatabase.LoadAssetAtPath<AudioClip>("Assets/Audio/GameMusic/SideBridge_EnchantedTiki.ogg");
            if(song.FindProperty("song").objectReferenceValue==null)throw new IOException("Side bridge music missing.");song.ApplyModifiedPropertiesWithoutUndo();
            EditorSceneManager.MarkSceneDirty(scene);EditorSceneManager.SaveScene(scene);
        }
        AssetDatabase.SaveAssets();
        var builds=EditorBuildSettings.scenes.Where(s=>File.Exists(s.path)).ToList();
        foreach(var path in new[]{Dynamic,Hold30})if(!builds.Any(s=>s.path==path))builds.Add(new EditorBuildSettingsScene(path,true));
        EditorBuildSettings.scenes=builds.ToArray();
    }
    static void SetString(UnityEngine.Object target,string name,string value)
    {var data=new SerializedObject(target);var property=data.FindProperty(name);if(property==null)throw new InvalidOperationException(target.GetType().Name+" lacks "+name);property.stringValue=value;data.ApplyModifiedPropertiesWithoutUndo();}

    static HipThrustPhaseTimeline BakeTimeline(Animator source,AnimationClip clip)
    {
        var timeline=ScriptableObject.CreateInstance<HipThrustPhaseTimeline>();timeline.sourceClip=clip;
        timeline.sourceStartFrame=330;timeline.repeatStartFrame=14;timeline.repeatEndFrame=1129;
        var host=UnityEngine.Object.Instantiate(source.gameObject);host.transform.SetParent(null);host.transform.SetPositionAndRotation(Vector3.zero,Quaternion.identity);host.transform.localScale=Vector3.one;
        foreach(var script in host.GetComponentsInChildren<MonoBehaviour>(true))script.enabled=false;
        host.SetActive(true);var animator=host.GetComponent<Animator>();animator.enabled=true;animator.Rebind();animator.Update(0);
        var keys=new Keyframe[Mathf.RoundToInt(clip.length*clip.frameRate)+1];
        try
        {
            for(int f=0;f<keys.Length;f++)
            {animator.Play("sidebridge_routine",0,f/(float)(keys.Length-1));animator.Update(0);if(!SideBridgePoseGeometry.TryMeasure(animator,out var pose))throw new Exception("Invalid side bridge skeleton");keys[f]=new Keyframe(f,pose.lift);}
        }
        finally{UnityEngine.Object.DestroyImmediate(host);}
        timeline.recordedHeight=new AnimationCurve(keys);
        for(int i=0;i<keys.Length;i++){AnimationUtility.SetKeyLeftTangentMode(timeline.recordedHeight,i,AnimationUtility.TangentMode.Linear);AnimationUtility.SetKeyRightTangentMode(timeline.recordedHeight,i,AnimationUtility.TangentMode.Linear);}
        // Source-frame boundaries inspected from the full left-support trajectory.
        int[,] raw={{435,480,695,734},{880,950,1085,1190},{1215,1259,1410,1459}};
        timeline.cycles=new HipThrustPhaseTimeline.Cycle[3];
        for(int i=0;i<3;i++)
        {
            int start=raw[i,0]-330,hold=raw[i,1]-330,lower=raw[i,2]-330,end=raw[i,3]-330;
            float high=Enumerable.Range(hold,lower-hold+1).Max(f=>keys[f].value);
            timeline.cycles[i]=new HipThrustPhaseTimeline.Cycle{liftStartFrame=start,holdStartFrame=hold,lowerStartFrame=lower,lowerEndFrame=end,lowHeight=keys[start].value,peakHeight=high};
        }
        timeline.Validate();return timeline;
    }
}
