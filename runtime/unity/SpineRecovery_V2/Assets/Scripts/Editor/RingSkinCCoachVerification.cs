using System;
using System.IO;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

/// <summary>Real menu/countdown/coach/note integration; calibration bypass is unsaved and test-only.</summary>
[InitializeOnLoad]
public static class RingSkinCCoachVerification
{
    const string Flag = "RingSkinCCoachVerification.Active";
    const string Output = "output/two-phase-haptics/";
    const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;
    static int index, step, segments, checkpoints, notes, samples;
    static bool complete, sawCountdown, sawReturn, paused;
    static float frozenProgress;
    static double entered, pauseAt;
    static RingSkinCCoachView view;
    static GameWorkoutScene entry;
    static string Action => RingSkinCCoachBuilder.Actions[index];
    static readonly HumanBodyBones[] Tips = { HumanBodyBones.LeftHand,HumanBodyBones.RightHand,HumanBodyBones.LeftFoot,HumanBodyBones.RightFoot,
        HumanBodyBones.LeftToes,HumanBodyBones.RightToes,HumanBodyBones.Head };
    static RingSkinCCoachVerification() { if (SessionState.GetBool(Flag,false)) EditorApplication.update += Tick; }

    public static void InstallAndRun()
    {
        RingSkinCCoachBuilder.Install();
        Run();
    }
    [MenuItem("Tools/Ring Skin C/Verify Migrated Coach Windows")]
    public static void Run()
    {
        Directory.CreateDirectory(Output);
        File.WriteAllText(Output+"verification.txt","Real startup/countdown/standard-difficulty session; no wearable device. Test-only calibration bypass and perfect judgement.\n");
        EditorSceneManager.OpenScene(RecordedCoachSceneTools.Recorded("BirdDog"));
        SessionState.SetBool(Flag,true); EditorApplication.update -= Tick; EditorApplication.update += Tick;
        EditorApplication.isPlaying = true;
    }
    static object Get(object o,string name) => o.GetType().GetField(name,Private).GetValue(o);
    static void Require(bool value,string message) { if(!value) throw new Exception(message); }
    static void Log(string message) => File.AppendAllText(Output+"verification.txt",message+"\n");
    static void UpdateView()
    {
        typeof(RingSkinCCoachView).GetMethod("LateUpdate",Private).Invoke(view,null);
        var framing=view.previewCamera.GetComponent<HipThrustPreviewFraming>();
        typeof(HipThrustPreviewFraming).GetMethod("LateUpdate",Private).Invoke(framing,null);
    }
    static void Tick()
    {
        if(!EditorApplication.isPlaying || EditorApplication.isCompiling) return;
        try
        {
            if(step==0)
            {
                view=UnityEngine.Object.FindObjectOfType<RingSkinCCoachView>();
                if(view==null || view.gameObject.scene.name!=Path.GetFileNameWithoutExtension(RecordedCoachSceneTools.Recorded(Action))) return;
                entry=view.entry; entered=EditorApplication.timeSinceStartup;
                segments=checkpoints=notes=samples=0; complete=sawCountdown=sawReturn=paused=false;
                var coachData = new SerializedObject(entry.coach);
                foreach(string difficulty in new[]{"simpleDifficulty","standardDifficulty","hardDifficulty"})
                    Require(coachData.FindProperty(difficulty).FindPropertyRelative("holdSeconds").floatValue==0,"Hold remains in "+difficulty);
                Require(entry.menu.IsTrainingMenuVisible,"Start menu missing");
                UpdateView(); Require(!view.IsVisible,"Coach panel covers start menu");
                Capture(Action+"-start-menu");
                entry.scorer.GetComponent<SupineGameCalibration>().skipCalibrationForPreview=true;
                entry.menu.SelectStandardDifficulty();
                entry.coach.SegmentPlaybackStarted += _=>segments++;
                entry.coach.ScoringCheckpointStarted += checkpoint=>
                {
                    checkpoints++;
                    Require(checkpoint.holdSeconds==0,"A checkpoint requests a hold");
                    if (Action=="BirdDog" && checkpoint.isUp) VerifyBirdDogPair(checkpoint.stateName);
                };
                entry.coach.ActionCompleted += ()=>complete=true;
                entry.rhythm.NoteResolved += _=>notes++;
                entry.rhythm.SetTutorialJudgementOverride(RhythmJudgement.Perfect);
                Click((Button)Get(entry.menu,"startButton"));
                step=1; return;
            }
            if(EditorApplication.timeSinceStartup-entered>150) throw new Exception("Timeout: step "+step);
            UpdateView();
            if(step==1)
            {
                if(view.countdown.gameObject.activeInHierarchy)
                {
                    Require(view.IsVisible,"Coach absent during countdown");
                    Require(view.PhaseText=="准备开始","Wrong countdown stage");
                    if(!sawCountdown) { Capture(Action+"-countdown"); sawCountdown=true; }
                    return;
                }
                Require(sawCountdown,"Countdown skipped");
                if(!entry.coach.IsPlayingAction && !complete) return;
                if(!complete) VerifyPose();
                if(!paused && view.Phase==0 && view.PhaseProgress>.4f)
                {
                    Capture(Action+"-extend");
                    entry.coach.TogglePause(); UpdateView();
                    frozenProgress=view.PhaseProgress; pauseAt=EditorApplication.timeSinceStartup;
                    step=2; return;
                }
                Require(!entry.coach.IsHoldingPose && !view.PhaseText.Contains("保持"),"Unexpected hold phase");
                if(view.Phase==1 && view.PhaseProgress>.4f && !sawReturn)
                {
                    Capture(Action+"-return"); sawReturn=true;
                }
                if(complete)
                {
                    Require(sawReturn,"Missing return phase");
                    Require(segments==18 && checkpoints==16 && notes==16,$"Gameplay changed: segments={segments}, checkpoints={checkpoints}, notes={notes}");
                    Require(!view.IsVisible,"Coach panel covers end menu");
                    Log($"PASS {Action}: actual start button, countdown, extend/return only, zero hold at every difficulty, pause/resume; {samples} pose samples including wrists/ankles/toes; 18 segments, 16 zero-hold checkpoints, 16 note results.");
                    Time.timeScale=1;
                    entry.StopTraining(); UpdateView();
                    Require(entry.menu.IsTrainingMenuVisible && !view.IsVisible,"Return menu visibility failed");
                    Log("PASS "+Action+": stopping restores menu and hides coach overlay.");
                    if(++index==2) { Finish(0); return; }
                    step=0;
                    SceneManager.LoadScene(Path.GetFileNameWithoutExtension(RecordedCoachSceneTools.Recorded(Action)));
                }
            }
            else if(step==2 && EditorApplication.timeSinceStartup-pauseAt>.5)
            {
                Require(Mathf.Abs(view.PhaseProgress-frozenProgress)<.00001f,"Phase advanced while paused");
                Require(view.PhaseText.Contains("已暂停"),"Missing pause label");
                VerifyPose(); Capture(Action+"-paused");
                entry.coach.TogglePause(); paused=true; step=1;
            }
        }
        catch(Exception e) { Log("FAIL "+Action+": "+e); Debug.LogException(e); Finish(1); }
    }
    static void VerifyPose()
    {
        Require(view.IsVisible,"Coach hidden during action");
        Require(view.previewAnimator.runtimeAnimatorController==view.coachAnimator.runtimeAnimatorController,"Different coach source");
        foreach(var bone in Tips)
        {
            var actual=view.previewAnimator.GetBoneTransform(bone); var source=view.coachAnimator.GetBoneTransform(bone);
            if(actual==null || source==null) continue;
            Require(Quaternion.Angle(actual.localRotation,source.localRotation)<.2f,"Preview rotation differs: "+bone);
            Vector3 p=view.previewCamera.WorldToViewportPoint(actual.position);
            Require(p.x>.005f && p.x<.995f && p.y>.005f && p.y<.995f && p.z>0,"Cropped bone: "+bone+" "+p);
        }
        var preview=GameObject.Find("RingSkinC Coach Display/Recorded Coach/Coach").GetComponent<RawImage>();
        Require(Mathf.Abs(preview.rectTransform.rect.width/preview.rectTransform.rect.height-(float)preview.texture.width/preview.texture.height)<.001f,"Preview stretched");
        var corners=new Vector3[4]; view.PanelRect.GetWorldCorners(corners);
        Require(corners.All(c=>c.x>=0 && c.x<=Screen.width && c.y>=0 && c.y<=Screen.height),"Coach panel outside viewport");
        samples++;
    }
    static void VerifyBirdDogPair(string stateName)
    {
        var args=new object[]{stateName,default(RhythmBodyTarget),default(RhythmBodyTarget)};
        Require((bool)typeof(RhythmModeController).GetMethod("TryGetSegmentTargets",Private).Invoke(entry.rhythm,args),"Missing bird-dog pair");
        var foot=(RhythmBodyTarget)args[1]; var hand=(RhythmBodyTarget)args[2];
        bool right=foot==RhythmBodyTarget.RightFoot;
        Require(hand==(right?RhythmBodyTarget.LeftHand:RhythmBodyTarget.RightHand),"Same-side bird-dog targets");
        float Y(HumanBodyBones bone)=>view.coachAnimator.GetBoneTransform(bone).position.y;
        Require(Y(right?HumanBodyBones.RightFoot:HumanBodyBones.LeftFoot)>Y(right?HumanBodyBones.LeftFoot:HumanBodyBones.RightFoot)+.015f,"Animated leg disagrees with target");
        Require(Y(right?HumanBodyBones.LeftHand:HumanBodyBones.RightHand)>Y(right?HumanBodyBones.RightHand:HumanBodyBones.LeftHand)+.1f,"Animated arm is not opposite the raised leg");
        Log("PASS BirdDog pose/targets: "+stateName+" -> "+foot+" + "+hand);
    }
    static void Click(Button button)
    {
        Canvas.ForceUpdateCanvases();
        var pointer=new PointerEventData(EventSystem.current) { position=RectTransformUtility.WorldToScreenPoint(null,button.transform.position) };
        var hits=new System.Collections.Generic.List<RaycastResult>(); EventSystem.current.RaycastAll(pointer,hits);
        Require(hits.Count>0 && hits[0].gameObject==button.gameObject,"Start button obscured");
        Require(button.interactable,"Start button unavailable"); button.onClick.Invoke();
    }
    static void Capture(string name)
    {
        UpdateView(); view.previewCamera.Render();
        var skin=UnityEngine.Object.FindObjectOfType<RhythmSkinCVisuals>(true);
        typeof(RhythmSkinCVisuals).GetMethod("LateUpdate",Private).Invoke(skin,null);
        UpdateView(); skin.presentationCamera.Render();
        var canvases=UnityEngine.Object.FindObjectsOfType<Canvas>().Where(c=>c.isActiveAndEnabled && c.isRootCanvas && c.renderMode==RenderMode.ScreenSpaceOverlay).ToArray();
        var host=new GameObject("Coach verification camera"); var camera=host.AddComponent<Camera>();
        camera.transform.position=new Vector3(1000,1000,-1000); camera.orthographic=true;
        camera.clearFlags=CameraClearFlags.SolidColor; camera.backgroundColor=Color.black;
        var rt=RenderTexture.GetTemporary(Mathf.Max(1,Screen.width),Mathf.Max(1,Screen.height),24,RenderTextureFormat.ARGB32,RenderTextureReadWrite.sRGB);
        var previous=RenderTexture.active;
        try
        {
            camera.targetTexture=rt;
            foreach(var canvas in canvases) { canvas.renderMode=RenderMode.ScreenSpaceCamera; canvas.worldCamera=camera; canvas.planeDistance=1; }
            Canvas.ForceUpdateCanvases(); camera.Render(); RenderTexture.active=rt;
            var image=new Texture2D(rt.width,rt.height,TextureFormat.RGB24,false);
            image.ReadPixels(new Rect(0,0,rt.width,rt.height),0,0); image.Apply();
            File.WriteAllBytes(Output+name+".png",image.EncodeToPNG()); UnityEngine.Object.DestroyImmediate(image);
        }
        finally
        {
            foreach(var canvas in canvases) { canvas.renderMode=RenderMode.ScreenSpaceOverlay; canvas.worldCamera=null; }
            RenderTexture.active=previous; UnityEngine.Object.DestroyImmediate(host); RenderTexture.ReleaseTemporary(rt);
        }
    }
    static void Finish(int code)
    {
        SessionState.SetBool(Flag,false); EditorApplication.update-=Tick; Time.timeScale=1;
        if(Application.isBatchMode) EditorApplication.Exit(code); else EditorApplication.isPlaying=false;
    }
}
