using System;
using System.Collections;
using System.IO;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

[InitializeOnLoad]
public static class RingBridgeGameVerification
{
    const string Flag = "RingBridgeGameVerification.Active";
    const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;
    static HipThrustTaikoController game;
    static HipThrustRingBridgeView view;
    static GameWorkoutScene entry;
    static int stage;
    static bool completed;
    static bool[] captured = new bool[3];
    static double began, pausedAt, frozenTime;
    static float frozenProgress;

    static RingBridgeGameVerification() { if (SessionState.GetBool(Flag,false)) EditorApplication.update += Tick; }
    [MenuItem("Tools/Ring Bridge/Verify Recorded Game")]
    public static void Run()
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode) throw new InvalidOperationException("Exit Play first.");
        Directory.CreateDirectory(RingBridgeGameBuilder.Output);
        File.WriteAllText(RingBridgeGameBuilder.Output+"verification.txt", "Actual menu/countdown/game transport; synthetic calibration bypass, no wearable device.\n");
        EditorSceneManager.OpenScene(RingBridgeGameBuilder.ScenePath);
        SessionState.SetBool(Flag,true); EditorApplication.update -= Tick; EditorApplication.update += Tick;
        EditorApplication.isPlaying = true;
    }
    public static void RunVisualProbe() { SessionState.SetBool("RingBridge.Probe", true); Run(); }
    static void Require(bool test,string message) { if (!test) throw new Exception(message); }
    static void Log(string value) => File.AppendAllText(RingBridgeGameBuilder.Output+"verification.txt", value+"\n");
    static void Tick()
    {
        if (!EditorApplication.isPlaying || EditorApplication.isCompiling) return;
        try
        {
            if (stage==0)
            {
                began=EditorApplication.timeSinceStartup;
                view=UnityEngine.Object.FindObjectOfType<HipThrustRingBridgeView>();
                entry=UnityEngine.Object.FindObjectOfType<GameWorkoutScene>(); game=view.transport;
                Require(game.PhaseTimeline!=null && !view.chart.autoPlay,"Demo clock is still active.");
                Require(view.displayMaterial.renderQueue==3000,"Artwork material can obscure the controls.");
                game.PhaseTimeline.Validate(); VerifyBoundaries();
                entry.scorer.GetComponent<SupineGameCalibration>().skipCalibrationForPreview=true;
                game.Completed += ()=>completed=true;
                entry.menu.SelectSimpleDifficulty(); entry.menu.StartTraining(); stage=1;
                view.SetCaptureResolution(1688,780);
            }
            if (EditorApplication.timeSinceStartup-began>80) throw new Exception("Runtime verification timeout stage "+stage);
            if (!game.IsRunning && !completed) return;
            if (SessionState.GetBool("RingBridge.Probe",false))
            {
                var cycle=game.PhaseTimeline.cycles[0];
                float[] frames={(cycle.liftStartFrame+cycle.holdStartFrame)*.5f,(cycle.holdStartFrame+cycle.lowerStartFrame)*.5f,(cycle.lowerStartFrame+cycle.lowerEndFrame)*.5f};
                game.TogglePause();
                for(int p=0;p<3;p++)
                {
                    typeof(HipThrustTaikoController).GetField("elapsed",Private).SetValue(game,(double)frames[p]/30);
                    typeof(HipThrustTaikoController).GetMethod("RenderTimeline",Private).Invoke(game,null);
                    typeof(HipThrustRingBridgeView).GetMethod("LateUpdate",Private).Invoke(view,null);
                    Capture("stage-"+p);
                }
                game.TogglePause();
                typeof(HipThrustRingBridgeView).GetMethod("LateUpdate",Private).Invoke(view,null);
                Require(view.IsPreviewVisible,"Coach must be visible by default.");
                CaptureDisplay();
                SessionState.SetBool("RingBridge.Probe",false); Finish(0); return;
            }
            if (stage==1)
            {
                float sourceFrame=game.SourceTime*game.PhaseTimeline.sourceClip.frameRate;
                var expected=game.PhaseTimeline.Evaluate(sourceFrame);
                Require(expected.phase==view.CurrentSample.phase,"Ring phase lags the sampled coach.");
                Require(Mathf.Abs(game.CoachAnimator.GetCurrentAnimatorStateInfo(0).normalizedTime-game.PreviewAnimator.GetCurrentAnimatorStateInfo(0).normalizedTime)<.0001f,"Coach and preview desynchronized.");
                if (expected.phase>=0 && expected.progress>.45f && expected.progress<.7f && !captured[expected.phase])
                {
                    Capture("stage-"+expected.phase); captured[expected.phase]=true;
                    if (expected.phase==1)
                    {
                        ClickVisibleButton(GameObject.Find("RingBridge Display/Pause").GetComponent<Button>());
                        Require(game.IsPaused,"Pause control did not pause transport.");
                        Require(view.IsPreviewVisible,"Coach must remain visible while paused.");
                        frozenTime=game.Elapsed; frozenProgress=view.CurrentSample.progress;
                        pausedAt=EditorApplication.timeSinceStartup; stage=2;
                    }
                }
                if (completed)
                {
                    Require(captured.All(v=>v),"Did not render all three phases.");
                    Require(game.HitCount+game.MissCount==8,"Expected two endpoint judgements per repetition.");
                    Log("PASS simple: actual menu/countdown, four cycles, eight endpoint judgements, all three artwork phases, coach/preview/ring synchronization, completion.");
                    VerifyRepeatsAndScore();
                    VerifyAspectRatios();
                    Finish(0);
                }
            }
            else if(stage==2 && EditorApplication.timeSinceStartup-pausedAt>.6)
            {
                Require(Math.Abs(game.Elapsed-frozenTime)<.000001 && Mathf.Abs(view.CurrentSample.progress-frozenProgress)<.000001,"Pause advanced the phase.");
                Log("PASS pause/resume: coach time and ring progress remain frozen together.");
                CaptureDisplay("game-paused");
                view.SetCaptureResolution(1688,780);
                typeof(HipThrustRingBridgeView).GetMethod("LateUpdate",Private).Invoke(view,null);
                ClickVisibleButton(GameObject.Find("RingBridge Display/Pause Menu/继续训练").GetComponent<Button>());
                Require(!game.IsPaused,"Resume control did not resume transport.");
                stage=1;
            }
        }
        catch(Exception ex) { Log("FAIL "+ex); Debug.LogException(ex); Finish(1); }
    }
    static void VerifyBoundaries()
    {
        var timeline=game.PhaseTimeline;
        Require(timeline.Evaluate(24).completedSeconds.z > .5f,"Intro falsely flashes a completed previous cycle.");
        int checkedSamples=0;
        foreach(var c in timeline.cycles)
        {
            int[] begins={c.liftStartFrame,c.holdStartFrame,c.lowerStartFrame};
            int[] ends={c.holdStartFrame,c.lowerStartFrame,c.lowerEndFrame};
            for(int phase=0;phase<3;phase++)
            {
                Require(timeline.Evaluate(begins[phase]).phase==phase,"Wrong exact boundary phase.");
                Require(Mathf.Abs(timeline.Evaluate(begins[phase]).progress)<1e-6,"Phase progress does not start at zero.");
                for(float f=begins[phase];f<ends[phase];f+=.25f)
                {
                    var s=timeline.Evaluate(f);
                    Require(s.phase==phase && Mathf.Abs(s.progress-(f-begins[phase])/(ends[phase]-begins[phase]))<1e-5,"Quarter-frame timing mismatch.");
                    if(phase==1) Require(s.poseHeight==1,"Centre figure does not hold at the top.");
                    checkedSamples++;
                }
                Require(timeline.Evaluate(ends[phase]-.01f).progress>.99f,"Progress is incomplete before transition.");
            }
        }
        Log("PASS boundaries: "+checkedSamples+" quarter-frame samples; exact lift/hold/lower transitions and centre hold.");
    }
    static void VerifyRepeatsAndScore()
    {
        foreach(int passes in new[]{2,3})
        {
            entry.PrepareMenu(passes==2?"standard":"hard");
            game.StartMode();
            Require(game.BeatCount==4*passes,"Difficulty repetition count mismatch.");
            var timeline=game.PhaseTimeline; double fps=timeline.sourceClip.frameRate;
            double loop=(timeline.repeatEndFrame-timeline.repeatStartFrame)/fps;
            for(int pass=0;pass<passes;pass++) foreach(var c in timeline.cycles)
                foreach(int f in new[]{c.liftStartFrame,c.holdStartFrame,c.lowerStartFrame,c.lowerEndFrame-1})
                    Require(Math.Abs(game.SourceTimeAt(f/fps+pass*loop)*fps-f)<.001,"Repeated phases drift from the recording.");
            game.StopMode();
        }
        // Exercise the actual score path with one synthetic HIT then MISS, without changing saved scene settings.
        game.StartMode();
        var beats=(IList)typeof(HipThrustTaikoController).GetField("beats",Private).GetValue(game);
        var resolve=typeof(HipThrustTaikoController).GetMethod("ResolveHit",Private);
        resolve.Invoke(game,new object[]{true,1000,beats[0]});
        Require(game.Score==1000 && game.ConsecutiveHits==1,"Real score/combo not updated on HIT.");
        resolve.Invoke(game,new object[]{false,1000,beats[0]});
        Require(game.Score==1000 && game.ConsecutiveHits==0,"MISS did not reset combo.");
        game.StopMode();
        Log("PASS standard/hard: 8/12 cycles, no repeat-boundary phase drift; real score and combo HIT/MISS paths.");
        Require(entry.hipThrustSceneOverride==Path.GetFileNameWithoutExtension(RingBridgeGameBuilder.ScenePath),"Workout does not retain the new bridge scene.");
    }
    static void Capture(string name)
    {
        Canvas.ForceUpdateCanvases();
        Vector3 scorePosition=view.presentationCamera.WorldToViewportPoint(view.score.transform.position);
        Require(scorePosition.x>0 && scorePosition.x<1 && scorePosition.y>0 && scorePosition.y<1,"Score HUD lies outside composition.");
        view.presentationCamera.Render();
        var rt=view.Composition; var previous=RenderTexture.active; RenderTexture.active=rt;
        var image=new Texture2D(rt.width,rt.height,TextureFormat.RGB24,false,true);
        image.ReadPixels(new Rect(0,0,rt.width,rt.height),0,0); image.Apply();
        File.WriteAllBytes(RingBridgeGameBuilder.Output+name+".png",image.EncodeToPNG());
        UnityEngine.Object.DestroyImmediate(image); RenderTexture.active=previous;
    }
    static void ClickVisibleButton(Button button)
    {
        Canvas.ForceUpdateCanvases();
        var pointer=new PointerEventData(EventSystem.current) {position=RectTransformUtility.WorldToScreenPoint(null,button.transform.position)};
        var hits=new System.Collections.Generic.List<RaycastResult>();
        EventSystem.current.RaycastAll(pointer,hits);
        Require(hits.Count>0 && hits[0].gameObject==button.gameObject,"Button is obscured: "+button.name);
        button.onClick.Invoke();
    }
    static void CaptureDisplay(string name="game-with-coach")
    {
        view.SetCaptureResolution(0,0);
        typeof(HipThrustRingBridgeView).GetMethod("LateUpdate",Private).Invoke(view,null);
        var framing=UnityEngine.Object.FindObjectOfType<HipThrustPreviewFraming>();
        typeof(HipThrustPreviewFraming).GetMethod("LateUpdate",Private).Invoke(framing,null);
        framing.previewCamera.Render(); view.presentationCamera.Render();
        var canvases=UnityEngine.Object.FindObjectsOfType<Canvas>().Where(c=>c.isActiveAndEnabled && c.isRootCanvas && c.renderMode==RenderMode.ScreenSpaceOverlay).ToArray();
        var host=new GameObject("Ring Bridge capture camera"); var camera=host.AddComponent<Camera>();
        camera.transform.position=new Vector3(1000,1000,-1000); camera.orthographic=true;
        camera.cullingMask=~(1<<29);
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
            File.WriteAllBytes(RingBridgeGameBuilder.Output+name+".png",image.EncodeToPNG());
            UnityEngine.Object.DestroyImmediate(image);
        }
        finally
        {
            foreach(var canvas in canvases) { canvas.renderMode=RenderMode.ScreenSpaceOverlay; canvas.worldCamera=null; }
            RenderTexture.active=previous; UnityEngine.Object.DestroyImmediate(host); RenderTexture.ReleaseTemporary(rt);
        }
    }
    static void VerifyAspectRatios()
    {
        foreach(var size in new[]{new Vector2Int(1920,1080),new Vector2Int(1024,768),new Vector2Int(2340,1080)})
        {
            view.SetCaptureResolution(size.x,size.y);
            var camera=view.presentationCamera;
            Require(camera.orthographicSize>=1.95f && camera.orthographicSize*camera.aspect>=4.2199f,"Artwork cropped at "+size);
            Vector3 center=view.artwork.transform.position;
            Vector3 origin=camera.WorldToViewportPoint(center);
            float x=(camera.WorldToViewportPoint(center+Vector3.right)-origin).x*size.x;
            float y=(camera.WorldToViewportPoint(center+Vector3.up)-origin).y*size.y;
            Require(Mathf.Abs(x-y)<.1f,"Ring stretched at "+size);
        }
        view.SetCaptureResolution(1688,780);
        Log("PASS UI: pause/resume controls receive the first pointer hit; coach remains visible; HUD is in frame; 16:9, 4:3 and wide-phone artwork fits without stretching.");
    }
    static void Finish(int code)
    {
        SessionState.SetBool(Flag,false); EditorApplication.update-=Tick;
        if(Application.isBatchMode) EditorApplication.Exit(code); else EditorApplication.isPlaying=false;
    }
}
