using System;
using System.IO;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>Real menu/transport smoke tests plus deterministic scoring and prescription boundary tests.</summary>
[InitializeOnLoad]
public static class HipThrustNewRulesVerification
{
    const string Flag="HipThrustNewRulesVerification.Active", Output="output/hipthrust-new-rules/";
    const BindingFlags Private=BindingFlags.Instance|BindingFlags.NonPublic;
    static GameWorkoutScene entry;
    static HipThrustTaikoController game;
    static HipThrustRingBridgeView view;
    static int stage,variant,checks;
    static bool countdownCaptured,completed;
    static double started,pauseStarted,frozen;
    static Quaternion[] frozenPose;
    static HipThrustNewRulesVerification() { if(SessionState.GetBool(Flag,false)) EditorApplication.update+=Tick; }

    [MenuItem("Tools/Ring Bridge/Verify New Scoring and 30 Second Scenes")]
    public static void Run()
    {
        Directory.CreateDirectory(Output);
        File.WriteAllText(Output+"verification.txt","Real Unity scenes/animations. Scoring inputs are synthetic; no patient or wearable validation.\n");
        TestRule(); TestTimeline(); TestPlan();
        EditorSceneManager.OpenScene(HipThrustNewRulesBuilder.NewScore);
        SessionState.SetBool(Flag,true); EditorApplication.update-=Tick; EditorApplication.update+=Tick;
        EditorApplication.isPlaying=true;
    }
    static void Check(bool value,string message) { checks++; if(!value) throw new Exception(message); }
    static void Log(string message)=>File.AppendAllText(Output+"verification.txt",message+"\n");
    static void Feed(HipThrustHoldRule rule,float seconds,float progress,float quality,bool valid=true)
    { for(int i=0;i<Mathf.RoundToInt(seconds/.01f);i++) rule.Step(progress,quality,.01f,valid); }
    static void TestRule()
    {
        string[] names={"simple","standard","hard"}; float[] gates={.20f,.40f,.55f},qualityGates={35,50,65},bases={85,80,75},seconds={3,5,7};
        for(int n=0;n<3;n++)
        {
            var rule=new HipThrustHoldRule(); rule.Reset(names[n]);
            Feed(rule,1,0,100); Check(rule.LiveScore==0,"Rest must be zero");
            Feed(rule,1,gates[n]-.01f,100); Check(rule.LiveScore==0,"Below progress threshold entered B");
            Feed(rule,1,gates[n],qualityGates[n]-.01f); Check(rule.LiveScore==0,"Below quality threshold entered B");
            Feed(rule,.29f,gates[n],qualityGates[n]); Check(rule.LiveScore==0,"Entry confirmed too early");
            Feed(rule,.01f,gates[n],qualityGates[n]); Check(Mathf.Abs(rule.LiveScore-bases[n])<.001f,"Wrong B entry score");
            Feed(rule,seconds[n]/2,1,100); Check(Mathf.Abs(rule.LiveScore-(100+bases[n])/2)<.02f,"Hold score not linear");
            float before=rule.LiveScore,hold=rule.ValidHoldSeconds;
            Feed(rule,.5f,1,0); Check(rule.Phase==HipThrustHoldRule.Stage.Holding && rule.LiveScore==0 && rule.ValidHoldSeconds==hold,"Poor posture retained a live score or earned time");
            Feed(rule,.2f,1,100,false); Check(rule.LiveScore==0 && rule.ValidHoldSeconds==hold,"Missing input earned points");
            Feed(rule,seconds[n]/2+.02f,1,100); Check(rule.LiveScore>99.99f,"Wrong full-score duration");
            Feed(rule,.11f,0,0); Check(rule.Phase==HipThrustHoldRule.Stage.Holding,"Exit jitter not rejected");
            Feed(rule,.01f,0,0); Check(rule.Phase==HipThrustHoldRule.Stage.Lowering && rule.LiveScore==0 && rule.LastRoundBest>99.99f,"B exit must clear live and retain best");
            Feed(rule,.07f,0,0); Check(rule.CompletedRounds==0,"Rest confirmed early");
            Feed(rule,.01f,0,0); Check(rule.CompletedRounds==1 && rule.RoundBest==0 && rule.LastRoundBest>99.99f,"Round reset wrong");
            rule.Reset(names[n]); Feed(rule,.1f,0,100);Feed(rule,.5f,1,100); Feed(rule,.12f,0,0); Feed(rule,.3f,gates[n]-.01f,0);
            Check(rule.CompletedRounds==0,"Partial return counted");Feed(rule,.08f,0,0);Check(rule.CompletedRounds==1,"Full return not counted");
            rule.Reset(names[n]); Feed(rule,.2f,1,100); Feed(rule,.1f,1,100,false); Feed(rule,.2f,1,100);
            Check(rule.LiveScore==0,"Disconnected frames incorrectly joined entry confirmation");
        }
        Log("PASS score: all 3 profiles; low-position arming; 0.30/0.12/0.08 second hysteresis; linear valid holds; bad posture scores zero; missing input; best retention; strict low return.");
    }
    static void TestTimeline()
    {
        var source=AssetDatabase.LoadAssetAtPath<HipThrustPhaseTimeline>(RingBridgeGameBuilder.TimelinePath);
        foreach(int count in new[]{1,3,7,100,2000})
        {
            var timeline=new HipThrustHold30Timeline(source,count);
            Check(timeline.Repetitions.Length==count,"Dose rounded to four");
            foreach(var rep in timeline.Repetitions)
            {
                Check(Math.Abs(rep.lower-rep.top-30)<1e-8,"Hold is not exactly 30 seconds");
                var c=source.cycles[rep.cycle];
                for(int f=c.liftStartFrame;f<=c.lowerEndFrame;f++)
                    Check(source.recordedHeight.Evaluate(f)<=source.recordedHeight.Evaluate(rep.peakFrame)+.000001f,"Frozen pose is not the recorded maximum");
                foreach(double offset in new[]{0d,3d,15d,29.999d})
                {
                    var sample=timeline.Evaluate(rep.top+offset,out float frame,out float remaining,out _);
                    Check(sample.phase==1 && frame==rep.peakFrame && Math.Abs(remaining-(30-offset))<.001 && Math.Abs(sample.progress-offset/30)<.0001,"Hold sample/remaining/progress mismatch");
                }
                Check(timeline.Evaluate(rep.top-.0001,out _,out _,out _).phase==0,"Lift ended early");
                Check(timeline.Evaluate(rep.lower,out float lowerFrame,out _,out _).phase==2 && lowerFrame==rep.peakFrame,"Lower boundary jumps away from the peak");
            }
            timeline.Evaluate(timeline.Duration,out float finalFrame,out _,out int complete);
            Check(complete==count && finalFrame==source.cycles[(count-1)%4].lowerEndFrame,"Final source plays extra repetitions");
        }
        Log("PASS exact doses 1/3/7/100/2000, every repetition peak, static 30-second hold, smooth source-time boundaries, exact completion without extra source cycles.");
    }
    static void TestPlan()
    {
        var plan=JsonUtility.FromJson<GameWorkoutPlan>("{\"actions\":[{\"action\":\"hipthrust_hold30\",\"difficulty\":\"standard\",\"sets\":2,\"reps\":3}]}").ValidatedCopy();
        Check(plan.actions[0].reps==3 && plan.actions[0].sets==2,"Executor lost prescription dose");
        Check(GameWorkoutPlan.SceneFor(plan.actions[0].action)==Path.GetFileNameWithoutExtension(HipThrustNewRulesBuilder.Hold30),"Wrong scene route");
        bool rejected=false; try { plan.actions[0].reps=0; plan.ValidatedCopy(); } catch(ArgumentException) { rejected=true; }
        Check(rejected,"Missing reps silently fell back to difficulty");
        Check(GameWorkoutPlan.SceneFor("hipthrust")=="HipThrust_RingBridge_Recorded","Legacy route changed");
        Log("PASS prescription plan route, exact sets/reps retention, invalid dose rejection, original route preserved.");
    }
    static void Tick()
    {
        if(!EditorApplication.isPlaying || EditorApplication.isCompiling) return;
        try
        {
            if(stage==0)
            {
                entry=UnityEngine.Object.FindObjectOfType<GameWorkoutScene>();
                if(entry==null || (variant==1 && entry.actionId!="hipthrust_hold30")) return;
                game=entry.taiko; view=UnityEngine.Object.FindObjectOfType<HipThrustRingBridgeView>();
                Check(game.RuleScoring!=null && entry.menu.IsTrainingMenuVisible,"Missing new rule or start menu");
                if(variant==1)
                {
                    var dose=game.PrescriptionDose;
                    Check(dose.IsPreview && dose.Total==4,"Preview prescription not explicitly loaded");
                    Check(!dose.SetPrescriptionJson("{\"prescriptionContractVersion\":2,\"items\":[{\"actionId\":\"7\",\"sets\":1,\"reps\":3}]}",0,out _),"Non-bridge Rx accepted");
                    string json="{\"prescriptionContractVersion\":2,\"items\":[{\"itemId\":10,\"actionId\":\"5\",\"sets\":2,\"reps\":3}]}";
                    Check(dose.SetPrescriptionJson(json,10,out _) && dose.Total==6 && !dose.IsPreview,"Doctor V2 dose not applied");
                    Check(!dose.SetDose(0,3,out _) && dose.Total==6,"Invalid update changed valid dose");
                    Check(entry.PrepareDose(new GameWorkoutAction{action="hipthrust_hold30",sets=2,reps=3},out _) && dose.Total==3,"Executor double-multiplied sets");
                }
                Capture("variant-"+variant+"-menu");
                entry.scorer.GetComponent<SupineGameCalibration>().skipCalibrationForPreview=true;
                completed=false; game.Completed+=()=>completed=true;
                entry.menu.SelectSimpleDifficulty(); entry.menu.StartTraining();
                started=EditorApplication.timeSinceStartup; countdownCaptured=false; stage=1;
            }
            if(EditorApplication.timeSinceStartup-started>70) throw new Exception("Runtime timeout stage "+stage);
            if(stage==1)
            {
                if(view.IsCountdownVisible && !countdownCaptured)
                { Check(view.IsPreviewVisible,"Coach hidden during countdown"); Capture("variant-"+variant+"-countdown"); countdownCaptured=true; }
                if(!game.IsRunning) return;
                Check(countdownCaptured,"Original countdown was skipped");
                Check(game.RuleScoring.TargetsReady,"A/B geometry capture failed");
                Check(view.IsPreviewVisible,"Coach preview missing");
                if(variant==0)
                {
                    Check(game.BeatCount==4 && game.Hold30Timeline==null,"First variant changed original duration/count");
                    Check(Math.Abs(game.Duration-game.PhaseTimeline.sourceClip.length)<.001,"First variant changed source timing");
                    Seek(HipThrustHold30Timeline.PeakFrame(game.PhaseTimeline,0)/30d);
                    TestGeometryAndRuntimeScore();
                    Capture("newscore-hold");
                }
                else
                {
                    Check(game.BeatCount==3,"Actual transport did not execute exact prescription count");
                    Check(!game.PrescriptionDose.SetDose(1,7,out _),"Mid-game dose mutation allowed");
                    var rep=game.Hold30Timeline.Repetitions[0];
                    Seek((rep.start+rep.top)/2); Capture("hold30-raise");
                    Seek(rep.top); frozenPose=Bones(game.CoachAnimator);
                    foreach(double offset in new[]{.1,15,29.999})
                    {
                        Seek(rep.top+offset); CheckBones(frozenPose,Bones(game.CoachAnimator));
                        CheckBones(Bones(game.CoachAnimator),Bones(game.PreviewAnimator));
                        Check(view.CurrentSample.phase==1 && Mathf.Abs(view.CurrentSample.progress-(float)(offset/30))<.0001,"Hold artwork clock mismatch");
                    }
                    Seek(rep.top+15); Capture("hold30-middle");
                }
                game.TogglePause(); frozen=game.Elapsed; pauseStarted=EditorApplication.timeSinceStartup; stage=2;
                return;
            }
            if(stage==2 && EditorApplication.timeSinceStartup-pauseStarted>.6)
            {
                Check(game.IsPaused && game.Elapsed==frozen,"Paused clock advanced");
                Check(view.IsPreviewVisible,"Pause hid coach"); Capture("variant-"+variant+"-paused");
                game.TogglePause();
                if(variant==1)
                {
                    var rep=game.Hold30Timeline.Repetitions[0]; Seek(rep.lower+.5); Check(view.CurrentSample.phase==2,"Did not lower after 30 seconds"); Capture("hold30-lower");
                    Seek(game.Hold30Timeline.Repetitions[2].end-.001); Check(game.CompletedDemonstrations==2,"Count advanced before final descent ended");
                }
                Seek(game.Duration); Call(game,"LateUpdate");
                Check(completed && !game.IsRunning,"Session not completed");
                var end=UnityEngine.Object.FindObjectsOfType<Canvas>(true).Single(c=>c.name=="EndMenuCanvas");
                Check(end.gameObject.activeSelf,"Result menu hidden"); Capture("variant-"+variant+"-result");
                game.StopAndShowMenu(); Call(view,"LateUpdate"); Check(entry.menu.IsTrainingMenuVisible && !end.gameObject.activeSelf,"Cannot return to menu");
                if(variant==0)
                {
                    foreach(string difficulty in new[]{"standard","hard"})
                    {
                        entry.PrepareMenu(difficulty); game.StartMode();
                        int passes=difficulty=="standard"?2:3;
                        Check(game.BeatCount==passes*4,"First variant difficulty count changed");
                        double loop=(game.PhaseTimeline.repeatEndFrame-game.PhaseTimeline.repeatStartFrame)/30d;
                        Check(Math.Abs(game.Duration-(game.PhaseTimeline.sourceClip.length+(passes-1)*loop))<.001,"First variant loop duration changed");
                        game.StopMode();
                    }
                    Log("PASS new-score scene: original 4/8/12 dose/timing, menu/countdown/coach, calibrated scoring path, pause/result/return.");
                    variant=1; stage=0; SceneManager.LoadScene(Path.GetFileNameWithoutExtension(HipThrustNewRulesBuilder.Hold30));
                }
                else
                {
                    Log("PASS 30-second scene: D01 prescription, executor per-set dose, exact 3 repetitions, coach + preview bones frozen (including hands/feet), artwork progress/countdown, pause/resume, descent/completion, result/menu.");
                    stage=3;
                }
            }
            else if(stage==3)
            {
                var plan=new GameWorkoutPlan {actions=new[]{new GameWorkoutAction {action="hipthrust_hold30",difficulty="standard",sets=2,reps=3}}};
                Check(GameWorkoutSession.Instance.SetPlan(plan,out string error),"SetPlan failed: "+error);
                Check(GameWorkoutSession.Instance.StartPlan(out error),"StartPlan failed: "+error);
                started=EditorApplication.timeSinceStartup; stage=4;
            }
            else if(stage==4 && game.IsRunning)
            {
                Check(game.BeatCount==3 && game.PrescriptionDose.Total==3,"Live executor dose is not 3");
                Seek(game.Hold30Timeline.Repetitions[0].top);
                frozenPose=Bones(game.CoachAnimator); frozen=game.Elapsed; pauseStarted=EditorApplication.timeSinceStartup; stage=5;
            }
            else if(stage==5)
            {
                var rep=game.Hold30Timeline.Repetitions[0];
                if(game.Elapsed<rep.lower)
                {
                    Check(view.CurrentSample.phase==1,"Real running clock left hold early");
                    CheckBones(frozenPose,Bones(game.CoachAnimator));
                    return;
                }
                Check(game.Elapsed-frozen>=30 && view.CurrentSample.phase==2,"Real 30-second timer did not enter lower");
                Log("PASS real-time playback: first prescription hold ran for 30 game seconds with unchanged coach bones before descending.");
                Seek(game.Duration); Call(game,"LateUpdate");
                Check(GameWorkoutSession.Instance.CompletedSets==1 && GameWorkoutSession.Instance.IsRunning,"First set incorrectly finished the prescription");
                stage=6; started=EditorApplication.timeSinceStartup;
            }
            else if(stage==6 && game.IsRunning)
            {
                Check(game.BeatCount==3 && game.PrescriptionDose.Total==3,"Second set dose changed");
                Seek(game.Duration); Call(game,"LateUpdate");
                Check(GameWorkoutSession.Instance.IsComplete && !GameWorkoutSession.Instance.IsRunning,"2 x 3 prescription did not finish exactly twice");
                Log("PASS live GameWorkoutSession: 2 sets x 3 reps, real automatic set transition/countdown, exactly 2 completion events.");
                Log("Runtime assertions: "+checks); Finish(0);
            }
        }
        catch(Exception ex) { Log("FAIL "+ex); Debug.LogException(ex); Finish(1); }
    }
    static void TestGeometryAndRuntimeScore()
    {
        var geometry=new HipThrustHoldGeometry(); var animator=game.CoachAnimator;
        Check(geometry.TryMeasurePose(animator,out var reference),"Cannot measure coach");
        Check(Mathf.Abs(geometry.CalculatePoseQuality(reference,reference,true)-100)<.001,"Identical pose quality not 100");
        var root=animator.transform; var position=root.position; var rotation=root.rotation; var scale=root.localScale;
        try
        {
            root.position+=new Vector3(8,-2,5); root.rotation=Quaternion.Euler(20,78,-11)*root.rotation; root.localScale*=1.7f;
            Check(geometry.TryMeasurePose(animator,out var changed),"Transformed capture failed");
            Check(Mathf.Abs(reference.bridgeLiftRatio-changed.bridgeLiftRatio)<.0001,"Support-plane ratio depends on root transform/scale");
            Check(geometry.CalculatePoseQuality(changed,reference,false)>99.99f,"Body-relative quality depends on root transform");
        }
        finally { root.SetPositionAndRotation(position,rotation); root.localScale=scale; }
        var scorer=game.RuleScoring; var original=entry.scorer.UserAnimator;
        typeof(DeadBugGamingPoseScorer).GetField("userAnimator",Private).SetValue(entry.scorer,animator);
        typeof(DeadBugGamingPoseScorer).GetField("calibrated",Private).SetValue(entry.scorer,true);
        var motion=UnityEngine.Object.FindObjectOfType<AvatarMotionSource>();
        try
        {
            scorer.Begin("simple");Seek(game.AssessmentRepetitions()[0].start);
            for(int i=0;i<15;i++){motion.ReportFrame(i*.01f,55);scorer.Tick(.01f);}
            Seek(game.Hold30Timeline!=null?game.Hold30Timeline.Repetitions[0].top+.1:HipThrustHold30Timeline.PeakFrame(game.PhaseTimeline,0)/30d);
            for(int i=0;i<(scorer.Rule.ContinuousHold?3100:350);i++) { motion.ReportFrame(i*.01f,55); scorer.Tick(.01f); }
            Check(game.Score==100 && scorer.Progress>.99f,$"Measured B pose did not reach new-rule score: score={game.Score}, progress={scorer.Progress}, phase={scorer.Rule.Phase}, armed={scorer.Rule.LowConfirmed}, fresh={scorer.InputFresh}, quality={scorer.Quality}, tracking={scorer.TrackingQuality}");
            // A packet arrives during pause, then the sensor stops. Resume must
            // not treat that old counter increment as fresh input.
            motion.ReportFrame(4,55); Call(scorer,"Update");
            typeof(HipThrustRuleScoring).GetField("lastFrameAt",Private).SetValue(scorer,Time.unscaledTime-1);
            float held=scorer.Rule.ValidHoldSeconds; scorer.Tick(.01f);
            Check(game.Score==0 && scorer.Rule.ValidHoldSeconds<=held,"Resume counted stale input received during pause");
            float best=scorer.Rule.SessionBest; motion.SetAvailable(false); scorer.Tick(.01f);
            Check(game.Score==0 && scorer.Rule.SessionBest==best,"Unavailable input kept earning live score");
        }
        finally
        {
            typeof(DeadBugGamingPoseScorer).GetField("userAnimator",Private).SetValue(entry.scorer,original);
            typeof(DeadBugGamingPoseScorer).GetField("calibrated",Private).SetValue(entry.scorer,false);
        }
        Log("PASS measured geometry: identical pose 100, translation/rotation/scale invariant; synthetic fresh VMC telemetry + calibrated Animator reaches 100; unavailable telemetry clears live score.");
    }
    static void Seek(double time) { typeof(HipThrustTaikoController).GetField("elapsed",Private).SetValue(game,time); Call(game,"RenderTimeline"); Call(view,"LateUpdate"); UnityEngine.Object.FindObjectOfType<RecordedDrumArtwork>().LateUpdate(); }
    static void Call(object obj,string method)=>obj.GetType().GetMethod(method,Private).Invoke(obj,null);
    static Quaternion[] Bones(Animator animator)=>Enumerable.Range(0,(int)HumanBodyBones.LastBone).Select(i=>animator.GetBoneTransform((HumanBodyBones)i)).Select(b=>b!=null?b.localRotation:Quaternion.identity).ToArray();
    static void CheckBones(Quaternion[] a,Quaternion[] b) { for(int i=0;i<a.Length;i++) Check(Quaternion.Angle(a[i],b[i])<.06f,"Frozen/preview bone mismatch "+(HumanBodyBones)i); }

    static void Capture(string name)
    {
        Call(view,"LateUpdate");
        var framing=UnityEngine.Object.FindObjectOfType<HipThrustPreviewFraming>(); Call(framing,"LateUpdate");
        framing.previewCamera.Render(); view.presentationCamera.Render();
        var canvases=UnityEngine.Object.FindObjectsOfType<Canvas>().Where(c=>c.isActiveAndEnabled && c.isRootCanvas && c.renderMode==RenderMode.ScreenSpaceOverlay).ToArray();
        var host=new GameObject("New rules verification capture"); var camera=host.AddComponent<Camera>();
        camera.transform.position=new Vector3(1000,1000,-1000); camera.orthographic=true; camera.cullingMask=~(1<<29);
        camera.clearFlags=CameraClearFlags.SolidColor; camera.backgroundColor=Color.black;
        var rt=RenderTexture.GetTemporary(Mathf.Max(1,Screen.width),Mathf.Max(1,Screen.height),24,RenderTextureFormat.ARGB32,RenderTextureReadWrite.sRGB);
        var previous=RenderTexture.active;
        try
        {
            camera.targetTexture=rt;
            foreach(var c in canvases) { c.renderMode=RenderMode.ScreenSpaceCamera; c.worldCamera=camera; c.planeDistance=1; }
            Canvas.ForceUpdateCanvases(); camera.Render(); RenderTexture.active=rt;
            var image=new Texture2D(rt.width,rt.height,TextureFormat.RGB24,false);
            image.ReadPixels(new Rect(0,0,rt.width,rt.height),0,0); image.Apply(); File.WriteAllBytes(Output+name+".png",image.EncodeToPNG()); UnityEngine.Object.DestroyImmediate(image);
        }
        finally
        {
            foreach(var c in canvases) { c.renderMode=RenderMode.ScreenSpaceOverlay; c.worldCamera=null; }
            RenderTexture.active=previous; UnityEngine.Object.DestroyImmediate(host); RenderTexture.ReleaseTemporary(rt);
        }
    }
    static void Finish(int code)
    {
        SessionState.SetBool(Flag,false); EditorApplication.update-=Tick;
        if(Application.isBatchMode) EditorApplication.Exit(code); else EditorApplication.isPlaying=false;
    }
}
