using System;
using System.IO;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

/// <summary>Deterministic assessment adversaries plus real scene/animation integration.
/// Synthetic calibration/telemetry are confined to this editor verification.</summary>
[InitializeOnLoad]
public static class TrainingAssessmentVerification
{
    const string Flag="TrainingAssessmentVerification",Output="output/training-assessment/";
    const BindingFlags Private=BindingFlags.Instance|BindingFlags.NonPublic;
    static readonly string[] Scenes={"BirdDog_RingSkinC_Recorded","DeadBug_RingSkinC_Recorded","HipThrust_RingBridge_NewScoring","HipThrust_RingBridge_Hold30","SideBridge_RingBridge_Dynamic","SideBridge_RingBridge_Hold30"};
    static int sceneIndex,stage,checks;
    static double started;
    static GameWorkoutScene entry;
    static GameTrainingAssessment assessment;
    static TrainingResultView view;
    static TrainingAssessmentVerification(){if(SessionState.GetBool(Flag,false))EditorApplication.update+=Tick;}
    [MenuItem("Tools/Training/Verify Percentage Settlement")]
    public static void Run()
    {
        Directory.CreateDirectory(Output);File.WriteAllText(Output+"verification.txt","Assessment v1. Synthetic replay; no wearable or clinical validation.\n");
        PureTests();TrainingUsageScenarios.Begin();EditorSceneManager.OpenScene("Assets/Scenes/"+Scenes[0]+".unity");
        SessionState.SetBool(Flag,true);EditorApplication.update-=Tick;EditorApplication.update+=Tick;EditorApplication.isPlaying=true;
    }
    static void Check(bool ok,string message){checks++;if(!ok)throw new Exception(message);}
    static void Log(string value)=>File.AppendAllText(Output+"verification.txt",value+"\n");
    static void PureTests()
    {
        var s=new TrainingQualitySummary();s.Reset(new[]{40d,40d},new[]{30d,30d},true,true,70);
        Check(!s.HasAssessment&&s.Score==0,"Empty session fabricated a score");
        s.Observe(0,40,true,100,true);s.RecordDose(0,300,300,true,true);
        Check(s.Score==50&&s.CreditedSeconds==30&&s.Qualified==1,"One great attempt compensated for a missed repetition");
        s.Observe(1,40,true,100,true);s.RecordDose(1,30,30,true,true);
        Check(s.Score==100,"Full measured prescription failed to score 100");
        s.Reset(new[]{40d},new[]{30d},true,true,70);s.Observe(0,40,true,100,true);s.RecordDose(0,5,5,false,false);
        Check(s.Score==55,"Five seconds produced a full 30-second assessment");
        s.Reset(new[]{40d},new[]{30d},true,true,70);s.Observe(0,40,true,100,true);s.RecordDose(0,30,30,false,false);
        Check(s.Score==80,"No lowering still earned completion points");
        s.Reset(new[]{40d},new[]{30d},true,true,70);s.Observe(0,20,true,100,true);s.Observe(0,20,false,100,true);s.RecordDose(0,15,15,false,false);
        Check(s.Score==40&&s.Coverage==.5f,"Missing data removed from the denominator");
        s.Reset(new[]{10d},new[]{10d},false,false,70);s.Observe(0,10,true,float.NaN,true);
        Check(!s.HasAssessment&&s.Score==0,"NaN became quality");
        foreach(int fps in new[]{30,60,120})
        {
            s.Reset(new[]{10d},new[]{10d},false,false,70);
            for(int f=0;f<10*fps;f++)s.Observe(0,1d/fps,true,80,true);
            s.RecordDose(0,0,0,true,true);Check(s.Score==90&&Math.Abs(s.TrackedSeconds-10)<.0001,"Frame-rate-dependent score");
            var attempt=new TrainingBridgeAttempt();attempt.Step(.1,0,true,true,false,false);
            for(int f=0;f<15*fps;f++)attempt.Step(1d/fps,1d/fps,true,false,true,false);
            attempt.Step(.3,.3,true,false,false,false);
            for(int f=0;f<15*fps;f++)attempt.Step(1d/fps,1d/fps,true,false,true,false);
            attempt.Step(.1,0,true,true,false,true);
            Check(attempt.Completed&&attempt.TotalHold>29.99&&attempt.LongestHold<15.01,"Split hold counted as continuous 30 seconds");
            attempt=new TrainingBridgeAttempt();attempt.Step(.1,0,true,true,false,false);
            for(int f=0;f<15*fps;f++)attempt.Step(1d/fps,1d/fps,true,false,true,false);
            attempt.BreakContinuity();
            for(int f=0;f<15*fps;f++)attempt.Step(1d/fps,1d/fps,true,false,true,false);
            Check(attempt.LongestHold<15.01,"Pause bridged two continuous holds");
        }
        Log("PASS pure model: no input, weighted components, dose caps, missing repetitions, 5/30 seconds, no lowering, missing data, NaN, 30/60/120 fps, interrupted continuous hold.");
    }
    static void Tick()
    {
        if(!EditorApplication.isPlaying||EditorApplication.isCompiling)return;
        try
        {
            if(stage==0)
            {
                entry=UnityEngine.Object.FindObjectOfType<GameWorkoutScene>();
                if(entry==null||entry.gameObject.scene.name!=Scenes[sceneIndex])return;
                view=entry.GetComponent<TrainingResultView>();if(view==null||view.Root==null)return;
                assessment=entry.GetComponent<GameTrainingAssessment>();Check(assessment!=null,"Scene missing settlement collector");
                entry.menu.SelectSimpleDifficulty();entry.scorer.GetComponent<SupineGameCalibration>().skipCalibrationForPreview=true;
                SetMenu(false);
                started=EditorApplication.timeSinceStartup;
                if(entry.taiko!=null)
                {
                    if(entry.taiko.PrescriptionDose!=null)entry.taiko.PrescriptionDose.SetDose(1,3,out _);
                    entry.taiko.StartMode();VerifyBridge();stage=2;
                }
                else
                {
                    Call(entry.coach,"StartSegmentedDeadBug");stage=1;
                }
            }
            else if(stage==1)
            {
                Check(EditorApplication.timeSinceStartup-started<30,"Coach never opened scoring session");
                if(!assessment.IsCollecting)return;
                entry.coach.StopAllCoroutines();VerifyRhythm();stage=2;
            }
            else if(stage==2)
            {
                var end=UnityEngine.Object.FindObjectsOfType<Canvas>(true).Single(c=>c.name=="EndMenuCanvas");
                SetMenu(false);end.gameObject.SetActive(true);Call(view,"LateUpdate");view.Refresh();
                Check(view.Root.activeInHierarchy&&view.ScoreLabel.text!="95","Settlement stayed static/hidden");
                VerifyVisibleLabels();
                Capture(Scenes[sceneIndex]+"-measured");
                // Real restarted session with no patient data must clear every prior result.
                Set(entry.scorer,"calibrated",false);
                if(entry.taiko!=null){entry.taiko.StartMode();Seek(entry.taiko.Duration);entry.taiko.RuleScoring.Tick(.02f);assessment.ObserveBridge(.02f);}
                else Call(assessment,"BeginRhythm");
                assessment.Finish();end.gameObject.SetActive(true);Call(view,"LateUpdate");view.Refresh();
                Check(view.ScoreLabel.text=="—"&&assessment.Summary.Qualified==0,"Preview reused measured results");
                VerifyVisibleLabels();
                Capture(Scenes[sceneIndex]+"-no-data");
                view.Root.GetComponentInChildren<Button>().onClick.Invoke();
                Check(entry.menu.IsTrainingMenuVisible&&!end.gameObject.activeSelf,"Result return button failed");
                Log("PASS "+Scenes[sceneIndex]+": real scene wiring, measured settlement, no-input restart, return button.");
                sceneIndex++;stage=0;
                if(sceneIndex==Scenes.Length){Log("Assertions: "+checks);Finish(0);}else SceneManager.LoadScene(Scenes[sceneIndex]);
            }
        }
        catch(Exception ex){Log("FAIL "+ex);Debug.LogException(ex);Finish(1);}
    }
    static void VerifyBridge()
    {
        var game=entry.taiko;var rule=game.RuleScoring;
        TrainingUsageScenarios.Bridge(entry,entry.scorer.UserAnimator);
        Set(entry.scorer,"userAnimator",game.CoachAnimator);Set(entry.scorer,"calibrated",true);
        var motion=UnityEngine.Object.FindObjectOfType<AvatarMotionSource>();
        foreach(int fps in new[]{30,60,120})
        {
            game.StartMode();
            for(int sample=1;sample<=Math.Ceiling(game.Duration*fps);sample++)
            {
                double now=Math.Min(game.Duration,sample/(double)fps);Seek(now);
                motion.ReportFrame((float)now,55);rule.Tick(1f/fps);assessment.ObserveBridge(1f/fps);
            }
            assessment.Finish();var s=assessment.Summary;
            Log($"REPLAY {Scenes[sceneIndex]} {fps} fps: score={s.Score}, quality={s.Quality:F3}, time={s.CreditedSeconds:F4}/{s.TargetSeconds:F4}, reps={s.Qualified}/{s.Planned}, coverage={s.Coverage:F5}");
            Check(s.Score>=99&&s.Qualified==s.Planned,"Exact bridge recording failed settlement: "+Scenes[sceneIndex]+" "+fps);
            Check(s.Coverage>.999&&s.Quality>99.9,"Bridge quality/coverage mismatch");
            Check(Math.Abs(s.QualifiedFollowingSeconds-s.ExpectedFollowingSeconds)<.001 && s.ExpectedFollowingSeconds<s.ExpectedSeconds,"Hold was included in moving follow time");
        }
        // Paused clock and stale data must not add measurements.
        double measured=assessment.Summary.TrackedSeconds;game.TogglePause();assessment.ObserveBridge(.08f);
        Check(assessment.Summary.TrackedSeconds==measured,"Pause added assessment time");game.TogglePause();
        if(assessment.Summary.IsContinuous)
        {
            game.StartMode();var plan=game.AssessmentRepetitions();int paused=-1;
            for(int sample=1;sample<=Math.Ceiling(game.Duration*30);sample++)
            {
                double now=Math.Min(game.Duration,sample/30d);Seek(now);
                for(int i=0;i<plan.Length;i++)if(paused<i&&now>=plan[i].top+15)
                {game.TogglePause();Call(assessment,"Update");game.TogglePause();paused=i;}
                motion.ReportFrame((float)now,55);rule.Tick(1f/30);assessment.ObserveBridge(1f/30);
            }
            assessment.Finish();Check(assessment.Summary.Qualified==0&&assessment.Summary.LongestHold<15.1,"Paused hold still qualified for 30 continuous seconds");
            Log($"PASS {Scenes[sceneIndex]} pause during each 30-second hold: score={assessment.Summary.Score}, qualified=0, longest={assessment.Summary.LongestHold:F2}s.");
        }
    }
    static void VerifyRhythm()
    {
        var coach=entry.coach;var source=entry.scorer;var motion=UnityEngine.Object.FindObjectOfType<AvatarMotionSource>();
        TrainingUsageScenarios.Rhythm(entry,source.UserAnimator);
        Set(source,"userAnimator",source.CoachAnimator);Set(source,"calibrated",true);
        var up=(string[])Get(coach,"upSegmentStates");var down=(string[])Get(coach,"downSegmentStates");
        Call(assessment,"BeginRhythm");Call(assessment,"BeginSegment",new CoachActionController.SegmentPlaybackInfo(up[0],1));
        var good=new AnatomicalPoseScore{score=100,rawScore=100,validSegmentCount=3};
        var bad=new AnatomicalPoseScore{score=0,rawScore=0,validSegmentCount=3};
        Set(source,"lastLimbScores",new AnatomicalLimbScores{leftArm=bad,rightArm=bad,leftLeg=good,rightLeg=good});
        motion.ReportFrame(0,55);Call(assessment,"Update");Call(assessment,"SampleRhythm",.05d);
        Check(assessment.Summary.Quality==0&&assessment.Summary.CreditedSeconds==0,"Wrong active arm was hidden by a good leg");
        motion.ReportFrame(0,0);Call(assessment,"Update");double captured=assessment.Summary.TrackedSeconds;
        Call(assessment,"SampleRhythm",.05d);Check(assessment.Summary.TrackedSeconds==captured,"Heartbeat with no bones counted as data");
        motion.ReportFrame(0,55);Call(assessment,"Update");Set(assessment,"lastPacketTime",Time.unscaledTime-1);
        Call(assessment,"SampleRhythm",.05d);Check(assessment.Summary.TrackedSeconds==captured,"Stale bones counted as data");
        var profile=Get(coach,"simpleDifficulty");float upSpeed=(float)profile.GetType().GetField("upSpeed").GetValue(profile);
        float downSpeed=(float)profile.GetType().GetField("downSpeed").GetValue(profile);
        foreach(int fps in new[]{30,60,120})
        {
            Call(assessment,"BeginRhythm");
            for(int pair=0;pair<up.Length;pair++)for(int direction=0;direction<2;direction++)
            {
                string state=direction==0?up[pair]:down[pair];float speed=direction==0?upSpeed:downSpeed;
                Call(assessment,"BeginSegment",new CoachActionController.SegmentPlaybackInfo(state,speed));
                double duration=coach.SegmentSeconds(state,speed);
                for(int i=1;i<=Math.Ceiling(duration*fps);i++)
                {
                    double elapsed=Math.Min(duration,i/(double)fps);source.CoachAnimator.Play(state,0,(float)(elapsed/duration));source.CoachAnimator.Update(0);
                    motion.ReportFrame((float)elapsed,55);Call(assessment,"Update");
                    // The synchronous replay has a frozen Unity wall clock. Reset
                    // engine history rather than accumulate thousands of same-time poses.
                    var engine=(AnatomicalMotionScoringEngine)Get(source,"scoringEngine");if(engine!=null)engine.ResetSession();
                    Call(source,"ScoreCurrentPose");
                    Call(assessment,"SampleRhythm",Math.Min(1d/fps,duration-(i-1)/(double)fps));
                }
                Call(assessment,"EndSegment",new CoachActionController.ScoringCheckpointInfo(state,0,1,pair,direction==0,0));
            }
            assessment.Finish();var s=assessment.Summary;
            Log($"REPLAY {Scenes[sceneIndex]} {fps} fps: score={s.Score}, quality={s.Quality:F3}, time={s.CreditedSeconds:F4}/{s.TargetSeconds:F4}, reps={s.Qualified}/{s.Planned}");
            Check(s.Score>=99&&s.Qualified==s.Planned,"Exact rhythm recording failed settlement");
        }
        // The active pair is contralateral and a bad limb cannot be hidden by its partner.
        Check(entry.rhythm.TryGetSegmentTargets(up[0],out var first,out var second),"Missing limb targets");
        Check((first==RhythmBodyTarget.LeftFoot&&second==RhythmBodyTarget.RightHand)||(first==RhythmBodyTarget.RightFoot&&second==RhythmBodyTarget.LeftHand),"Wrong ipsilateral assessment mapping");
    }
    static void Seek(double time){Set(entry.taiko,"elapsed",time);Call(entry.taiko,"RenderTimeline");}
    static void VerifyVisibleLabels()
    {
        var labels=view.Root.GetComponentsInChildren<TMPro.TMP_Text>().Select(t=>t.text).ToArray();
        Check(labels.Length==8,"Result contains extra explanatory labels");
        foreach(var required in new[]{"总成绩","动作质量","合格保持时长","合格跟随时长","合格次数"})
            Check(labels.Any(t=>t.Contains(required)),"Missing requested result metric "+required);
        foreach(var removed in new[]{"权重","数据覆盖","缺测","校准","本组测得结果","合格线","（动作）","规则"})
            Check(!labels.Any(t=>t.Contains(removed)),"Unwanted result explanation "+removed);
    }
    static void SetMenu(bool visible)
    {
        var menu=(GameObject)Get(entry.menu,"menuRoot");menu.SetActive(visible);
    }
    static object Get(object o,string n)=>o.GetType().GetField(n,Private).GetValue(o);
    static void Set(object o,string n,object value)=>o.GetType().GetField(n,Private).SetValue(o,value);
    static object Call(object o,string n,params object[] args)=>o.GetType().GetMethod(n,Private).Invoke(o,args);
    static void Capture(string name)
    {
        var canvases=UnityEngine.Object.FindObjectsOfType<Canvas>().Where(c=>c.isActiveAndEnabled&&c.isRootCanvas&&c.renderMode==RenderMode.ScreenSpaceOverlay).ToArray();
        var host=new GameObject("Assessment capture");var camera=host.AddComponent<Camera>();camera.transform.position=new Vector3(1000,1000,-1000);camera.orthographic=true;camera.clearFlags=CameraClearFlags.SolidColor;camera.backgroundColor=Color.black;
        var rt=RenderTexture.GetTemporary(1920,1080,24,RenderTextureFormat.ARGB32,RenderTextureReadWrite.sRGB);var previous=RenderTexture.active;
        try
        {
            camera.targetTexture=rt;foreach(var c in canvases){c.renderMode=RenderMode.ScreenSpaceCamera;c.worldCamera=camera;c.planeDistance=1;}Canvas.ForceUpdateCanvases();camera.Render();RenderTexture.active=rt;
            var image=new Texture2D(rt.width,rt.height,TextureFormat.RGB24,false);image.ReadPixels(new Rect(0,0,rt.width,rt.height),0,0);image.Apply();File.WriteAllBytes(Output+name+".png",image.EncodeToPNG());UnityEngine.Object.DestroyImmediate(image);
        }
        finally{foreach(var c in canvases){c.renderMode=RenderMode.ScreenSpaceOverlay;c.worldCamera=null;}RenderTexture.active=previous;UnityEngine.Object.DestroyImmediate(host);RenderTexture.ReleaseTemporary(rt);}
    }
    static void Finish(int code){SessionState.SetBool(Flag,false);EditorApplication.update-=Tick;if(Application.isBatchMode)EditorApplication.Exit(code);else EditorApplication.isPlaying=false;}
}
