using System;
using System.IO;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

[InitializeOnLoad]
public static class SideBridgeGameVerification
{
    const string Flag="SideBridgeGameVerification.Active",Output="output/sidebridge/";
    const BindingFlags Private=BindingFlags.Instance|BindingFlags.NonPublic;
    static int variant,stage,checks;
    static GameWorkoutScene entry;
    static HipThrustTaikoController game;
    static HipThrustRingBridgeView view;
    static double started,pausedAt,frozen;
    static Quaternion[] frozenPose;
    static bool countdown,capturedMid;
    static string Scene=>variant==0?SideBridgeGameBuilder.Dynamic:SideBridgeGameBuilder.Hold30;
    static SideBridgeGameVerification(){if(SessionState.GetBool(Flag,false))EditorApplication.update+=Tick;}
    [MenuItem("Tools/Ring Bridge/Verify Left Side Bridge Games")]
    public static void Run()
    {
        Directory.CreateDirectory(Output);
        File.WriteAllText(Output+"verification.txt","Real scenes/recorded HumanPose; synthetic calibration and telemetry for scoring tests. No wearable/patient validation.\n");
        VerifyRule();VerifyTimeline();VerifyOrientation();
        EditorSceneManager.OpenScene(SideBridgeGameBuilder.Dynamic);
        SessionState.SetBool(Flag,true);EditorApplication.update-=Tick;EditorApplication.update+=Tick;EditorApplication.isPlaying=true;
    }
    static void Check(bool value,string message){checks++;if(!value)throw new Exception(message);}
    static void Log(string value)=>File.AppendAllText(Output+"verification.txt",value+"\n");
    static void Feed(SideBridgeHoldRule rule,float seconds,float progress,float quality,bool valid=true)
    {for(int i=0;i<Mathf.RoundToInt(seconds*100);i++)rule.Step(progress,quality,.01f,valid);}
    static void VerifyRule()
    {
        foreach(string difficulty in new[]{"simple","standard","hard"})
        {
            var rule=new SideBridgeHoldRule();rule.Reset(difficulty);rule.BeginRepetition(3,3);
            Feed(rule,.1f,0,100);
            Feed(rule,5,.4f,100);Check(rule.LiveScore==0,"Partial side lift earned hold score");
            Feed(rule,9,1,100);Check(rule.LiveScore==100,"Valid side hold did not reach 100");
            Feed(rule,1,1,0);Check(rule.LiveScore==0&&rule.ContinuousHoldSeconds==0,"Low quality retained current full score");
            Feed(rule,.2f,1,100,false);Check(rule.LiveScore==0,"Invalid input retained live score");
            Feed(rule,.6f,0,100);Check(rule.CompletedRounds==1,"A-B-A not counted once");
            Feed(rule,3,0,100);Check(rule.CompletedRounds==1,"Rest produced extra repetitions");
        }
        var legacy=new HipThrustHoldRule();legacy.Reset("simple");for(int i=0;i<10;i++)legacy.Step(0,100,.01f,true);for(int i=0;i<360;i++)legacy.Step(.25f,100,.01f,true);Check(legacy.LiveScore==100,"Hip entry threshold changed");
        Log("PASS side-specific thresholds reject partial lift; good holds score 100; poor quality clears live score/continuous hold; invalid input scores zero; true low return counts once; original hip thresholds preserved.");
    }
    static void VerifyTimeline()
    {
        var source=AssetDatabase.LoadAssetAtPath<HipThrustPhaseTimeline>(SideBridgeGameBuilder.TimelinePath);source.Validate();
        foreach(int count in new[]{1,3,7,100,2000})foreach(bool dynamic in new[]{true,false})
        {
            var time=new HipThrustHold30Timeline(source,count,dynamic,!dynamic,true);
            Check(time.Repetitions.Length==count,"Dose rounded to source cycle count");
            foreach(var rep in time.Repetitions)
            {
                var c=source.cycles[rep.cycle];
                Check(rep.start<rep.top&&rep.top<rep.lower&&rep.lower<rep.end,"Invalid phase order");
                double hold=dynamic?(c.lowerStartFrame-c.holdStartFrame)/30d:30;
                Check(Math.Abs(rep.lower-rep.top-hold)<.000001,"Wrong hold duration");
                foreach(float fraction in new[]{0,.25f,.5f,.9999f})
                {
                    var sample=time.Evaluate(rep.top+hold*fraction,out float frame,out _,out _);
                    Check(sample.phase==1&&Mathf.Abs(sample.progress-fraction)<.0001f,"Hold progress mismatch");
                    if(!dynamic)Check(Mathf.Abs(frame-rep.peakFrame)<.001,"30-second hold moved coach");
                }
                var lower=time.Evaluate(rep.lower+.2,out float lowered,out _,out _);
                Check(lower.phase==2&&lowered>=c.lowerStartFrame,"Recorded plateau appended after hold");
            }
            time.Evaluate(time.Duration,out _,out _,out int completed);Check(completed==count,"Extra/missing final repetitions");
            if(count>3)
            {
                double wrap=time.Repetitions[2].end+.25;
                Check(time.TryGetWrapProgress(wrap,out float p)&&Mathf.Abs(p-.5f)<.001,"Loop return blend missing");
                Check(time.Evaluate(wrap,out _,out _,out _).phase==-1,"Loop return accidentally scored another phase");
            }
            if(count>1)
            {
                double rest=time.Repetitions[0].end+.2;
                Check(time.Evaluate(rest,out float frame,out _,out _).phase==-1&&Mathf.Abs(frame-(source.cycles[0].lowerEndFrame+6))<.001,"Recorded rest gap was skipped");
            }
        }
        Log("PASS exact doses 1/3/7/100/2000, dynamic recorded timing, 30.000-second frozen peaks and progress, immediate descending phase, recorded rest gaps and 0.5-second loop return blend.");
    }
    static void VerifyOrientation()
    {
        var scene=EditorSceneManager.OpenScene(SideBridgeGameBuilder.Dynamic);
        var source=scene.GetRootGameObjects().SelectMany(r=>r.GetComponentsInChildren<HipThrustTaikoController>(true)).Single();
        var host=UnityEngine.Object.Instantiate(source.CoachAnimator.gameObject);host.transform.SetParent(null);host.transform.SetPositionAndRotation(Vector3.zero,Quaternion.identity);host.transform.localScale=Vector3.one;
        foreach(var script in host.GetComponentsInChildren<MonoBehaviour>(true))script.enabled=false;
        host.SetActive(true);var avatar=host.GetComponent<Animator>();avatar.enabled=true;avatar.Rebind();avatar.Update(0);
        var frames=JsonUtility.FromJson<CoachMotionPackage>(File.ReadAllText(SideBridgeGameBuilder.Root+"SideBridge.coachrecording")).segments[0].frames;
        var bones=Enumerable.Range(0,(int)HumanBodyBones.LastBone).Select(i=>avatar.GetBoneTransform((HumanBodyBones)i)).Where(b=>b!=null).ToArray();
        float rotationError=0,positionError=0;
        try
        {
            using(var handler=new HumanPoseHandler(avatar.avatar,avatar.transform))
            for(int f=330;f<=1470;f++)
            {
                avatar.Play("sidebridge_routine",0,(f-330)/1140f);avatar.Update(0);
                var rotations=bones.Select(b=>b.rotation).ToArray();var positions=bones.Select(b=>b.position).ToArray();
                var expected=new HumanPose();frames[f].ApplyTo(ref expected);handler.SetHumanPose(ref expected);
                for(int b=0;b<bones.Length;b++){rotationError=Mathf.Max(rotationError,Quaternion.Angle(rotations[b],bones[b].rotation));positionError=Mathf.Max(positionError,Vector3.Distance(positions[b],bones[b].position));}
            }
            using(var handler=new HumanPoseHandler(avatar.avatar,avatar.transform))
            for(int endpoint=0;endpoint<2;endpoint++)
            {
                avatar.Play("sidebridge_wrap",0,endpoint);avatar.Update(0);
                var rotations=bones.Select(b=>b.rotation).ToArray();var positions=bones.Select(b=>b.position).ToArray();
                var expected=new HumanPose();frames[endpoint==0?1459:435].ApplyTo(ref expected);handler.SetHumanPose(ref expected);
                for(int b=0;b<bones.Length;b++){rotationError=Mathf.Max(rotationError,Quaternion.Angle(rotations[b],bones[b].rotation));positionError=Mathf.Max(positionError,Vector3.Distance(positions[b],bones[b].position));}
            }
        }
        finally{UnityEngine.Object.DestroyImmediate(host);}
        Check(rotationError<1&&positionError<.01f,"Original hand/foot orientation not preserved");
        Log(FormattableString.Invariant($"PASS original HumanPose: 1141 frames plus both loop-blend endpoints, all humanoid bones including hands/feet/toes; max rotation {rotationError:F5} deg, position {positionError:F6} m; no arm mirroring."));
    }
    static void Tick()
    {
        if(!EditorApplication.isPlaying||EditorApplication.isCompiling)return;
        try
        {
            if(stage==0)
            {
                view=UnityEngine.Object.FindObjectOfType<HipThrustRingBridgeView>();
                if(view==null||view.gameObject.scene.name!=Path.GetFileNameWithoutExtension(Scene))return;
                game=view.transport;entry=view.entry;started=EditorApplication.timeSinceStartup;countdown=false;
                Check(entry.menu.IsTrainingMenuVisible,"Missing start menu");Capture(variant+"-menu");
                entry.scorer.GetComponent<SupineGameCalibration>().skipCalibrationForPreview=true;
                entry.menu.SelectSimpleDifficulty();Check(game.PrescriptionDose.SetDose(1,3,out _),"Dose rejected");
                ((Button)typeof(DeadBugStartMenuController).GetField("startButton",Private).GetValue(entry.menu)).onClick.Invoke();
                stage=1;return;
            }
            Check(EditorApplication.timeSinceStartup-started<85,"Runtime timeout "+stage);
            if(stage==1)
            {
                if(view.IsCountdownVisible&&!countdown)
                {
                    Check(view.IsPreviewVisible,"No countdown coach");Capture(variant+"-countdown");
                    Check(((TMPro.TMP_Text)typeof(HipThrustRingBridgeView).GetField("holdMetric",Private).GetValue(view)).text.Contains("准备开始计时"),"Countdown showed stale hold measurements");
                    Check(((TMPro.TMP_Text)typeof(HipThrustRingBridgeView).GetField("repetitionMetric",Private).GetValue(view)).text.Contains("合格 0/3"),"Countdown showed previous-set completion");
                    countdown=true;
                }
                if(!game.IsRunning)return;
                Check(countdown&&game.RuleScoring.TargetsReady&&game.BeatCount==3,"Startup/targets/dose failed");
                Check(!game.PrescriptionDose.SetDose(1,7,out _),"Dose mutated during training");
                Check(game.RuleScoring.sideBridge!=null,"Wrong scoring profile");
                var music=UnityEngine.Object.FindObjectOfType<RhythmSongClock>().GetComponent<AudioSource>();
                Check(music.isPlaying&&music.clip.name=="SideBridge_EnchantedTiki"&&music.loop,"Side bridge music missing");
                VerifyGeometry();
                var rep=game.Hold30Timeline.Repetitions[0];
                Seek((rep.start+rep.top)*.5);Capture(variant+"-raise");CheckPreview();
                Seek((rep.top+rep.lower)*.5);Capture(variant+"-hold");CheckPreview();
                Seek((rep.lower+rep.end)*.5);Capture(variant+"-lower");CheckPreview();
                Seek(rep.top+.1);game.TogglePause();frozen=game.Elapsed;frozenPose=Bones(game.CoachAnimator);pausedAt=EditorApplication.timeSinceStartup;stage=2;
            }
            else if(stage==2&&EditorApplication.timeSinceStartup-pausedAt>.4)
            {
                Check(game.Elapsed==frozen,"Pause advanced phase");CheckBones(frozenPose,Bones(game.CoachAnimator));
                game.TogglePause();
                if(variant==0){FinishStandalone();return;}
                Seek(game.Hold30Timeline.Repetitions[0].top);frozenPose=Bones(game.CoachAnimator);capturedMid=false;stage=3;
            }
            else if(stage==3)
            {
                var rep=game.Hold30Timeline.Repetitions[0];
                if(game.Elapsed<rep.lower)
                {
                    Check(view.CurrentSample.phase==1,"Hold ended before 30 seconds");CheckBones(frozenPose,Bones(game.CoachAnimator));CheckPreview();
                    if(!capturedMid&&game.Elapsed-rep.top>15){Capture("hold30-halfway");capturedMid=true;}
                    return;
                }
                Check(view.CurrentSample.phase==2,"30-second hold did not enter lowering");
                Log("PASS real-time 30-second hold: coach and preview frozen at recorded peak; progress advances; pause freezes both; lower starts after exactly 30 game seconds.");
                FinishStandalone();
            }
            else if(stage==4)
            {
                var plan=new GameWorkoutPlan{actions=new[]{new GameWorkoutAction{action="sidebridge_hold30",difficulty="standard",sets=2,reps=3}}};
                Check(GameWorkoutSession.Instance.SetPlan(plan,out string error),error);Check(GameWorkoutSession.Instance.StartPlan(out error),error);stage=5;started=EditorApplication.timeSinceStartup;
            }
            else if(stage==5&&game.IsRunning)
            {
                Check(game.BeatCount==3,"Executor first set dose changed");Seek(game.Duration);Call(game,"LateUpdate");
                Check(GameWorkoutSession.Instance.CompletedSets==1&&GameWorkoutSession.Instance.IsRunning,"First set prematurely completed plan");stage=6;
            }
            else if(stage==6&&game.IsRunning)
            {
                Check(game.BeatCount==3,"Executor second set dose changed");Seek(game.Duration);Call(game,"LateUpdate");
                Check(GameWorkoutSession.Instance.IsComplete&&!GameWorkoutSession.Instance.IsRunning,"2 x 3 executor did not finish exactly");
                Log("PASS live prescription executor: two sets x three left-side bridge repetitions, automatic next-set countdown, exactly two completed sets.");
                Log("Assertions: "+checks);Finish(0);
            }
        }
        catch(Exception error){Log("FAIL "+error);Debug.LogException(error);Finish(1);}
    }
    static void FinishStandalone()
    {
        Seek(game.Duration);Call(game,"LateUpdate");Check(!game.IsRunning&&game.CompletedDemonstrations==3,"Session did not finish exactly three reps");
        Check(game.RuleScoring.sideBridge.Rule.QualifiedRounds==0,"Coach playback without patient input counted as qualified repetitions");
        entry.StopTraining();Check(entry.menu.IsTrainingMenuVisible,"Return menu failed");
        Log("PASS "+Path.GetFileNameWithoutExtension(Scene)+": real menu/countdown, all three phases, matching coach preview, music, pause/resume, exact dose, return menu.");
        if(variant++==0){stage=0;SceneManager.LoadScene(Path.GetFileNameWithoutExtension(SideBridgeGameBuilder.Hold30));}
        else {variant=1;stage=4;}
    }
    static void VerifyGeometry()
    {
        var rule=game.RuleScoring;var geometry=rule.sideBridge.Geometry;var animator=game.CoachAnimator;
        int peak=HipThrustHold30Timeline.PeakFrame(game.PhaseTimeline,0);
        animator.Play(game.RoutineStateName,0,peak/30f/game.PhaseTimeline.sourceClip.length);animator.Update(0);
        Check(SideBridgePoseGeometry.TryMeasure(animator,out var target)&&target.IsLeftSupport,"Reference is not left support");
        Check(geometry.Quality(target,target)>99.99f,"Identical pose not 100");
        var root=animator.transform;var pos=root.position;var rotation=root.rotation;var scale=root.localScale;
        try
        {
            root.position+=new Vector3(3,2,5);root.rotation=Quaternion.Euler(0,137,0)*rotation;root.localScale*=1.4f;
            Check(SideBridgePoseGeometry.TryMeasure(animator,out var changed)&&changed.IsLeftSupport,"Heading/scale rejected");
            Check(Mathf.Abs(changed.lift-target.lift)<.0001&&geometry.Quality(changed,target)>99.99f,"Side score depends on heading/scale/translation");
            root.rotation=Quaternion.Euler(0,0,180)*rotation;
            Check(SideBridgePoseGeometry.TryMeasure(animator,out changed)&&!changed.IsLeftSupport,"Opposite support accepted");
        }
        finally{root.SetPositionAndRotation(pos,rotation);root.localScale=scale;}
        var free=animator.GetBoneTransform(HumanBodyBones.RightLowerArm);var freeRotation=free.localRotation;free.localRotation*=Quaternion.Euler(55,20,30);
        Check(SideBridgePoseGeometry.TryMeasure(animator,out var moved)&&geometry.Quality(moved,target)>99.99f,"Free arm penalized");free.localRotation=freeRotation;
        var original=entry.scorer.UserAnimator;
        typeof(DeadBugGamingPoseScorer).GetField("userAnimator",Private).SetValue(entry.scorer,animator);
        typeof(DeadBugGamingPoseScorer).GetField("calibrated",Private).SetValue(entry.scorer,true);
        var motion=UnityEngine.Object.FindObjectOfType<AvatarMotionSource>();
        try
        {
            rule.Begin("simple");
            animator.Play(game.RoutineStateName,0,game.PhaseTimeline.cycles[0].liftStartFrame/30f/game.PhaseTimeline.sourceClip.length);animator.Update(0);
            for(int i=0;i<15;i++){motion.ReportFrame(i*.01f,55);rule.Tick(.01f);}
            animator.Play(game.RoutineStateName,0,peak/30f/game.PhaseTimeline.sourceClip.length);animator.Update(0);
            // Settle the short measurement filter before the instructed hold opens.
            typeof(HipThrustTaikoController).GetField("elapsed",Private).SetValue(game,game.Hold30Timeline.Repetitions[0].top-.1);
            for(int i=0;i<30;i++){motion.ReportFrame(i*.01f,55);rule.Tick(.01f);}
            for(int i=0;i<(rule.sideBridge.Rule.RequiresContinuousHold?3000:1000);i++)
            {
                typeof(HipThrustTaikoController).GetField("elapsed",Private).SetValue(game,game.Hold30Timeline.Repetitions[0].top+(i+1)*.01);
                motion.ReportFrame(i*.01f,55);rule.Tick(.01f);
            }
            Check(rule.LiveScore==100&&rule.MatchesSideBridgeCoach(),"Good live pose not scored/HIT");
            typeof(HipThrustRuleScoring).GetField("lastFrameAt",Private).SetValue(rule,Time.unscaledTime-1);rule.Tick(.01f);
            Check(rule.LiveScore==0&&!rule.MatchesSideBridgeCoach(),"Stale input scored or HIT");
            motion.SetAvailable(false);rule.Tick(.01f);Check(rule.LiveScore==0&&!rule.MatchesSideBridgeCoach(),"Disconnected input HIT");
            rule.Begin("simple");var rep=game.Hold30Timeline.Repetitions[0];
            for(int i=0;i<=(rep.end+.3)*30;i++)
            {
                Seek(i/30d);motion.ReportFrame(i/30f,55);rule.Tick(1/30f);
                if(i==Mathf.FloorToInt((float)(rep.lower-.1)*30))Capture(variant+"-measured-hold");
            }
            Check(rule.sideBridge.Rule.CompletedRounds==1&&rule.sideBridge.Rule.QualifiedRounds==1,"Fresh calibrated full recording did not count exactly one qualified repetition");
            Capture(variant+"-measured-return");
            Check(((TMPro.TMP_Text)typeof(HipThrustRingBridgeView).GetField("repetitionMetric",Private).GetValue(view)).text.Contains("合格 1/3"),"HUD confused qualified and demonstrated repetitions");
        }
        finally
        {
            typeof(DeadBugGamingPoseScorer).GetField("userAnimator",Private).SetValue(entry.scorer,original);
            typeof(DeadBugGamingPoseScorer).GetField("calibrated",Private).SetValue(entry.scorer,false);
            Seek(0);rule.Begin("simple");
        }
        Log("PASS geometry and scoring: left support gate, normalized elbow/ankle lift, body-scale/translation/yaw invariance, free arm ignored, fresh calibrated input scores 100, stale/disconnected input cannot score or HIT.");
    }
    static void Seek(double time){typeof(HipThrustTaikoController).GetField("elapsed",Private).SetValue(game,time);Call(game,"RenderTimeline");Call(view,"LateUpdate");UnityEngine.Object.FindObjectOfType<RecordedDrumArtwork>().LateUpdate();}
    static void Call(object target,string method)=>target.GetType().GetMethod(method,Private).Invoke(target,null);
    static Quaternion[] Bones(Animator animator)=>Enumerable.Range(0,(int)HumanBodyBones.LastBone).Select(i=>animator.GetBoneTransform((HumanBodyBones)i)).Select(b=>b!=null?b.localRotation:Quaternion.identity).ToArray();
    static void CheckBones(Quaternion[] a,Quaternion[] b){for(int i=0;i<a.Length;i++)Check(Quaternion.Angle(a[i],b[i])<.07f,"Frozen/preview mismatch "+(HumanBodyBones)i);}
    static void CheckPreview()=>CheckBones(Bones(game.CoachAnimator),Bones(game.PreviewAnimator));
    static void Capture(string name)
    {
        Call(view,"LateUpdate");var framing=UnityEngine.Object.FindObjectOfType<HipThrustPreviewFraming>();Call(framing,"LateUpdate");framing.previewCamera.Render();view.presentationCamera.Render();
        var canvases=UnityEngine.Object.FindObjectsOfType<Canvas>().Where(c=>c.isActiveAndEnabled&&c.isRootCanvas&&c.renderMode==RenderMode.ScreenSpaceOverlay).ToArray();
        var host=new GameObject("Side bridge verification camera");var camera=host.AddComponent<Camera>();camera.transform.position=new Vector3(1000,1000,-1000);camera.orthographic=true;camera.cullingMask=~(1<<29);camera.clearFlags=CameraClearFlags.SolidColor;camera.backgroundColor=Color.black;
        var rt=RenderTexture.GetTemporary(Mathf.Max(1,Screen.width),Mathf.Max(1,Screen.height),24,RenderTextureFormat.ARGB32,RenderTextureReadWrite.sRGB);var previous=RenderTexture.active;
        try
        {
            camera.targetTexture=rt;foreach(var c in canvases){c.renderMode=RenderMode.ScreenSpaceCamera;c.worldCamera=camera;c.planeDistance=1;}Canvas.ForceUpdateCanvases();camera.Render();RenderTexture.active=rt;
            var image=new Texture2D(rt.width,rt.height,TextureFormat.RGB24,false);image.ReadPixels(new Rect(0,0,rt.width,rt.height),0,0);image.Apply();File.WriteAllBytes(Output+name+".png",image.EncodeToPNG());UnityEngine.Object.DestroyImmediate(image);
        }
        finally
        {
            foreach(var c in canvases){c.renderMode=RenderMode.ScreenSpaceOverlay;c.worldCamera=null;}RenderTexture.active=previous;UnityEngine.Object.DestroyImmediate(host);RenderTexture.ReleaseTemporary(rt);
        }
    }
    static void Finish(int code){SessionState.SetBool(Flag,false);EditorApplication.update-=Tick;Time.timeScale=1;if(Application.isBatchMode)EditorApplication.Exit(code);else EditorApplication.isPlaying=false;}
}
