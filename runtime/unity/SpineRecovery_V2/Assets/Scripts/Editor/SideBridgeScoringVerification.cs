using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>Real recorded-clip replay and adversarial scoring inputs; no patient-data claims.</summary>
public static class SideBridgeScoringVerification
{
    const BindingFlags Private=BindingFlags.Instance|BindingFlags.NonPublic;
    static readonly StringBuilder report=new StringBuilder();
    static int checks;
    [MenuItem("Tools/Ring Bridge/Verify Side Bridge Scoring")]
    public static void Run()
    {
        report.Clear();checks=0;
        report.AppendLine("Real Unity/recorded HumanPose. Adversarial inputs are synthetic; no wearable/clinical validation.");
        try
        {
            VerifyBoundaries();VerifyRecordings();
            report.AppendLine("PASS assertions: "+checks);
        }
        catch(Exception error){report.AppendLine("FAIL "+error);throw;}
        finally{Directory.CreateDirectory("output/sidebridge-scoring");File.WriteAllText("output/sidebridge-scoring/scoring.txt",report.ToString());}
    }
    static void Check(bool pass,string message){checks++;if(!pass)throw new Exception(message);}
    static void Feed(SideBridgeHoldRule rule,float seconds,float progress,float quality=100,bool valid=true)
    {for(int i=0;i<Mathf.RoundToInt(seconds*100);i++)rule.Step(progress,quality,.01f,valid);}
    static void VerifyBoundaries()
    {
        var unbound=new GameObject("Unbound side scoring test");
        try
        {
            var policy=unbound.AddComponent<SideBridgePoseScoring>();policy.Begin("standard");policy.Step(1,100,.01f,true);
            Check(policy.Rule.LiveScore==0,"Missing references produced a score");
        }
        finally{UnityEngine.Object.DestroyImmediate(unbound);}
        foreach(string difficulty in new[]{"simple","standard","hard"})
        {
            var rule=new SideBridgeHoldRule();rule.Reset(difficulty);rule.BeginRepetition(30,rule.DefaultScoreSeconds);
            Feed(rule,31,1);Check(rule.LiveScore==0&&rule.QualifiedRounds==0,"Starting at the top bypassed low-position arming");
            Feed(rule,.1f,0);Feed(rule,10,1);Check(rule.LiveScore==100,"Short hold score missing");
            Check(rule.QualifiedRounds==0,"Short full score counted as a 30-second repetition");
            Feed(rule,.3f,1,0);Check(rule.LiveScore==0&&rule.ContinuousHoldSeconds==0,"Poor quality preserved current full score/continuous timer");
            Feed(rule,20,1);Feed(rule,.3f,0);
            Check(rule.CompletedRounds==1&&rule.QualifiedRounds==0,"Two interrupted short holds were merged into 30 seconds");
            rule.BeginRepetition(30,rule.DefaultScoreSeconds);Feed(rule,.1f,0);Feed(rule,28.9f,1);Feed(rule,.3f,0);
            Check(rule.QualifiedRounds==0,"28.9 seconds bypassed the bounded 30-second qualification tolerance");
            rule.BeginRepetition(30,rule.DefaultScoreSeconds);Feed(rule,.1f,0);Feed(rule,30,1);
            Check(rule.QualifiedRounds==0,"High position counted before returning low");
            Feed(rule,.3f,0);Check(rule.QualifiedRounds==1,"True 30-second hold and full return did not count");
            Feed(rule,31,1);Feed(rule,.3f,0);Check(rule.QualifiedRounds==1,"Same demonstrated repetition counted twice");
            rule.BeginRepetition(3,3);Feed(rule,.1f,0);Feed(rule,4,1);Feed(rule,1,1,100,false);Feed(rule,4,1);Feed(rule,.3f,0);
            Check(rule.QualifiedRounds==1,"Long tracking gap joined unrelated high/low poses");
        }
        var partial=new SideBridgeHoldRule();partial.Reset("standard");partial.BeginRepetition(3,3);
        Feed(partial,.1f,0);Feed(partial,4,1);Feed(partial,1,.55f);Feed(partial,4,1);Feed(partial,1,.55f);
        Check(partial.CompletedRounds==0,"55%-100%-55% partial repetitions counted");
        Feed(partial,.07f,0);Check(partial.CompletedRounds==0,"Low position confirmation too early");
        Feed(partial,.01f,0);Check(partial.CompletedRounds==1,"True low return not counted");
        partial.Reset("standard");Feed(partial,.1f,0);Feed(partial,1,1);float held=partial.ContinuousHoldSeconds;
        Feed(partial,.1f,.6f);Check(partial.LiveScore==0&&Mathf.Abs(partial.ContinuousHoldSeconds-held)<.001,"Sub-threshold height added hold seconds");
        Feed(partial,.15f,.6f);Check(partial.ContinuousHoldSeconds==0,"Sustained height sag did not reset continuity");
        partial.Reset("standard");partial.BeginRepetition(30,5);Feed(partial,.1f,0);
        for(int i=0;i<1000;i++)partial.Step(1,100,.01f,true,0);
        Check(partial.ContinuousHoldSeconds==0,"Raised time outside the prescribed hold filled its goal");
        Feed(partial,28.9f,1);Feed(partial,.3f,0);Check(partial.QualifiedRounds==0,"Pre-hold time topped up an insufficient hold");
        var hip=new HipThrustHoldRule();hip.Reset("simple");for(int i=0;i<10;i++)hip.Step(0,100,.01f,true);for(int i=0;i<360;i++)hip.Step(.25f,100,.01f,true);
        Check(hip.LiveScore==100,"Original hip scoring changed");
        report.AppendLine("PASS low-position arming, strict return, duplicate prevention, 30-second dose, bad-quality/height reset, lost input, and original hip score.");
    }
    static void VerifyRecordings()
    {
        foreach(bool dynamic in new[]{true,false})
        {
            var scene=EditorSceneManager.OpenScene(dynamic?SideBridgeGameBuilder.Dynamic:SideBridgeGameBuilder.Hold30);
            var transport=scene.GetRootGameObjects().SelectMany(r=>r.GetComponentsInChildren<HipThrustTaikoController>(true)).Single();
            var policy=transport.GetComponent<SideBridgePoseScoring>();
            var host=UnityEngine.Object.Instantiate(transport.CoachAnimator.gameObject);
            host.transform.SetParent(null);host.transform.SetPositionAndRotation(Vector3.zero,Quaternion.identity);host.transform.localScale=Vector3.one;
            foreach(var script in host.GetComponentsInChildren<MonoBehaviour>(true))script.enabled=false;
            host.SetActive(true);var avatar=host.GetComponent<Animator>();avatar.enabled=true;avatar.Rebind();avatar.Update(0);
            typeof(HipThrustTaikoController).GetField("coachAnimator",Private).SetValue(transport,avatar);
            var timeline=new HipThrustHold30Timeline(transport.PhaseTimeline,6,dynamic,!dynamic,true);
            typeof(HipThrustTaikoController).GetField("hold30Timeline",Private).SetValue(transport,timeline);
            try
            {
                foreach(string difficulty in new[]{"simple","standard","hard"})
                foreach(int fps in new[]{30,60,120})
                {
                    typeof(DeadBugGamingPoseScorer).GetField("userAnimator",Private).SetValue(transport.RuleScoring.calibration,transport.CoachAnimator);
                    policy.Begin(difficulty);Check(policy.CaptureTargets(transport),"Reference profiles invalid");
                    float dt=1f/fps;int lastCompleted=0;
                    for(int i=0;i<=Math.Ceiling(timeline.Duration*fps);i++)
                    {
                        double time=i/(double)fps;typeof(HipThrustTaikoController).GetField("elapsed",Private).SetValue(transport,time);
                        timeline.Evaluate(time,out float frame,out _,out _);
                        if(timeline.TryGetWrapProgress(time,out float wrap))avatar.Play("sidebridge_wrap",0,wrap);
                        else avatar.Play("sidebridge_routine",0,frame/transport.PhaseTimeline.sourceClip.frameRate/transport.PhaseTimeline.sourceClip.length);
                        avatar.Update(0);policy.UpdateReference();
                        bool valid=policy.Evaluate(avatar,out float progress,out float quality,1f/fps);
                        policy.Step(progress,quality,dt,valid);
                        if(policy.Rule.CompletedRounds>lastCompleted)
                        {
                            lastCompleted=policy.Rule.CompletedRounds;
                            Check(policy.Rule.LastRoundBest>99.95f,$"Exact coach replay scored {policy.Rule.LastRoundBest} in {dynamic}/{difficulty}/{fps}/rep{lastCompleted}");
                        }
                    }
                    Check(policy.Rule.CompletedRounds==6&&policy.Rule.QualifiedRounds==6,$"Exact coach replay not 6/6: {dynamic}/{difficulty}/{fps}, complete={policy.Rule.CompletedRounds}, qualified={policy.Rule.QualifiedRounds}");
                    report.AppendLine($"PASS {(dynamic?"dynamic":"hold30")} {difficulty} {fps}fps: all six recorded repetitions score 100 and qualify; targets {string.Join(", ",policy.RecordedHoldTargets.Select(t=>t.ToString("F3")))} seconds.");
                }
                var target=policy.Geometry.Top;
                var bad=new SideBridgePoseGeometry.Pose {lift=target.lift,trunkAngle=target.trunkAngle,kneeAngle=target.kneeAngle,elbowAngle=target.elbowAngle+45,
                    shoulderOverElbow=target.shoulderOverElbow+.2f,shoulderStack=target.shoulderStack,pelvisStack=target.pelvisStack,torsoVertical=target.torsoVertical,
                    thigh=target.thigh,shin=target.shin,shoulderAxis=target.shoulderAxis,pelvisAxis=target.pelvisAxis};
                Check(policy.Geometry.Quality(bad,target)>=79.9f&&!SideBridgePoseGeometry.HasStableSupport(bad,target),"Composite quality masked invalid support arm");
                bad.elbowAngle=target.elbowAngle;bad.shoulderOverElbow=target.shoulderOverElbow;
                bad.shoulderAxis=Quaternion.AngleAxis(35,Vector3.forward)*target.shoulderAxis;
                Check(!SideBridgePoseGeometry.HasStableSupport(bad,target),"Shoulder-pelvis twist accepted");
            }
            finally{UnityEngine.Object.DestroyImmediate(host);}
        }
        report.AppendLine("PASS independent support-arm/alignment gates cannot be compensated by other quality terms.");
    }
}
