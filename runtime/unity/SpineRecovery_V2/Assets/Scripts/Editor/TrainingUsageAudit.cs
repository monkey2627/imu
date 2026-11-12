using System;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEngine;

public static class TrainingUsageAudit
{
    static int failures,checks;
    static string output;
    public static void RunStrict(){Run();if(failures>0)throw new Exception("Real-use scoring audit failed: "+failures);}
    [MenuItem("Tools/Training/Audit Real Usage Boundaries")]
    public static void Run()
    {
        output="output/training-assessment/usage-audit.txt";Directory.CreateDirectory(Path.GetDirectoryName(output));
        File.WriteAllText(output,"Synthetic adversaries for real-use failure modes.\n");checks=failures=0;
        foreach(string difficulty in new[]{"simple","standard","hard"})
        {
            var rule=new HipThrustHoldRule();rule.Reset(difficulty);
            Feed(rule,1,1,100);Check(rule.LiveScore==0,difficulty+": already raised at start must not create a repetition/hold score");
            rule.Reset(difficulty);Feed(rule,.2f,0,100);Feed(rule,10,1,100);Feed(rule,.3f,1,0);
            Check(rule.LiveScore==0,difficulty+": bad posture must not retain full live score");
            rule.Reset(difficulty);Feed(rule,.2f,0,100);Feed(rule,10,1,100);Feed(rule,.12f,0,0);
            Feed(rule,.4f,difficulty=="simple"?.19f:difficulty=="hard"?.54f:.39f,0);
            Check(rule.CompletedRounds==0,difficulty+": partial return must not complete a repetition");
        }
        var host=new GameObject("Uncalibrated hit audit");
        try
        {
            var judge=host.AddComponent<RhythmPoseJudge>();
            var good=new AnatomicalPoseScore{score=100,rawScore=100,validSegmentCount=3};
            typeof(RhythmPoseJudge).GetMethod("HandleLimbScoresChanged",BindingFlags.Instance|BindingFlags.NonPublic).Invoke(judge,new object[]{new AnatomicalLimbScores{leftArm=good,rightArm=good,leftLeg=good,rightLeg=good}});
            foreach(var limb in new[]{RhythmBodyTarget.LeftHand,RhythmBodyTarget.RightHand,RhythmBodyTarget.LeftFoot,RhythmBodyTarget.RightFoot})
                Check(!judge.TryHit(limb,out _),limb+": cached 100 without calibrated fresh input must not HIT");
        }
        finally{UnityEngine.Object.DestroyImmediate(host);}
        host=new GameObject("Frozen receiver audit");
        try
        {
            var receiver=host.AddComponent<EVMC4U.ExternalReceiver>();var source=host.AddComponent<AvatarMotionSource>();
            var bridge=host.AddComponent<VmcGameModeSourceBridge>();bridge.Configure(receiver,source);
            source.ReportFrame(0,55);int before=source.LastPacketframeCounterInFrame;
            receiver.Freeze=true;receiver.LastBonePacketCounterInFrame=55;
            typeof(VmcGameModeSourceBridge).GetMethod("Update",BindingFlags.Instance|BindingFlags.NonPublic).Invoke(bridge,null);
            Check(!source.HasFreshFrame&&source.LastPacketframeCounterInFrame==before,"Frozen receiver must not replay last bone packet as fresh data");
        }
        finally{UnityEngine.Object.DestroyImmediate(host);}
        File.AppendAllText(output,$"Checks: {checks}; failures: {failures}\n");
    }
    static void Feed(HipThrustHoldRule rule,float seconds,float progress,float quality)
    {for(int i=0;i<Mathf.RoundToInt(seconds*100);i++)rule.Step(progress,quality,.01f,true);}
    static void Check(bool ok,string message){checks++;if(!ok)failures++;File.AppendAllText(output,(ok?"PASS ":"FAIL ")+message+"\n");}
}
