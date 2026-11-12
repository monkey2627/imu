using System;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

// Replay the actual recorded clips on an independent patient rig, including the
// game's real supine calibration. Unlike settlement tests, query HIT at note arrival.
[InitializeOnLoad]
public static class GameplayHitVerification
{
    const string Flag="GameplayHitVerification", Output="output/scoring-tolerance/gameplay-hit.txt";
    const BindingFlags Private=BindingFlags.Instance|BindingFlags.NonPublic;
    static readonly string[] Scenes={"BirdDog_RingSkinC_Recorded","DeadBug_RingSkinC_Recorded"};
    static int scene,stage;
    static GameWorkoutScene entry;
    static GameplayHitVerification(){if(SessionState.GetBool(Flag,false))EditorApplication.update+=Tick;}
    public static void Run()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Output));
        File.WriteAllText(Output,"Recorded coach -> independent patient, actual calibration, endpoint HIT replay; synthetic input, not physical IMUs.\n");
        SessionState.SetBool(Flag,true);EditorApplication.update+=Tick;
        EditorSceneManager.OpenScene("Assets/Scenes/"+Scenes[0]+".unity");EditorApplication.isPlaying=true;
    }
    static void Tick()
    {
        if(!EditorApplication.isPlaying||EditorApplication.isCompiling)return;
        try
        {
            if(stage==0)
            {
                entry=UnityEngine.Object.FindObjectOfType<GameWorkoutScene>();
                if(entry==null||entry.gameObject.scene.name!=Scenes[scene]||entry.GetComponent<TrainingResultView>()?.Root==null)return;
                entry.menu.SelectStandardDifficulty();
                ((GameObject)Get(entry.menu,"menuRoot")).SetActive(false);
                Set(entry.scorer,"calibrated",true);
                Call(entry.coach,"StartSegmentedDeadBug");stage=1;return;
            }
            if(stage==1)
            {
                if(!entry.GetComponent<GameTrainingAssessment>().IsCollecting)return;
                entry.coach.StopAllCoroutines();
                foreach(string difficulty in new[]{"simple","standard","hard"})
                {
                    if(difficulty=="simple")entry.menu.SelectSimpleDifficulty();else if(difficulty=="hard")entry.menu.SelectHardDifficulty();else entry.menu.SelectStandardDifficulty();
                    ((GameObject)Get(entry.menu,"menuRoot")).SetActive(false);
                    Replay();
                }
                if(++scene==Scenes.Length){Finish(0);return;}
                stage=0;SceneManager.LoadScene(Scenes[scene]);
            }
        }
        catch(Exception ex){File.AppendAllText(Output,"FAIL "+ex+"\n");Debug.LogException(ex);Finish(1);}
    }
    static void Replay()
    {
        var scorer=entry.scorer;var coach=scorer.CoachAnimator;var user=scorer.UserAnimator;
        var engine=(AnatomicalMotionScoringEngine)Get(scorer,"scoringEngine");
        var assessment=entry.GetComponent<GameTrainingAssessment>();
        var motion=UnityEngine.Object.FindObjectOfType<AvatarMotionSource>();
        var judge=UnityEngine.Object.FindObjectOfType<RhythmPoseJudge>();
        if(Mathf.Abs(judge.SuccessThreshold-GameDifficultyPolicy.HitThreshold(entry.menu.SelectedDifficultyKey))>.001f)
            throw new Exception("Hidden menu lost the selected HIT difficulty");
        var ups=(string[])Get(entry.coach,"upSegmentStates");var downs=(string[])Get(entry.coach,"downSegmentStates");
        var profile=Call(entry.coach,"GetSelectedDifficultyProfile");
        float speed=(float)profile.GetType().GetField("upSpeed").GetValue(profile);
        float downSpeed=(float)profile.GetType().GetField("downSpeed").GetValue(profile);
        var durations=new double[ups.Length];for(int i=0;i<ups.Length;i++)durations[i]=entry.coach.SegmentSeconds(ups[i],speed)+entry.coach.SegmentSeconds(downs[i],downSpeed);
        Set(entry.coach,"<AssessmentDurations>k__BackingField",durations);
        user.gameObject.SetActive(true);coach.gameObject.SetActive(true);
        using(var c=new HumanPoseHandler(coach.avatar,coach.transform))
        using(var u=new HumanPoseHandler(user.avatar,user.transform))
        foreach(string scenario in new[]{"exact","range80","range70","half","half_leg_only","rest","leg15","leg_minus12","leg_side15","knee25","calibration15","range80_calibration15","calibration_ready_pose","foot45","delay500","delay800","wrong_leg_direction","wrong_knee","wrong_pair","wrong_floor","no_input"})
        {
            engine.ClearCalibration();
            var calibration=scorer.GetComponent<SupineGameCalibration>();
            if(!calibration.Prepare(coach))throw new Exception("Missing calibration reference");
            var neutral=new HumanPose();c.GetHumanPose(ref neutral);u.SetHumanPose(ref neutral);
            if(scenario=="calibration15"||scenario=="range80_calibration15")RotateLegs(user,15);
            var calibrationCoach=engine.CaptureRawPose(coach);
            if(scenario=="calibration_ready_pose"&&entry.actionId=="deadbug")
            {
                calibration.Release();coach.Play(ups[0],0,0);coach.Update(0);
                var ready=new HumanPose();c.GetHumanPose(ref ready);u.SetHumanPose(ref ready);
            }
            var apply=scorer.GetType().GetMethod("ApplyGameCalibration");
            if(apply==null)engine.SetStandingCalibration(engine.CaptureRawPose(user),calibrationCoach);
            else apply.Invoke(scorer,new object[]{engine.CaptureRawPose(user),calibrationCoach});
            calibration.Release();
            Set(scorer,"calibrated",true);Set(scorer,"sessionActive",true);
            Call(assessment,"BeginRhythm");
            int hits=0;float lowestQuality=100,lowestProgress=100;
            for(int pair=0;pair<ups.Length;pair++)
            {
                Call(assessment,"BeginSegment",new CoachActionController.SegmentPlaybackInfo(ups[pair],speed));
                entry.rhythm.TryGetSegmentTargets(ups[pair],out var a,out var b);
                var capture=judge.GetType().GetMethod("CaptureTarget");
                var hitReference=capture!=null?capture.Invoke(judge,new object[]{ups[pair]}):null;
                var boundJudge=judge.GetType().GetMethod("TryHit",new[]{typeof(RhythmBodyTarget),typeof(RhythmMovementRange),typeof(float).MakeByRefType()});
                coach.Play(ups[pair],0,0);coach.Update(0);var rest=new HumanPose();c.GetHumanPose(ref rest);
                var restDirections=engine.CaptureRawPose(coach);
                float up=entry.coach.SegmentSeconds(ups[pair],speed),down=entry.coach.SegmentSeconds(downs[pair],downSpeed);
                bool hitA=false,hitB=false;float bestA=0,bestB=0,bestProgressA=0,bestProgressB=0;
                for(int f=0;f<=Mathf.CeilToInt((up+Mathf.Min(1.2f,down))*60);f++)
                {
                    float now=f/60f,patientTime=Mathf.Max(0,now-(scenario=="delay500"?.5f:scenario=="delay800"?.8f:0));
                    Sample(patientTime);var pose=new HumanPose();c.GetHumanPose(ref pose);
                    if(scenario=="rest")pose=rest;
                    if(scenario=="range80"||scenario=="range80_calibration15"||scenario=="range70"||scenario=="half")
                        for(int m=0;m<pose.muscles.Length;m++)pose.muscles[m]=Mathf.Lerp(rest.muscles[m],pose.muscles[m],scenario.StartsWith("range80")?.8f:scenario=="range70"?.7f:.5f);
                    if(scenario=="wrong_pair")
                        for(int m=0;m<pose.muscles.Length;m++)if(HumanTrait.MuscleName[m].StartsWith("Left"))
                        {int other=Array.IndexOf(HumanTrait.MuscleName,"Right"+HumanTrait.MuscleName[m].Substring(4));if(other>=0){float swap=pose.muscles[m];pose.muscles[m]=pose.muscles[other];pose.muscles[other]=swap;}}
                    u.SetHumanPose(ref pose);
                    if(scenario=="half_leg_only")HalfLegs(user,engine,restDirections);
                    Quaternion patientRotation=user.transform.rotation;
                    if(scenario=="wrong_floor")user.transform.rotation=Quaternion.Euler(0,0,180)*patientRotation;
                    if(scenario=="leg15")RotateLegs(user,15);
                    if(scenario=="leg_minus12")RotateLegs(user,-12);
                    if(scenario=="leg_side15")foreach(var bone in new[]{HumanBodyBones.LeftUpperLeg,HumanBodyBones.RightUpperLeg})user.GetBoneTransform(bone).localRotation*=Quaternion.Euler(0,0,15);
                    if(scenario=="wrong_leg_direction")foreach(var bone in new[]{HumanBodyBones.LeftUpperLeg,HumanBodyBones.RightUpperLeg})user.GetBoneTransform(bone).localRotation*=Quaternion.Euler(0,0,90);
                    if(scenario=="knee25")foreach(var bone in new[]{HumanBodyBones.LeftLowerLeg,HumanBodyBones.RightLowerLeg})user.GetBoneTransform(bone).localRotation*=Quaternion.Euler(25,0,0);
                    if(scenario=="wrong_knee")foreach(var bone in new[]{HumanBodyBones.LeftLowerLeg,HumanBodyBones.RightLowerLeg})user.GetBoneTransform(bone).localRotation*=Quaternion.Euler(100,0,0);
                    if(scenario=="foot45")foreach(var bone in new[]{HumanBodyBones.LeftFoot,HumanBodyBones.RightFoot})user.GetBoneTransform(bone).localRotation*=Quaternion.Euler(45,0,0);
                    Sample(now);motion.ReportFrame(now,scenario=="no_input"?0:55);
                    foreach(var h in (System.Collections.IEnumerable)Get(engine,"coachHistory"))
                    {var t=h.GetType().GetField("timestamp");t.SetValue(h,(float)t.GetValue(h)-1f/60);}
                    engine.Evaluate(user,coach,1f/60,out var limbs);Set(scorer,"lastLimbScores",limbs);Call(judge,"HandleLimbScoresChanged",limbs);
                    if(now>=up)
                    {
                        hitA|=Hit(a,out float qa);hitB|=Hit(b,out float qb);
                        bestA=Mathf.Max(bestA,qa);bestB=Mathf.Max(bestB,qb);
                        if(hitReference is RhythmMovementRange reference)
                        {
                            reference.TryEndpoint(scorer,a,out float pa,out _);reference.TryEndpoint(scorer,b,out float pb,out _);
                            bestProgressA=Mathf.Max(bestProgressA,pa*100);bestProgressB=Mathf.Max(bestProgressB,pb*100);
                        }
                    }
                    user.transform.rotation=patientRotation;
                }
                hits+=(hitA?1:0)+(hitB?1:0);lowestQuality=Mathf.Min(lowestQuality,bestA,bestB);lowestProgress=Mathf.Min(lowestProgress,bestProgressA,bestProgressB);
                File.AppendAllText(Output,$"{entry.actionId} {entry.menu.SelectedDifficultyKey} {scenario} {ups[pair]} {a}={hitA}/{bestA:F1}/{bestProgressA:F1} {b}={hitB}/{bestB:F1}/{bestProgressB:F1}\n");
                if((scenario=="half_leg_only"||scenario=="wrong_leg_direction"||scenario=="wrong_knee")&&((IsLeg(a)&&hitA)||(IsLeg(b)&&hitB)))throw new Exception("Correct arm concealed wrong leg: "+scenario);
                void Sample(float t){coach.Play(t<=up?ups[pair]:downs[pair],0,t<=up?t/up:Mathf.Clamp01((t-up)/down));coach.Update(0);}
                bool Hit(RhythmBodyTarget target,out float score)
                {
                    if(boundJudge==null)return judge.TryHit(target,out score);
                    object[] args={target,hitReference,0f};bool result=(bool)boundJudge.Invoke(judge,args);score=(float)args[2];return result;
                }
            }
            File.AppendAllText(Output,$"TOTAL {entry.actionId} {entry.menu.SelectedDifficultyKey} {scenario} hits={hits}/{ups.Length*2} minimumQuality={lowestQuality:F1} minimumProgress={lowestProgress:F1}\n");
            if((scenario=="exact"||scenario=="range80"||scenario=="leg15"||scenario=="leg_minus12"||scenario=="leg_side15"||scenario=="knee25"||scenario=="calibration15"||scenario=="foot45"||scenario=="delay500"||scenario=="delay800")&&hits!=ups.Length*2)
                throw new Exception("Normal motion cannot HIT: "+scenario+" "+entry.menu.SelectedDifficultyKey);
            if((scenario=="half"||scenario=="rest"||scenario=="wrong_pair"||scenario=="wrong_floor"||scenario=="no_input")&&hits!=0)
                throw new Exception("Invalid motion earned HIT: "+scenario);
            if((scenario=="calibration_ready_pose"||scenario=="range70")&&entry.menu.SelectedDifficultyKey=="simple"&&hits!=ups.Length*2)
                throw new Exception("Entry difficulty too strict: "+scenario);
            if(scenario=="range80_calibration15"&&entry.menu.SelectedDifficultyKey!="hard"&&hits!=ups.Length*2)
                throw new Exception("Mild amplitude/calibration combination too strict: "+entry.menu.SelectedDifficultyKey);
        }
        VerifyNoteBinding(scorer,coach,user,engine,assessment,motion,judge,ups);
    }
    static void VerifyNoteBinding(DeadBugGamingPoseScorer scorer,Animator coach,Animator user,
        AnatomicalMotionScoringEngine engine,GameTrainingAssessment assessment,AvatarMotionSource motion,
        RhythmPoseJudge judge,string[] ups)
    {
        engine.ClearCalibration();Set(scorer,"calibrated",true);Set(scorer,"sessionActive",true);
        entry.rhythm.StartRhythmMode();
        Call(entry.rhythm,"HandleCoachSegmentStarted",new CoachActionController.SegmentPlaybackInfo(ups[0],1));
        var notes=new System.Collections.Generic.List<RhythmNoteView>((System.Collections.Generic.List<RhythmNoteView>)Get(entry.rhythm,"activeNotes"));
        if(notes.Count!=2||notes[0].JudgementReference==null||notes[1].JudgementReference==null)
            throw new Exception("Spawned notes did not capture their own target");
        coach.Play(ups[0],0,1);coach.Update(0);
        using(var c=new HumanPoseHandler(coach.avatar,coach.transform))
        using(var u=new HumanPoseHandler(user.avatar,user.transform))
        {var pose=new HumanPose();c.GetHumanPose(ref pose);u.SetHumanPose(ref pose);}
        Call(assessment,"BeginSegment",new CoachActionController.SegmentPlaybackInfo(ups[1],1));
        coach.Play((string)Get(entry.coach,"endSegmentState"),0,1);coach.Update(0);
        motion.ReportFrame(1,55);engine.Evaluate(user,coach,1f/60,out var limbs);
        Set(scorer,"lastLimbScores",limbs);Call(judge,"HandleLimbScoresChanged",limbs);
        foreach(var note in notes)
            if(entry.rhythm.JudgeAtLine(note).Judgement==RhythmJudgement.Miss)
                throw new Exception("Pending note followed the next segment/outro instead of its own target");
        motion.SetAvailable(false);
        foreach(var note in notes)
            if(entry.rhythm.JudgeAtLine(note).Judgement!=RhythmJudgement.Miss)
                throw new Exception("Bound note accepted stale input");
        entry.rhythm.StopRhythmMode();
        File.AppendAllText(Output,$"PASS {entry.actionId} {entry.menu.SelectedDifficultyKey}: actual spawned notes retain targets across segment/outro and reject lost input.\n");
    }
    static void RotateLegs(Animator a,float angle){foreach(var b in new[]{HumanBodyBones.LeftUpperLeg,HumanBodyBones.RightUpperLeg})a.GetBoneTransform(b).localRotation*=Quaternion.Euler(angle,0,0);}
    static void HalfLegs(Animator patient,AnatomicalMotionScoringEngine engine,System.Collections.Generic.Dictionary<HumanBodyBones,Vector3> rest)
    {
        // Half of each anatomical leg arc while leaving arms/torso unchanged.
        // Humanoid muscle interpolation alone is not half a physical excursion.
        var actual=engine.CaptureRawPose(patient);AnatomicalMotionScoringEngine.TryGetBodyFrame(patient,out var frame);
        foreach(var pair in new[]{(HumanBodyBones.LeftUpperLeg,HumanBodyBones.LeftLowerLeg),
            (HumanBodyBones.LeftLowerLeg,HumanBodyBones.LeftFoot),(HumanBodyBones.RightUpperLeg,HumanBodyBones.RightLowerLeg),
            (HumanBodyBones.RightLowerLeg,HumanBodyBones.RightFoot)})
        {
            var bone=patient.GetBoneTransform(pair.Item1);var child=patient.GetBoneTransform(pair.Item2);
            Vector3 target=frame*Vector3.Slerp(rest[pair.Item1],actual[pair.Item1],.5f);
            bone.rotation=Quaternion.FromToRotation(child.position-bone.position,target)*bone.rotation;
        }
    }
    static bool IsLeg(RhythmBodyTarget target)=>target==RhythmBodyTarget.LeftFoot||target==RhythmBodyTarget.RightFoot;
    static object Get(object o,string n)=>o.GetType().GetField(n,Private).GetValue(o);
    static void Set(object o,string n,object v)=>o.GetType().GetField(n,Private).SetValue(o,v);
    static object Call(object o,string n,params object[] v)=>o.GetType().GetMethod(n,Private).Invoke(o,v);
    static void Finish(int code){SessionState.SetBool(Flag,false);EditorApplication.update-=Tick;EditorApplication.Exit(code);}
}
