using System;
using System.IO;
using System.Reflection;
using UnityEngine;

/// <summary>Uses separate patient and coach skeletons: no shared-pose shortcut.</summary>
public static class TrainingToleranceScenarios
{
    const BindingFlags Private=BindingFlags.Instance|BindingFlags.NonPublic;
    const string Output="output/training-assessment/tolerance-scenarios.txt";
    public static void Begin(){File.WriteAllText(Output,"Separate humanoid patient replay, synthetic packet transport; no physical IMU validation.\n");}
    public static void Bridge(GameWorkoutScene entry,Animator patient)
    {
        var game=entry.taiko;var scoring=game.RuleScoring;var assessment=entry.GetComponent<GameTrainingAssessment>();
        var motion=UnityEngine.Object.FindObjectOfType<AvatarMotionSource>();
        Set(entry.scorer,"userAnimator",patient);Set(entry.scorer,"calibrated",true);
        Quaternion rotation=patient.transform.rotation;
        using(var coachHandler=new HumanPoseHandler(game.CoachAnimator.avatar,game.CoachAnimator.transform))
        using(var patientHandler=new HumanPoseHandler(patient.avatar,patient.transform))
        {
            foreach(string scenario in new[]{"follow","range_90","range_80","delay_350ms","delay_500ms","jitter_3deg","jitter_5deg","gap_1frame_mid","gap_1frame_end","gap_100ms_mid","spike_100ms_mid","gap_300ms_mid","rest","half_range","wrong_orientation","disconnect","heartbeat_only","early_lower"})
            {
                if(SkipProbe(scenario))continue;
                patient.transform.rotation=rotation;
                var bones=patient.GetComponentsInChildren<Transform>(true);var beforePosition=new Vector3[bones.Length];var beforeRotation=new Quaternion[bones.Length];
                for(int b=0;b<bones.Length;b++){beforePosition[b]=bones[b].localPosition;beforeRotation[b]=bones[b].localRotation;}
                game.StartMode();
                TrainingUsageScenarios.AssertRestored(bones,beforePosition,beforeRotation);
                var plan=game.AssessmentRepetitions();Seek(plan[0].start);var rest=new HumanPose();coachHandler.GetHumanPose(ref rest);
                int invalidHits=0,earlyFullScores=0,partialHighScores=0;double validSamples=0;
                for(int sample=1;sample<=Math.Ceiling(game.Duration*30);sample++)
                {
                    double time=Math.Min(game.Duration,sample/30d);Seek(time);int index=0;
                    for(int i=1;i<plan.Length&&time>=plan[i].start;i++)index=i;
                    var rep=plan[index];bool inSecondHoldHalf=time>rep.top+(rep.lower-rep.top)/2&&time<rep.lower;
                    double delay=scenario=="delay_350ms"?.35:scenario=="delay_500ms"?.5:0; if(delay>0)Seek(Math.Max(0,time-delay));
                    var pose=new HumanPose();coachHandler.GetHumanPose(ref pose);
                    if(scenario=="rest"||(scenario=="early_lower"&&inSecondHoldHalf)||(scenario=="spike_100ms_mid"&&time>=rep.top+(rep.lower-rep.top)/2&&time<rep.top+(rep.lower-rep.top)/2+.1))pose=rest;
                    if(scenario=="range_90"||scenario=="range_80"||scenario=="half_range")
                    {
                        for(int m=0;m<pose.muscles.Length;m++)pose.muscles[m]=Mathf.Lerp(rest.muscles[m],pose.muscles[m],scenario=="range_90"?.9f:scenario=="range_80"?.8f:.5f);
                        pose.bodyPosition=Vector3.Lerp(rest.bodyPosition,pose.bodyPosition,scenario=="range_90"?.9f:scenario=="range_80"?.8f:.5f);pose.bodyRotation=Quaternion.Slerp(rest.bodyRotation,pose.bodyRotation,scenario=="range_90"?.9f:scenario=="range_80"?.8f:.5f);
                    }
                    patientHandler.SetHumanPose(ref pose);
                    patient.transform.rotation=scenario=="wrong_orientation"?Quaternion.Euler(0,0,180)*rotation:rotation;
                    Jitter(patient,scenario,sample);
                    Seek(time);motion.ReportFrame((float)time,scenario=="heartbeat_only"?0:55);
                    double mid=rep.top+(rep.lower-rep.top)/2; bool absent=Gap(scenario,time,mid,rep.lower)||(scenario=="disconnect"&&inSecondHoldHalf);if(absent)motion.SetAvailable(false);
                    scoring.Tick(1f/30);assessment.ObserveBridge(1f/30);
                    if(scenario=="follow"&&assessment.Summary.IsContinuous&&time>=rep.top&&time<rep.lower-.05&&game.Score==100)earlyFullScores++;
                    if((absent||scenario=="heartbeat_only")&&(bool)Call(game,"PoseMatchesCoach"))invalidHits++;
                    if(scoring.LiveScore>0)validSamples++;
                    if(scenario=="half_range"&&time>rep.top+TrainingScoringTolerance.MaximumResponseDelay+.1&&time<rep.lower&&scoring.LiveScore>0)partialHighScores++;
                }
                assessment.Finish();var s=assessment.Summary;
                File.AppendAllText(Output,$"{game.gameObject.scene.name} | {entry.menu.SelectedDifficultyKey} | {scenario} | score={s.Score} quality={s.Quality:F1} hold={s.CreditedSeconds:F2}/{s.TargetSeconds:F2} qualified={s.Qualified}/{s.Planned} live_frames={validSamples} invalid_hits={invalidHits}\n");
                if(invalidHits!=0)throw new Exception("Invalid telemetry produced HIT: "+scenario);
                if(partialHighScores!=0)throw new Exception("Sustained half lift earned live hold points");
                if(earlyFullScores!=0)throw new Exception("Full live score before the prescribed 30-second hold ends");
                if(scenario=="heartbeat_only"&&s.HasAssessment)throw new Exception("Heartbeat created patient measurements");
                if((scenario=="rest"||scenario=="half_range"||scenario=="wrong_orientation"||scenario=="early_lower")&&s.Qualified>0)
                    throw new Exception("Invalid motion earned qualified repetitions: "+scenario);
                CheckResult(s,scenario);
            }
        }
        patient.transform.rotation=rotation;Set(entry.scorer,"userAnimator",game.CoachAnimator);
        void Seek(double t){Set(game,"elapsed",t);Call(game,"RenderTimeline");}
    }
    public static void Rhythm(GameWorkoutScene entry,Animator patient)
    {
        var source=entry.scorer;var assessment=entry.GetComponent<GameTrainingAssessment>();var coach=source.CoachAnimator;
        var motion=UnityEngine.Object.FindObjectOfType<AvatarMotionSource>();var judge=UnityEngine.Object.FindObjectOfType<RhythmPoseJudge>();
        Set(source,"userAnimator",patient);Set(source,"calibrated",true);Set(source,"sessionActive",true);
        var ups=(string[])Get(entry.coach,"upSegmentStates");var downs=(string[])Get(entry.coach,"downSegmentStates");
        var profile=Call(entry.coach,"GetSelectedDifficultyProfile");float upSpeed=(float)profile.GetType().GetField("upSpeed").GetValue(profile),downSpeed=(float)profile.GetType().GetField("downSpeed").GetValue(profile);
        int loops=(int)profile.GetType().GetField("loopCount").GetValue(profile);
        var durations=new double[ups.Length*loops];
        for(int i=0;i<durations.Length;i++)durations[i]=entry.coach.SegmentSeconds(ups[i%ups.Length],upSpeed)+entry.coach.SegmentSeconds(downs[i%ups.Length],downSpeed);
        typeof(CoachActionController).GetField("<AssessmentDurations>k__BackingField",Private).SetValue(entry.coach,durations);
        Quaternion rotation=patient.transform.rotation;
        using(var coachHandler=new HumanPoseHandler(coach.avatar,coach.transform))
        using(var patientHandler=new HumanPoseHandler(patient.avatar,patient.transform))
        foreach(string scenario in new[]{"follow","range_90","range_80","delay_350ms","delay_500ms","jitter_3deg","jitter_5deg","gap_1frame_mid","gap_1frame_end","gap_100ms_mid","spike_100ms_mid","gap_300ms_mid","rest","half_range","wrong_orientation","disconnect","heartbeat_only","wrong_arm","wrong_leg_direction","wrong_knee"})
        {
            if(SkipProbe(scenario))continue;
            patient.transform.rotation=rotation;
            Call(assessment,"BeginRhythm");((AnatomicalMotionScoringEngine)Get(source,"scoringEngine")).ResetSession();coach.Play(ups[0],0,0);coach.Update(0);var rest=new HumanPose();coachHandler.GetHumanPose(ref rest);int invalidHits=0;
            for(int pair=0;pair<durations.Length;pair++)for(int direction=0;direction<2;direction++)
            {
                string state=direction==0?ups[pair%ups.Length]:downs[pair%ups.Length];float speed=direction==0?upSpeed:downSpeed;
                double duration=entry.coach.SegmentSeconds(state,speed);
                Call(assessment,"BeginSegment",new CoachActionController.SegmentPlaybackInfo(state,speed));
                for(int frame=1;frame<=Math.Ceiling(duration*30);frame++)
                {
                    double elapsed=Math.Min(duration,frame/30d);
                    coach.Play(state,0,(float)(Math.Max(0,elapsed-(scenario=="delay_350ms"?.35:scenario=="delay_500ms"?.5:0))/duration));coach.Update(0);
                    var pose=new HumanPose();coachHandler.GetHumanPose(ref pose);
                    if(scenario=="rest"||(scenario=="spike_100ms_mid"&&elapsed>=duration*.5&&elapsed<duration*.5+.1))pose=rest;
                    if(scenario=="range_90"||scenario=="range_80"||scenario=="half_range")for(int m=0;m<pose.muscles.Length;m++)pose.muscles[m]=Mathf.Lerp(rest.muscles[m],pose.muscles[m],scenario=="range_90"?.9f:scenario=="range_80"?.8f:.5f);
                    if(scenario=="wrong_arm")for(int m=0;m<pose.muscles.Length;m++)
                    {
                        string name=HumanTrait.MuscleName[m];
                        if(name.StartsWith("Left")&&(name.Contains("Arm")||name.Contains("Forearm")||name.Contains("Shoulder")))
                        {int other=Array.IndexOf(HumanTrait.MuscleName,"Right"+name.Substring(4));if(other>=0){float swap=pose.muscles[m];pose.muscles[m]=pose.muscles[other];pose.muscles[other]=swap;}}
                    }
                    patientHandler.SetHumanPose(ref pose);
                    patient.transform.rotation=scenario=="wrong_orientation"?Quaternion.Euler(0,0,180)*rotation:rotation;
                    Jitter(patient,scenario,frame);
                    coach.Play(state,0,(float)(elapsed/duration));coach.Update(0);
                    motion.ReportFrame((float)elapsed,scenario=="heartbeat_only"?0:55);
                    bool absent=Gap(scenario,elapsed,duration*.5,duration)||(scenario=="disconnect"&&elapsed>duration*.5);if(absent)motion.SetAvailable(false);
                    Call(assessment,"Update");ScoreWithHistory(source,patient,coach,1f/30);
                    Call(judge,"HandleLimbScoresChanged",source.LastLimbScores);
                    if((absent||scenario=="heartbeat_only")&&judge.TryHit(RhythmBodyTarget.LeftHand,out _))invalidHits++;
                    Call(assessment,"SampleRhythm",Math.Min(1d/30,duration-(frame-1)/30d));
                }
                Call(assessment,"EndSegment",new CoachActionController.ScoringCheckpointInfo(state,0,1,pair,direction==0,0));
            }
            // The final return has the same bounded response window as intermediate
            // phases. These samples only confirm the endpoint, never add dose/time.
            if(scenario=="delay_350ms"||scenario=="delay_500ms")
            {
                double observedBefore=assessment.Summary.ObservedSeconds;
                Call(assessment,"RequestFinish");
                string state=downs[downs.Length-1];double duration=entry.coach.SegmentSeconds(state,downSpeed);
                double delay=scenario=="delay_350ms"?.35:.5;
                for(int frame=1;frame<=17;frame++)
                {
                    coach.Play(state,0,(float)Math.Min(1,(duration+frame/30d-delay)/duration));coach.Update(0);
                    var pose=new HumanPose();coachHandler.GetHumanPose(ref pose);patientHandler.SetHumanPose(ref pose);
                    // The outro may already be upright; endpoint evidence belongs
                    // to the captured floor exercise, not the coach's new pose.
                    coach.Play((string)Get(entry.coach,"endSegmentState"),0,1);coach.Update(0);motion.ReportFrame(frame/30f,55);
                    Call(assessment,"Update");Call(assessment,"ProcessPendingEndpoints",1d/30);
                }
                Call(assessment,"LateUpdate");
                if(!assessment.IsFinished||assessment.Summary.ObservedSeconds!=observedBefore)throw new Exception("Deferred final endpoint failed or invented time");
            }
            assessment.Finish();var s=assessment.Summary;
            File.AppendAllText(Output,$"{entry.gameObject.scene.name} | {entry.menu.SelectedDifficultyKey} | {scenario} | score={s.Score} quality={s.Quality:F1} following={s.CreditedSeconds:F2}/{s.TargetSeconds:F2} qualified={s.Qualified}/{s.Planned} invalid_hits={invalidHits}\n");
            if(invalidHits>0)throw new Exception("Rhythm HIT without input");
            if((scenario=="rest"||scenario=="wrong_arm"||scenario=="half_range"||scenario=="wrong_orientation"||scenario=="wrong_leg_direction"||scenario=="wrong_knee")&&s.Qualified>0)throw new Exception("Incorrect limb movement earned repetitions");
            CheckResult(s,scenario);
        }
        patient.transform.rotation=rotation;Set(source,"userAnimator",coach);
    }
    // Age history by the replay step while Unity wall time is frozen inside this loop.
    // Exercise the real filter and delay matching, with no production source changes.
    static void ScoreWithHistory(DeadBugGamingPoseScorer source,Animator patient,Animator coach,float dt)
    {
        var engine=(AnatomicalMotionScoringEngine)Get(source,"scoringEngine");
        var history=(System.Collections.IEnumerable)Get(engine,"coachHistory");
        foreach(var sample in history)
        {
            var time=sample.GetType().GetField("timestamp",BindingFlags.Public|BindingFlags.Instance);
            time.SetValue(sample,(float)time.GetValue(sample)-dt);
        }
        engine.Evaluate(patient,coach,dt,out var limbs);
        if(!source.HasMatchingFloorPosture)
        {
            limbs.leftArm.score=limbs.leftArm.rawScore=0;limbs.rightArm.score=limbs.rightArm.rawScore=0;
            limbs.leftLeg.score=limbs.leftLeg.rawScore=0;limbs.rightLeg.score=limbs.rightLeg.rawScore=0;
        }
        Set(source,"lastLimbScores",limbs);
    }
    static bool Gap(string scenario,double time,double mid,double end)
    {
        if(scenario=="gap_1frame_mid")return time>=mid&&time<mid+1d/30;
        if(scenario=="gap_100ms_mid")return time>=mid&&time<mid+.1; if(scenario=="gap_300ms_mid")return time>=mid&&time<mid+.3;
        if(scenario=="gap_1frame_end")return time>end-1d/30&&time<=end;
        return false;
    }
    static bool SkipProbe(string scenario)=>UnityEditor.SessionState.GetBool("ToleranceFast",false)&&
        scenario!="follow"&&scenario!="delay_350ms"&&scenario!="delay_500ms"&&scenario!="jitter_3deg"&&scenario!="jitter_5deg"&&scenario!="spike_100ms_mid";
    static void Jitter(Animator patient,string scenario,int frame)
    {
        if(scenario=="wrong_leg_direction")foreach(var bone in new[]{HumanBodyBones.LeftUpperLeg,HumanBodyBones.RightUpperLeg})
            patient.GetBoneTransform(bone).localRotation*=Quaternion.Euler(0,0,90);
        if(scenario=="wrong_knee")foreach(var bone in new[]{HumanBodyBones.LeftLowerLeg,HumanBodyBones.RightLowerLeg})
            patient.GetBoneTransform(bone).localRotation*=Quaternion.Euler(100,0,0);
        float degrees=scenario=="jitter_3deg"?3:scenario=="jitter_5deg"?5:0;
        if(degrees==0)return;
        int i=0;
        foreach(var bone in new[]{HumanBodyBones.Hips,HumanBodyBones.LeftUpperArm,HumanBodyBones.RightUpperArm,HumanBodyBones.LeftUpperLeg,HumanBodyBones.RightUpperLeg})
        {
            var transform=patient.GetBoneTransform(bone);
            transform.localRotation*=Quaternion.Euler(0,0,Mathf.Sin(frame*1.7f+i++*1.1f)*degrees);
        }
    }
    static object Get(object o,string n)=>o.GetType().GetField(n,Private).GetValue(o);
    static void CheckResult(TrainingQualitySummary s,string scenario)
    {
        if(scenario=="rest"&&s.Score!=0)throw new Exception("Stationary rest earned training points");
        if(scenario=="follow"&&(s.Score<98||s.Qualified!=s.Planned))throw new Exception("Correct retargeted motion unfairly penalized");
        if((scenario=="delay_350ms"||scenario=="delay_500ms")&&(s.Score<90||s.Qualified!=s.Planned))throw new Exception("Bounded response delay unfairly penalized");
        if((scenario=="jitter_3deg"||scenario=="gap_1frame_mid"||scenario=="gap_1frame_end"||scenario=="gap_100ms_mid"||scenario=="spike_100ms_mid")&&
            (s.Score<94||s.Qualified!=s.Planned))throw new Exception("Brief measurement error erased repetition credit: "+scenario);
        if(scenario=="jitter_5deg"&&(s.Score<90||s.Qualified<s.Planned*2f/3))throw new Exception("Sensor jitter over-penalized");
        if(scenario=="gap_300ms_mid"&&s.IsContinuous&&s.Qualified>0)throw new Exception("Long interrupted hold qualified");
        if(scenario=="heartbeat_only"&&s.HasAssessment)throw new Exception("Heartbeat invented measurements");
        if(scenario=="half_range"&&s.Score>=60)throw new Exception("Half range earned a passing session score");
        if((scenario=="wrong_leg_direction"||scenario=="wrong_knee")&&s.Score>=60)throw new Exception("Gross leg error earned a passing session score");
    }
    static void Set(object o,string n,object value)=>o.GetType().GetField(n,Private).SetValue(o,value);
    static object Call(object o,string n,params object[] args)=>o.GetType().GetMethod(n,Private).Invoke(o,args);
}
