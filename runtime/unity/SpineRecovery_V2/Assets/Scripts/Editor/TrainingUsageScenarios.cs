using System;
using System.IO;
using System.Reflection;
using UnityEngine;

/// <summary>Uses separate patient and coach skeletons: no shared-pose shortcut.</summary>
public static class TrainingUsageScenarios
{
    const BindingFlags Private=BindingFlags.Instance|BindingFlags.NonPublic;
    const string Output="output/training-assessment/usage-scenarios.txt";
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
            foreach(string scenario in new[]{"follow","rest","half_range","wrong_orientation","early_lower","disconnect","heartbeat_only","delay_200ms","small_jitter"})
            {
                patient.transform.rotation=rotation;
                var bones=patient.GetComponentsInChildren<Transform>(true);var beforePosition=new Vector3[bones.Length];var beforeRotation=new Quaternion[bones.Length];
                for(int b=0;b<bones.Length;b++){beforePosition[b]=bones[b].localPosition;beforeRotation[b]=bones[b].localRotation;}
                game.StartMode();
                AssertRestored(bones,beforePosition,beforeRotation);
                var plan=game.AssessmentRepetitions();Seek(plan[0].start);var rest=new HumanPose();coachHandler.GetHumanPose(ref rest);
                int invalidHits=0,earlyFullScores=0;double validSamples=0;
                for(int sample=1;sample<=Math.Ceiling(game.Duration*30);sample++)
                {
                    double time=Math.Min(game.Duration,sample/30d);Seek(time);int index=0;
                    for(int i=1;i<plan.Length&&time>=plan[i].start;i++)index=i;
                    var rep=plan[index];bool inSecondHoldHalf=time>rep.top+(rep.lower-rep.top)/2&&time<rep.lower;
                    if(scenario=="delay_200ms")Seek(Math.Max(0,time-.2));
                    var pose=new HumanPose();coachHandler.GetHumanPose(ref pose);
                    if(scenario=="rest"||(scenario=="early_lower"&&inSecondHoldHalf))pose=rest;
                    if(scenario=="half_range")
                    {
                        for(int m=0;m<pose.muscles.Length;m++)pose.muscles[m]=Mathf.Lerp(rest.muscles[m],pose.muscles[m],.5f);
                        pose.bodyPosition=Vector3.Lerp(rest.bodyPosition,pose.bodyPosition,.5f);pose.bodyRotation=Quaternion.Slerp(rest.bodyRotation,pose.bodyRotation,.5f);
                    }
                    patientHandler.SetHumanPose(ref pose);
                    patient.transform.rotation=scenario=="wrong_orientation"?Quaternion.Euler(0,0,180)*rotation:rotation;
                    if(scenario=="small_jitter")patient.GetBoneTransform(HumanBodyBones.Hips).localRotation*=Quaternion.Euler(0,0,Mathf.Sin(sample)*1.5f);
                    Seek(time);motion.ReportFrame((float)time,scenario=="heartbeat_only"?0:55);
                    bool absent=scenario=="disconnect"&&inSecondHoldHalf;if(absent)motion.SetAvailable(false);
                    scoring.Tick(1f/30);assessment.ObserveBridge(1f/30);
                    if(scenario=="follow"&&assessment.Summary.IsContinuous&&time>=rep.top&&time<rep.lower-.05&&game.Score==100)earlyFullScores++;
                    if((absent||scenario=="heartbeat_only")&&(bool)Call(game,"PoseMatchesCoach"))invalidHits++;
                    if(scoring.LiveScore>0)validSamples++;
                }
                assessment.Finish();var s=assessment.Summary;
                File.AppendAllText(Output,$"{game.gameObject.scene.name} | {scenario} | score={s.Score} quality={s.Quality:F1} hold={s.CreditedSeconds:F2}/{s.TargetSeconds:F2} qualified={s.Qualified}/{s.Planned} live_frames={validSamples} invalid_hits={invalidHits}\n");
                if(invalidHits!=0)throw new Exception("Invalid telemetry produced HIT: "+scenario);
                if(earlyFullScores!=0)throw new Exception("Full live score before 30-second goal");
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
        var profile=Get(entry.coach,"simpleDifficulty");float upSpeed=(float)profile.GetType().GetField("upSpeed").GetValue(profile),downSpeed=(float)profile.GetType().GetField("downSpeed").GetValue(profile);
        Quaternion rotation=patient.transform.rotation;
        using(var coachHandler=new HumanPoseHandler(coach.avatar,coach.transform))
        using(var patientHandler=new HumanPoseHandler(patient.avatar,patient.transform))
        foreach(string scenario in new[]{"follow","rest","wrong_arm","half_range","wrong_orientation","disconnect","heartbeat_only","delay_200ms","small_jitter"})
        {
            patient.transform.rotation=rotation;
            Call(assessment,"BeginRhythm");coach.Play(ups[0],0,0);coach.Update(0);var rest=new HumanPose();coachHandler.GetHumanPose(ref rest);int invalidHits=0;
            for(int pair=0;pair<ups.Length;pair++)for(int direction=0;direction<2;direction++)
            {
                string state=direction==0?ups[pair]:downs[pair];float speed=direction==0?upSpeed:downSpeed;
                double duration=entry.coach.SegmentSeconds(state,speed);
                Call(assessment,"BeginSegment",new CoachActionController.SegmentPlaybackInfo(state,speed));
                for(int frame=1;frame<=Math.Ceiling(duration*30);frame++)
                {
                    double elapsed=Math.Min(duration,frame/30d);
                    coach.Play(state,0,(float)(Math.Max(0,elapsed-(scenario=="delay_200ms"?.2:0))/duration));coach.Update(0);
                    var pose=new HumanPose();coachHandler.GetHumanPose(ref pose);
                    if(scenario=="rest")pose=rest;
                    if(scenario=="half_range")for(int m=0;m<pose.muscles.Length;m++)pose.muscles[m]=Mathf.Lerp(rest.muscles[m],pose.muscles[m],.5f);
                    if(scenario=="wrong_arm")for(int m=0;m<pose.muscles.Length;m++)
                    {
                        string name=HumanTrait.MuscleName[m];
                        if(name.StartsWith("Left")&&(name.Contains("Arm")||name.Contains("Forearm")||name.Contains("Shoulder")))
                        {int other=Array.IndexOf(HumanTrait.MuscleName,"Right"+name.Substring(4));if(other>=0){float swap=pose.muscles[m];pose.muscles[m]=pose.muscles[other];pose.muscles[other]=swap;}}
                    }
                    patientHandler.SetHumanPose(ref pose);
                    patient.transform.rotation=scenario=="wrong_orientation"?Quaternion.Euler(0,0,180)*rotation:rotation;
                    if(scenario=="small_jitter")patient.GetBoneTransform(HumanBodyBones.Hips).localRotation*=Quaternion.Euler(0,0,Mathf.Sin(frame)*1.5f);
                    coach.Play(state,0,(float)(elapsed/duration));coach.Update(0);
                    motion.ReportFrame((float)elapsed,scenario=="heartbeat_only"?0:55);
                    bool absent=scenario=="disconnect"&&elapsed>duration*.5;if(absent)motion.SetAvailable(false);
                    Call(assessment,"Update");var engine=(AnatomicalMotionScoringEngine)Get(source,"scoringEngine");engine.ResetSession();Call(source,"ScoreCurrentPose");
                    Call(judge,"HandleLimbScoresChanged",source.LastLimbScores);
                    if((absent||scenario=="heartbeat_only")&&judge.TryHit(RhythmBodyTarget.LeftHand,out _))invalidHits++;
                    Call(assessment,"SampleRhythm",Math.Min(1d/30,duration-(frame-1)/30d));
                }
                Call(assessment,"EndSegment",new CoachActionController.ScoringCheckpointInfo(state,0,1,pair,direction==0,0));
            }
            assessment.Finish();var s=assessment.Summary;
            File.AppendAllText(Output,$"{entry.gameObject.scene.name} | {scenario} | score={s.Score} quality={s.Quality:F1} following={s.CreditedSeconds:F2}/{s.TargetSeconds:F2} qualified={s.Qualified}/{s.Planned} invalid_hits={invalidHits}\n");
            if(invalidHits>0)throw new Exception("Rhythm HIT without input");
            if((scenario=="rest"||scenario=="wrong_arm"||scenario=="half_range"||scenario=="wrong_orientation")&&s.Qualified>0)throw new Exception("Incorrect limb movement earned repetitions");
            CheckResult(s,scenario);
        }
        patient.transform.rotation=rotation;Set(source,"userAnimator",coach);
    }
    static object Get(object o,string n)=>o.GetType().GetField(n,Private).GetValue(o);
    public static void AssertRestored(Transform[] bones,Vector3[] positions,Quaternion[] rotations)
    {
        for(int b=0;b<bones.Length;b++)
        {
            var q=bones[b].localRotation;var r=rotations[b];
            var actual=new Vector4(q.x,q.y,q.z,q.w);var expected=new Vector4(r.x,r.y,r.z,r.w);
            // Unity normalizes rotations on assignment. Component round-off (and
            // equivalent q/-q signs) is not a change to the skeleton's pose.
            float rotationError=Mathf.Min(Vector4.Distance(actual,expected),Vector4.Distance(actual,-expected));
            float positionError=Vector3.Distance(bones[b].localPosition,positions[b]);
            if(positionError>1e-6f||rotationError>1e-5f)
                throw new Exception($"Reference capture changed patient bone {bones[b].name}: position={positionError}, quaternion={rotationError}");
        }
    }
    static void CheckResult(TrainingQualitySummary s,string scenario)
    {
        if(scenario=="rest"&&s.Score!=0)throw new Exception("Stationary rest earned training points");
        if((scenario=="follow"||scenario=="small_jitter")&&(s.Score<98||s.Qualified!=s.Planned))throw new Exception("Correct retargeted motion or tiny noise unfairly penalized");
        if(scenario=="delay_200ms"&&s.Score<85)throw new Exception("Normal response delay unfairly penalized");
        if(scenario=="half_range"&&s.Score>=60)throw new Exception("Half range earned a passing session score");
    }
    static void Set(object o,string n,object value)=>o.GetType().GetField(n,Private).SetValue(o,value);
    static object Call(object o,string n,params object[] args)=>o.GetType().GetMethod(n,Private).Invoke(o,args);
}
