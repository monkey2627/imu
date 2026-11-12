using System.Collections.Generic;
using UnityEngine;

/// <summary>Normalizes each active arm/leg to its recorded start/end excursion.
/// Absolute angular tolerance alone can mistake half of a small movement for a full rep.</summary>
public sealed class RhythmMovementRange
{
    readonly Dictionary<string,Reference> references=new Dictionary<string,Reference>();
    Reference current;
    readonly List<ProgressSample> history=new List<ProgressSample>();
    double elapsed;
    struct ProgressSample { public double time;public Vector2 value;public Dictionary<HumanBodyBones,Vector3> pose; }
    sealed class Reference { public Dictionary<HumanBodyBones,Vector3> rest,top;public float front; }
    public void Begin(DeadBugGamingPoseScorer scorer,string upState)
    {
        history.Clear();elapsed=0;
        if(references.TryGetValue(upState,out current))return;
        var coach=scorer.CoachAnimator;var state=coach.GetCurrentAnimatorStateInfo(0);float speed=coach.speed;
        try
        {
            current=new Reference();coach.Play(upState,0,0);coach.Update(0);current.rest=scorer.CaptureAssessmentPose(false);
            DeadBugGamingPoseScorer.FloorOrientation(coach,out current.front,out _);
            coach.Play(upState,0,1);coach.Update(0);current.top=scorer.CaptureAssessmentPose(false);references[upState]=current;
        }
        finally{coach.Play(state.fullPathHash,0,state.normalizedTime);coach.Update(0);coach.speed=speed;}
    }
    public float FollowingQuality(DeadBugGamingPoseScorer scorer,RhythmBodyTarget a,RhythmBodyTarget b,double seconds=0,float delayHint=0)
    {
        if(current==null)return 0;
        var user=scorer.CaptureAssessmentPose(true);var coach=scorer.CaptureAssessmentPose(false);
        elapsed+=System.Math.Max(0,seconds);
        if(!Progress(user,a,out float actualA)||!Progress(user,b,out float actualB)||
            !Progress(coach,a,out float expectedA)||!Progress(coach,b,out float expectedB))return 0;
        history.Add(new ProgressSample{time=elapsed,value=new Vector2(expectedA,expectedB),pose=coach});
        while(history.Count>1&&history[0].time<elapsed-TrainingScoringTolerance.MaximumResponseDelay)history.RemoveAt(0);
        float best=0,bestObjective=float.NegativeInfinity;
        foreach(var sample in history)
        {
            // Both contralateral limbs must match the SAME recent coach instant.
            // The angular engine's estimate breaks ties; no future pose is accepted.
            float error=Mathf.Max(Mathf.Abs(actualA-sample.value.x),Mathf.Abs(actualB-sample.value.y));
            float quality=100*(1-Mathf.InverseLerp(.12f,.5f,error));
            if(!HasPlausibleLegDirections(user,sample.pose,a)||!HasPlausibleLegDirections(user,sample.pose,b))quality=0;
            float objective=quality-5*Mathf.Abs((float)(elapsed-sample.time)-delayHint);
            if(objective>bestObjective){bestObjective=objective;best=quality;}
        }
        return best;
    }
    public void Skip(double seconds){elapsed+=System.Math.Max(0,seconds);}
    public RhythmMovementRange CaptureReference()=>new RhythmMovementRange{current=current};
    public bool HasMatchingPosture(DeadBugGamingPoseScorer scorer)
        =>current!=null&&DeadBugGamingPoseScorer.FloorOrientation(scorer.UserAnimator,out float front,out float tilt)&&
            Mathf.Abs(tilt)<.7f&&Mathf.Abs(front)>.35f&&front*current.front>0;
    public bool AtEndpoint(DeadBugGamingPoseScorer scorer,RhythmBodyTarget a,RhythmBodyTarget b,bool up,Vector2 minimum)
    {
        if(current==null||!DeadBugGamingPoseScorer.FloorOrientation(scorer.UserAnimator,out float front,out float tilt)||
            Mathf.Abs(tilt)>=.7f||Mathf.Abs(front)<=.35f||front*current.front<=0)return false;
        var pose=scorer.CaptureAssessmentPose(true);
        if(!Progress(pose,a,out float x)||!Progress(pose,b,out float y))return false;
        if(!HasPlausibleLegDirections(pose,a,up)||!HasPlausibleLegDirections(pose,b,up))return false;
        return up?x>=minimum.x&&y>=minimum.y:x<=1-minimum.x&&y<=1-minimum.y;
    }
    public float EndpointQuality(DeadBugGamingPoseScorer scorer,RhythmBodyTarget target)
    {
        return TryEndpoint(scorer,target,out _,out float quality)?quality:0;
    }
    public bool TryEndpoint(DeadBugGamingPoseScorer scorer,RhythmBodyTarget target,out float progress,out float quality)
        =>TryEndpoint(scorer,target,out progress,out quality,out _,out _,out _);
    public bool TryEndpoint(DeadBugGamingPoseScorer scorer,RhythmBodyTarget target,out float progress,out float quality,
        out string failureReason,out float thighAngle,out float shinAngle)
    {
        failureReason="";thighAngle=shinAngle=-1;
        progress=quality=0;
        if(!HasMatchingPosture(scorer)){failureReason="posture_mismatch";return false;}
        var pose=scorer.CaptureAssessmentPose(true);
        if(!Progress(pose,target,out progress)){failureReason="missing_or_invalid_movement_reference";return false;}
        HumanBodyBones primary=PrimaryBone(target);
        if(!pose.TryGetValue(primary,out var actual)||!current.top.TryGetValue(primary,out var expected))
        {failureReason="missing_thigh_tracking";return false;}
        thighAngle=Vector3.Angle(actual,expected);
        HumanBodyBones shin=target==RhythmBodyTarget.LeftFoot?HumanBodyBones.LeftLowerLeg:HumanBodyBones.RightLowerLeg;
        if(pose.TryGetValue(shin,out var actualShin)&&current.top.TryGetValue(shin,out var expectedShin))
            shinAngle=Vector3.Angle(actualShin,expectedShin);
        if(!HasPlausibleLegDirections(pose,target,true)){failureReason="leg_direction_mismatch";return false;}
        float direction=DirectionQuality(actual,expected);
        HumanBodyBones secondary=target==RhythmBodyTarget.LeftHand?HumanBodyBones.LeftLowerArm:
            target==RhythmBodyTarget.RightHand?HumanBodyBones.RightLowerArm:
            target==RhythmBodyTarget.LeftFoot?HumanBodyBones.LeftLowerLeg:HumanBodyBones.RightLowerLeg;
        // Foot pitch/toe orientation does not describe hip extension. Mild knee
        // flexion contributes feedback, but cannot outweigh a correctly raised thigh.
        if(pose.TryGetValue(secondary,out actual)&&current.top.TryGetValue(secondary,out expected))
        {
            direction=.85f*direction+.15f*DirectionQuality(actual,expected);
        }
        quality=Mathf.Min(direction,100*Mathf.InverseLerp(.4f,.9f,progress));
        return true;
    }
    bool HasPlausibleLegDirections(Dictionary<HumanBodyBones,Vector3> pose,RhythmBodyTarget target,bool up)
        =>HasPlausibleLegDirections(pose,up?current.top:current.rest,target);
    static bool HasPlausibleLegDirections(Dictionary<HumanBodyBones,Vector3> pose,Dictionary<HumanBodyBones,Vector3> expected,RhythmBodyTarget target)
    {
        if(target!=RhythmBodyTarget.LeftFoot&&target!=RhythmBodyTarget.RightFoot)return true;
        HumanBodyBones thigh=PrimaryBone(target),shin=target==RhythmBodyTarget.LeftFoot?HumanBodyBones.LeftLowerLeg:HumanBodyBones.RightLowerLeg;
        // Extra amplitude tolerance never permits a wrong-direction thigh or a
        // grossly folded knee. Toe/ankle orientation is deliberately irrelevant.
        return pose.TryGetValue(thigh,out var actualThigh)&&expected.TryGetValue(thigh,out var expectedThigh)&&
            Vector3.Angle(actualThigh,expectedThigh)<=50f&&pose.TryGetValue(shin,out var actualShin)&&
            expected.TryGetValue(shin,out var expectedShin)&&Vector3.Angle(actualShin,expectedShin)<=65f;
    }
    static float DirectionQuality(Vector3 actual,Vector3 expected)
        =>100*(1-Mathf.InverseLerp(18f,70f,Vector3.Angle(actual,expected)));
    public Vector2 CurrentProgress(DeadBugGamingPoseScorer scorer,RhythmBodyTarget a,RhythmBodyTarget b)
    {
        var pose=scorer.CaptureAssessmentPose(true);Progress(pose,a,out float x);Progress(pose,b,out float y);return new Vector2(x,y);
    }
    bool Progress(Dictionary<HumanBodyBones,Vector3> pose,RhythmBodyTarget target,out float result)
    {
        HumanBodyBones bone=PrimaryBone(target);
        result=0;
        if(!pose.TryGetValue(bone,out var actual)||!current.rest.TryGetValue(bone,out var rest)||!current.top.TryGetValue(bone,out var top))return false;
        Vector3 excursion=top-rest;
        if(excursion.sqrMagnitude<.005f)return false;
        result=Mathf.Clamp01(Vector3.Dot(actual-rest,excursion)/excursion.sqrMagnitude);return true;
    }
    static HumanBodyBones PrimaryBone(RhythmBodyTarget target)=>target==RhythmBodyTarget.LeftHand?HumanBodyBones.LeftUpperArm:
        target==RhythmBodyTarget.RightHand?HumanBodyBones.RightUpperArm:
        target==RhythmBodyTarget.LeftFoot?HumanBodyBones.LeftUpperLeg:HumanBodyBones.RightUpperLeg;
}
