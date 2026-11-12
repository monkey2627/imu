using UnityEngine;

/// <summary>Left-support references, achievable recorded hold targets, and measured repetitions.</summary>
public sealed class SideBridgePoseScoring : MonoBehaviour
{
    public readonly SideBridgeHoldRule Rule=new SideBridgeHoldRule();
    public SideBridgePoseGeometry Geometry => references!=null?references[activeCycle]:null;
    public bool PostureReady { get; private set; }
    public bool SupportReady { get; private set; }
    public float TrackingQuality { get; private set; }
    public float[] RecordedHoldTargets { get; private set; }
    SideBridgePoseGeometry[] references;
    SideBridgePoseGeometry[] coachReferences;
    HipThrustTaikoController transport;
    int activeRepetition=-1,activeCycle;
    string difficulty;
    float filteredProgress;
    bool hasFilteredProgress;

    public void Begin(string selectedDifficulty)
    {
        difficulty=selectedDifficulty;Rule.Reset(difficulty);activeRepetition=-1;activeCycle=0;
        PostureReady=SupportReady=false;TrackingQuality=0;
        hasFilteredProgress=false;
    }

    public bool CaptureTargets(HipThrustTaikoController source)
    {
        transport=source;var animator=source.CoachAnimator;var timeline=source.PhaseTimeline;
        if(animator==null||timeline==null)return false;
        using(var patient=new PatientPoseReference(animator,source.RuleScoring!=null&&source.RuleScoring.calibration!=null?source.RuleScoring.calibration.UserAnimator:null))
        {
        Animator measured=patient.Animator;
        references=new SideBridgePoseGeometry[timeline.cycles.Length];RecordedHoldTargets=new float[references.Length];
        coachReferences=new SideBridgePoseGeometry[references.Length];
        for(int i=0;i<references.Length;i++)
        {
            var cycle=timeline.cycles[i];Sample(cycle.liftStartFrame);
            if(!SideBridgePoseGeometry.TryMeasure(measured,out var rest)||!SideBridgePoseGeometry.TryMeasure(animator,out var coachRest))return false;
            Sample(HipThrustHold30Timeline.PeakFrame(timeline,i));
            if(!SideBridgePoseGeometry.TryMeasure(measured,out var top)||!SideBridgePoseGeometry.TryMeasure(animator,out var coachTop))return false;
            // A brief recorded overshoot must not make the whole stable plateau
            // look like a partial lift. Use its median height for amplitude scaling;
            // the coach still freezes at the original actual peak in the 30s mode.
            var plateau=new float[cycle.lowerStartFrame-cycle.holdStartFrame+1];
            var coachPlateau=new float[plateau.Length];
            for(int frame=cycle.holdStartFrame;frame<=cycle.lowerStartFrame;frame++)
            {
                Sample(frame);if(!SideBridgePoseGeometry.TryMeasure(measured,out var holdPose)||!SideBridgePoseGeometry.TryMeasure(animator,out var coachPose))return false;
                plateau[frame-cycle.holdStartFrame]=holdPose.lift;
                coachPlateau[frame-cycle.holdStartFrame]=coachPose.lift;
            }
            System.Array.Sort(plateau);
            System.Array.Sort(coachPlateau);
            float stableLift=plateau[plateau.Length/2];
            var geometry=references[i]=new SideBridgePoseGeometry();
            if(!geometry.SetTargets(rest,top,rest.lift,stableLift))return false;
            coachReferences[i]=new SideBridgePoseGeometry();
            if(!coachReferences[i].SetTargets(coachRest,coachTop,coachRest.lift,coachPlateau[coachPlateau.Length/2]))return false;
            // Measure the achievable total valid high-position time of this dynamic cycle.
            // This uses the same entry confirmation and quality gates as the player.
            float achievable=float.PositiveInfinity;
            foreach(int sampleRate in new[]{30,60,120})
            {
                var referenceRule=new SideBridgeHoldRule();referenceRule.Reset(difficulty);
                referenceRule.BeginRepetition(600,600,false);
                double duration=(cycle.lowerEndFrame-cycle.liftStartFrame)/(double)timeline.sourceClip.frameRate;
                for(int sample=0;sample<=System.Math.Floor(duration*sampleRate);sample++)
                {
                    Sample(cycle.liftStartFrame+sample*timeline.sourceClip.frameRate/sampleRate);
                    bool valid=SideBridgePoseGeometry.TryMeasure(measured,out var pose)&&SideBridgePoseGeometry.HasStableSupport(pose,top);
                    referenceRule.Step(valid?geometry.Progress(pose):0,valid?geometry.Quality(pose,top):0,1f/sampleRate,valid);
                }
                achievable=Mathf.Min(achievable,referenceRule.RoundValidHoldSeconds);
            }
            if(achievable<.4f)return false;
            // At most two source frames compensate for render/IMU sampling alignment.
            RecordedHoldTargets[i]=Mathf.Max(.3f,achievable-2/timeline.sourceClip.frameRate);
        }
        SelectRepetition(0);return true;
        void Sample(float frame)
        {animator.Play(source.RoutineStateName,0,frame/timeline.sourceClip.frameRate/timeline.sourceClip.length);animator.Update(0);patient.Apply();}
        }
    }

    public void UpdateReference()
    {
        if(transport.Hold30Timeline==null)return;
        int index=0;var reps=transport.Hold30Timeline.Repetitions;
        for(int i=1;i<reps.Length&&transport.Elapsed>=reps[i].start;i++)index=i;
        if(index!=activeRepetition)SelectRepetition(index);
    }

    void SelectRepetition(int index)
    {
        activeRepetition=index;activeCycle=transport.Hold30Timeline!=null?transport.Hold30Timeline.Repetitions[index].cycle:0;
        bool dynamic=transport.PrescriptionDose!=null&&transport.PrescriptionDose.preserveRecordedTiming;
        float goal=dynamic?RecordedHoldTargets[activeCycle]:30;
        Rule.BeginRepetition(goal,goal,!dynamic);
        hasFilteredProgress=false;
    }

    public bool Evaluate(Animator animator,out float progress,out float quality,float deltaTime=.0333333f)
    {
        progress=quality=TrackingQuality=0;PostureReady=SupportReady=false;
        if(Geometry==null||!SideBridgePoseGeometry.TryMeasure(animator,out var pose))return false;
        PostureReady=pose.IsLeftSupport;
        if(!PostureReady){hasFilteredProgress=false;return false;}
        // Filter before clamping: clipping peaks first turns symmetric sensor
        // noise into a downward bias and unfairly fails a correct high hold.
        float rawProgress=Geometry.RawProgress(pose);
        // A short filter removes normalized-height amplification of sensor jitter.
        // A genuine low return bypasses smoothing; it can never extend a hold.
        float alpha=1-Mathf.Exp(-Mathf.Clamp(deltaTime,0,.1f)/.08f);
        filteredProgress=!hasFilteredProgress||rawProgress<.3f||(filteredProgress<.3f&&rawProgress>.65f)
            ?rawProgress:Mathf.Lerp(filteredProgress,rawProgress,alpha);
        hasFilteredProgress=true;
        progress=Mathf.Clamp01(filteredProgress);quality=Geometry.Quality(pose,Geometry.Top);
        SupportReady=SideBridgePoseGeometry.HasStableSupport(pose,Geometry.Top);
        if(SupportReady&&SideBridgePoseGeometry.TryMeasure(transport.CoachAnimator,out var expected))
        {
            float difference=Mathf.Abs(progress-coachReferences[activeCycle].Progress(expected));
            TrackingQuality=Mathf.Min(Geometry.Quality(pose,expected),100*(1-Mathf.InverseLerp(.12f,.5f,difference)));
        }
        return SupportReady;
    }

    public void InvalidateInput(){TrackingQuality=0;PostureReady=SupportReady=false;hasFilteredProgress=false;}

    public void Step(float progress,float quality,float deltaTime,bool valid)
    {
        if(activeRepetition<0||transport==null||Geometry==null)
        {Rule.Step(0,0,deltaTime,false,0);return;}
        float eligible=-1;
        if(Rule.RequiresContinuousHold&&transport.Hold30Timeline!=null)
        {
            var rep=transport.Hold30Timeline.Repetitions[activeRepetition];
            // Raised time before/after the instructed hold cannot fill its 30-second goal.
            eligible=(float)System.Math.Max(0,System.Math.Min(transport.Elapsed,rep.lower)-System.Math.Max(transport.Elapsed-deltaTime,rep.top));
        }
        Rule.Step(progress,quality,deltaTime,valid,eligible);
    }

    public bool Matches(Animator user,Animator coach)
    {
        return Geometry!=null&&SideBridgePoseGeometry.TryMeasure(user,out var actual)&&
            SideBridgePoseGeometry.TryMeasure(coach,out var expected)&&expected.IsLeftSupport&&
            SideBridgePoseGeometry.HasStableSupport(actual,expected)&&
            Mathf.Abs(Geometry.Progress(actual)-coachReferences[activeCycle].Progress(expected))<=GameDifficultyPolicy.BridgeProgressTolerance(difficulty)&&
            Geometry.Quality(actual,expected)>=GameDifficultyPolicy.FollowingThreshold(difficulty);
    }
}
