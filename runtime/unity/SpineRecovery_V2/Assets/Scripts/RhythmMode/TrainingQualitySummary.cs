using System;
using UnityEngine;

/// <summary>Version 1 session assessment. All denominators come from the prescribed
/// coach timeline, never from the number of successful samples. No live/HIT rules are changed.</summary>
public sealed class TrainingQualitySummary
{
    public sealed class Repetition
    {
        public double expectedSeconds, targetSeconds;
        public double observedSeconds, trackedSeconds, qualityIntegral, qualifiedSeconds;
        public double expectedFollowingSeconds, followingQualifiedSeconds;
        public double creditedSeconds, longestHold;
        public bool completed, qualified;
    }
    public Repetition[] Repetitions { get; private set; } = Array.Empty<Repetition>();
    public bool IsHold { get; private set; }
    public bool IsContinuous { get; private set; }
    public float Threshold { get; private set; }
    public int Planned => Repetitions.Length;
    public int Completed { get { int n=0; foreach(var r in Repetitions) if(r.completed)n++; return n; } }
    public int Qualified { get { int n=0; foreach(var r in Repetitions) if(r.qualified)n++; return n; } }
    public double ExpectedSeconds => Sum(r=>r.expectedSeconds);
    public double ObservedSeconds => Sum(r=>r.observedSeconds);
    public double TrackedSeconds => Sum(r=>r.trackedSeconds);
    public double TargetSeconds => Sum(r=>r.targetSeconds);
    public double ExpectedFollowingSeconds => Sum(r=>r.expectedFollowingSeconds);
    public double QualifiedFollowingSeconds => Sum(r=>r.followingQualifiedSeconds);
    public double CreditedSeconds => Sum(r=>Math.Min(r.targetSeconds,IsHold?r.creditedSeconds:r.qualifiedSeconds));
    public double LongestHold { get { double best=0; foreach(var r in Repetitions)best=Math.Max(best,r.longestHold); return best; } }
    public float Coverage => Ratio(TrackedSeconds,ExpectedSeconds);
    public float Quality => 100*Ratio(Sum(r=>r.qualityIntegral),100*ExpectedSeconds);
    public float TimeCompletion => Ratio(CreditedSeconds,TargetSeconds);
    public float RepetitionCompletion => Ratio(Qualified,Planned);
    public bool HasAssessment => TrackedSeconds >= .1 && ExpectedSeconds > 0;
    // 100 is reserved for all planned repetitions and dose, even after integer rounding.
    public int Score => Mathf.Min(Qualified==Planned && TimeCompletion>=.9999f && Coverage>=.9999f?100:99,
        Mathf.RoundToInt(.5f*Quality+30*TimeCompletion+20*RepetitionCompletion));

    public void Reset(double[] durations,double[] targets,bool hold,bool continuous,float threshold,double[] followingTargets=null)
    {
        if(durations.Length!=targets.Length)throw new ArgumentException("Mismatched assessment plan");
        IsHold=hold;IsContinuous=continuous;Threshold=threshold;
        Repetitions=new Repetition[durations.Length];
        for(int i=0;i<durations.Length;i++)Repetitions[i]=new Repetition {
            expectedSeconds=Safe(durations[i]),targetSeconds=Safe(targets[i]),
            expectedFollowingSeconds=followingTargets==null?Safe(durations[i]):Safe(followingTargets[i]) };
    }

    public void Observe(int index,double seconds,bool tracked,float quality,bool qualified,double followingSeconds=-1)
    {
        if(index<0||index>=Planned)return;
        var r=Repetitions[index];
        double dt=Math.Min(Safe(seconds),Math.Max(0,r.expectedSeconds-r.observedSeconds));
        r.observedSeconds+=dt;
        if(!tracked || float.IsNaN(quality)||float.IsInfinity(quality))return;
        r.trackedSeconds+=dt;r.qualityIntegral+=Mathf.Clamp(quality,0,100)*dt;
        if(qualified)
        {
            r.qualifiedSeconds+=dt;
            r.followingQualifiedSeconds+=Math.Min(Math.Max(0,r.expectedFollowingSeconds-r.followingQualifiedSeconds),
                followingSeconds<0?dt:Math.Min(dt,Safe(followingSeconds)));
        }
    }

    public void RecordDose(int index,double seconds,double longest,bool completed,bool qualified)
    {
        if(index<0||index>=Planned)return;
        var r=Repetitions[index];
        r.creditedSeconds=Math.Max(r.creditedSeconds,Math.Min(r.targetSeconds,Safe(seconds)));
        r.longestHold=Math.Max(r.longestHold,Safe(longest));
        r.completed|=completed;r.qualified|=completed&&qualified;
    }
    public void RejectStationary(int index)
    {
        var r=Repetitions[index];r.qualityIntegral=r.qualifiedSeconds=r.followingQualifiedSeconds=r.creditedSeconds=r.longestHold=0;
        r.completed=r.qualified=false;
    }
    public void CapMovementQuality(int index,float maximumQuality)
    {
        var r=Repetitions[index];
        r.qualityIntegral=Math.Min(r.qualityIntegral,Mathf.Clamp(maximumQuality,0,100)*r.expectedSeconds);
    }
    double Sum(Func<Repetition,double> value){double n=0;foreach(var r in Repetitions)n+=value(r);return n;}
    static double Safe(double n)=>double.IsNaN(n)||double.IsInfinity(n)?0:Math.Max(0,n);
    static float Ratio(double n,double d)=>d>0?Mathf.Clamp01((float)(n/d)):0;
}

/// <summary>Settlement-only hip bridge attempt: observe low, high, then a genuine low
/// return; a prolonged hold cannot compensate for another missed repetition.</summary>
public sealed class TrainingBridgeAttempt
{
    public double TotalHold { get; private set; }
    public double LongestHold { get; private set; }
    public bool Completed { get; private set; }
    public bool Armed { get; private set; }
    double lowSeconds,continuous,badSeconds,missingSeconds;
    bool raised;
    public void BreakContinuity(){continuous=0;badSeconds=0;}
    public void Step(double dt,double holdDelta,bool tracked,bool low,bool high,bool lowering)
    {
        if(dt<=0||double.IsNaN(dt)||double.IsInfinity(dt)||Completed)return;
        if(!tracked)
        {
            lowSeconds=0;missingSeconds+=dt;badSeconds+=dt;
            if(TrainingScoringTolerance.GapExpired(badSeconds))continuous=0;
            if(missingSeconds>=.5){Armed=raised=false;TotalHold=LongestHold=0;}
            return;
        }
        missingSeconds=0;
        if(low){lowSeconds+=dt;if(lowSeconds>=.08){if(!raised)Armed=true;else if(lowering)Completed=true;}}
        else lowSeconds=0;
        if(!Armed)return;
        if(high && holdDelta>0)
        {raised=true;badSeconds=0;TotalHold+=holdDelta;continuous+=holdDelta;LongestHold=Math.Max(LongestHold,continuous);}
        else if(holdDelta>0)
        {badSeconds+=holdDelta;if(TrainingScoringTolerance.GapExpired(badSeconds))continuous=0;}
    }
}
