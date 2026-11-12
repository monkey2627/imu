using System;
using UnityEngine;

/// <summary>Resamples the recorded poses, freezes the actual peak for 30 seconds, and executes an exact dose.</summary>
public sealed class HipThrustHold30Timeline
{
    public const float HoldSeconds = 30;
    public readonly struct Repetition
    {
        public readonly int cycle, peakFrame;
        public readonly double start, top, lower, end;
        public Repetition(int cycle, int peak, double start, double top, double end, double hold=HoldSeconds)
        { this.cycle=cycle; peakFrame=peak; this.start=start; this.top=top; lower=top+hold; this.end=end; }
    }
    public Repetition[] Repetitions { get; }
    public double Duration { get; }
    readonly HipThrustPhaseTimeline source;
    readonly float fps;
    public bool UsesRecordedTiming { get; }
    readonly bool skipPlateau;
    public bool PreservesSourceGaps { get; }
    public const float WrapSeconds=.5f;

    public HipThrustHold30Timeline(HipThrustPhaseTimeline source, int count, bool preserveRecordedTiming=false, bool skipRecordedPlateauAfterHold=false, bool preserveSourceGaps=false)
    {
        if (count < 1 || count > 2000) throw new ArgumentOutOfRangeException(nameof(count), "处方总次数必须为 1–2000。");
        source.Validate(); this.source=source; fps=source.sourceClip.frameRate;
        UsesRecordedTiming=preserveRecordedTiming;
        skipPlateau=skipRecordedPlateauAfterHold&&!preserveRecordedTiming;
        PreservesSourceGaps=preserveSourceGaps;
        Repetitions=new Repetition[count];
        double start=source.cycles[0].liftStartFrame/(double)fps;
        for(int i=0;i<count;i++)
        {
            int index=i%source.cycles.Length; var cycle=source.cycles[index];
            int peak=PeakFrame(source, index);
            double top=start+((UsesRecordedTiming?cycle.holdStartFrame:peak)-cycle.liftStartFrame)/(double)fps;
            double hold=UsesRecordedTiming?(cycle.lowerStartFrame-cycle.holdStartFrame)/(double)fps:HoldSeconds;
            double end=top+hold+(cycle.lowerEndFrame-(UsesRecordedTiming||skipPlateau?cycle.lowerStartFrame:peak))/(double)fps;
            Repetitions[i]=new Repetition(index,peak,start,top,end,hold); start=end;
            if(PreservesSourceGaps && i<count-1)
            {
                int next=(index+1)%source.cycles.Length;
                start+=next==0?WrapSeconds:(source.cycles[next].liftStartFrame-cycle.lowerEndFrame)/(double)fps;
            }
        }
        // A short final rest allows the final endpoint judgement to resolve. Never play subsequent source repetitions.
        Duration=start+.5;
    }

    public static int PeakFrame(HipThrustPhaseTimeline source,int index)
    {
        var cycle=source.cycles[index]; int peak=cycle.holdStartFrame;
        for(int f=cycle.liftStartFrame+1;f<cycle.lowerEndFrame;f++)
            if(source.recordedHeight.Evaluate(f)>source.recordedHeight.Evaluate(peak)) peak=f;
        return peak;
    }

    public HipThrustPhaseTimeline.Sample Evaluate(double time,out float sourceFrame,out float remaining,out int completed)
    {
        var sample=new HipThrustPhaseTimeline.Sample {phase=-1,cycle=-1,completedSeconds=Vector3.one*999};
        remaining=0; completed=0;
        sourceFrame=(float)Math.Max(0,time)*fps;
        Repetition previous=default;
        foreach(var rep in Repetitions)
        {
            var c=source.cycles[rep.cycle];
            if(time>=rep.top) sample.completedSeconds.x=Mathf.Min(sample.completedSeconds.x,(float)(time-rep.top));
            if(time>=rep.lower) sample.completedSeconds.y=Mathf.Min(sample.completedSeconds.y,(float)(time-rep.lower));
            if(time>=rep.end) sample.completedSeconds.z=Mathf.Min(sample.completedSeconds.z,(float)(time-rep.end));
            if(time>=rep.end) { completed++; sourceFrame=c.lowerEndFrame; previous=rep; continue; }
            if(time<rep.start)
            {
                if(PreservesSourceGaps && completed>0)
                    sourceFrame=rep.cycle==0?c.liftStartFrame:source.cycles[previous.cycle].lowerEndFrame+(float)(time-previous.end)*fps;
                break;
            }
            sample.cycle=rep.cycle;
            double begin,end;
            if(time<rep.top)
            { sample.phase=0; begin=rep.start; end=rep.top; sourceFrame=c.liftStartFrame+(float)(time-begin)*fps; }
            else if(time<rep.lower)
            { sample.phase=1; begin=rep.top; end=rep.lower; sourceFrame=UsesRecordedTiming?c.holdStartFrame+(float)(time-begin)*fps:rep.peakFrame; }
            else
            {
                sample.phase=2; begin=rep.lower; end=rep.end;
                sourceFrame=(UsesRecordedTiming||skipPlateau?c.lowerStartFrame:rep.peakFrame)+(float)(time-begin)*fps;
                // The side-bridge capture contains a long stationary plateau after
                // its peak. Do not append that plateau to the requested 30 seconds.
                if(skipPlateau) sourceFrame=Mathf.Lerp(rep.peakFrame,sourceFrame,Mathf.SmoothStep(0,1,(float)(time-begin)/.15f));
            }
            sample.progress=Mathf.Clamp01((float)((time-begin)/(end-begin)));
            remaining=(float)(end-time);
            sample.holdSeconds=sample.phase==0?0:sample.phase==1?(float)(time-rep.top):(float)(rep.lower-rep.top);
            sample.poseHeight=sample.phase==1?1:Mathf.InverseLerp(c.lowHeight,source.recordedHeight.Evaluate(rep.peakFrame),source.recordedHeight.Evaluate(sourceFrame));
            break;
        }
        return sample;
    }

    public bool TryGetWrapProgress(double time,out float progress)
    {
        progress=0;
        if(!PreservesSourceGaps)return false;
        for(int i=1;i<Repetitions.Length;i++)
            if(Repetitions[i].cycle==0 && time>=Repetitions[i-1].end && time<Repetitions[i].start)
            {progress=Mathf.Clamp01((float)((time-Repetitions[i-1].end)/(Repetitions[i].start-Repetitions[i-1].end)));return true;}
        return false;
    }
}
