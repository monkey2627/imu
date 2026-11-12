using UnityEngine;
using System.Collections.Generic;

/// <summary>Opt-in scoring for the two new scenes. The legacy menu/calibration and transport remain in charge.</summary>
public sealed class HipThrustRuleScoring : MonoBehaviour
{
    public HipThrustTaikoController transport;
    public DeadBugGamingPoseScorer calibration;
    public SideBridgePoseScoring sideBridge;
    public readonly HipThrustHoldRule Rule = new HipThrustHoldRule();
    public float Progress { get; private set; }
    public float Quality { get; private set; }
    public bool TargetsReady { get; private set; }
    public bool InputFresh { get; private set; }
    public bool HasMeasurement { get; private set; }
    public float TrackingQuality { get; private set; }
    public float CoachProgress { get; private set; }
    public float MatchedDelay { get; private set; }
    sealed class CoachPoseSample {public double time;public HipThrustHoldGeometry.TargetPose pose;public float progress;}
    readonly List<CoachPoseSample> coachHistory=new List<CoachPoseSample>();
    public float LiveScore => sideBridge!=null?sideBridge.Rule.LiveScore:Rule.LiveScore;
    public float LastRoundBest => sideBridge!=null?sideBridge.Rule.LastRoundBest:Rule.LastRoundBest;
    public string Status => !TargetsReady ? "动作参考不可用" :
        sideBridge!=null&&(calibration==null||!calibration.IsCalibrated)
            ? calibration!=null&&calibration.IsCalibrationBypassed?"流程预览 · 未记录训练成绩":"请先完成动作校准" :
        !InputFresh ? "等待动作数据" :
        sideBridge!=null && !sideBridge.PostureReady ? "请保持左侧支撑、肩髋叠放" :
        sideBridge!=null ? SideStatus :
        Rule.Phase == HipThrustHoldRule.Stage.Holding ? "保持计分" : "抬起进入保持位后计分";
    string SideStatus => !sideBridge.SupportReady?"请调整支撑肘和肩髋排列":
        sideBridge.Rule.RoundFinished?"本次已记录 · 跟随下一次示范":
        !sideBridge.Rule.LowConfirmed?"请先回到低位":
        sideBridge.Rule.HoldingCorrectly?"姿势达标 · 有效保持中":
        sideBridge.Rule.Phase==SideBridgeHoldRule.Stage.Lowering?"请完全放下后计次":"请按教练调整抬起幅度和姿势";
    readonly HipThrustHoldGeometry geometry = new HipThrustHoldGeometry();
    readonly HipThrustHoldGeometry coachGeometry = new HipThrustHoldGeometry();
    AvatarMotionSource motion;
    int lastCounter;
    float lastFrameAt = float.NegativeInfinity;
    float minimumTrackingQuality;
    float progressTolerance;

    public void Begin(string difficulty)
    {
        Rule.Reset(difficulty); Progress=Quality=MatchedDelay=0; InputFresh=false;coachHistory.Clear();
        minimumTrackingQuality=GameDifficultyPolicy.FollowingThreshold(difficulty);
        progressTolerance=GameDifficultyPolicy.BridgeProgressTolerance(difficulty);
        if(sideBridge!=null) sideBridge.Begin(difficulty);
        bool continuous=transport.Hold30Timeline!=null&&!transport.Hold30Timeline.UsesRecordedTiming;
        Rule.ContinuousHold=continuous;if(continuous)Rule.SetFullScoreSeconds(30);
        motion=FindObjectOfType<AvatarMotionSource>();
        lastCounter=motion!=null?motion.LastPacketframeCounterInFrame:0;
        lastFrameAt=float.NegativeInfinity;
        TargetsReady=CaptureTargets();
        if(!TargetsReady) Debug.LogError(transport.ActionDisplayName+"评分无法采样录制动作的 A/B 参考姿态。",this);
    }

    bool CaptureTargets()
    {
        var animator=transport.CoachAnimator; var timeline=transport.PhaseTimeline;
        if(animator==null || !animator.isHuman || timeline==null) return false;
        var state=animator.GetCurrentAnimatorStateInfo(0); float speed=animator.speed;
        try
        {
            if(sideBridge!=null) return sideBridge.CaptureTargets(transport);
            using(var patient=new PatientPoseReference(animator,calibration!=null?calibration.UserAnimator:null))
            {
            Sample(timeline.cycles[0].liftStartFrame);
            patient.Apply();
            if(!geometry.TryMeasurePose(patient.Animator,out geometry.restTarget)||!coachGeometry.TryMeasurePose(animator,out coachGeometry.restTarget)) return false;
            Sample(HipThrustHold30Timeline.PeakFrame(timeline,0));
            patient.Apply();
            if(!geometry.TryMeasurePose(patient.Animator,out geometry.holdTarget)||!coachGeometry.TryMeasurePose(animator,out coachGeometry.holdTarget)) return false;
            if(geometry.restTarget.hasBridgeLift && geometry.holdTarget.hasBridgeLift &&
               geometry.restTarget.bridgeLiftRatio>geometry.holdTarget.bridgeLiftRatio)
            { var swap=geometry.restTarget; geometry.restTarget=geometry.holdTarget; geometry.holdTarget=swap; }
            return true;
            }
        }
        finally { animator.Play(state.fullPathHash,0,state.normalizedTime); animator.Update(0); animator.speed=speed; }
        void Sample(int frame)
        { animator.Play(transport.RoutineStateName,0,frame/timeline.sourceClip.frameRate/timeline.sourceClip.length); animator.Update(0); }
    }

    // Observe packet freshness even while the transport is paused. Otherwise a
    // packet received early in a long pause could appear newly received on resume.
    void Update() { ObserveInput(); }

    void ObserveInput()
    {
        if(motion==null) motion=FindObjectOfType<AvatarMotionSource>();
        if(motion!=null && motion.LastPacketframeCounterInFrame!=lastCounter)
        { lastCounter=motion.LastPacketframeCounterInFrame; lastFrameAt=Time.unscaledTime; }
    }

    public void Tick(float deltaTime)
    {
        ObserveInput();
        InputFresh=motion!=null && motion.HasFreshFrame && Time.unscaledTime-lastFrameAt<=.5f;
        bool valid=TargetsReady && InputFresh && deltaTime>0 && deltaTime<=.1001f && calibration!=null && calibration.IsCalibrated;
        if(sideBridge!=null&&TargetsReady)sideBridge.UpdateReference();
        Progress=Quality=TrackingQuality=CoachProgress=0;HasMeasurement=false;
        if(valid)
        {
            if(sideBridge!=null)
            {
                HasMeasurement=SideBridgePoseGeometry.TryMeasure(calibration.UserAnimator,out _);
                valid=sideBridge.Evaluate(calibration.UserAnimator,out float progress,out float quality,deltaTime);
                Progress=progress;Quality=quality;
                TrackingQuality=sideBridge.TrackingQuality;
            }
            else
            {
                valid=geometry.TryMeasurePose(calibration.UserAnimator,out var pose);
                HasMeasurement=valid;
                valid=valid&&HipThrustHoldGeometry.HasSupineSupport(pose,geometry.holdTarget);
                if(valid)
                {
                    Progress=geometry.CalculateProgress(pose); Quality=geometry.CalculatePoseQuality(pose,geometry.holdTarget,true);
                    if(geometry.TryMeasurePose(transport.CoachAnimator,out var expected))
                    {
                        double now=transport.Elapsed;
                        coachHistory.Add(new CoachPoseSample{time=now,pose=expected,progress=coachGeometry.CalculateProgress(expected)});
                        while(coachHistory.Count>1&&(coachHistory[0].time<now-TrainingScoringTolerance.MaximumResponseDelay||coachHistory.Count>160))coachHistory.RemoveAt(0);
                        float bestObjective=float.NegativeInfinity,previousDelay=MatchedDelay;
                        foreach(var sample in coachHistory)
                        {
                            float difference=Mathf.Abs(Progress-sample.progress),delay=(float)(now-sample.time);
                            if(delay<0||delay>TrainingScoringTolerance.MaximumResponseDelay)continue;
                            float quality=Mathf.Min(100*(1-Mathf.InverseLerp(.12f,.5f,difference)),geometry.CalculatePoseQuality(pose,sample.pose,false));
                            float objective=quality-100*Mathf.Max(0,difference-.05f)-2*Mathf.Abs(delay-previousDelay);
                            if(objective>bestObjective){bestObjective=objective;TrackingQuality=quality;CoachProgress=sample.progress;MatchedDelay=delay;}
                        }
                    }
                }
            }
        }
        if(sideBridge!=null)
        {
            if(!HasMeasurement||!TargetsReady||!InputFresh||calibration==null||!calibration.IsCalibrated)sideBridge.InvalidateInput();
            sideBridge.Step(Progress,Quality,deltaTime,valid);
        }
        else
        {
            float eligible=-1;
            if(Rule.ContinuousHold&&transport.Hold30Timeline!=null)
            {
                var reps=transport.Hold30Timeline.Repetitions;int index=0;
                for(int i=1;i<reps.Length&&transport.Elapsed>=reps[i].start;i++)index=i;
                var rep=reps[index];
                eligible=(float)System.Math.Max(0,System.Math.Min(transport.Elapsed,rep.lower)-System.Math.Max(transport.Elapsed-deltaTime,rep.top));
            }
            // The forgiving A/B entry thresholds must not turn a sustained half
            // lift into a high live score while the settlement correctly rejects it.
            float holdQuality=Mathf.Abs(Progress-CoachProgress)<=progressTolerance&&TrackingQuality>=minimumTrackingQuality
                ?Mathf.Min(Quality,TrackingQuality):0;
            Rule.Step(Progress,holdQuality,deltaTime,valid,eligible);
        }
    }
    public void BreakContinuity(){Rule.BreakContinuity();if(sideBridge!=null)sideBridge.Rule.BreakContinuity();}
    public bool MatchesHipCoach() => sideBridge==null&&TargetsReady&&InputFresh&&HasMeasurement&&calibration!=null&&calibration.IsCalibrated&&
        TrackingQuality>=minimumTrackingQuality&&Mathf.Abs(Progress-CoachProgress)<=progressTolerance;

    public bool MatchesSideBridgeCoach() => sideBridge!=null && TargetsReady && InputFresh && HasMeasurement && calibration!=null && calibration.IsCalibrated &&
        sideBridge.Matches(calibration.UserAnimator,transport.CoachAnimator);

    public string Result(int demonstrated) => sideBridge!=null
        ? $"示范结束  {demonstrated} 次{transport.ActionDisplayName}\n实际合格 {sideBridge.Rule.QualifiedRounds}/{demonstrated} 次 · 完整往返 {sideBridge.Rule.CompletedRounds} 次\n有效保持累计 {sideBridge.Rule.TotalValidHoldSeconds:0.0} 秒\n最高保持分 {sideBridge.Rule.SessionBest:0}/100（不代表处方已完成）"
        : $"训练完成  示范 {demonstrated} 次{transport.ActionDisplayName}\n最高 {Rule.SessionBest:0} / 100　上轮 {Rule.LastRoundBest:0}\n识别往返 {Rule.CompletedRounds} 次";
}
