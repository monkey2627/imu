using UnityEngine;

/// <summary>One measured attempt per demonstrated side bridge. A true low position arms each attempt.</summary>
public sealed class SideBridgeHoldRule
{
    public enum Stage { Rest, Raising, Holding, Lowering }
    public Stage Phase { get; private set; }
    public bool InputValid { get; private set; }
    public bool HoldingCorrectly { get; private set; }
    public bool LowConfirmed { get; private set; }
    public bool RoundFinished { get; private set; }
    public float ContinuousHoldSeconds => (float)continuousHold;
    public float BestContinuousSeconds => (float)bestContinuous;
    public float TotalValidHoldSeconds => (float)totalValidHold;
    public float RoundValidHoldSeconds => (float)roundValidHold;
    public bool RequiresContinuousHold { get; private set; }
    public float TargetHoldSeconds { get; private set; }
    public float RoundBest { get; private set; }
    public float LastRoundBest { get; private set; }
    public float LastRoundHoldSeconds { get; private set; }
    public float SessionBest { get; private set; }
    public int CompletedRounds { get; private set; }
    public int QualifiedRounds { get; private set; }
    public float DisplayHoldSeconds => RoundFinished ? LastRoundHoldSeconds : RequiresContinuousHold?ContinuousHoldSeconds:RoundValidHoldSeconds;
    public float LiveScore => InputValid && HoldingCorrectly && Phase == Stage.Holding && !RoundFinished
        ? Mathf.Lerp(entryScore,100,Mathf.Clamp01((RequiresContinuousHold?ContinuousHoldSeconds:RoundValidHoldSeconds)/scoreSeconds)) : 0;
    public float EnterProgress { get; private set; }
    public float SustainProgress => EnterProgress-.05f;
    public float EnterQuality { get; private set; }
    public float ValidQuality { get; private set; }
    public float DefaultScoreSeconds { get; private set; }
    public void BreakContinuity(){if(RequiresContinuousHold)continuousHold=pendingHold=0;HoldingCorrectly=false;confirmation=badSeconds=0;}
    float exit, rest, entryScore, scoreSeconds, confirmation, badSeconds, missingSeconds;
    double continuousHold,bestContinuous,totalValidHold,roundValidHold,pendingHold;

    public void Reset(string difficulty)
    {
        if(difficulty=="simple") {EnterProgress=.65f;exit=.48f;rest=.22f;EnterQuality=60;ValidQuality=55;entryScore=85;DefaultScoreSeconds=3;}
        else if(difficulty=="hard") {EnterProgress=.85f;exit=.68f;rest=.25f;EnterQuality=80;ValidQuality=75;entryScore=75;DefaultScoreSeconds=7;}
        else {EnterProgress=.75f;exit=.58f;rest=.23f;EnterQuality=70;ValidQuality=65;entryScore=80;DefaultScoreSeconds=5;}
        CompletedRounds=QualifiedRounds=0;LastRoundBest=LastRoundHoldSeconds=SessionBest=0;totalValidHold=0;
        BeginRepetition(DefaultScoreSeconds,DefaultScoreSeconds);
    }

    public void BeginRepetition(float targetSeconds,float fullScoreSeconds,bool continuous=true)
    {
        TargetHoldSeconds=Mathf.Max(.3f,targetSeconds);scoreSeconds=Mathf.Max(.3f,fullScoreSeconds);
        RequiresContinuousHold=continuous;
        RoundFinished=false;ResetAttempt();
    }

    void ResetAttempt()
    {
        Phase=Stage.Rest;LowConfirmed=InputValid=HoldingCorrectly=false;
        continuousHold=bestContinuous=roundValidHold=pendingHold=0;RoundBest=confirmation=badSeconds=missingSeconds=0;
    }

    public void Step(float progress,float quality,float deltaTime,bool validInput,float eligibleHoldDelta=-1)
    {
        float dt=float.IsNaN(deltaTime)||float.IsInfinity(deltaTime)?0:Mathf.Clamp(deltaTime,0,.1f);
        float holdDt=float.IsNaN(eligibleHoldDelta)||float.IsInfinity(eligibleHoldDelta)?0:eligibleHoldDelta<0?dt:Mathf.Clamp(eligibleHoldDelta,0,dt);
        InputValid=validInput&&deltaTime<=.1001f&&!float.IsNaN(progress)&&!float.IsInfinity(progress)&&!float.IsNaN(quality)&&!float.IsInfinity(quality);
        HoldingCorrectly=false;
        if(!InputValid)
        {
            float missingDt=float.IsNaN(deltaTime)||float.IsInfinity(deltaTime)?0:Mathf.Max(0,deltaTime);
            confirmation=0;pendingHold=0;missingSeconds+=missingDt;badSeconds+=missingDt;
            if(TrainingScoringTolerance.GapExpired(badSeconds))continuousHold=0;
            // A long tracking loss cannot bridge two unrelated poses into a repetition.
            if(missingSeconds>=.5f&&!RoundFinished)ResetAttempt();
            return;
        }
        missingSeconds=0;
        if(RoundFinished)return;
        switch(Phase)
        {
            case Stage.Rest:
                if(progress<=rest)
                {confirmation+=dt;if(confirmation+.00001f>=.08f)LowConfirmed=true;}
                else
                {
                    confirmation=0;
                    if(LowConfirmed){Phase=Stage.Raising;StepRaising(progress,quality,dt,holdDt);}
                }
                break;
            case Stage.Raising: StepRaising(progress,quality,dt,holdDt);break;
            case Stage.Holding:
                HoldingCorrectly=progress>=SustainProgress&&quality>=ValidQuality;
                if(HoldingCorrectly){badSeconds=0;AddHold(holdDt);}
                else
                {
                    badSeconds+=dt;
                    // Reject isolated noisy samples without counting them as valid time.
                    if(TrainingScoringTolerance.GapExpired(badSeconds))continuousHold=0;
                }
                if(progress<=exit)
                {
                    confirmation+=dt;
                    if(confirmation+.00001f>=.12f){Phase=Stage.Lowering;confirmation=0;HoldingCorrectly=false;}
                }
                else confirmation=0;
                break;
            case Stage.Lowering:
                if(progress<=rest)
                {
                    confirmation+=dt;
                    if(confirmation+.00001f>=.08f)
                    {
                        float achieved=RequiresContinuousHold?BestContinuousSeconds:RoundValidHoldSeconds;
                        CompletedRounds++;if(TrainingScoringTolerance.HoldQualified(achieved,TargetHoldSeconds,RequiresContinuousHold))QualifiedRounds++;
                        LastRoundBest=RoundBest;LastRoundHoldSeconds=achieved;
                        RoundFinished=true;Phase=Stage.Rest;continuousHold=0;
                    }
                }
                else confirmation=0;
                break;
        }
        RoundBest=Mathf.Max(RoundBest,LiveScore);SessionBest=Mathf.Max(SessionBest,RoundBest);
    }

    void StepRaising(float progress,float quality,float dt,float holdDt)
    {
        if(progress>=EnterProgress&&quality>=EnterQuality)
        {
            badSeconds=0;confirmation+=dt;pendingHold+=holdDt;
            if(confirmation+.00001f>=.3f)
            {
                Phase=Stage.Holding;HoldingCorrectly=true;
                // These already observed good samples are real hold time, not a timer bonus.
                AddHold((float)pendingHold);confirmation=0;pendingHold=0;
            }
        }
        else
        {
            // A brief dip near the entry boundary does not erase all preceding
            // stable samples. Only actually good entry samples add hold time.
            badSeconds+=dt;
            if(badSeconds>.1f||progress<SustainProgress||quality<ValidQuality){confirmation=0;pendingHold=0;}
            if(progress<=rest){Phase=Stage.Rest;LowConfirmed=true;badSeconds=0;}
        }
    }

    void AddHold(float dt)
    {
        continuousHold+=dt;totalValidHold+=dt;roundValidHold+=dt;
        bestContinuous=System.Math.Max(bestContinuous,continuousHold);
    }
}
