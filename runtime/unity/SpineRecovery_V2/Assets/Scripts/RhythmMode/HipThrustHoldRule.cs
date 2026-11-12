using UnityEngine;

/// <summary>The delivery package's continuous A/B rule, independent of the coach clock.</summary>
public sealed class HipThrustHoldRule
{
    public enum Stage { Rest, Raising, Holding, Lowering }
    public Stage Phase { get; private set; }
    public float LiveScore => InputValid && HoldingCorrectly && Phase == Stage.Holding
        ? Mathf.Lerp(entryScore, 100, Mathf.Clamp01(ValidHoldSeconds / maximumSeconds)) : 0;
    public float ValidHoldSeconds { get; private set; }
    public float RoundBest { get; private set; }
    public float LastRoundBest { get; private set; }
    public float SessionBest { get; private set; }
    public int CompletedRounds { get; private set; }
    public bool InputValid { get; private set; }
    public bool HoldingCorrectly { get; private set; }
    public bool LowConfirmed { get; private set; }
    public bool ContinuousHold { get; set; }
    float raise, enter, exit, rest, enterQuality, validQuality, entryScore, maximumSeconds;
    float confirmation, phaseElapsed, badSeconds, missingSeconds,pendingHold;
    public void SetFullScoreSeconds(float seconds){maximumSeconds=Mathf.Max(.1f,seconds);}
    public void BreakContinuity(){if(ContinuousHold)ValidHoldSeconds=0;HoldingCorrectly=false;confirmation=badSeconds=pendingHold=0;}

    public void Reset(string difficulty)
    {
        Phase = Stage.Rest;
        ValidHoldSeconds = RoundBest = LastRoundBest = SessionBest = confirmation = phaseElapsed = 0;
        CompletedRounds = 0; InputValid = false;
        LowConfirmed=HoldingCorrectly=ContinuousHold=false;badSeconds=missingSeconds=pendingHold=0;
        if (difficulty == "simple")
        { raise=.08f; enter=.20f; exit=.14f; rest=.10f; enterQuality=35; validQuality=30; entryScore=85; maximumSeconds=3; }
        else if (difficulty == "hard")
        { raise=.20f; enter=.55f; exit=.40f; rest=.18f; enterQuality=65; validQuality=60; entryScore=75; maximumSeconds=7; }
        else
        { raise=.15f; enter=.40f; exit=.28f; rest=.15f; enterQuality=50; validQuality=45; entryScore=80; maximumSeconds=5; }
    }

    public void Step(float progress, float quality, float deltaTime, bool validInput,float eligibleHoldDelta=-1)
    {
        InputValid = validInput && deltaTime<=.1001f && !float.IsNaN(progress) && !float.IsInfinity(progress)
            && !float.IsNaN(quality) && !float.IsInfinity(quality);
        HoldingCorrectly=false;
        float dt = float.IsNaN(deltaTime)||float.IsInfinity(deltaTime)?0:Mathf.Clamp(deltaTime, 0, .1f);
        float holdDt=float.IsNaN(eligibleHoldDelta)||float.IsInfinity(eligibleHoldDelta)?0:eligibleHoldDelta<0?dt:Mathf.Clamp(eligibleHoldDelta,0,dt);
        if (!InputValid)
        {
            float missingDt=float.IsNaN(deltaTime)||float.IsInfinity(deltaTime)?0:Mathf.Max(0,deltaTime);
            confirmation=pendingHold=0;missingSeconds+=missingDt;badSeconds+=missingDt;
            if(ContinuousHold&&TrainingScoringTolerance.GapExpired(badSeconds))ValidHoldSeconds=0;
            if(missingSeconds>=.5f){LowConfirmed=false;ValidHoldSeconds=RoundBest=0;Change(Stage.Rest);}
            return;
        }
        missingSeconds=0;
        phaseElapsed += dt;
        switch (Phase)
        {
            case Stage.Rest:
                if(progress<=rest){confirmation+=dt;if(confirmation+.000001f>=.08f)LowConfirmed=true;}
                else confirmation=0;
                if (LowConfirmed && progress >= raise) Change(Stage.Raising);
                break;
            case Stage.Raising:
                if (progress >= enter && quality >= enterQuality)
                {
                    confirmation += dt;
                    pendingHold+=holdDt;
                    if (confirmation + .000001f >= .30f) { ValidHoldSeconds = ContinuousHold?pendingHold:0;pendingHold=0;Change(Stage.Holding);HoldingCorrectly=true; }
                }
                else
                {
                    confirmation = pendingHold = 0;
                    if (progress <= rest) Change(Stage.Rest);
                }
                break;
            case Stage.Holding:
                HoldingCorrectly=quality>=validQuality&&progress>=enter;
                if(HoldingCorrectly){badSeconds=0;ValidHoldSeconds+=holdDt;}
                else {badSeconds+=dt;if(ContinuousHold&&TrainingScoringTolerance.GapExpired(badSeconds))ValidHoldSeconds=0;}
                RoundBest = Mathf.Max(RoundBest, LiveScore);
                if (progress <= exit)
                {
                    confirmation += dt;
                    if (confirmation + .000001f >= .12f)
                    { LastRoundBest = RoundBest; Change(Stage.Lowering); }
                }
                else confirmation = 0;
                break;
            case Stage.Lowering:
                if (progress <= rest)
                {
                    confirmation += dt;
                    if (confirmation + .000001f >= .08f)
                    { CompletedRounds++; RoundBest = 0; Change(Stage.Rest); }
                }
                else confirmation = 0;
                break;
        }
        RoundBest = Mathf.Max(RoundBest, LiveScore);
        SessionBest = Mathf.Max(SessionBest, RoundBest);
    }

    void Change(Stage next) { Phase = next; confirmation = phaseElapsed = 0; }

    // A side bridge must approach its measured top; the supine bridge's deliberately
    // permissive 20/40/55% entry thresholds would reward a visibly sagging side bridge.
    public void ResetSideBridge(string difficulty)
    {
        Reset(difficulty);
        if (difficulty == "simple") { enter=.65f; exit=.48f; rest=.22f; enterQuality=60; validQuality=55; }
        else if (difficulty == "hard") { enter=.85f; exit=.68f; rest=.25f; enterQuality=80; validQuality=75; }
        else { enter=.75f; exit=.58f; rest=.23f; enterQuality=70; validQuality=65; }
    }
}
