using System;
using System.IO;
using UnityEngine;

public static class TrainingToleranceVerification
{
    static int checks;
    public static void Run()
    {
        checks=0;
        foreach(int fps in new[]{30,60,120})
        foreach(double gap in new[]{1d/fps,.1,.12,.2,.2001,.25,.6})
        {
            var attempt=new TrainingBridgeAttempt();attempt.Step(.1,0,true,true,false,false);
            Feed(15,true);attempt.Step(gap,gap,false,false,false,false);Feed(15-gap,true);
            attempt.Step(.1,0,true,true,false,true);
            bool brief=gap<=TrainingScoringTolerance.ShortGapSeconds;
            Check(!brief||Math.Abs(attempt.LongestHold-(30-gap))<.0001,"Short gap reset or added hold evidence");
            Check(brief||attempt.LongestHold<15.01,"Long loss joined distinct holds");
            Check(TrainingScoringTolerance.HoldQualified(attempt.LongestHold,30,true)==brief,"Qualification ignored bounded gap policy");
            void Feed(double seconds,bool valid)
            {for(double t=0;t<seconds-.000001;t+=1d/fps){double dt=Math.Min(1d/fps,seconds-t);attempt.Step(dt,dt,valid,false,true,false);}}
        }
        foreach(string difficulty in new[]{"simple","standard","hard"})
        {
            var side=new SideBridgeHoldRule();side.Reset(difficulty);side.BeginRepetition(30,30);
            FeedSide(.2f,0,true);FeedSide(15,1,true);float before=side.ContinuousHoldSeconds;
            side.Step(1,100,.12f,true); // Rendering stall: unavailable interval, not .1s of invented evidence.
            Check(Math.Abs(before-side.ContinuousHoldSeconds)<.001,"Render stall altered measured continuous time");
            Check(side.LiveScore==0,"Render stall kept live feedback valid");
            FeedSide(14.9f,1,true);Check(side.LiveScore<100,"29.9 seconds earned full live marks");
            FeedSide(.3f,0,true);Check(side.QualifiedRounds==1,"Nearly complete hold lost all repetition credit");
            Check(side.LastRoundHoldSeconds<30,"Qualification rounded the measured duration up");
            side.BeginRepetition(30,30);FeedSide(.2f,0,true);FeedSide(15,1,true);side.BreakContinuity();FeedSide(15,1,true);FeedSide(.3f,0,true);
            Check(side.QualifiedRounds==1,"Pause joined two 15-second holds");
            var hip=new HipThrustHoldRule();hip.Reset(difficulty);hip.ContinuousHold=true;hip.SetFullScoreSeconds(30);
            for(int i=0;i<20;i++)hip.Step(0,100,.01f,true);
            for(int i=0;i<1500;i++)hip.Step(1,100,.01f,true);
            float held=hip.ValidHoldSeconds;hip.Step(1,100,.12f,true);
            Check(Math.Abs(held-hip.ValidHoldSeconds)<.001&&hip.LiveScore==0,"Hip render stall counted time or reset the attempt");
            for(int i=0;i<30;i++)hip.Step(1,100,.01f,false);
            Check(hip.ValidHoldSeconds==0,"Hip extended missing input preserved continuity");
            void FeedSide(float seconds,float progress,bool valid)
            {for(int i=0;i<Mathf.RoundToInt(seconds*100);i++)side.Step(progress,100,.01f,valid);}
        }
        Check(!TrainingScoringTolerance.HoldQualified(28.9,30,true),"Long shortfall passed");
        Check(!TrainingScoringTolerance.HoldQualified(15,30,true),"Half dose passed");
        Check(!TrainingScoringTolerance.HoldQualified(double.NaN,30,true),"NaN passed");
        Check(!TrainingScoringTolerance.GapExpired(.2f),"Float rounding rejected exactly 200ms");
        var mixed=new TrainingBridgeAttempt();mixed.Step(.1,0,true,true,false,false);mixed.Step(10,10,true,false,true,false);
        mixed.Step(.1,.1,false,false,false,false);mixed.Step(.15,.15,true,false,false,false);mixed.Step(10,10,true,false,true,false);
        Check(mixed.LongestHold<10.01,"Mixed missing/bad samples reset the gap budget and joined separate holds");
        Directory.CreateDirectory("output/training-assessment");
        File.WriteAllText("output/training-assessment/tolerance-unit.txt",$"PASS {checks} checks: short/long/mixed timing, actual evidence duration, qualification versus full marks, render stalls, pause, all difficulties and 30/60/120 fps.\n");
    }
    static void Check(bool value,string message){checks++;if(!value)throw new Exception(message);}
}
