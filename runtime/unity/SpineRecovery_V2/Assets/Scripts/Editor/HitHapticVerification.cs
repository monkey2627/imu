using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

// Receives the actual motor HTTP requests on loopback; never contacts a wearable.
[InitializeOnLoad]
public static class HitHapticVerification
{
    const string Flag="HitHapticVerification.Active", Output="output/two-phase-haptics/haptics.txt";
    const BindingFlags Private=BindingFlags.Instance|BindingFlags.NonPublic;
    static int sceneIndex,caseIndex,step;
    static double sentAt;
    static GameWorkoutScene entry;
    static RhythmHitHapticAdapter adapter;
    static CoachHapticFeedbackController haptics;
    static MotorReceiver receiver;
    static readonly List<Case> cases=new List<Case>();
    static readonly List<string> requests=new List<string>();
    sealed class Case { public string name,expected; public Action run; }
    static HitHapticVerification() { if(SessionState.GetBool(Flag,false)) EditorApplication.update+=Tick; }
    [MenuItem("Tools/Ring Skin C/Verify HIT-only Tracker Requests")]
    public static void Run()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Output));
        File.WriteAllText(Output,"Loopback motor HTTP verification. No physical trackers used.\n");
        EditorSceneManager.OpenScene(RecordedCoachSceneTools.Recorded("BirdDog"));
        SessionState.SetBool(Flag,true); EditorApplication.update-=Tick; EditorApplication.update+=Tick;
        EditorApplication.isPlaying=true;
    }
    static void Require(bool condition,string message) { if(!condition) throw new Exception(message); }
    static void Log(string text)=>File.AppendAllText(Output,text+"\n");
    static void Tick()
    {
        if(!EditorApplication.isPlaying || EditorApplication.isCompiling) return;
        try
        {
            if(step==0)
            {
                entry=UnityEngine.Object.FindObjectOfType<GameWorkoutScene>();
                if(entry==null || entry.actionId!=new[]{"birddog","deadbug","hipthrust"}[sceneIndex]) return;
                var adapters=UnityEngine.Object.FindObjectsOfType<RhythmHitHapticAdapter>();
                Require(adapters.Length==1 && adapters[0].isActiveAndEnabled,"Expected exactly one active haptic adapter");
                adapter=adapters[0];
                haptics=(CoachHapticFeedbackController)typeof(RhythmHitHapticAdapter).GetField("hapticController",Private).GetValue(adapter);
                Require(haptics!=null && haptics.isActiveAndEnabled,"Inactive haptic transport");
                haptics.StopAllCoroutines(); haptics.StopAllHaptics();
                typeof(CoachHapticFeedbackController).GetMethod("ClearAutomaticallyDiscoveredEndpoints",Private).Invoke(haptics,null);
                receiver=new MotorReceiver();
                foreach(TrackerWearLocation location in Enum.GetValues(typeof(TrackerWearLocation)))
                    haptics.ConfigureEndpoint(location,"127.0.0.1:"+receiver.Port+"/"+location,100);
                haptics.SetEnabled(true);
                VerifyWaistBindings(); BuildCases(); caseIndex=0; step=1;
                return;
            }
            if(step==1)
            {
                requests.Clear(); while(receiver.Requests.TryDequeue(out _)) {}
                cases[caseIndex].run(); sentAt=EditorApplication.timeSinceStartup; step=2; return;
            }
            while(receiver.Requests.TryDequeue(out string request)) requests.Add(request);
            if(EditorApplication.timeSinceStartup-sentAt<.18) return;
            var test=cases[caseIndex];
            if(test.expected!=null && requests.Count==0 && EditorApplication.timeSinceStartup-sentAt<2) return;
            Require(test.expected==null ? requests.Count==0 : requests.Count==1 && requests[0]==test.expected,
                entry.actionId+" "+test.name+": expected "+(test.expected??"silence")+", got "+string.Join(", ",requests));
            Log("PASS "+entry.actionId+" "+test.name+" -> "+(test.expected??"no motor request"));
            caseIndex++;
            if(caseIndex<cases.Count) { step=1; return; }
            receiver.Dispose(); receiver=null;
            if(++sceneIndex==3) { Finish(0); return; }
            step=0;
            SceneManager.LoadScene(sceneIndex==1?"DeadBug_RingSkinC_Recorded":"HipThrust_RingBridge_Recorded");
        }
        catch(Exception e) { Log("FAIL "+e); Debug.LogException(e); Finish(1); }
    }
    static void VerifyWaistBindings()
    {
        foreach(string name in new[]{"body:waist","waist","body:hip","hip"})
        {
            var args=new object[]{name,default(TrackerWearLocation)};
            bool mapped=(bool)typeof(CoachHapticFeedbackController).GetMethod("TryMapBodyPosition",BindingFlags.Static|BindingFlags.NonPublic).Invoke(null,args);
            Require(mapped && (TrackerWearLocation)args[1]==TrackerWearLocation.Abdomen,"Waist binding misrouted");
        }
    }
    static void BuildCases()
    {
        cases.Clear();
        if(entry.taiko!=null)
        {
            var resolve=typeof(HipThrustTaikoController).GetMethod("ResolveHit",Private);
            cases.Add(new Case{name="up HIT",expected=Motor(TrackerWearLocation.Abdomen),run=()=>resolve.Invoke(entry.taiko,new object[]{true,1000,null})});
            cases.Add(new Case{name="down HIT",expected=Motor(TrackerWearLocation.Abdomen),run=()=>resolve.Invoke(entry.taiko,new object[]{true,1000,null})});
            cases.Add(new Case{name="MISS",run=()=>resolve.Invoke(entry.taiko,new object[]{false,1000,null})});
            foreach(RhythmBodyTarget target in Enum.GetValues(typeof(RhythmBodyTarget)))
            {
                var local=target;
                cases.Add(new Case{name="legacy "+target+" must be silent",run=()=>Resolve(local,RhythmJudgement.Perfect)});
            }
            cases.Add(new Case{name="re-enable without duplicate subscription",expected=Motor(TrackerWearLocation.Abdomen),run=()=>{
                adapter.enabled=false; adapter.enabled=true; resolve.Invoke(entry.taiko,new object[]{true,1000,null}); }});
        }
        else
        {
            var locations=new[]{TrackerWearLocation.LeftForearm,TrackerWearLocation.RightForearm,TrackerWearLocation.LeftLowerLeg,TrackerWearLocation.RightLowerLeg,TrackerWearLocation.Abdomen};
            foreach(RhythmBodyTarget target in Enum.GetValues(typeof(RhythmBodyTarget)))
                foreach(RhythmJudgement judgement in new[]{RhythmJudgement.Perfect,RhythmJudgement.Great,RhythmJudgement.Good,RhythmJudgement.Miss,RhythmJudgement.None})
                {
                    var localTarget=target; var localJudgement=judgement;
                    cases.Add(new Case{name=target+" "+judgement,expected=(judgement==RhythmJudgement.Miss||judgement==RhythmJudgement.None)?null:Motor(locations[(int)target]),run=()=>Resolve(localTarget,localJudgement)});
                }
            cases.Add(new Case{name="re-enable without duplicate subscription",expected=Motor(TrackerWearLocation.RightForearm),run=()=>{
                adapter.enabled=false; adapter.enabled=true; Resolve(RhythmBodyTarget.RightHand,RhythmJudgement.Perfect); }});
        }
        cases.Add(new Case{name="idle is silent",run=()=>{}});
    }
    static string Motor(TrackerWearLocation location)=>"/"+location+"/motor?duration_ms=100";
    static void Resolve(RhythmBodyTarget target,RhythmJudgement judgement)=>entry.rhythm.ResolveNote(null,new RhythmNoteResult(target,judgement,1,0,Vector3.zero));
    static void Finish(int code)
    {
        receiver?.Dispose(); receiver=null; SessionState.SetBool(Flag,false); EditorApplication.update-=Tick;
        if(Application.isBatchMode) EditorApplication.Exit(code); else EditorApplication.isPlaying=false;
    }
    sealed class MotorReceiver:IDisposable
    {
        readonly TcpListener listener=new TcpListener(IPAddress.Loopback,0);
        readonly Thread worker;
        volatile bool disposed;
        public readonly ConcurrentQueue<string> Requests=new ConcurrentQueue<string>();
        public int Port=>((IPEndPoint)listener.LocalEndpoint).Port;
        public MotorReceiver() { listener.Start(); worker=new Thread(Read){IsBackground=true}; worker.Start(); }
        void Read()
        {
            while(!disposed)
            {
                try
                {
                    using(var client=listener.AcceptTcpClient())
                    using(var stream=client.GetStream())
                    {
                        client.ReceiveTimeout=1000;
                        var reader=new StreamReader(stream,Encoding.ASCII,false,1024,true);
                        string line=reader.ReadLine();
                        if(line!=null) Requests.Enqueue(line.Split(' ')[1]);
                        while(!string.IsNullOrEmpty(reader.ReadLine())) {}
                        byte[] response=Encoding.ASCII.GetBytes("HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK");
                        stream.Write(response,0,response.Length);
                    }
                }
                catch(Exception) { if(disposed) return; }
            }
        }
        public void Dispose() { disposed=true; listener.Stop(); worker.Join(1500); }
    }
}
