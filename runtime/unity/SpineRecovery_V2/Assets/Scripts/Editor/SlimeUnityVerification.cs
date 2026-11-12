using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>Synthetic VMC input through the installed SlimeVR server, then the real Unity UDP receiver.</summary>
[InitializeOnLoad]
public static class SlimeUnityVerification
{
    const string Flag="SlimeUnityVerification",Report="output/android-test/slime-unity.txt";
    static int index,stage,checks;static double at;static UdpClient udp;
    static GameWorkoutScene entry;static uOSC.uOscServer receiver;
    static readonly Dictionary<string,Quaternion> incoming=new Dictionary<string,Quaternion>();
    static readonly HumanBodyBones[] Observed={HumanBodyBones.Hips,HumanBodyBones.LeftUpperArm,HumanBodyBones.RightUpperArm,HumanBodyBones.LeftHand,HumanBodyBones.RightHand,HumanBodyBones.LeftUpperLeg,HumanBodyBones.RightUpperLeg,HumanBodyBones.LeftFoot,HumanBodyBones.RightFoot};
    static readonly string[] InputBones={"Hips","Spine","Chest","UpperChest","Neck","Head","LeftShoulder","RightShoulder","LeftUpperArm","RightUpperArm","LeftLowerArm","RightLowerArm","LeftHand","RightHand","LeftUpperLeg","RightUpperLeg","LeftLowerLeg","RightLowerLeg","LeftFoot","RightFoot","LeftToes","RightToes"};
    static Quaternion[] initial;
    static SlimeUnityVerification(){if(SessionState.GetBool(Flag,false))EditorApplication.update+=Tick;}
    public static void Run()
    {
        Directory.CreateDirectory("output/android-test");File.WriteAllText(Report,"Synthetic poses -> installed SlimeVR :39540 -> VMC :39539 -> actual Unity user rig. No physical IMU.\n");
        EditorSceneManager.OpenScene("Assets/Scenes/"+GameTestMenu.Scenes[0]+".unity");SessionState.SetBool(Flag,true);EditorApplication.update-=Tick;EditorApplication.update+=Tick;EditorApplication.isPlaying=true;
    }
    static void Check(bool value,string message){checks++;if(!value)throw new Exception(message);}
    static void Log(string message)=>File.AppendAllText(Report,message+"\n");
    static void Tick()
    {
        if(!EditorApplication.isPlaying||EditorApplication.isCompiling)return;
        try {
            if(stage==0) {
                entry=UnityEngine.Object.FindObjectOfType<GameWorkoutScene>();if(entry==null||entry.gameObject.scene.name!=GameTestMenu.Scenes[index]||entry.GetComponent<GameTestMenu>().Panel==null)return;
                entry.scorer.UserAnimator.gameObject.SetActive(true);receiver=UnityEngine.Object.FindObjectOfType<uOSC.uOscServer>();
                Check(receiver!=null&&receiver.port==39539,"Missing real VMC receiver");incoming.Clear();receiver.onDataReceived.AddListener(Observe);
                udp=new UdpClient();at=EditorApplication.timeSinceStartup;stage=1;
            } else if(stage==1||stage==2) {
                SendInput(stage==2);if(EditorApplication.timeSinceStartup-at<1.5)return;
                Check(UnityEngine.Object.FindObjectOfType<AvatarMotionSource>().HasFreshFrame,"SlimeVR did not produce fresh motion");
                var current=Observed.Select(b=>entry.scorer.UserAnimator.GetBoneTransform(b).localRotation).ToArray();
                for(int i=0;i<Observed.Length;i++) {
                    Check(incoming.ContainsKey(Observed[i].ToString()),"SlimeVR omitted "+Observed[i]);
                    Check(Quaternion.Angle(current[i],incoming[Observed[i].ToString()])<1,"Model differs from SlimeVR named-bone output: "+Observed[i]);
                    if(stage==2)Check(Quaternion.Angle(initial[i],current[i])>3,"SlimeVR did not move "+Observed[i]);
                }
                if(stage==1){initial=current;stage=2;at=EditorApplication.timeSinceStartup;return;}
                Check(!entry.scorer.GetComponent<SupineGameCalibration>().skipCalibrationForPreview,"Live build bypassed calibration");
                Check(!entry.GetComponent<GameTrainingAssessment>().Summary.HasAssessment,"Idle network test fabricated a result");
                Log("PASS "+GameTestMenu.Scenes[index]+": real SlimeVR drives hips, both arms/hands, both legs/feet; named-bone rotations match, no idle score.");
                receiver.onDataReceived.RemoveListener(Observe);udp.Dispose();udp=null;index++;
                if(index==GameTestMenu.Scenes.Length){Log("Assertions: "+checks);Finish(0);}else{entry.GetComponent<GameTestMenu>().SelectGame(index);stage=0;}
            }
        }catch(Exception ex){Log("FAIL "+ex);Debug.LogException(ex);Finish(1);}
    }
    static void Observe(uOSC.Message message)
    {
        if(message.address=="/VMC/Ext/Bone/Pos"&&message.values.Length>=8)
            incoming[(string)message.values[0]]=new Quaternion((float)message.values[4],(float)message.values[5],(float)message.values[6],(float)message.values[7]);
    }
    static void SendInput(bool moved)
    {
        foreach(string bone in InputBones) {
            Quaternion q=Quaternion.identity;
            if(moved&&Array.Exists(Observed,b=>b.ToString()==bone))q=Quaternion.Euler(bone.StartsWith("Left")?30:-22,12,bone=="Hips"?20:8);
            using(var stream=new MemoryStream()) {
                Str(stream,"/VMC/Ext/Bone/Pos");Str(stream,",sfffffff");Str(stream,bone);
                foreach(float v in new[]{0f,0f,0f,q.x,q.y,q.z,q.w}){var bytes=BitConverter.GetBytes(v);if(BitConverter.IsLittleEndian)Array.Reverse(bytes);stream.Write(bytes,0,4);}
                var data=stream.ToArray();udp.Send(data,data.Length,"127.0.0.1",39540);
            }
        }
    }
    static void Str(Stream stream,string value){var bytes=Encoding.UTF8.GetBytes(value);stream.Write(bytes,0,bytes.Length);stream.WriteByte(0);while(stream.Position%4!=0)stream.WriteByte(0);}
    static void Finish(int code){udp?.Dispose();SessionState.SetBool(Flag,false);EditorApplication.update-=Tick;EditorApplication.Exit(code);}
}
