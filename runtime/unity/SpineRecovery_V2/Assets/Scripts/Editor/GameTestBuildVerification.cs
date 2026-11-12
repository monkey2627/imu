using System;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

[InitializeOnLoad]
public static class GameTestBuildVerification
{
    const string Flag="GameTestBuildVerification", Output="output/android-test/";
    static int index, stage, checks;
    static double at;
    static double pausedSongTime;
    static GameWorkoutScene entry;
    static GameTestMenu menu;
    static UdpClient udp;
    static Quaternion[] expected;
    static Vector3[] positions;
    static readonly HumanBodyBones[] Bones={HumanBodyBones.Hips,HumanBodyBones.LeftUpperArm,HumanBodyBones.RightUpperArm,HumanBodyBones.LeftHand,HumanBodyBones.RightHand,HumanBodyBones.LeftUpperLeg,HumanBodyBones.RightUpperLeg,HumanBodyBones.LeftFoot,HumanBodyBones.RightFoot};
    static GameTestBuildVerification(){if(SessionState.GetBool(Flag,false))EditorApplication.update+=Tick;}
    public static void Run()
    {
        Directory.CreateDirectory(Output);File.WriteAllText(Output+"navigation-vmc.txt","Real six-scene navigation and OSC UDP receiver test. Synthetic poses, no physical trackers.\n");
        EditorSceneManager.OpenScene("Assets/Scenes/"+GameTestMenu.Scenes[0]+".unity");
        SessionState.SetBool(Flag,true);EditorApplication.update-=Tick;EditorApplication.update+=Tick;EditorApplication.isPlaying=true;
    }
    static void Check(bool value,string message){checks++;if(!value)throw new Exception(message);}
    static void Log(string message)=>File.AppendAllText(Output+"navigation-vmc.txt",message+"\n");
    static void Tick()
    {
        if(!EditorApplication.isPlaying||EditorApplication.isCompiling)return;
        try {
            if(stage==0) {
                entry=UnityEngine.Object.FindObjectOfType<GameWorkoutScene>();
                if(entry==null||entry.gameObject.scene.name!=GameTestMenu.Scenes[index])return;
                menu=entry.GetComponent<GameTestMenu>();if(menu==null||menu.Panel==null)return;
                Check(entry.menu.IsTrainingMenuVisible,"New scene did not return to start menu");
                Check(!entry.scorer.GetComponent<SupineGameCalibration>().skipCalibrationForPreview,"Hardware test silently bypassed calibration");
                Check(!GameWorkoutSession.Instance.IsRunning,"Old prescription survived manual scene switch");
                Check(!entry.GetComponent<GameTrainingAssessment>().Summary.HasAssessment,"Old scores survived switch");
                Check(UnityEngine.Object.FindObjectsOfType<GameTestMenu>().Length==1,"Duplicate navigation");
                Check(menu.GameButtons.Length==6&&menu.GameButtons.All(b=>b.interactable),"Missing action buttons");
                menu.Open();at=EditorApplication.timeSinceStartup;stage=1;
            } else if(stage==1&&EditorApplication.timeSinceStartup-at>.2) {
                Check(menu.Panel.activeInHierarchy,"Start-menu action selection inaccessible");
                menu.Close();entry.menu.SelectSimpleDifficulty();
                // Preview bypass is confined to this editor UI test. Real APK requires calibration.
                entry.scorer.GetComponent<SupineGameCalibration>().skipCalibrationForPreview=true;
                entry.menu.StartTraining();at=EditorApplication.timeSinceStartup;stage=2;
            } else if(stage==2) {
                Check(EditorApplication.timeSinceStartup-at<30,"Countdown never finished");
                if(!(entry.taiko!=null?entry.taiko.IsRunning:entry.coach.IsPlayingAction))return;
                menu.Open();pausedSongTime=entry.rhythm.SongTime;at=EditorApplication.timeSinceStartup;stage=3;
            } else if(stage==3&&EditorApplication.timeSinceStartup-at>.3) {
                Check(menu.IsPaused&&menu.Panel.activeInHierarchy,"Pause did not show six game choices");
                Check(Time.timeScale==0&&AudioListener.pause,"Pause did not freeze the game and audio");
                Check(Math.Abs(entry.rhythm.SongTime-pausedSongTime)<.04,"Pause advanced rhythm notes");
                Capture(GameTestMenu.Scenes[index]+"-pause");
                menu.Close();Check(!menu.IsPaused&&Time.timeScale==1&&!AudioListener.pause,"Resume failed");
                menu.Open();menu.ReturnToMenu();Check(entry.menu.IsTrainingMenuVisible&&!menu.IsPaused,"Return failed");
                // Activate only the user rig; the real receiver must drive its transforms.
                var user=entry.scorer.UserAnimator;user.gameObject.SetActive(true);
                var server=UnityEngine.Object.FindObjectOfType<uOSC.uOscServer>();Check(server!=null&&server.port==39539,"Wrong VMC input port");
                server.port=39549;udp=new UdpClient();
                expected=new Quaternion[Bones.Length];positions=new Vector3[Bones.Length];
                for(int i=0;i<Bones.Length;i++) {var b=user.GetBoneTransform(Bones[i]);positions[i]=b.localPosition;expected[i]=Quaternion.Euler(i*3+12,i*2-9,i*4-16);}
                at=EditorApplication.timeSinceStartup;stage=4;
            } else if(stage==4) {
                SendPose(39549);if(EditorApplication.timeSinceStartup-at<1)return;
                var motion=UnityEngine.Object.FindObjectOfType<AvatarMotionSource>();
                Check(motion!=null&&motion.HasFreshFrame,"UDP bones did not reach scoring telemetry");
                for(int i=0;i<Bones.Length;i++) {
                    var b=entry.scorer.UserAnimator.GetBoneTransform(Bones[i]);
                    Check(Quaternion.Angle(b.localRotation,expected[i])<.15f,"VMC bone rotation mismatch: "+Bones[i]+" actual="+b.localRotation+" expected="+expected[i]);
                    Check(Vector3.Distance(b.localPosition,positions[i])<.0001f,"VMC stretched bone: "+Bones[i]);
                }
                udp.Dispose();udp=null;at=EditorApplication.timeSinceStartup;stage=5;
            } else if(stage==5&&EditorApplication.timeSinceStartup-at>1) {
                Check(!UnityEngine.Object.FindObjectOfType<AvatarMotionSource>().HasFreshFrame,"Disconnected input remained fresh");
                Log("PASS "+GameTestMenu.Scenes[index]+": selection, calibration gate, countdown, pause/resume, clean menu; real UDP drives torso/left-right hands and feet, preserves bone lengths, expires on disconnect.");
                index++;
                if(index==GameTestMenu.Scenes.Length){Log("Assertions: "+checks);Finish(0);return;}
                menu.Open();menu.GameButtons[index].onClick.Invoke();stage=0;
            }
        } catch(Exception ex){Log("FAIL "+ex);Debug.LogException(ex);Finish(1);}
    }
    static void SendPose(int port)
    {
        Send(port,"/VMC/Ext/OK",1);Send(port,"/VMC/Ext/T",Time.realtimeSinceStartup);
        for(int i=0;i<Bones.Length;i++){var q=expected[i];Send(port,"/VMC/Ext/Bone/Pos",Bones[i].ToString(),0f,0f,0f,q.x,q.y,q.z,q.w);}
    }
    static void Send(int port,string address,params object[] args)
    {
        using(var stream=new MemoryStream()) {
            WriteString(stream,address);WriteString(stream,","+string.Concat(args.Select(a=>a is string?"s":a is int?"i":"f")));
            foreach(var arg in args) {if(arg is string s)WriteString(stream,s);else{byte[] bytes=arg is int n?BitConverter.GetBytes(n):BitConverter.GetBytes((float)arg);if(BitConverter.IsLittleEndian)Array.Reverse(bytes);stream.Write(bytes,0,4);}}
            byte[] data=stream.ToArray();udp.Send(data,data.Length,"127.0.0.1",port);
        }
    }
    static void WriteString(Stream s,string value){var data=Encoding.UTF8.GetBytes(value);s.Write(data,0,data.Length);s.WriteByte(0);while(s.Position%4!=0)s.WriteByte(0);}
    static void Capture(string name)
    {
        var canvases=UnityEngine.Object.FindObjectsOfType<Canvas>().Where(c=>c.isActiveAndEnabled&&c.isRootCanvas&&c.renderMode==RenderMode.ScreenSpaceOverlay).ToArray();
        var host=new GameObject("Test capture");var camera=host.AddComponent<Camera>();camera.transform.position=new Vector3(1000,1000,-1000);camera.orthographic=true;camera.clearFlags=CameraClearFlags.SolidColor;camera.backgroundColor=Color.black;
        var rt=RenderTexture.GetTemporary(1920,1080,24,RenderTextureFormat.ARGB32,RenderTextureReadWrite.sRGB);var previous=RenderTexture.active;
        try {
            camera.targetTexture=rt;foreach(var c in canvases){c.renderMode=RenderMode.ScreenSpaceCamera;c.worldCamera=camera;c.planeDistance=1;}
            Canvas.ForceUpdateCanvases();camera.Render();RenderTexture.active=rt;var image=new Texture2D(rt.width,rt.height,TextureFormat.RGB24,false);
            image.ReadPixels(new Rect(0,0,rt.width,rt.height),0,0);image.Apply();File.WriteAllBytes(Output+name+".png",image.EncodeToPNG());UnityEngine.Object.DestroyImmediate(image);
        } finally {foreach(var c in canvases){c.renderMode=RenderMode.ScreenSpaceOverlay;c.worldCamera=null;}RenderTexture.active=previous;UnityEngine.Object.DestroyImmediate(host);RenderTexture.ReleaseTemporary(rt);}
    }
    static void Finish(int code){udp?.Dispose();SessionState.SetBool(Flag,false);EditorApplication.update-=Tick;EditorApplication.Exit(code);}
}
