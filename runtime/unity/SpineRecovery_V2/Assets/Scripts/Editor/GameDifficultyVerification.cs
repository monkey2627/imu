using System;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

[InitializeOnLoad]
public static class GameDifficultyVerification
{
    const string Flag="GameDifficultyVerification",Output="output/scoring-tolerance/difficulty-dose.txt";
    const BindingFlags Private=BindingFlags.Instance|BindingFlags.NonPublic;
    static int index;
    static GameDifficultyVerification(){if(SessionState.GetBool(Flag,false))EditorApplication.update+=Tick;}
    public static void Run()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Output));File.WriteAllText(Output,"Six production games: same standard dose/timing, ordered acceptance thresholds.\n");
        SessionState.SetBool(Flag,true);EditorApplication.update+=Tick;
        EditorSceneManager.OpenScene("Assets/Scenes/"+GameTestMenu.Scenes[0]+".unity");EditorApplication.isPlaying=true;
    }
    static void Tick()
    {
        if(!EditorApplication.isPlaying||EditorApplication.isCompiling)return;
        try
        {
            var entry=UnityEngine.Object.FindObjectOfType<GameWorkoutScene>();
            if(entry==null||entry.gameObject.scene.name!=GameTestMenu.Scenes[index]||entry.GetComponent<TrainingResultView>()?.Root==null)return;
            entry.scorer.GetComponent<SupineGameCalibration>().skipCalibrationForPreview=true;
            string baseline=null;
            foreach(string difficulty in new[]{"standard","simple","hard"})
            {
                entry.PrepareMenu(difficulty);
                string volume=entry.menu.TimesText.text,signature;
                if(entry.taiko==null)
                {
                    var profile=typeof(CoachActionController).GetMethod("GetSelectedDifficultyProfile",Private).Invoke(entry.coach,null);
                    var expected=typeof(CoachActionController).GetField("standardDifficulty",Private).GetValue(entry.coach);
                    if(!ReferenceEquals(profile,expected))throw new Exception("Difficulty changed the standard coach dose");
                    signature=volume;foreach(var f in profile.GetType().GetFields())signature+="|"+f.Name+"="+f.GetValue(profile);
                }
                else
                {
                    entry.taiko.StartMode();signature=volume+"|duration="+entry.taiko.Duration;
                    foreach(var rep in entry.taiko.AssessmentRepetitions())signature+=$"|{rep.start:R},{rep.top:R},{rep.lower:R},{rep.end:R}";
                    entry.taiko.StopAndShowMenu();
                }
                if(baseline==null)baseline=signature;else if(signature!=baseline)throw new Exception("Difficulty changed dose/timing: "+entry.actionId);
                File.AppendAllText(Output,$"PASS {entry.actionId} {difficulty} {signature}\n");
            }
            if(++index==GameTestMenu.Scenes.Length)
            {
                foreach(float progress in new[]{0f,.3f,.5f,.59f,.6f,.67f,.68f,.77f,.78f,.85f,1f})
                foreach(float quality in new[]{0f,20f,34f,35f,44f,45f,59f,60f,80f,100f})
                foreach(var target in new[]{RhythmBodyTarget.LeftHand,RhythmBodyTarget.RightHand,RhythmBodyTarget.LeftFoot,RhythmBodyTarget.RightFoot})
                {
                    bool previous=true;
                    foreach(string level in new[]{"simple","standard","hard"})
                    {
                        bool hit=progress>=GameDifficultyPolicy.MinimumExtension(level,target)&&quality>=GameDifficultyPolicy.HitThreshold(level);
                        if(hit&&!previous)throw new Exception("Non-monotone HIT difficulty");previous=hit;
                    }
                }
                File.AppendAllText(Output,"PASS 440 threshold boundary combinations: harder never accepts an easier-rejected pose.\n");Finish(0);
            }
            else SceneManager.LoadScene(GameTestMenu.Scenes[index]);
        }
        catch(Exception ex){File.AppendAllText(Output,"FAIL "+ex+"\n");Debug.LogException(ex);Finish(1);}
    }
    static void Finish(int code){SessionState.SetBool(Flag,false);EditorApplication.update-=Tick;EditorApplication.Exit(code);}
}
