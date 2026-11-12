using System;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

[InitializeOnLoad]
public static class TrainingToleranceSceneVerification
{
    const string Flag="TrainingToleranceSceneVerification";
    static readonly string[] AllScenes={"BirdDog_RingSkinC_Recorded","DeadBug_RingSkinC_Recorded","HipThrust_RingBridge_NewScoring","HipThrust_RingBridge_Hold30","SideBridge_RingBridge_Dynamic","SideBridge_RingBridge_Hold30"};
    static readonly string[] ProbeScenes={"DeadBug_RingSkinC_Recorded","HipThrust_RingBridge_NewScoring","SideBridge_RingBridge_Dynamic","SideBridge_RingBridge_Hold30"};
    static readonly string[] HipScenes={"HipThrust_RingBridge_NewScoring","HipThrust_RingBridge_Hold30"};
    static readonly string[] RhythmScenes={"BirdDog_RingSkinC_Recorded","DeadBug_RingSkinC_Recorded"};
    static string[] Scenes=>SessionState.GetBool("ToleranceRhythmOnly",false)?RhythmScenes:SessionState.GetBool("ToleranceHipOnly",false)?HipScenes:SessionState.GetBool("ToleranceFast",false)?ProbeScenes:AllScenes;
    static int sceneIndex,difficultyIndex,stage;
    static GameWorkoutScene entry;
    static readonly BindingFlags Private=BindingFlags.Instance|BindingFlags.NonPublic;
    static TrainingToleranceSceneVerification(){if(SessionState.GetBool(Flag,false))EditorApplication.update+=Tick;}
    public static void Run(){SetMode(false,false,false);Start();}
    public static void RunFast(){SetMode(true,false,false);Start();}
    public static void RunHipOnly(){SetMode(false,true,false);Start();}
    public static void RunRhythmOnly(){SetMode(false,false,true);Start();}
    static void SetMode(bool fast,bool hip,bool rhythm){SessionState.SetBool("ToleranceFast",fast);SessionState.SetBool("ToleranceHipOnly",hip);SessionState.SetBool("ToleranceRhythmOnly",rhythm);}
    static void Start()
    {
        TrainingToleranceScenarios.Begin();
        EditorSceneManager.OpenScene("Assets/Scenes/"+Scenes[0]+".unity");
        SessionState.SetBool(Flag,true);EditorApplication.update-=Tick;EditorApplication.update+=Tick;EditorApplication.isPlaying=true;
    }
    static void Tick()
    {
        if(!EditorApplication.isPlaying||EditorApplication.isCompiling)return;
        try
        {
            if(stage==0)
            {
                entry=UnityEngine.Object.FindObjectOfType<GameWorkoutScene>();
                if(entry==null||entry.gameObject.scene.name!=Scenes[sceneIndex]||entry.GetComponent<TrainingResultView>()?.Root==null)return;
                entry.menu.SelectSimpleDifficulty();
                entry.scorer.GetComponent<SupineGameCalibration>().skipCalibrationForPreview=true;
                ((GameObject)entry.menu.GetType().GetField("menuRoot",Private).GetValue(entry.menu)).SetActive(false);
                if(entry.taiko==null){entry.coach.GetType().GetMethod("StartSegmentedDeadBug",Private).Invoke(entry.coach,null);stage=1;}
                else {if(entry.taiko.PrescriptionDose!=null)entry.taiko.PrescriptionDose.SetDose(1,3,out _);stage=2;}
            }
            else if(stage==1)
            {
                if(!entry.GetComponent<GameTrainingAssessment>().IsCollecting)return;
                entry.coach.StopAllCoroutines();stage=2;
            }
            else
            {
                if(difficultyIndex==0)entry.menu.SelectSimpleDifficulty();
                else if(difficultyIndex==1)entry.menu.SelectStandardDifficulty();
                else entry.menu.SelectHardDifficulty();
                var patient=entry.scorer.UserAnimator;
                if(entry.taiko!=null)TrainingToleranceScenarios.Bridge(entry,patient);
                else TrainingToleranceScenarios.Rhythm(entry,patient);
                entry.scorer.GetType().GetField("userAnimator",Private).SetValue(entry.scorer,patient);
                difficultyIndex++;
                if(difficultyIndex<3)return;
                difficultyIndex=0;stage=0;sceneIndex++;
                if(sceneIndex==Scenes.Length)Finish(0);else SceneManager.LoadScene(Scenes[sceneIndex]);
            }
        }
        catch(Exception ex){File.AppendAllText("output/training-assessment/tolerance-scenarios.txt","FAIL "+ex);Debug.LogException(ex);Finish(1);}
    }
    static void Finish(int code){SessionState.SetBool(Flag,false);EditorApplication.update-=Tick;EditorApplication.Exit(code);}
}
