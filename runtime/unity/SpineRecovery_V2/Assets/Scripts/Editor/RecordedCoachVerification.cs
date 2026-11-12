using System;
using System.IO;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

/// <summary>Runs the real menu, countdown, animation sequence and note events without a tracker.
/// Calibration bypass and perfect-note overrides exist only in the unsaved Play-mode test.</summary>
[InitializeOnLoad]
public static class RecordedCoachVerification
{
    const string Flag = "RecordedCoachVerification.Active";
    const string StartIndex = "RecordedCoachVerification.StartIndex";
    const string Output = RecordedCoachSceneTools.Output;
    const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;
    static int index = SessionState.GetInt(StartIndex, 0), phase, segments, checkpoints, notes;
    static double entered, pausedAt;
    static bool completed, captured;
    static GameWorkoutScene entry;
    static Animator coach, preview;
    static string action;
    static double pausedTime;
    static float peakHeight, lowHeight;

    static RecordedCoachVerification() { if (SessionState.GetBool(Flag, false)) EditorApplication.update += Tick; }

    [MenuItem("Tools/Recorded Coach/Verify Three Games in Play Mode")]
    public static void Run() => RunAt(0);

    public static void RunHipThrust() => RunAt(2);

    static void RunAt(int first)
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode) throw new Exception("Exit Play first.");
        for (int i=0; i<SceneManager.sceneCount; i++)
            if (SceneManager.GetSceneAt(i).isDirty) throw new Exception("Save your current scene before verification.");
        RecordedCoachOrientationVerification.Run();
        RecordedCoachSceneTools.ValidatePhases();
        Directory.CreateDirectory(Output);
        if (first == 0) File.WriteAllText(Output + "play-verification.txt", "Real menu/countdown/animation and note events. Synthetic calibration bypass; no wearable hardware used.\n");
        index = first; phase = 0;
        SessionState.SetInt(StartIndex, first);
        EditorSceneManager.OpenScene(RecordedCoachSceneTools.Recorded(RecordedCoachSceneTools.Actions[first]));
        SessionState.SetBool(Flag, true);
        EditorApplication.update -= Tick;
        EditorApplication.update += Tick;
        EditorApplication.isPlaying = true;
    }

    static void Require(bool value, string message) { if (!value) throw new Exception(message); }
    static object Get(object o, string field) => o.GetType().GetField(field, Private | BindingFlags.Public).GetValue(o);
    static void Log(string message) { File.AppendAllText(Output + "play-verification.txt", message + "\n"); }
    static void Tick()
    {
        if (!EditorApplication.isPlaying || EditorApplication.isCompiling) return;
        try
        {
            if (phase == 0)
            {
                action = RecordedCoachSceneTools.Actions[index];
                entry = UnityEngine.Object.FindObjectOfType<GameWorkoutScene>();
                if (entry == null || entry.gameObject.scene.name != Path.GetFileNameWithoutExtension(RecordedCoachSceneTools.Recorded(action))) return;
                entered = EditorApplication.timeSinceStartup;
                completed = captured = false; segments = checkpoints = notes = 0;
                coach = (Animator)Get(entry.coach, "coachAnimator");
                preview = (Animator)Get(UnityEngine.Object.FindObjectOfType<ActionPreviewCard>(true), "previewAnimator");
                Require(coach != null && preview != null && coach.isHuman && preview.isHuman, action + ": invalid animator");
                Require(coach.runtimeAnimatorController == preview.runtimeAnimatorController, action + ": coach/preview source mismatch");
                Require(AssetDatabase.GetAssetPath(coach.runtimeAnimatorController).StartsWith(RecordedCoachSceneTools.Root), "Old coach controller remains");
                Require(entry.useRecordedCoachAnimations, "New workout variant flag missing");
                Require(GameWorkoutPlan.SceneFor(entry.actionId, true) == SceneManager.GetActiveScene().name, "Workout route mismatch");
                var calibration = entry.scorer.GetComponent<SupineGameCalibration>();
                calibration.skipCalibrationForPreview = true;
                entry.coach.SegmentPlaybackStarted += _ => segments++;
                entry.coach.ScoringCheckpointStarted += _ => checkpoints++;
                if (entry.taiko != null) entry.taiko.Completed += () => completed = true;
                else
                {
                    entry.coach.ActionCompleted += () => completed = true;
                    entry.rhythm.NoteResolved += _ => notes++;
                    entry.rhythm.SetTutorialJudgementOverride(RhythmJudgement.Perfect);
                }
                entry.menu.SelectSimpleDifficulty();
                Require(entry.scorer.CanStartTraining, "Test calibration bypass did not take effect");
                entry.menu.StartTraining();
                phase = 1;
            }
            if (EditorApplication.timeSinceStartup - entered > 110) throw new Exception(action + ": runtime timeout phase=" + phase);
            if (phase == 1)
            {
                if (entry.taiko != null)
                {
                    if (!entry.taiko.IsRunning || entry.taiko.Elapsed < 4) return;
                    Require(entry.taiko.BeatCount == 4, "HipThrust chart does not contain four pairs");
                    Require(Math.Abs(entry.taiko.CoachAnimator.GetCurrentAnimatorStateInfo(0).normalizedTime - entry.taiko.PreviewAnimator.GetCurrentAnimatorStateInfo(0).normalizedTime) < .01f, "Hip coach and preview clocks differ");
                    CaptureInterface(action);
                    peakHeight = HipHeight(coach);
                    entry.taiko.TogglePause(); pausedTime = entry.taiko.Elapsed; pausedAt = EditorApplication.timeSinceStartup;
                    phase = 2;
                }
                else
                {
                    var state = coach.GetCurrentAnimatorStateInfo(0);
                    if (!captured && state.IsName(action.ToLowerInvariant() + "_1_up") && state.normalizedTime > .55f)
                    {
                        Require(preview.GetCurrentAnimatorStateInfo(0).fullPathHash == state.fullPathHash, action + ": preview not following coach segment");
                        CaptureInterface(action); captured = true;
                    }
                    if (completed)
                    {
                        Require(captured && segments == 10 && checkpoints == 8, action + ": incomplete phase sequence: " + segments + "/" + checkpoints);
                        Require(notes == 8, action + ": expected 8 limb-note resolutions, got " + notes);
                        Log("PASS " + action + ": start menu, countdown, 10 segments, 8 checkpoints, 8 note resolutions, shared new coach/preview controller, recorded workout route.");
                        Next();
                    }
                }
            }
            else if (phase == 2 && EditorApplication.timeSinceStartup - pausedAt > .5)
            {
                Require(Math.Abs(entry.taiko.Elapsed - pausedTime) < .0001, "Hip timeline advanced while paused");
                entry.taiko.TogglePause(); phase = 3;
            }
            else if (phase == 3)
            {
                if (entry.taiko.Elapsed > 7 && entry.taiko.Elapsed < 7.3) lowHeight = HipHeight(coach);
                if (completed)
                {
                    Require(Math.Abs(peakHeight-lowHeight) > .01, "Hip motion did not change the coach pose");
                    Require(entry.taiko.HitCount + entry.taiko.MissCount > 0, "Hip chart never reached judgement");
                    Log("PASS HipThrust: start menu/countdown, 4 recorded beat pairs, synchronized coach/preview, pause/resume, real judgement and completion.");
                    Next();
                }
            }
        }
        catch (Exception e)
        {
            Log("FAIL " + action + ": " + e);
            Debug.LogException(e);
            Finish(1);
        }
    }

    static float HipHeight(Animator a) => a.GetBoneTransform(HumanBodyBones.Hips).position.y -
        (a.GetBoneTransform(HumanBodyBones.LeftLowerLeg).position.y + a.GetBoneTransform(HumanBodyBones.RightLowerLeg).position.y)*.5f;

    static void Next()
    {
        entry.StopTraining();
        if (entry.taiko == null) Require(!entry.rhythm.IsRunning, "Rhythm session remains active after stop");
        index++;
        if (index == RecordedCoachSceneTools.Actions.Length) { Finish(0); return; }
        phase = 0;
        SceneManager.LoadScene(Path.GetFileNameWithoutExtension(RecordedCoachSceneTools.Recorded(RecordedCoachSceneTools.Actions[index])));
    }

    static void Finish(int code)
    {
        SessionState.SetBool(Flag, false);
        SessionState.SetInt(StartIndex, 0);
        EditorApplication.update -= Tick;
        if (Application.isBatchMode) EditorApplication.Exit(code);
        else EditorApplication.isPlaying = false;
    }

    static void CaptureInterface(string name)
    {
        if (name == "HipThrust")
        {
            // Batchmode does not provide a reliable GameView framebuffer.
            // Read the actual in-game preview target after its framing update.
            var framing = UnityEngine.Object.FindObjectOfType<HipThrustPreviewFraming>();
            typeof(HipThrustPreviewFraming).GetMethod("LateUpdate", Private).Invoke(framing, null);
            var camera = framing.previewCamera;
            camera.Render();
            var previousTarget = RenderTexture.active;
            RenderTexture.active = camera.targetTexture;
            var texture = new Texture2D(camera.targetTexture.width, camera.targetTexture.height, TextureFormat.RGBA32, false);
            try
            {
                texture.ReadPixels(new Rect(0, 0, texture.width, texture.height), 0, 0); texture.Apply();
                File.WriteAllBytes(Output + name + "-game-preview.png", texture.EncodeToPNG());
            }
            finally { RenderTexture.active = previousTarget; UnityEngine.Object.DestroyImmediate(texture); }
            return;
        }
        foreach (var camera in UnityEngine.Object.FindObjectsOfType<Camera>())
            if (camera.enabled && camera.targetTexture != null) camera.Render();
        var canvases = UnityEngine.Object.FindObjectsOfType<Canvas>().Where(c => c.isRootCanvas && c.isActiveAndEnabled).ToArray();
        var modes = canvases.Select(c => c.renderMode).ToArray();
        var cameras = canvases.Select(c => c.worldCamera).ToArray();
        var distances = canvases.Select(c => c.planeDistance).ToArray();
        var host = new GameObject("Recorded verification camera");
        var capture = host.AddComponent<Camera>();
        capture.orthographic = true; capture.clearFlags = CameraClearFlags.SolidColor; capture.backgroundColor = Color.black;
        capture.transform.position = new Vector3(1000,1000,-1000);
        var rt = RenderTexture.GetTemporary(1688,780,24,RenderTextureFormat.ARGB32,RenderTextureReadWrite.sRGB);
        var previous = RenderTexture.active;
        try
        {
            capture.targetTexture = rt;
            for (int i=0;i<canvases.Length;i++) { canvases[i].renderMode = RenderMode.ScreenSpaceCamera; canvases[i].worldCamera = capture; canvases[i].planeDistance = 1; }
            Canvas.ForceUpdateCanvases(); capture.Render(); RenderTexture.active = rt;
            var image = new Texture2D(rt.width,rt.height,TextureFormat.RGB24,false);
            image.ReadPixels(new Rect(0,0,rt.width,rt.height),0,0); image.Apply();
            File.WriteAllBytes(Output + name + "-game.png",image.EncodeToPNG());
            UnityEngine.Object.DestroyImmediate(image);
        }
        finally
        {
            for(int i=0;i<canvases.Length;i++) { canvases[i].renderMode=modes[i]; canvases[i].worldCamera=cameras[i]; canvases[i].planeDistance=distances[i]; }
            RenderTexture.active=previous; UnityEngine.Object.DestroyImmediate(host); RenderTexture.ReleaseTemporary(rt);
        }
    }
}
