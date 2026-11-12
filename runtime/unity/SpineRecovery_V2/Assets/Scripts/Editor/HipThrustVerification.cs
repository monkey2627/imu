using System;
using System.Collections;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;

/// <summary>Exercises the real menu and countdown. Timeline probes are Play-mode only.</summary>
[InitializeOnLoad]
public static class HipThrustVerification
{
    const string Flag = "HipThrustVerification.Active";
    const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;
    static int phase;
    static double started, pausedAt;
    static HipThrustTaikoController game;
    static DeadBugStartMenuController menu;
    static string report;
    static SupineGameCalibration testPose;
    static HipThrustVerification() { if (SessionState.GetBool(Flag, false)) EditorApplication.update += Tick; }

    [MenuItem("Tools/Hip Thrust/Verify Game in Play Mode")]
    public static void Run()
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode) throw new Exception("Exit Play first");
        EditorSceneManager.OpenScene("Assets/Scenes/HipThrust.unity");
        SessionState.SetBool(Flag, true);
        phase = 0;
        EditorApplication.update -= Tick;
        EditorApplication.update += Tick;
        EditorApplication.isPlaying = true;
    }
    static object Get(object target, string name) => target.GetType().GetField(name, Private | BindingFlags.Public).GetValue(target);
    static void Call(string name) => typeof(HipThrustTaikoController).GetMethod(name, Private).Invoke(game, null);
    static void Require(bool condition, string message) { if (!condition) throw new Exception(message); }
    static void Seek(double time)
    {
        typeof(HipThrustTaikoController).GetField("elapsed", Private).SetValue(game, time);
        Call("RenderTimeline");
        Canvas.ForceUpdateCanvases();
    }
    static float Height(Animator animator) => animator.GetBoneTransform(HumanBodyBones.Hips).position.y -
        (animator.GetBoneTransform(HumanBodyBones.LeftLowerLeg).position.y + animator.GetBoneTransform(HumanBodyBones.RightLowerLeg).position.y) / 2;
    static void Tick()
    {
        if (!EditorApplication.isPlaying || EditorApplication.isCompiling) return;
        try
        {
            if (phase == 0)
            {
                started = EditorApplication.timeSinceStartup;
                game = UnityEngine.Object.FindObjectOfType<HipThrustTaikoController>(true);
                menu = UnityEngine.Object.FindObjectOfType<DeadBugStartMenuController>(true);
                report = "Actual start button + calibration + countdown.\n";
                var scorer = UnityEngine.Object.FindObjectOfType<DeadBugGamingPoseScorer>(true);
                var calibration = scorer.GetComponent<SupineGameCalibration>();
                if (calibration != null)
                {
                    calibration.skipCalibrationForPreview = false;
                    scorer.BeginCalibration();
                    var bridge = UnityEngine.Object.FindObjectOfType<VmcGameModeSourceBridge>();
                    if (bridge != null) bridge.enabled = false;
                    var receiver = UnityEngine.Object.FindObjectOfType<EVMC4U.ExternalReceiver>();
                    if (receiver != null) receiver.enabled = false;
                    UnityEngine.Object.FindObjectOfType<AvatarMotionSource>().SetAvailable(true);
                    testPose = new GameObject("Editor only hip test pose").AddComponent<SupineGameCalibration>();
                    testPose.referenceClip = calibration.referenceClip;
                    testPose.referenceSeconds = calibration.referenceSeconds;
                    Require(testPose.Prepare(scorer.UserAnimator), "Test supine pose unavailable");
                }
                phase = 1;
            }
            if (testPose != null) testPose.SampleReference();
            if (EditorApplication.timeSinceStartup - started > 80) throw new Exception("Start/calibration timed out, phase " + phase);
            if (phase == 1)
            {
                var scorer = UnityEngine.Object.FindObjectOfType<DeadBugGamingPoseScorer>(true);
                if (!scorer.IsCalibrated) return;
                Require(menu.gameObject.activeInHierarchy, "Start menu not visible");
                menu.SelectSimpleDifficulty();
                var button = (Button)new SerializedObject(menu).FindProperty("startButton").objectReferenceValue;
                Require(button.interactable, "Calibrated start button disabled");
                button.onClick.Invoke();
                phase = 2;
            }
            else if (phase == 2 && game.IsRunning && game.Elapsed > 7.4)
            {
                Require(game.BeatCount == 4, "Simple mode must contain four repetitions");
                game.TogglePause();
                pausedAt = game.Elapsed;
                ScreenCapture.CaptureScreenshot("output/hipthrust/game-notes.png");
                started = EditorApplication.timeSinceStartup;
                phase = 3;
            }
            else if (phase == 3 && EditorApplication.timeSinceStartup - started > 2)
            {
                Require(game.Elapsed == pausedAt, "Pause moved the timeline");
                IList chart = (IList)Get(game, "beats");
                object first = chart[0];
                double redTime = (double)Get(first, "upTime");
                Seek(redTime + 0.16);
                var tail = (Image)Get(first, "hold");
                var red = (Image)Get(first, "up");
                var blue = (Image)Get(first, "down");
                var judge = ((Image)Get(game, "judgementImage")).rectTransform;
                Require(red.gameObject.activeSelf && Vector3.Distance(red.transform.position, judge.position) < 0.01f,
                    "Red must remain visible at judgment after arrival");
                Vector3 tailLeft = tail.rectTransform.TransformPoint(new Vector3(tail.rectTransform.rect.xMin, 0f));
                Vector3 judgeRight = judge.TransformPoint(new Vector3(judge.rect.xMax, 0f));
                Require(Vector3.Distance(tailLeft, judgeRight) < 0.01f, "Ribbon overlaps judgment ring");
                Vector3 tailRight = tail.rectTransform.TransformPoint(new Vector3(tail.rectTransform.rect.xMax, 0f));
                Require(Vector3.Distance(tailRight, blue.transform.position) < 0.01f, "Ribbon disconnected from blue note");
                // Simulate a successful pose match in the editor only, through the real hit-resolution path.
                first.GetType().GetField("upHit").SetValue(first, true);
                Call("UpdateJudgements");
                var particles = (HipThrustHitParticles)Get(game, "hitParticles");
                Require(game.HitCount == 1 && particles.ParticleCount >= 40 && particles.RingVisible, "HIT did not emit ring and tail particles");
                Canvas.ForceUpdateCanvases();
                var mesh = particles.canvasRenderer.GetMesh();
                File.WriteAllText("output/hipthrust/particle-render.txt", $"active={particles.gameObject.activeInHierarchy} enabled={particles.enabled} cull={particles.canvasRenderer.cull} rect={particles.rectTransform.rect} color={particles.color} vertices={mesh.vertexCount}");
                Require(!particles.canvasRenderer.cull && mesh.vertexCount > 100, "Particle mesh was not rendered");
                var framing = UnityEngine.Object.FindObjectOfType<HipThrustPreviewFraming>();
                Require(framing != null, "Preview framing missing");
                typeof(HipThrustPreviewFraming).GetMethod("LateUpdate", Private).Invoke(framing, null);
                Camera camera = framing.previewCamera;
                camera.Render();
                var previous = RenderTexture.active;
                RenderTexture.active = camera.targetTexture;
                var pixels = new Texture2D(camera.targetTexture.width, camera.targetTexture.height, TextureFormat.RGBA32, false);
                pixels.ReadPixels(new Rect(0, 0, pixels.width, pixels.height), 0, 0);
                pixels.Apply();
                Require(pixels.GetPixel(0, 0).a < 0.01f && pixels.GetPixel(pixels.width - 1, pixels.height - 1).a < 0.01f, "Preview background is not transparent");
                File.WriteAllBytes("output/hipthrust/transparent-coach.png", pixels.EncodeToPNG());
                UnityEngine.Object.DestroyImmediate(pixels);
                RenderTexture.active = previous;
                ScreenCapture.CaptureScreenshot("output/hipthrust/game-hit-particles.png");
                report += "PASS: red latches over judge; ribbon stays outside ring and connects blue; simulated HIT emits unchanged ring and tail particles; transparent preview.\n";
                started = EditorApplication.timeSinceStartup;
                phase = 6;
            }
            else if (phase == 6 && EditorApplication.timeSinceStartup - started > 2)
            {
                IList beats = (IList)Get(game, "beats");
                var judge = ((Image)Get(game, "judgementImage")).rectTransform;
                for (int i = 0; i < beats.Count; i++)
                {
                    object beat = beats[i];
                    double up = (double)Get(beat, "upTime"), down = (double)Get(beat, "downTime");
                    Seek(up);
                    var red = (Image)Get(beat, "up");
                    Require(Vector3.Distance(red.transform.position, judge.position) < 0.01f, "Red does not meet judgment center");
                    float high = Height(game.CoachAnimator);
                    float norm = game.CoachAnimator.GetCurrentAnimatorStateInfo(0).normalizedTime;
                    Require(Math.Abs(norm * 25.5 - game.SourceTime) < 0.002, "Coach time drift");
                    Require(Math.Abs(norm - game.PreviewAnimator.GetCurrentAnimatorStateInfo(0).normalizedTime) < 0.0001, "Preview time drift");
                    Seek((up + down) * 0.5);
                    var tail = (Image)Get(beat, "hold");
                    float middleWidth = tail.rectTransform.rect.width;
                    Require(red.gameObject.activeSelf && Vector3.Distance(red.transform.position, judge.position) < 0.01f, "Red did not stay on judge");
                    Seek(down - 0.001);
                    Require(red.gameObject.activeSelf && ((Image)Get(beat, "down")).gameObject.activeSelf, "Pair disappeared before blue arrival");
                    Require(tail.rectTransform.rect.width <= middleWidth && !tail.gameObject.activeSelf, "Ribbon must shrink away before overlapping ring");
                    Seek(down);
                    var blue = (Image)Get(beat, "down");
                    Require(Vector3.Distance(blue.transform.position, judge.position) < 0.01f, "Blue does not meet judgment center");
                    Require(!red.gameObject.activeSelf && !blue.gameObject.activeSelf && !tail.gameObject.activeSelf, "Pair must disappear together at blue arrival");
                    float low = Height(game.CoachAnimator);
                    Require(high - low > 0.06f, "Red/blue coach hip heights not distinct");
                    report += $"Pair {i + 1}: red={up:F4}s blue={down:F4}s hip difference={high-low:F4}m; alignment PASS.\n";
                }
                Seek(game.Duration);
                game.TogglePause();
                Call("LateUpdate");
                Require(!game.IsRunning && ((GameObject)Get(game, "endMenuRoot")).activeSelf, "Completion panel missing");
                ScreenCapture.CaptureScreenshot("output/hipthrust/game-complete.png");
                started = EditorApplication.timeSinceStartup;
                phase = 4;
            }
            else if (phase == 4 && EditorApplication.timeSinceStartup - started > 2)
            {
                game.StopAndShowMenu();
                Require(menu.gameObject.activeInHierarchy, "Return to menu failed");
                menu.SelectHardDifficulty();
                ((Button)new SerializedObject(menu).FindProperty("startButton").objectReferenceValue).onClick.Invoke();
                phase = 5;
            }
            else if (phase == 5 && game.IsRunning)
            {
                Require(game.BeatCount == 12, "Hard mode must have 12 repetitions");
                Require(Math.Abs(game.SourceTimeAt(7.983333333 + 15.2) - 7.983333333) < 0.001, "Repeated coach time mismatch");
                Require(Math.Abs(game.Duration - 55.9) < 0.001, "Repeated routine duration mismatch");
                report += "PASS: pause; completion; return to menu; restart; 12-repetition chart and repeated animation clock.\n";
                File.WriteAllText("output/hipthrust/verification.txt", report);
                Finish();
            }
        }
        catch (Exception ex)
        {
            File.WriteAllText("output/hipthrust/verification.txt", "FAIL phase " + phase + ": " + ex);
            Debug.LogException(ex);
            Finish();
        }
    }
    static void Finish()
    {
        SessionState.SetBool(Flag, false);
        EditorApplication.update -= Tick;
        EditorApplication.isPlaying = false;
    }
}
