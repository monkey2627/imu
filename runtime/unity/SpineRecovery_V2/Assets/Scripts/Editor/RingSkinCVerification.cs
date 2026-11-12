using System;
using System.IO;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>Batch-mode visual and real note-lifecycle smoke check. Never saves its play-mode overrides.</summary>
[InitializeOnLoad]
public static class RingSkinCVerification
{
    private const string Flag = "RingSkinCVerification.Active";
    private static int phase;
    private static double phaseStart;
    private static int hits;
    private static int misses;
    private static RhythmModeController controller;
    private static RhythmSkinCVisuals skin;
    private static string Output => SessionState.GetString("RingSkinCVerification.Output", "output/ringskinc");
    private const string LiveFlag = "RingSkinCVerification.Live";

    static RingSkinCVerification()
    {
        if (SessionState.GetBool(Flag, false)) EditorApplication.update += Tick;
    }

    public static void Run()
    {
        RunScene(DeadBugRingSkinCBuilder.Destination, false);
    }

    public static void RunBirdDog() => RunScene(DeadBugRingSkinCBuilder.BirdDogDestination, false);

    [MenuItem("Tools/Ring Skin C/Verify BirdDog in Play Mode")]
    public static void VerifyBirdDogLive() => RunScene(DeadBugRingSkinCBuilder.BirdDogDestination, true);

    [MenuItem("Tools/Ring Skin C/Verify DeadBug in Play Mode")]
    public static void VerifyDeadBugLive() => RunScene(DeadBugRingSkinCBuilder.Destination, true);

    private static void RunScene(string path, bool live)
    {
        Require(!EditorApplication.isPlayingOrWillChangePlaymode, "Exit Play mode before verification");
        for (int i = 0; i < UnityEngine.SceneManagement.SceneManager.sceneCount; i++)
            Require(!UnityEngine.SceneManagement.SceneManager.GetSceneAt(i).isDirty, "Save your scene changes before verification");
        SessionState.SetBool(LiveFlag, live);
        SessionState.SetString("RingSkinCVerification.Output", "output/ringskinc/" + (path.Contains("BirdDog") ? "birddog" : "deadbug"));
        Directory.CreateDirectory(Output);
        EditorSceneManager.OpenScene(path, OpenSceneMode.Single);
        skin = UnityEngine.Object.FindObjectOfType<RhythmSkinCVisuals>(true);
        Require(skin != null, "Skin component missing");
        Require(UnityEngine.Object.FindObjectsOfType<RingSkinC.RingDemoChart>(true).Length == 0, "Demo chart must not run in the game");
        foreach (SpriteRenderer r in skin.artwork.GetComponentsInChildren<SpriteRenderer>(true))
            Require(r.sprite != null, "Missing sprite on " + r.name);
        if (path.Contains("BirdDog"))
        {
            var resolve = typeof(RhythmModeController).GetMethod("TryGetSegmentTargets", BindingFlags.NonPublic | BindingFlags.Instance);
            for (int step = 1; step <= 4; step++)
            {
                object[] args = { "birddog_" + step + "_up", default(RhythmBodyTarget), default(RhythmBodyTarget) };
                Require((bool)resolve.Invoke(skin.controller, args), "BirdDog lift state missing: " + args[0]);
                var targets = new[] { (RhythmBodyTarget)args[1], (RhythmBodyTarget)args[2] };
                Require(step % 2 == 1
                    ? targets.Contains(RhythmBodyTarget.LeftFoot) && targets.Contains(RhythmBodyTarget.RightHand)
                    : targets.Contains(RhythmBodyTarget.RightFoot) && targets.Contains(RhythmBodyTarget.LeftHand), "BirdDog diagonal pair changed");
            }
        }
        Capture("scene-authored.png");
        phase = hits = misses = 0;
        SessionState.SetBool(Flag, true);
        EditorApplication.update -= Tick;
        EditorApplication.update += Tick;
        EditorApplication.isPlaying = true;
    }

    public static void DiagnoseSprites()
    {
        foreach (string path in new[] { "Assets/RingSkinC/Sprites/c_note_melody.png", "Assets/ImageSource/DeadBugWhite.png" })
        {
            AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate);
            Debug.Log("SPRITE_DIAG " + path + " => " + string.Join(", ", AssetDatabase.LoadAllAssetsAtPath(path).Select(x => x.name + ":" + x.GetType().Name)));
            var importer = (TextureImporter)AssetImporter.GetAtPath(path);
            Debug.Log("SPRITE_DIAG_SETTINGS " + importer.textureType + " " + importer.spriteImportMode + " " + importer.spritePixelsPerUnit);
        }
    }

    private static void Tick()
    {
        if (!EditorApplication.isPlaying || EditorApplication.isCompiling) return;
        try
        {
            if (phase == 0)
            {
                phaseStart = EditorApplication.timeSinceStartup;
                phase = 1;
                return;
            }
            double elapsed = EditorApplication.timeSinceStartup - phaseStart;
            if (phase == 1 && elapsed > 2)
            {
                skin = UnityEngine.Object.FindObjectOfType<RhythmSkinCVisuals>(true);
                controller = skin.controller;
                Require(!skin.presentationCamera.enabled, "Skin must be hidden on the initial menu");
                VerifyBackgroundCoverage();
                // Simulate the existing session-start boundary, without a tracker, for visual verification.
                foreach (Canvas canvas in UnityEngine.Object.FindObjectsOfType<Canvas>(true))
                {
                    if (canvas.name == "StartMenuCanvas") canvas.gameObject.SetActive(false);
                    if (canvas.name == "GamingCanvas") canvas.gameObject.SetActive(true);
                }
                controller.StartRhythmMode();
                controller.SetTutorialJudgementOverride(RhythmJudgement.Perfect);
                controller.NoteResolved += result => { if (result.IsHit) hits++; else misses++; };
                for (int lane = 0; lane < 4; lane++) Spawn((RhythmBodyTarget)lane, 2.2f);
                Require(UnityEngine.Object.FindObjectsOfType<RhythmNoteView>().Length == 4, "Expected four original note views");
                phase = 2;
                phaseStart = EditorApplication.timeSinceStartup;
            }
            else if (phase == 2 && elapsed > 1.8)
            {
                Capture("game-notes.png");
                CaptureInterface();
                phase = 3;
            }
            else if (phase == 3 && elapsed > 2.4)
            {
                Capture("game-hits.png");
                Require(hits == 4, "Original resolution event did not report four hits: " + hits);
                controller.SetTutorialJudgementOverride(RhythmJudgement.Miss);
                Spawn(RhythmBodyTarget.LeftHand, 0.5f);
                phase = 4;
                phaseStart = EditorApplication.timeSinceStartup;
            }
            else if (phase == 4 && elapsed > 1.85)
            {
                Capture("game-miss.png");
                Require(misses == 1, "Expected miss after the original entry/hold window");
                controller.StopRhythmMode();
                phase = 5;
                phaseStart = EditorApplication.timeSinceStartup;
            }
            else if (phase == 5 && elapsed > 0.5)
            {
                Require(UnityEngine.Object.FindObjectsOfType<RhythmNoteView>().Length == 0, "Notes leaked after stop");
                Require(!skin.presentationCamera.enabled, "Skin camera still visible after stop");
                File.WriteAllText(Output + "/verification.txt", "PASS: authored sprites; no demo chart; menu hides skin; four gameplay notes; four HIT events; original MISS window; stop clears notes and hides skin.\n");
                Finish(0);
            }
            if (elapsed > 45) throw new Exception("Verification timed out");
        }
        catch (Exception ex)
        {
            Debug.LogException(ex);
            File.WriteAllText(Output + "/verification.txt", "FAIL: " + ex);
            Finish(1);
        }
    }

    private static void Spawn(RhythmBodyTarget target, float delay)
    {
        typeof(RhythmModeController).GetMethod("SpawnNote", BindingFlags.NonPublic | BindingFlags.Instance)
            .Invoke(controller, new object[] { target, controller.SongTime + delay, delay });
    }

    private static void VerifyBackgroundCoverage()
    {
        Camera camera = skin.presentationCamera;
        Transform ring = skin.artwork.transform.Find("Ring");
        Vector3 ringScale = ring.localScale;
        float aspect = camera.aspect, size = camera.orthographicSize;
        foreach (float ratio in new[] { 844f / 390f, 16f / 9f, 4f / 3f, 9f / 16f, 32f / 9f })
        {
            camera.aspect = ratio;
            camera.orthographicSize = skin.layout.transform.lossyScale.x * skin.designScale * Mathf.Max(1.95f, 4.22f / ratio) / skin.viewScale;
            skin.FitBackgroundToView();
            Bounds bounds = skin.artwork.transform.Find("Background/Sky").GetComponent<SpriteRenderer>().bounds;
            Vector3 center = camera.transform.position;
            float height = camera.orthographicSize, width = height * ratio;
            Require(bounds.min.x <= center.x - width + 0.001f && bounds.max.x >= center.x + width - 0.001f &&
                bounds.min.y <= center.y - height + 0.001f && bounds.max.y >= center.y + height - 0.001f, "Background does not cover aspect " + ratio);
            Require(ring.localScale == ringScale, "Background fitting must not resize the ring");
        }
        camera.aspect = aspect;
        camera.orthographicSize = size;
        skin.FitBackgroundToView();
    }

    private static void Capture(string filename)
    {
        Camera camera = skin.presentationCamera;
        RenderTexture rt = RenderTexture.GetTemporary(1688, 780, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
        RenderTexture old = RenderTexture.active;
        RenderTexture oldTarget = camera.targetTexture;
        camera.targetTexture = rt;
        camera.aspect = 844f / 390f;
        camera.orthographicSize = skin.layout.transform.lossyScale.x * skin.designScale * 1.95f / Mathf.Clamp(skin.viewScale, 0.5f, 1f);
        skin.FitBackgroundToView();
        camera.Render();
        RenderTexture.active = rt;
        Texture2D image = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
        image.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        image.Apply();
        File.WriteAllBytes(Output + "/" + filename, image.EncodeToPNG());
        UnityEngine.Object.DestroyImmediate(image);
        RenderTexture.active = old;
        camera.targetTexture = oldTarget;
        RenderTexture.ReleaseTemporary(rt);
    }

    private static void CaptureInterface()
    {
        var display = GameObject.Find("RingSkinC_Display").GetComponentInChildren<UnityEngine.UI.RawImage>();
        Texture oldDisplay = display.texture;
        RenderTexture oldSkinTarget = skin.presentationCamera.targetTexture;
        RenderTexture artwork = RenderTexture.GetTemporary(1688, 780, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
        skin.presentationCamera.targetTexture = artwork;
        skin.presentationCamera.aspect = 844f / 390f;
        skin.presentationCamera.orthographicSize = skin.layout.transform.lossyScale.x * skin.designScale * 1.95f / Mathf.Clamp(skin.viewScale, 0.5f, 1f);
        skin.FitBackgroundToView();
        skin.presentationCamera.Render();
        display.texture = artwork;
        var canvases = UnityEngine.Object.FindObjectsOfType<Canvas>().Where(c => c.isRootCanvas).ToArray();
        var modes = canvases.Select(c => c.renderMode).ToArray();
        var cameras = canvases.Select(c => c.worldCamera).ToArray();
        var distances = canvases.Select(c => c.planeDistance).ToArray();
        GameObject host = new GameObject("Verification UI Camera");
        Camera camera = host.AddComponent<Camera>();
        camera.orthographic = true;
        camera.cullingMask = ~0;
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.backgroundColor = Color.black;
        camera.transform.position = new Vector3(1000, 1000, -1000);
        RenderTexture rt = RenderTexture.GetTemporary(1688, 780, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        camera.targetTexture = rt;
        for (int i = 0; i < canvases.Length; i++)
        {
            canvases[i].renderMode = RenderMode.ScreenSpaceCamera;
            canvases[i].worldCamera = camera;
            canvases[i].planeDistance = 1f;
        }
        Canvas.ForceUpdateCanvases();
        camera.Render();
        RenderTexture old = RenderTexture.active;
        RenderTexture.active = rt;
        Texture2D image = new Texture2D(1688, 780, TextureFormat.RGB24, false);
        image.ReadPixels(new Rect(0, 0, 1688, 780), 0, 0);
        image.Apply();
        File.WriteAllBytes(Output + "/game-interface.png", image.EncodeToPNG());
        UnityEngine.Object.DestroyImmediate(image);
        RenderTexture.active = old;
        for (int i = 0; i < canvases.Length; i++)
        {
            canvases[i].renderMode = modes[i];
            canvases[i].worldCamera = cameras[i];
            canvases[i].planeDistance = distances[i];
        }
        UnityEngine.Object.DestroyImmediate(host);
        RenderTexture.ReleaseTemporary(rt);
        display.texture = oldDisplay;
        skin.presentationCamera.targetTexture = oldSkinTarget;
        RenderTexture.ReleaseTemporary(artwork);
    }

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }

    private static void Finish(int code)
    {
        SessionState.SetBool(Flag, false);
        EditorApplication.update -= Tick;
        Debug.Log("RING_SKIN_C_VERIFIED exit=" + code);
        if (SessionState.GetBool(LiveFlag, false)) EditorApplication.isPlaying = false;
        else EditorApplication.Exit(code);
    }
}
