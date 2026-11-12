using System;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEditor.Animations;
using UnityEditor.SceneManagement;
using UnityEditor.Events;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public static class HipThrustAnimationTools
{
    [MenuItem("Tools/Hip Thrust/Polish Feedback and Transparent Preview")]
    public static void PolishPresentation()
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode) throw new InvalidOperationException("Exit Play mode first.");
        var scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene();
        if (scene.path != "Assets/Scenes/HipThrust.unity") scene = EditorSceneManager.OpenScene("Assets/Scenes/HipThrust.unity");
        var transforms = scene.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<Transform>(true)).ToArray();
        Transform Named(string name) => transforms.Single(t => t.name == name);
        var card = (RectTransform)Named("HipThrustCard");
        var root = (RectTransform)Named("HipThrustCanvasVisuals");
        var camera = Named("HipThrustPreviewCamera").GetComponent<Camera>();
        var animator = Named("HipThrustPreview").GetComponentInChildren<Animator>(true);
        var raw = card.GetComponentInChildren<RawImage>(true);
        card.SetParent(root, false);
        Layout(card, new Vector2(0, 435), new Vector2(420, 180));
        card.GetComponent<Image>().color = Color.clear;
        card.GetComponent<Image>().raycastTarget = false;
        raw.transform.SetParent(card, false);
        foreach (Transform child in card) if (child != raw.transform) child.gameObject.SetActive(false);
        raw.gameObject.SetActive(true);
        Layout(raw.rectTransform, Vector2.zero, new Vector2(420, 180));
        raw.uvRect = new Rect(0, 0, 1, 1);
        raw.color = Color.white;
        raw.raycastTarget = false;
        const string path = "Assets/RenderTextures/RT_HipThrustTaikoPreview.renderTexture";
        var texture = AssetDatabase.LoadAssetAtPath<RenderTexture>(path);
        if (texture == null)
        {
            texture = new RenderTexture(840, 360, 24, RenderTextureFormat.ARGB32) { name = "RT_HipThrustTaikoPreview", antiAliasing = 2 };
            AssetDatabase.CreateAsset(texture, path);
        }
        raw.texture = texture;
        camera.targetTexture = texture;
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.backgroundColor = Color.clear;
        camera.allowHDR = false;
        camera.nearClipPlane = 0.01f;
        camera.farClipPlane = 30f;
        var framing = camera.GetComponent<HipThrustPreviewFraming>() ?? camera.gameObject.AddComponent<HipThrustPreviewFraming>();
        framing.avatar = animator;
        framing.previewCamera = camera;
        var phase = Named("LivePhase").GetComponent<TMP_Text>();
        Layout(phase.rectTransform, new Vector2(0, 350), new Vector2(600, 36));
        phase.fontSize = 24;
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        Debug.Log("HIPTHRUST_PRESENTATION_POLISHED");
    }

    [MenuItem("Tools/Hip Thrust/Repair Game and Bake Actual Rhythm")]
    public static void Repair()
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode) throw new InvalidOperationException("Exit Play mode first.");
        var scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene();
        if (scene.path != "Assets/Scenes/HipThrust.unity")
            scene = EditorSceneManager.OpenScene("Assets/Scenes/HipThrust.unity");
        var roots = scene.GetRootGameObjects();
        T Find<T>() where T : Component => roots.SelectMany(r => r.GetComponentsInChildren<T>(true)).First();
        GameObject Named(string name) => roots.SelectMany(r => r.GetComponentsInChildren<Transform>(true)).Single(t => t.name == name).gameObject;
        var taiko = Find<HipThrustTaikoController>();
        var coach = Find<CoachActionController>();
        var menu = Find<DeadBugStartMenuController>();
        var countdown = Find<DeadBugCountdownController>();
        var scorer = Find<DeadBugGamingPoseScorer>();
        var animator = Named("coach").GetComponent<Animator>();
        var preview = Named("HipThrustPreview").GetComponentInChildren<Animator>(true);
        var card = Find<ActionPreviewCard>();
        AnimationClip clip = AssetDatabase.LoadAllAssetsAtPath("Assets/Movements/hipthrust.fbx")
            .OfType<AnimationClip>().Single(c => c.name == "animation_pose");
        const string controllerPath = "Assets/Controllers/HipThrustTaiko.controller";
        if (!File.Exists(controllerPath)) AssetDatabase.CopyAsset("Assets/Controllers/HTPreview.controller", controllerPath);
        var controller = AssetDatabase.LoadAssetAtPath<AnimatorController>(controllerPath);
        var machine = controller.layers[0].stateMachine;
        var state = machine.states.Select(s => s.state).FirstOrDefault(s => s.name == "hipthrust_routine") ?? machine.AddState("hipthrust_routine");
        state.motion = clip;
        EditorUtility.SetDirty(controller);
        animator.runtimeAnimatorController = preview.runtimeAnimatorController = controller;
        Set(coach, "coachAnimator", animator);
        Set(card, "previewAnimator", preview);
        Set(countdown, "deadBugPreviewCard", card);

        Set(taiko, "sourceClip", clip);
        Set(taiko, "hipThrustAnimatorController", controller);
        Set(taiko, "coachAnimator", animator);
        Set(taiko, "coachPreviewAnimator", preview);
        Set(taiko, "poseScorer", scorer);
        Set(taiko, "songClock", Find<RhythmSongClock>());
        Set(taiko, "startMenu", menu);
        Set(taiko, "startMenuRoot", Named("StartMenuCanvas"));
        Set(taiko, "endMenuRoot", Named("EndMenuCanvas"));
        var menuSerialized = new SerializedObject(menu);
        Set(taiko, "repetitionsText", menuSerialized.FindProperty("timesText").objectReferenceValue);
        var properties = new SerializedObject(taiko);
        var sourceBeats = BakeBeats(clip);
        var beatsProperty = properties.FindProperty("sourceBeats");
        beatsProperty.arraySize = sourceBeats.Length;
        for (int i = 0; i < sourceBeats.Length; i++)
        {
            var item = beatsProperty.GetArrayElementAtIndex(i);
            item.FindPropertyRelative("upFrame").intValue = sourceBeats[i].upFrame;
            item.FindPropertyRelative("holdEndFrame").intValue = sourceBeats[i].holdEndFrame;
            item.FindPropertyRelative("downFrame").intValue = sourceBeats[i].downFrame;
        }
        properties.ApplyModifiedPropertiesWithoutUndo();
        RectTransform visualRoot = (RectTransform)Named("HipThrustCanvasVisuals").transform;
        // Cover the baked example score/difficulty with real session values, using the same red panel.
        RectTransform scorePanel = Panel(visualRoot, "LiveScorePanel", new Vector2(-603, 185), new Vector2(205, 272));
        TMP_Text score = Label(scorePanel, "LiveScore", "0", new Vector2(0, 61), new Vector2(220, 65), 48);
        Label(scorePanel, "ScoreTitle", "SCORE", new Vector2(0, 116), new Vector2(220, 35), 22);
        Label(scorePanel, "DifficultyTitle", "DIFFICULTY", new Vector2(0, -30), new Vector2(220, 35), 22);
        TMP_Text difficulty = Label(scorePanel, "LiveDifficulty", "标准", new Vector2(0, -83), new Vector2(220, 45), 30);
        TMP_Text phase = Label(visualRoot, "LivePhase", "准备动作", new Vector2(250, 365), new Vector2(800, 48), 30);
        Set(taiko, "scoreLabel", score);
        Set(taiko, "difficultyLabel", difficulty);
        Set(taiko, "phaseLabel", phase);
        RectTransform titlePanel = Panel(Named("StartMenuCanvas").transform, "HipThrustTitlePanel", new Vector2(-410, 40), new Vector2(610, 130));
        titlePanel.GetComponent<Image>().color = new Color(0.90f, 0.95f, 1f);
        Label(titlePanel, "Title", "Hip Thrust 臀桥", new Vector2(0, 25), new Vector2(600, 60), 46).color = new Color(0.15f, 0.24f, 0.35f);
        Label(titlePanel, "Description", "红色抬臀 · 蓝色落臀 · 拖尾保持", new Vector2(0, -30), new Vector2(600, 40), 23).color = new Color(0.25f, 0.35f, 0.45f);
        Transform endRoot = Named("EndMenuCanvas").transform;
        foreach (Transform child in endRoot) child.gameObject.SetActive(false);
        RectTransform resultPanel = Panel(endRoot, "HipThrustResultPanel", Vector2.zero, new Vector2(1920, 1080));
        resultPanel.gameObject.SetActive(true);
        resultPanel.GetComponent<Image>().color = new Color(0.89f, 0.95f, 1f);
        resultPanel.anchorMin = Vector2.zero;
        resultPanel.anchorMax = Vector2.one;
        resultPanel.sizeDelta = Vector2.zero;
        TMP_Text result = Label(resultPanel, "HipThrustResult", "训练完成", new Vector2(0, 80), new Vector2(1200, 200), 48);
        result.color = new Color(0.15f, 0.24f, 0.35f);
        Set(taiko, "resultLabel", result);
        RectTransform returnPanel = Panel(resultPanel, "ReturnToSelection", new Vector2(0, -130), new Vector2(360, 90));
        returnPanel.GetComponent<Image>().color = new Color(0.15f, 0.4f, 0.68f);
        returnPanel.GetComponent<Image>().raycastTarget = true;
        var returnButton = returnPanel.GetComponent<Button>() ?? returnPanel.gameObject.AddComponent<Button>();
        returnButton.onClick = new Button.ButtonClickedEvent();
        UnityEventTools.AddPersistentListener(returnButton.onClick, taiko.StopAndShowMenu);
        Label(returnPanel, "Label", "返回选择", Vector2.zero, new Vector2(340, 80), 36);
        Named("NoteClipArea").transform.SetAsLastSibling();
        Named("HipThrustSunlight").transform.SetAsLastSibling();
        // Retain the shared menu/countdown implementation, repair only this scene's disabled entry.
        Named("StartMenuCanvas").transform.localScale = Vector3.one;
        Named("StartMenuCanvas").SetActive(true);
        Named("EndMenuCanvas").transform.localScale = Vector3.one;
        Named("EndMenuCanvas").SetActive(false);
        Named("GamingCanvas").SetActive(false);
        foreach (Button button in roots.SelectMany(r => r.GetComponentsInChildren<Button>(true)))
        {
            for (int i = button.onClick.GetPersistentEventCount() - 1; i >= 0; i--)
            {
                if (button.onClick.GetPersistentTarget(i) != coach) continue;
                string method = button.onClick.GetPersistentMethodName(i);
                if (method == "TogglePause")
                {
                    UnityEventTools.RemovePersistentListener(button.onClick, i);
                    UnityEventTools.AddPersistentListener(button.onClick, taiko.TogglePause);
                }
                else if (method == "ReturnIdle")
                {
                    UnityEventTools.RemovePersistentListener(button.onClick, i);
                    UnityEventTools.AddPersistentListener(button.onClick, taiko.StopAndShowMenu);
                }
            }
        }
        EditorSceneManager.SaveScene(scene);
        AssetDatabase.SaveAssetIfDirty(controller);
        PolishPresentation();
        Debug.Log("HIPTHRUST_REPAIRED: full 25.5s source, four sampled pairs, shared animation/notes clock.");
    }

    private static HipThrustTaikoController.HipBeat[] BakeBeats(AnimationClip clip)
    {
        var sample = UnityEngine.Object.Instantiate(AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Movements/hipthrust.fbx"));
        sample.hideFlags = HideFlags.HideAndDontSave;
        try
        {
            var animator = sample.GetComponent<Animator>();
            int count = Mathf.RoundToInt(clip.length * clip.frameRate) + 1;
            float[] height = new float[count];
            for (int f = 0; f < count; f++)
            {
                clip.SampleAnimation(sample, f / clip.frameRate);
                height[f] = animator.GetBoneTransform(HumanBodyBones.Hips).position.y -
                    (animator.GetBoneTransform(HumanBodyBones.LeftLowerLeg).position.y + animator.GetBoneTransform(HumanBodyBones.RightLowerLeg).position.y) * 0.5f;
            }
            // Windows isolate the four repetitions from the standing/lying transitions in this asset.
            int[,] windows = { { 430, 600 }, { 650, 840 }, { 890, 1110 }, { 1150, 1305 } };
            var beats = new HipThrustTaikoController.HipBeat[4];
            var report = new StringBuilder("Source: Assets/Movements/hipthrust.fbx / animation_pose; 60 fps; imported clip starts at FBX frame 170.\nUp = enter high plateau within 1 cm of local peak. Hold ends on leaving plateau. Down = enter low plateau within 1 cm of following minimum.\n");
            for (int i = 0; i < 4; i++)
            {
                int peak = Enumerable.Range(windows[i, 0], windows[i, 1] - windows[i, 0]).OrderByDescending(f => height[f]).First();
                int bottom = Enumerable.Range(peak + 1, windows[i, 1] - peak).OrderBy(f => height[f]).First();
                int up = peak, holdEnd = peak, down = bottom;
                while (up > windows[i, 0] && height[up - 1] >= height[peak] - 0.01f) up--;
                while (holdEnd < bottom && height[holdEnd] >= height[peak] - 0.01f) holdEnd++;
                while (down > holdEnd && height[down - 1] <= height[bottom] + 0.01f) down--;
                beats[i] = new HipThrustTaikoController.HipBeat { upFrame = up, holdEndFrame = holdEnd, downFrame = down };
                report.AppendLine(FormattableString.Invariant($"Pair {i + 1}: red frame {up} ({up / clip.frameRate:F4}s), hold end {holdEnd} ({holdEnd / clip.frameRate:F4}s), blue frame {down} ({down / clip.frameRate:F4}s)."));
            }
            Directory.CreateDirectory("output/hipthrust");
            File.WriteAllText("output/hipthrust/beat-map.txt", report.ToString());
            return beats;
        }
        finally { UnityEngine.Object.DestroyImmediate(sample); }
    }

    private static void Set(UnityEngine.Object target, string field, UnityEngine.Object value)
    {
        var so = new SerializedObject(target);
        so.FindProperty(field).objectReferenceValue = value;
        so.ApplyModifiedPropertiesWithoutUndo();
    }
    private static RectTransform Panel(Transform parent, string name, Vector2 position, Vector2 size)
    {
        var existing = parent.Find(name);
        var image = existing != null ? existing.GetComponent<Image>() : new GameObject(name, typeof(RectTransform), typeof(Image)).GetComponent<Image>();
        image.transform.SetParent(parent, false);
        image.color = new Color(0.91f, 0.22f, 0.14f);
        image.raycastTarget = false;
        Layout(image.rectTransform, position, size);
        return image.rectTransform;
    }
    private static TMP_Text Label(Transform parent, string name, string value, Vector2 position, Vector2 size, float fontSize)
    {
        var existing = parent.Find(name);
        var label = existing != null ? existing.GetComponent<TMP_Text>() : new GameObject(name, typeof(RectTransform)).AddComponent<TextMeshProUGUI>();
        label.transform.SetParent(parent, false);
        var font = AssetDatabase.LoadAssetAtPath<TMP_FontAsset>("Assets/Fonts/SourceHanSans/SourceHanSansSC-Heavy SDF.asset");
        label.font = font;
        label.text = value;
        label.fontSize = fontSize;
        label.alignment = TextAlignmentOptions.Center;
        label.raycastTarget = false;
        Layout(label.rectTransform, position, size);
        return label;
    }
    private static void Layout(RectTransform rect, Vector2 position, Vector2 size)
    {
        rect.anchorMin = rect.anchorMax = rect.pivot = Vector2.one * 0.5f;
        rect.localScale = Vector3.one;
        rect.localRotation = Quaternion.identity;
        rect.anchoredPosition = position;
        rect.sizeDelta = size;
    }

    [MenuItem("Tools/Hip Thrust/Inspect Source Animation")]
    public static void Inspect()
    {
        const string path = "Assets/Movements/hipthrust.fbx";
        Directory.CreateDirectory("output/hipthrust");
        AnimationClip[] clips = AssetDatabase.LoadAllAssetsAtPath(path).OfType<AnimationClip>().Where(c => !c.name.StartsWith("__")).ToArray();
        File.WriteAllLines("output/hipthrust/clips.txt", clips.Select(c => $"{c.name}: {c.length:R}s @ {c.frameRate}fps"));
        var sample = UnityEngine.Object.Instantiate(AssetDatabase.LoadAssetAtPath<GameObject>(path));
        sample.hideFlags = HideFlags.HideAndDontSave;
        try
        {
            Animator animator = sample.GetComponent<Animator>();
            AnimationClip full = clips.Single(c => c.name == "animation_pose");
            var csv = new StringBuilder("time,hipsY,chestY,leftKneeY,rightKneeY\n");
            for (int frame = 0; frame <= Mathf.RoundToInt(full.length * full.frameRate); frame++)
            {
                float time = frame / full.frameRate;
                full.SampleAnimation(sample, time);
                float Y(HumanBodyBones bone) => animator.GetBoneTransform(bone).position.y;
                csv.AppendLine(FormattableString.Invariant($"{time:R},{Y(HumanBodyBones.Hips):R},{Y(HumanBodyBones.Chest):R},{Y(HumanBodyBones.LeftLowerLeg):R},{Y(HumanBodyBones.RightLowerLeg):R}"));
            }
            File.WriteAllText("output/hipthrust/hip-height.csv", csv.ToString());
            Debug.Log("HIPTHRUST_ANIMATION_SAMPLED");
        }
        finally { UnityEngine.Object.DestroyImmediate(sample); }
    }
}
