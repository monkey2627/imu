using System;
using System.IO;
using System.Linq;
using TMPro;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;

public static class RingBridgeGameBuilder
{
    public const string ScenePath = "Assets/Scenes/HipThrust_RingBridge_Recorded.unity";
    public const string TimelinePath = "Assets/RingBridge/RecordedHipThrustTimeline.asset";
    public const string Output = "output/ringbridge/";

    [MenuItem("Tools/Ring Bridge/Create Recorded Hip Thrust Game")]
    public static void Build()
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode) throw new InvalidOperationException("Exit Play mode first.");
        if (File.Exists(ScenePath)) throw new IOException("The ring bridge game exists; edit it in place.");
        Directory.CreateDirectory(Output);
        if (!AssetDatabase.CopyAsset(RecordedCoachSceneTools.Recorded("HipThrust"), ScenePath)) throw new IOException("Cannot copy the recorded game.");
        var scene = EditorSceneManager.OpenScene(ScenePath);
        var all = scene.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<Transform>(true)).ToArray();
        T Find<T>() where T : Component => scene.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<T>(true)).Single();
        Transform Named(string name) => all.Single(t => t.name == name);
        var taiko = Find<HipThrustTaikoController>();
        var entry = Find<GameWorkoutScene>();
        var data = new SerializedObject(taiko);
        var canvas = (Canvas)data.FindProperty("gamingCanvas").objectReferenceValue;
        var animator = (Animator)data.FindProperty("coachAnimator").objectReferenceValue;
        var clip = RecordedCoachSceneTools.LoadClips("HipThrust")["hipthrust_routine"];
        var timeline = BakeTimeline(animator, clip);
        AssetDatabase.CreateAsset(timeline, TimelinePath);
        data.FindProperty("phaseTimeline").objectReferenceValue = timeline;
        data.ApplyModifiedPropertiesWithoutUndo();

        // Only the new scene loses the old taiko artwork; its transport/menu/calibration stay wired.
        Named("HipThrustCanvasVisuals").gameObject.SetActive(false);
        foreach (var image in canvas.GetComponentsInChildren<Graphic>(true)) image.enabled = false;
        var card = Find<ActionPreviewCard>();
        var cardData = new SerializedObject(card);
        cardData.FindProperty("targetSceneName").stringValue = Path.GetFileNameWithoutExtension(ScenePath);
        cardData.ApplyModifiedPropertiesWithoutUndo();
        entry.hideWorkoutToggle = true;
        entry.hipThrustSceneOverride = Path.GetFileNameWithoutExtension(ScenePath);
        var description = all.FirstOrDefault(t => t.name == "Description" && t.GetComponent<TMP_Text>() != null);
        if (description != null) description.GetComponent<TMP_Text>().text = "粉色抬起 · 金色保持 · 蓝色放下";

        GameObject art = RingBridge.EditorTools.RingBridgeBuilder.CreateArtwork();
        art.name = "Recorded Ring Bridge Artwork";
        art.transform.position = new Vector3(1000, 1000, 0);
        foreach (Transform t in art.GetComponentsInChildren<Transform>(true)) t.gameObject.layer = 29;
        var chart = art.GetComponent<RingBridge.RingBridgeChart>(); chart.autoPlay = false;
        var cameraObject = new GameObject("RingBridge Presentation Camera");
        Camera camera = cameraObject.AddComponent<Camera>();
        camera.transform.position = art.transform.position + Vector3.back * 10;
        camera.orthographic = true; camera.orthographicSize = 1.95f;
        camera.cullingMask = 1 << 29; camera.clearFlags = CameraClearFlags.SolidColor;
        camera.backgroundColor = new Color(.059f,.063f,.204f);
        camera.allowHDR = false; camera.allowMSAA = false; camera.nearClipPlane = .1f; camera.farClipPlane = 25;
        ConfigureArtwork(art);
        var hud = art.transform.Find("HUD").GetComponent<Canvas>();
        hud.renderMode = RenderMode.WorldSpace; hud.worldCamera = camera; hud.sortingOrder = 100;
        hud.GetComponent<CanvasScaler>().enabled = false;
        var rect = (RectTransform)hud.transform;
        rect.anchorMin = rect.anchorMax = rect.pivot = Vector2.one*.5f;
        rect.sizeDelta = new Vector2(844,390); rect.anchoredPosition3D = Vector3.zero; rect.localPosition = Vector3.zero; rect.localScale = Vector3.one*.01f;
        UnityEngine.Object.DestroyImmediate(hud.GetComponent<GraphicRaycaster>());
        var view = new GameObject("Recorded Ring Bridge View").AddComponent<HipThrustRingBridgeView>();
        view.transport = taiko; view.entry = entry; view.artwork = art; view.chart = chart;
        view.presentationCamera = camera; view.gamingCanvas = canvas;
        view.score = hud.transform.Find("Score").GetComponent<Text>();
        view.combo = hud.transform.Find("Combo").GetComponent<Text>();
        view.score.text = "0"; view.combo.text = "0 COMBO";
        view.font = entry.font;
        ConfigureDisplayMaterial(view);
        view.coachPreview = Named("HipThrustPreviewCamera").GetComponent<Camera>().targetTexture;
        chart.RenderRecorded(timeline.Evaluate(0));
        EditorSceneManager.MarkSceneDirty(scene); EditorSceneManager.SaveScene(scene);
        var build = EditorBuildSettings.scenes.ToList();
        build.Add(new EditorBuildSettingsScene(ScenePath, true)); EditorBuildSettings.scenes = build.ToArray();
        AssetDatabase.SaveAssets();
        File.WriteAllText(Output + "build-result.txt", "Created " + ScenePath + ". Existing six scenes preserved.\n");
    }

    public static void RefreshVisuals()
    {
        var scene = EditorSceneManager.OpenScene(ScenePath);
        var view = scene.GetRootGameObjects().SelectMany(r=>r.GetComponentsInChildren<HipThrustRingBridgeView>(true)).Single();
        ConfigureArtwork(view.artwork);
        ConfigureDisplayMaterial(view);
        var rect = (RectTransform)view.artwork.transform.Find("HUD");
        rect.anchorMin = rect.anchorMax = rect.pivot = Vector2.one*.5f;
        rect.anchoredPosition3D = Vector3.zero; rect.localPosition = Vector3.zero;
        EditorSceneManager.MarkSceneDirty(scene); EditorSceneManager.SaveScene(scene); AssetDatabase.SaveAssets();
    }

    static void ConfigureArtwork(GameObject art)
    {
        Directory.CreateDirectory("Assets/RingBridge/Materials"); AssetDatabase.Refresh();
        Material MaterialFor(string name, float white)
        {
            string path="Assets/RingBridge/Materials/"+name+".mat";
            var material=AssetDatabase.LoadAssetAtPath<Material>(path);
            if(material==null) { material=new Material(Shader.Find("RingBridge/RecordedSprite")); AssetDatabase.CreateAsset(material,path); }
            material.SetFloat("_WhiteMask",white); EditorUtility.SetDirty(material); return material;
        }
        var normal=MaterialFor("RecordedArtwork",0); var stars=MaterialFor("WhiteStarMask",1);
        foreach(var renderer in art.GetComponentsInChildren<SpriteRenderer>(true))
        {
            renderer.sharedMaterial=renderer.name.StartsWith("rb_sparkle_")?stars:normal;
            renderer.maskInteraction=SpriteMaskInteraction.None;
        }
        foreach(var mask in art.GetComponentsInChildren<SpriteMask>(true)) mask.enabled=false;
    }

    static void ConfigureDisplayMaterial(HipThrustRingBridgeView view)
    {
        const string path="Assets/RingBridge/Materials/RecordedDisplay.mat";
        var material=AssetDatabase.LoadAssetAtPath<Material>(path);
        if(material==null)
        {
            material=new Material(Shader.Find("RingSkinC/DisplayGammaComposite"));
            AssetDatabase.CreateAsset(material,path);
        }
        // Match the UI queue so controls later in the canvas draw above the artwork.
        material.renderQueue=3000; EditorUtility.SetDirty(material); view.displayMaterial=material;
    }

    static HipThrustPhaseTimeline BakeTimeline(Animator source, AnimationClip clip)
    {
        var timeline = ScriptableObject.CreateInstance<HipThrustPhaseTimeline>();
        timeline.sourceClip = clip;
        var sample = UnityEngine.Object.Instantiate(source.gameObject);
        sample.hideFlags = HideFlags.HideAndDontSave;
        sample.transform.SetParent(null); sample.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity); sample.transform.localScale = Vector3.one;
        foreach (var script in sample.GetComponentsInChildren<MonoBehaviour>(true)) script.enabled = false;
        sample.SetActive(true);
        var animator = sample.GetComponent<Animator>(); animator.enabled = true; animator.Rebind(); animator.Update(0);
        int count = Mathf.RoundToInt(clip.length*clip.frameRate)+1;
        var heights = new float[count];
        try
        {
            for (int f = 0; f < count; f++)
            {
                animator.Play("hipthrust_routine", 0, f/(float)(count-1)); animator.Update(0);
                Vector3 Bone(HumanBodyBones b) => animator.GetBoneTransform(b).position;
                heights[f] = Bone(HumanBodyBones.Hips).y - (Bone(HumanBodyBones.LeftLowerLeg).y+Bone(HumanBodyBones.RightLowerLeg).y)*.5f;
            }
        }
        finally { UnityEngine.Object.DestroyImmediate(sample); }
        var smooth = new float[count];
        for (int f=0;f<count;f++)
            for (int j=-2;j<=2;j++) smooth[f] += heights[Mathf.Clamp(f+j,0,count-1)] * (3-Mathf.Abs(j))/9f;
        int Min(int a,int b) => Enumerable.Range(a-180,b-a).OrderBy(f=>smooth[f]).First();
        int Max(int a,int b) => Enumerable.Range(a-180,b-a).OrderByDescending(f=>smooth[f]).First();
        int[] lows={Min(205,225),Min(390,410),Min(575,595),Min(770,790),Min(930,946)};
        int[] peaks={Max(250,310),Max(450,490),Max(640,680),Max(830,860)};
        timeline.cycles=new HipThrustPhaseTimeline.Cycle[4];
        var report=new System.Text.StringBuilder("cycle,liftStartRaw,holdStartRaw,lowerStartRaw,lowerEndRaw,liftSeconds,holdSeconds,lowerSeconds\n");
        for(int i=0;i<4;i++)
        {
            float low=Mathf.Min(smooth[lows[i]],smooth[lows[i+1]]), high=smooth[peaks[i]], threshold=high-.1f*(high-low);
            int hold=Enumerable.Range(lows[i],peaks[i]-lows[i]).First(f=>smooth[f]>=threshold);
            int lower=Enumerable.Range(peaks[i],lows[i+1]-peaks[i]).First(f=>smooth[f]<threshold);
            timeline.cycles[i]=new HipThrustPhaseTimeline.Cycle {liftStartFrame=lows[i],holdStartFrame=hold,lowerStartFrame=lower,lowerEndFrame=lows[i+1],lowHeight=low,peakHeight=high};
            report.AppendLine(FormattableString.Invariant($"{i+1},{lows[i]+180},{hold+180},{lower+180},{lows[i+1]+180},{(hold-lows[i])/30f:F3},{(lower-hold)/30f:F3},{(lows[i+1]-lower)/30f:F3}"));
        }
        timeline.recordedHeight=new AnimationCurve(Enumerable.Range(0,count).Select(f=>new Keyframe(f,smooth[f])).ToArray());
        for(int k=0;k<count;k++) {AnimationUtility.SetKeyLeftTangentMode(timeline.recordedHeight,k,AnimationUtility.TangentMode.Linear);AnimationUtility.SetKeyRightTangentMode(timeline.recordedHeight,k,AnimationUtility.TangentMode.Linear);}
        timeline.Validate();
        File.WriteAllText(Output+"phase-boundaries.csv",report.ToString());
        return timeline;
    }
}
