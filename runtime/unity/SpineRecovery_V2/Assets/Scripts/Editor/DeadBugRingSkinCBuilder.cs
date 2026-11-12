using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>Copies an exercise scene and wires only presentation overrides.</summary>
public static class DeadBugRingSkinCBuilder
{
    public const string Source = "Assets/Scenes/DeadBug_Classic.unity";
    public const string Destination = "Assets/Scenes/DeadBug_RingSkinC.unity";
    public const string BirdDogSource = "Assets/Scenes/BirdDog_Classic.unity";
    public const string BirdDogDestination = "Assets/Scenes/BirdDog_RingSkinC.unity";
    private const string Prefab = "Assets/RingSkinC/Prefabs/RingSkinC_GameVisuals.prefab";
    private static readonly string[] Zones = { "melody", "harmony", "bass", "drums" };

    [MenuItem("Tools/Ring Skin C/Create DeadBug Scene")]
    public static void Build()
    {
        BuildCopy(Source, Destination);
    }

    [MenuItem("Tools/Ring Skin C/Create BirdDog Scene")]
    public static void BuildBirdDog()
    {
        BuildCopy(BirdDogSource, BirdDogDestination);
    }

    private static void BuildCopy(string source, string destination)
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode) throw new InvalidOperationException("Exit Play mode before creating scenes.");
        if (File.Exists(destination)) throw new InvalidOperationException("The new scene already exists; edit it in place.");
        if (!AssetDatabase.CopyAsset(source, destination)) throw new IOException("Cannot copy scene: " + source);
        Scene previous = SceneManager.GetActiveScene();
        Scene scene = EditorSceneManager.OpenScene(destination, OpenSceneMode.Additive);
        SceneManager.SetActiveScene(scene);
        try
        {
            RhythmModeController controller = FindInScene<RhythmModeController>(scene);
            RhythmRingLayout layout = FindInScene<RhythmRingLayout>(scene);
            RhythmHitFeedback feedback = FindInScene<RhythmHitFeedback>(scene);
            if (controller == null || layout == null || feedback == null) throw new InvalidOperationException("Missing original rhythm components.");
            float designScale = layout.JudgementRadius / 1.43f;
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(Prefab);
            if (prefab == null)
            {
                GameObject generated = RingSkinC.EditorTools.RingSkinCBuilder.BuildGameVisuals();
                Directory.CreateDirectory("Assets/RingSkinC/Prefabs");
                prefab = PrefabUtility.SaveAsPrefabAsset(generated, Prefab);
                UnityEngine.Object.DestroyImmediate(generated);
            }
            GameObject art = (GameObject)PrefabUtility.InstantiatePrefab(prefab, layout.transform);
            art.transform.localPosition = new Vector3(0f, 0.2f * designScale, 0f);
            art.transform.localScale = Vector3.one * designScale;

            RhythmSkinCVisuals skin = controller.gameObject.AddComponent<RhythmSkinCVisuals>();
            skin.controller = controller;
            skin.layout = layout;
            skin.artwork = art;
            skin.designScale = designScale;
            skin.core = art.transform.Find("Ring/Core");
            for (int i = 0; i < 4; i++)
            {
                skin.noteBodies[i] = Sprite("c_note_" + Zones[i]);
                skin.noteGlows[i] = Sprite("c_note_" + Zones[i] + "_glow");
                skin.noteTrails[i] = Sprite("c_note_" + Zones[i] + "_trail");
                skin.hitRipples[i] = Sprite("c_hit_" + Zones[i]);
                skin.zoneFlashes[i] = art.transform.Find("Ring/Zone_" + Zones[i] + "/HitFlash").GetComponent<SpriteRenderer>();
            }
            skin.noteBodies[4] = Sprite("c_center_cue");
            skin.hitRipples[4] = Sprite("c_hit_center_wave");
            skin.perfectText = Sprite("c_text_perfect");
            skin.perfectCenterText = Sprite("c_text_perfect_center");
            Set(layout, "completeVisualRoot", art);
            Set(controller, "visualSkin", skin);
            Set(controller, "visualEffectRoot", art);
            Set(feedback, "visualSkin", skin);
            // Arrival circles are graphics only; retain their radius because it also participates in the original movement.
            var serializedController = new SerializedObject(controller);
            serializedController.FindProperty("showArrivalIndicator").boolValue = false;
            serializedController.ApplyModifiedPropertiesWithoutUndo();
            var layoutProperties = new SerializedObject(layout);
            string[] colorProperties = { "leftHandColor", "rightHandColor", "leftFootColor", "rightFootColor" };
            string[] colors = { "#FFC1DC", "#FFF0A6", "#B8FFF1", "#D7CCFF" };
            for (int i = 0; i < 4; i++)
            {
                ColorUtility.TryParseHtmlString(colors[i], out Color color);
                layoutProperties.FindProperty(colorProperties[i]).colorValue = color;
            }
            layoutProperties.ApplyModifiedPropertiesWithoutUndo();

            // Dedicated camera displays the authored composition without moving the tracked models or the game world.
            const int skinLayer = 30;
            Transform oldEffect = controller.transform.Find("LegacyRhythmVisualEffects");
            if (oldEffect != null) oldEffect.gameObject.SetActive(false);
            foreach (Transform child in controller.GetComponentsInChildren<Transform>(true)) child.gameObject.layer = skinLayer;
            GameObject cameraObject = new GameObject("RingSkinC_PresentationCamera");
            Camera camera = cameraObject.AddComponent<Camera>();
            camera.orthographic = true;
            camera.orthographicSize = 1.5f * designScale * 1.95f / skin.viewScale;
            camera.transform.position = layout.transform.position + new Vector3(0f, 0.2f * 1.5f * designScale, -10f);
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(12f / 255f, 14f / 255f, 46f / 255f);
            camera.cullingMask = 1 << skinLayer;
            camera.depth = 20f;
            camera.nearClipPlane = 0.1f;
            camera.farClipPlane = 30f;
            camera.allowHDR = false;
            skin.presentationCamera = camera;
            ConfigurePresentation(skin);

            EditorSceneManager.SaveScene(scene, destination);
            var buildScenes = EditorBuildSettings.scenes.ToList();
            if (buildScenes.All(s => s.path != destination)) buildScenes.Add(new EditorBuildSettingsScene(destination, true));
            EditorBuildSettings.scenes = buildScenes.ToArray();
            AssetDatabase.SaveAssets();
            Debug.Log("RING_SKIN_C_CREATED: " + destination);
        }
        finally
        {
            if (previous.IsValid() && previous.isLoaded) SceneManager.SetActiveScene(previous);
            EditorSceneManager.CloseScene(scene, true);
        }
    }

    private static T FindInScene<T>(Scene scene) where T : Component =>
        scene.GetRootGameObjects().SelectMany(root => root.GetComponentsInChildren<T>(true)).SingleOrDefault();

    public static void UpdatePresentation()
    {
        var scene = EditorSceneManager.OpenScene(Destination, OpenSceneMode.Single);
        ConfigurePresentation(UnityEngine.Object.FindObjectOfType<RhythmSkinCVisuals>(true));
        EditorSceneManager.SaveScene(scene);
        AssetDatabase.SaveAssets();
    }

    private static void ConfigurePresentation(RhythmSkinCVisuals skin)
    {
        skin.spriteMaterial = Material("SpriteGammaComposite");
        skin.displayMaterial = Material("DisplayGammaComposite");
        foreach (SpriteRenderer renderer in skin.artwork.GetComponentsInChildren<SpriteRenderer>(true))
            renderer.sharedMaterial = skin.spriteMaterial;
        EditorUtility.SetDirty(skin);
    }

    private static Material Material(string name)
    {
        string path = "Assets/RingSkinC/" + name + ".mat";
        Material material = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (material == null)
        {
            Shader shader = Shader.Find("RingSkinC/" + name);
            if (shader == null) throw new InvalidOperationException("Missing skin shader: " + name);
            material = new Material(shader);
            AssetDatabase.CreateAsset(material, path);
        }
        return material;
    }

    private static Sprite Sprite(string name)
    {
        Sprite sprite = AssetDatabase.LoadAssetAtPath<Sprite>("Assets/RingSkinC/Sprites/" + name + ".png");
        if (sprite == null) throw new InvalidOperationException("Missing sprite: " + name);
        return sprite;
    }

    private static void Set(UnityEngine.Object obj, string field, UnityEngine.Object value)
    {
        var serialized = new SerializedObject(obj);
        serialized.FindProperty(field).objectReferenceValue = value;
        serialized.ApplyModifiedPropertiesWithoutUndo();
    }
}
