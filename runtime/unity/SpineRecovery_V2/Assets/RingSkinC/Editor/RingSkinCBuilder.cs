using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;

namespace RingSkinC.EditorTools
{
    /// <summary>
    /// Menu: Tools > Ring Skin C > Build Prefab + Demo Scene
    /// 1. Sets every texture in Assets/RingSkinC/Sprites to Sprite, 300 PPU (3x art, 1 design px = 0.01 unit).
    /// 2. Assembles starry background, ring, judge glow, core, demo chart and HUD.
    /// 3. Saves Prefabs/RingSkinC.prefab and Scenes/RingSkinC_Demo.unity.
    /// </summary>
    public static class RingSkinCBuilder
    {
        const string Root = "Assets/RingSkinC";
        const string SpriteDir = Root + "/Sprites";
        const string FontDir = Root + "/Fonts";
        const string PrefabPath = Root + "/Prefabs/RingSkinC.prefab";
        const string ScenePath = Root + "/Scenes/RingSkinC_Demo.unity";

        // Order matches RingDemoChart: 0 melody (UL), 1 harmony (UR), 2 bass (LL), 3 drums (LR)
        static readonly string[] Zones = { "melody", "harmony", "bass", "drums" };

        // The game uses the original authored assembly without the sample chart or dummy HUD.
        public static GameObject BuildGameVisuals()
        {
            ConfigureSprites();
            GameObject root = new GameObject("RingSkinC_Visuals");
            BuildBackground(root.transform);
            SpriteRenderer[] flashes;
            BuildRing(root.transform, out flashes);
            return root;
        }

        [MenuItem("Tools/Ring Skin C/Build Prefab + Demo Scene")]
        public static void Build()
        {
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;

            ConfigureSprites();

            GameObject root = new GameObject("RingSkinC");
            BuildBackground(root.transform);
            SpriteRenderer[] flashes;
            Transform core = BuildRing(root.transform, out flashes);
            BuildChart(root, core, flashes);
            BuildHud(root.transform);

            Directory.CreateDirectory(Root + "/Prefabs");
            GameObject prefab = PrefabUtility.SaveAsPrefabAsset(root, PrefabPath);
            Object.DestroyImmediate(root);

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            GameObject camGo = new GameObject("Main Camera");
            camGo.tag = "MainCamera";
            Camera cam = camGo.AddComponent<Camera>();
            cam.orthographic = true;
            cam.orthographicSize = 1.95f; // 390 design px tall
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = Hex("#0C0E2E");
            camGo.transform.position = new Vector3(0f, 0f, -10f);
            PrefabUtility.InstantiatePrefab(prefab);

            Directory.CreateDirectory(Root + "/Scenes");
            EditorSceneManager.SaveScene(scene, ScenePath);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();

            EditorUtility.DisplayDialog("Ring Skin C",
                "Built:\n" + PrefabPath + "\n" + ScenePath +
                "\n\nSet the Game view to a landscape phone resolution (e.g. 2532x1170) and press Play.", "OK");
        }

        // ------------------------------------------------------------------ import settings

        static void ConfigureSprites()
        {
            string[] guids = AssetDatabase.FindAssets("t:Texture2D", new[] { SpriteDir });
            try
            {
                for (int i = 0; i < guids.Length; i++)
                {
                    string path = AssetDatabase.GUIDToAssetPath(guids[i]);
                    EditorUtility.DisplayProgressBar("Ring Skin C", "Importing " + Path.GetFileName(path), (float)i / guids.Length);
                    TextureImporter ti = AssetImporter.GetAtPath(path) as TextureImporter;
                    if (ti == null) continue;
                    bool ok = ti.textureType == TextureImporterType.Sprite
                              && ti.textureShape == TextureImporterShape.Texture2D
                              && Mathf.Approximately(ti.spritePixelsPerUnit, 300f)
                              && !ti.mipmapEnabled
                              && ti.textureCompression == TextureImporterCompression.Uncompressed;
                    if (ok && AssetDatabase.LoadAssetAtPath<Sprite>(path) != null) continue;
                    ti.textureType = TextureImporterType.Sprite;
                    ti.textureShape = TextureImporterShape.Texture2D;
                    ti.spriteImportMode = SpriteImportMode.Single;
                    ti.spritePixelsPerUnit = 300f;
                    ti.mipmapEnabled = false;
                    ti.alphaIsTransparency = true;
                    ti.wrapMode = TextureWrapMode.Clamp;
                    ti.filterMode = FilterMode.Bilinear;
                    ti.npotScale = TextureImporterNPOTScale.None;
                    ti.maxTextureSize = 4096;
                    ti.textureCompression = TextureImporterCompression.Uncompressed;
                    TextureImporterSettings s = new TextureImporterSettings();
                    ti.ReadTextureSettings(s);
                    s.spriteMeshType = SpriteMeshType.FullRect;
                    s.spriteAlignment = (int)SpriteAlignment.Center;
                    ti.SetTextureSettings(s);
                    ti.SaveAndReimport();
                }
            }
            finally
            {
                EditorUtility.ClearProgressBar();
            }
        }

        // ------------------------------------------------------------------ background

        static void BuildBackground(Transform parent)
        {
            Transform bg = Group("Background", parent, Vector3.zero);
            SR(bg, "Sky", "c_bg_sky", 0, Vector3.zero);
            SR(bg, "StarsMoonConstellation", "c_bg_far", 1, Vector3.zero);

            // Twinkling sparkles: design px position, phase offset (s)
            Vector2[] sparklePos = { new Vector2(96f, 100f), new Vector2(232f, 150f), new Vector2(612f, 40f), new Vector2(790f, 80f), new Vector2(812f, 250f) };
            float[] sparklePhase = { 0f, 0.6f, 1.2f, 1.8f, 0.9f };
            Transform sparkles = Group("Sparkles", bg, Vector3.zero);
            for (int i = 0; i < sparklePos.Length; i++)
            {
                Twinkle tw = SR(sparkles, "Sparkle" + i, "c_fx_sparkle", 2, DesignSpace.ToWorld(sparklePos[i].x, sparklePos[i].y))
                    .gameObject.AddComponent<Twinkle>();
                tw.phase = sparklePhase[i];
            }
            // Brightest star of the constellation breathes on a 3 s cycle
            SpriteRenderer brightStar = SR(sparkles, "ConstellationStar", "c_fx_sparkle", 2, DesignSpace.ToWorld(676f, 162f));
            brightStar.color = Hex("#BFE9FF");
            brightStar.gameObject.AddComponent<Twinkle>().period = 3f;

            SpriteRenderer meteor = SR(bg, "Meteor", "c_fx_meteor", 3, Vector3.zero);
            meteor.color = new Color(1f, 1f, 1f, 0f);
            meteor.gameObject.AddComponent<MeteorLoop>();

            SR(bg, "Mountains", "c_bg_mountains", 4, Vector3.zero);
            SR(bg, "Clouds", "c_bg_clouds", 5, Vector3.zero);

            Transform lanterns = Group("Lanterns", bg, Vector3.zero);
            Vector2[] lanternPos = { new Vector2(108f, 249.5f), new Vector2(203f, 204f), new Vector2(741.5f, 257f), new Vector2(798.5f, 229.5f) };
            float[] lanternScale = { 1f, 0.75f, 0.9f, 0.6f };
            for (int i = 0; i < lanternPos.Length; i++)
            {
                SpriteRenderer l = SR(lanterns, "Lantern" + i, "c_fx_lantern", 6, DesignSpace.ToWorld(lanternPos[i].x, lanternPos[i].y));
                l.transform.localScale = new Vector3(lanternScale[i], lanternScale[i], 1f);
            }
            lanterns.gameObject.AddComponent<FloatBob>();
        }

        // ------------------------------------------------------------------ ring

        static Transform BuildRing(Transform parent, out SpriteRenderer[] flashes)
        {
            Transform ring = Group("Ring", parent, DesignSpace.RingCenter);

            SR(ring, "Ticks", "c_ring_ticks", 10, Vector3.zero).gameObject.AddComponent<SpinForever>();

            flashes = new SpriteRenderer[Zones.Length];
            for (int i = 0; i < Zones.Length; i++)
            {
                string z = Zones[i];
                Transform zone = Group("Zone_" + z, ring, Vector3.zero);
                SR(zone, "Base", "c_zone_" + z, 20, Vector3.zero);
                SR(zone, "ArcGlow", "c_zone_" + z + "_glow", 21, Vector3.zero);
                SpriteRenderer flash = SR(zone, "HitFlash", "c_zone_" + z + "_flash", 22, Vector3.zero);
                flash.color = new Color(1f, 1f, 1f, 0f);
                flashes[i] = flash;
            }

            BeatPulseAlpha jb = SR(ring, "JudgeRingBeatGlow", "c_judge_ring_glow", 31, Vector3.zero)
                .gameObject.AddComponent<BeatPulseAlpha>();
            jb.from = 1f;
            jb.to = 0.15f;

            Transform core = Group("Core", ring, Vector3.zero);
            SR(core, "Glow", "c_core_glow", 40, Vector3.zero);
            SR(core, "Orb", "c_core", 41, Vector3.zero);
            return core;
        }

        // ------------------------------------------------------------------ demo chart

        static void BuildChart(GameObject root, Transform core, SpriteRenderer[] flashes)
        {
            RingDemoChart chart = root.AddComponent<RingDemoChart>();
            chart.core = core;
            chart.zoneFlash = flashes;
            for (int i = 0; i < 4; i++)
            {
                string z = Zones[i];
                chart.noteBody[i] = S("c_note_" + z);
                chart.noteGlow[i] = S("c_note_" + z + "_glow");
                chart.noteTrail[i] = S("c_note_" + z + "_trail");
                chart.hitRipple[i] = S("c_hit_" + z);
            }
            chart.centerCue = S("c_center_cue");
            chart.centerWave = S("c_hit_center_wave");
            chart.textPerfect = S("c_text_perfect");
            chart.textPerfectCenter = S("c_text_perfect_center");
        }

        // ------------------------------------------------------------------ HUD

        static void BuildHud(Transform parent)
        {
            GameObject canvasGo = new GameObject("HUD", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            canvasGo.transform.SetParent(parent, false);
            Canvas canvas = canvasGo.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            CanvasScaler scaler = canvasGo.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(844f, 390f); // 1 canvas unit = 1 design px
            scaler.screenMatchMode = CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
            scaler.matchWidthOrHeight = 1f;
            Transform hud = canvasGo.transform;

            Font builtin = BuiltinFont();
            Font fredoka = AssetDatabase.LoadAssetAtPath<Font>(FontDir + "/Fredoka-SemiBold.ttf");
            if (fredoka == null) fredoka = builtin;

            RectTransform pause = UI(hud, "PauseButton", new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(18f, -8f), new Vector2(44f, 44f));
            Image pauseImg = pause.gameObject.AddComponent<Image>();
            pauseImg.sprite = S("c_hud_pause");
            pause.gameObject.AddComponent<Button>().targetGraphic = pauseImg;

            RectTransform title = UI(hud, "Title", new Vector2(0f, 1f), new Vector2(0f, 0.5f), new Vector2(74f, -30f), new Vector2(260f, 16f));
            Text titleText = AddText(title, builtin, 12, FontStyle.Bold, Color.white, TextAnchor.MiddleLeft);
            titleText.text = "云端信笺 · HARD 10";

            RectTransform score = UI(hud, "Score", new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(-18f, -12f), new Vector2(240f, 30f));
            Text scoreText = AddText(score, fredoka, 24, FontStyle.Normal, Color.white, TextAnchor.UpperRight);
            scoreText.text = "0,524,180";

            RectTransform combo = UI(hud, "Combo", new Vector2(1f, 1f), new Vector2(1f, 0.5f), new Vector2(-18f, -44f), new Vector2(240f, 16f));
            Text comboText = AddText(combo, fredoka, 11, FontStyle.Normal, Hex("#D8DCFF"), TextAnchor.MiddleRight);
            comboText.text = "128 COMBO";
            combo.gameObject.AddComponent<BeatBump>();
        }

        // ------------------------------------------------------------------ helpers

        static Sprite S(string name)
        {
            Sprite s = AssetDatabase.LoadAssetAtPath<Sprite>(SpriteDir + "/" + name + ".png");
            if (s == null) Debug.LogError("[RingSkinC] Missing sprite: " + name);
            return s;
        }

        static Transform Group(string name, Transform parent, Vector3 localPos)
        {
            GameObject g = new GameObject(name);
            g.transform.SetParent(parent, false);
            g.transform.localPosition = localPos;
            return g.transform;
        }

        static SpriteRenderer SR(Transform parent, string name, string sprite, int order, Vector3 localPos)
        {
            GameObject g = new GameObject(name);
            g.transform.SetParent(parent, false);
            g.transform.localPosition = localPos;
            SpriteRenderer r = g.AddComponent<SpriteRenderer>();
            r.sprite = S(sprite);
            r.sortingOrder = order;
            return r;
        }

        static RectTransform UI(Transform parent, string name, Vector2 anchor, Vector2 pivot, Vector2 pos, Vector2 size)
        {
            GameObject g = new GameObject(name, typeof(RectTransform));
            g.transform.SetParent(parent, false);
            RectTransform rt = (RectTransform)g.transform;
            rt.anchorMin = anchor;
            rt.anchorMax = anchor;
            rt.pivot = pivot;
            rt.anchoredPosition = pos;
            rt.sizeDelta = size;
            return rt;
        }

        static Text AddText(RectTransform rt, Font font, int size, FontStyle style, Color color, TextAnchor align)
        {
            Text t = rt.gameObject.AddComponent<Text>();
            t.font = font;
            t.fontSize = size;
            t.fontStyle = style;
            t.color = color;
            t.alignment = align;
            t.horizontalOverflow = HorizontalWrapMode.Overflow;
            t.verticalOverflow = VerticalWrapMode.Overflow;
            t.raycastTarget = false;
            return t;
        }

        static Font BuiltinFont()
        {
            Font f = null;
            try { f = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf"); } catch { }
            if (f == null)
            {
                try { f = Resources.GetBuiltinResource<Font>("Arial.ttf"); } catch { }
            }
            return f;
        }

        static Color Hex(string hex)
        {
            Color c;
            return ColorUtility.TryParseHtmlString(hex, out c) ? c : Color.magenta;
        }
    }
}
