using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;

namespace RingSkinD.EditorTools
{
    /// <summary>
    /// Menu: Tools > Ring Skin D > Build Prefab + Demo Scene
    /// 1. Sets every texture in Assets/RingSkinD/Sprites to Sprite, 300 PPU (3x art, 1 design px = 0.01 unit).
    /// 2. Assembles background, ring, LED quadrants, judge ring, core, demo chart and HUD.
    /// 3. Saves Prefabs/RingSkinD.prefab and Scenes/RingSkinD_Demo.unity.
    /// </summary>
    public static class RingSkinDBuilder
    {
        const string Root = "Assets/RingSkinD";
        const string SpriteDir = Root + "/Sprites";
        const string FontDir = Root + "/Fonts";
        const string PrefabPath = Root + "/Prefabs/RingSkinD.prefab";
        const string ScenePath = Root + "/Scenes/RingSkinD_Demo.unity";

        // Order matches RingDemoChart: 0 melody (UL), 1 harmony (UR), 2 bass (LL), 3 drums (LR)
        static readonly string[] Zones = { "melody", "harmony", "bass", "drums" };

        [MenuItem("Tools/Ring Skin D/Build Prefab + Demo Scene")]
        public static void Build()
        {
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;

            ConfigureSprites();

            GameObject root = new GameObject("RingSkinD");
            BuildBackground(root.transform);
            Transform core = BuildRing(root.transform);
            BuildChart(root, core);
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
            cam.backgroundColor = Hex("#07081A");
            camGo.transform.position = new Vector3(0f, 0f, -10f);
            PrefabUtility.InstantiatePrefab(prefab);

            Directory.CreateDirectory(Root + "/Scenes");
            EditorSceneManager.SaveScene(scene, ScenePath);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();

            EditorUtility.DisplayDialog("Ring Skin D",
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
                    EditorUtility.DisplayProgressBar("Ring Skin D", "Importing " + Path.GetFileName(path), (float)i / guids.Length);
                    TextureImporter ti = AssetImporter.GetAtPath(path) as TextureImporter;
                    if (ti == null) continue;
                    bool ok = ti.textureType == TextureImporterType.Sprite
                              && Mathf.Approximately(ti.spritePixelsPerUnit, 300f)
                              && !ti.mipmapEnabled
                              && ti.textureCompression == TextureImporterCompression.Uncompressed;
                    if (ok) continue;
                    ti.textureType = TextureImporterType.Sprite;
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
            SR(bg, "Bg", "d_bg", 0, Vector3.zero);

            SpriteRenderer horizonGlow = SR(bg, "HorizonGlow", "d_bg_horizon_glow", 1, Vector3.zero);
            BeatPulseAlpha hb = horizonGlow.gameObject.AddComponent<BeatPulseAlpha>();
            hb.from = 1f;
            hb.to = 0.35f;

            SR(bg, "Skyline", "d_bg_skyline", 2, Vector3.zero);
            SR(bg, "HorizonLine", "d_bg_horizon_line", 3, Vector3.zero);
            SR(bg, "FloorRays", "d_bg_floor_rays", 4, Vector3.zero);

            Transform floor = Group("FloorLines", bg, Vector3.zero);
            SpriteRenderer[] lines = new SpriteRenderer[5];
            for (int i = 0; i < lines.Length; i++)
                lines[i] = SR(floor, "Line" + i, "d_fx_floor_line", 5, DesignSpace.ToWorld(422f, 292f));
            floor.gameObject.AddComponent<FloorLines>().lines = lines;

            SR(bg, "RooftopLightL", "d_fx_rooftop_light_magenta", 6, DesignSpace.ToWorld(74f, 184f))
                .gameObject.AddComponent<BlinkLight>();
            SR(bg, "RooftopLightR", "d_fx_rooftop_light_cyan", 6, DesignSpace.ToWorld(770f, 170f))
                .gameObject.AddComponent<BlinkLight>().phase = 0.5f;
        }

        // ------------------------------------------------------------------ ring

        static Transform BuildRing(Transform parent)
        {
            Transform ring = Group("Ring", parent, DesignSpace.RingCenter);

            SR(ring, "Ticks", "d_ring_ticks", 10, Vector3.zero).gameObject.AddComponent<SpinForever>();

            foreach (string z in Zones)
            {
                Transform zone = Group("Zone_" + z, ring, Vector3.zero);
                SR(zone, "Base", "d_zone_" + z, 20, Vector3.zero);
                SpriteRenderer[] rows = new SpriteRenderer[4];
                for (int r = 0; r < 4; r++)
                    rows[r] = SR(zone, "LED" + (r + 1), "d_zone_" + z + "_led" + (r + 1), 21, Vector3.zero);
                SpriteRenderer flash = SR(zone, "ChargeFlash", "d_zone_" + z + "_flash", 22, Vector3.zero);
                flash.color = new Color(1f, 1f, 1f, 0f);

                LedChargeQuadrant charge = zone.gameObject.AddComponent<LedChargeQuadrant>();
                charge.rows = rows;
                charge.flash = flash;
                charge.offset = (z == "harmony" || z == "bass") ? 3f : 0f; // diagonal pairs alternate
            }

            SR(ring, "ChromaMagenta", "d_judge_ring_chroma_magenta", 30, new Vector3(-0.02f, 0f, 0f))
                .gameObject.AddComponent<ChromaGlitch>().sign = 1f;
            SR(ring, "ChromaCyan", "d_judge_ring_chroma_cyan", 30, new Vector3(0.02f, 0f, 0f))
                .gameObject.AddComponent<ChromaGlitch>().sign = -1f;
            BeatPulseAlpha jb = SR(ring, "JudgeRingGlow", "d_judge_ring_glow", 31, Vector3.zero)
                .gameObject.AddComponent<BeatPulseAlpha>();
            jb.from = 1f;
            jb.to = 0.15f;
            SR(ring, "JudgeRing", "d_judge_ring", 32, Vector3.zero);

            Transform core = Group("Core", ring, Vector3.zero);
            SR(core, "Glow", "d_core_glow", 40, Vector3.zero);
            SR(core, "Hex", "d_core", 41, Vector3.zero);
            SR(core, "Label", "d_text_core", 42, Vector3.zero);
            return core;
        }

        // ------------------------------------------------------------------ demo chart

        static void BuildChart(GameObject root, Transform core)
        {
            RingDemoChart chart = root.AddComponent<RingDemoChart>();
            chart.core = core;
            for (int i = 0; i < 4; i++)
            {
                string z = Zones[i];
                chart.noteBody[i] = S("d_note_" + z);
                chart.noteGlow[i] = S("d_note_" + z + "_glow");
                chart.noteTrail[i] = S("d_note_" + z + "_trail");
                chart.hitBurst[i] = S("d_hit_" + z);
            }
            chart.centerCue = S("d_center_cue");
            chart.centerWave = S("d_hit_center_hex");
            chart.textPerfect = S("d_text_perfect");
            chart.textPerfectCenter = S("d_text_perfect_center");
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
            Font orbitron = AssetDatabase.LoadAssetAtPath<Font>(FontDir + "/Orbitron-Bold.ttf");
            Font mono = AssetDatabase.LoadAssetAtPath<Font>(FontDir + "/JetBrainsMono-Medium.ttf");

            // Pause button (44x44 at left 18, vertically centered in the 60 px top bar)
            RectTransform pause = UI(hud, "PauseButton", new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(18f, -8f), new Vector2(44f, 44f));
            Image pauseImg = pause.gameObject.AddComponent<Image>();
            pauseImg.sprite = S("d_hud_pause");
            pause.gameObject.AddComponent<Button>().targetGraphic = pauseImg;

            // Song title
            RectTransform title = UI(hud, "Title", new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(74f, -16f), new Vector2(260f, 16f));
            Text titleText = AddText(title, builtin, 12, FontStyle.Bold, Hex("#E8F7FF"), TextAnchor.MiddleLeft);
            titleText.supportRichText = true;
            titleText.text = "云端信笺 <color=#FF2BD6>// HARD_10</color>";

            // Progress bar
            RectTransform track = UI(hud, "Progress", new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(74f, -36f), new Vector2(180f, 4f));
            track.gameObject.AddComponent<Image>().sprite = S("d_hud_progress_track");
            RectTransform fill = UI(track, "Fill", Vector2.zero, new Vector2(0f, 0.5f), Vector2.zero, Vector2.zero);
            fill.anchorMin = Vector2.zero;
            fill.anchorMax = Vector2.one;
            fill.offsetMin = Vector2.zero;
            fill.offsetMax = Vector2.zero;
            Image fillImg = fill.gameObject.AddComponent<Image>();
            fillImg.sprite = S("d_hud_progress_fill");
            fillImg.type = Image.Type.Filled;
            fillImg.fillMethod = Image.FillMethod.Horizontal;
            fillImg.fillOrigin = (int)Image.OriginHorizontal.Left;
            fillImg.fillAmount = 0.58f;

            // Score
            RectTransform score = UI(hud, "Score", new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(-18f, -10f), new Vector2(240f, 30f));
            Text scoreText = AddText(score, orbitron != null ? orbitron : builtin, 24, FontStyle.Normal, Hex("#E8F7FF"), TextAnchor.UpperRight);
            scoreText.text = "0524180";

            // Sync / combo label, bumps on every beat
            RectTransform sync = UI(hud, "Sync", new Vector2(1f, 1f), new Vector2(1f, 0.5f), new Vector2(-18f, -44f), new Vector2(240f, 16f));
            Text syncText = AddText(sync, mono != null ? mono : builtin, 11, FontStyle.Normal, Hex("#F5FF3B"), TextAnchor.MiddleRight);
            syncText.text = "SYNC 98.62% · 128 CMB";
            sync.gameObject.AddComponent<BeatBump>();
        }

        // ------------------------------------------------------------------ helpers

        static Sprite S(string name)
        {
            Sprite s = AssetDatabase.LoadAssetAtPath<Sprite>(SpriteDir + "/" + name + ".png");
            if (s == null) Debug.LogError("[RingSkinD] Missing sprite: " + name);
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
