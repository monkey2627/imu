using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;

namespace RingBridge.EditorTools
{
    /// <summary>
    /// Menu: Tools > Ring Bridge > Build Prefab + Demo Scene
    /// 1. Sets every texture in Assets/RingBridge/Sprites to Sprite, 200 PPU (2x art, 1 design px = 0.01 unit).
    /// 2. Assembles background, the three ring sections with masked progress arcs, centre figure and HUD.
    /// 3. Saves Prefabs/RingBridge.prefab and Scenes/RingBridge_Demo.unity.
    /// </summary>
    public static class RingBridgeBuilder
    {
        const string Root = "Assets/RingBridge";
        const string SpriteDir = Root + "/Sprites";
        const string FontDir = Root + "/Fonts";
        const string PrefabPath = Root + "/Prefabs/RingBridge.prefab";
        const string ScenePath = Root + "/Scenes/RingBridge_Demo.unity";
        const float RingX = 422f, RingY = 215f;
        const float Pad = 3.5f;

        [MenuItem("Tools/Ring Bridge/Build Prefab + Demo Scene")]
        public static void Build()
        {
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
            GameObject root = CreateArtwork();
            Directory.CreateDirectory(Root + "/Prefabs");
            GameObject prefab = PrefabUtility.SaveAsPrefabAsset(root, PrefabPath);
            Object.DestroyImmediate(root);

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            GameObject camGo = new GameObject("Main Camera");
            camGo.tag = "MainCamera";
            Camera cam = camGo.AddComponent<Camera>();
            cam.orthographic = true;
            cam.orthographicSize = 1.95f;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = Hex("#0F1034");
            camGo.transform.position = new Vector3(0f, 0f, -10f);
            PrefabUtility.InstantiatePrefab(prefab);
            Directory.CreateDirectory(Root + "/Scenes");
            EditorSceneManager.SaveScene(scene, ScenePath);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
        }

        public static GameObject CreateArtwork()
        {
            ConfigureSprites();

            GameObject root = new GameObject("RingBridge");
            RingBridgeChart chart = root.AddComponent<RingBridgeChart>();

            BuildBackground(root.transform);
            Transform ring = Group("Ring", root.transform);
            chart.sections = new BridgeSection[]
            {
                BuildSection(ring, "lift", "抬起", 0.5f, 2.0f, 180f, 270f, 10),
                BuildSection(ring, "hold", "保持", 2.0f, 5.0f, 270f, 360f, 20),
                BuildSection(ring, "lower", "放下", 5.0f, 6.5f, 0f, 180f, 30),
            };
            BuildCentre(ring, chart);
            chart.combo = BuildHud(root.transform);
            return root;
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
                    EditorUtility.DisplayProgressBar("Ring Bridge", "Importing " + Path.GetFileName(path), (float)i / guids.Length);
                    TextureImporter ti = AssetImporter.GetAtPath(path) as TextureImporter;
                    if (ti == null) continue;
                    bool ok = ti.textureType == TextureImporterType.Sprite
                              && ti.textureShape == TextureImporterShape.Texture2D
                              && Mathf.Approximately(ti.spritePixelsPerUnit, 200f)
                              && !ti.mipmapEnabled
                              && ti.textureCompression == TextureImporterCompression.Uncompressed;
                    if (ok) continue;
                    ti.textureType = TextureImporterType.Sprite;
                    ti.textureShape = TextureImporterShape.Texture2D;
                    ti.spriteImportMode = SpriteImportMode.Single;
                    ti.spritePixelsPerUnit = 200f;
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
            Transform bg = Group("Background", parent);
            Place(bg, "rb_bg");
            for (int i = 0; i < BridgeLayout.Sparkles.Length; i++)
            {
                Twinkle tw = Place(bg, "rb_sparkle_" + i).gameObject.AddComponent<Twinkle>();
                tw.period = BridgeLayout.Sparkles[i].period;
                tw.phase = BridgeLayout.Sparkles[i].phase;
            }
            Twinkle glow = Place(bg, "rb_star_glow").gameObject.AddComponent<Twinkle>();
            glow.period = 3f;

            MeteorLoop meteor = Place(bg, "rb_meteor").gameObject.AddComponent<MeteorLoop>();
            meteor.headToSpriteCenterPx = new Vector2(27f, -10f);
            Hidden(meteor.GetComponent<SpriteRenderer>());

            Place(bg, "rb_lanterns").gameObject.AddComponent<FloatBob>();
            Place(bg, "rb_spin_ring").gameObject.AddComponent<SpinForever>().secondsPerTurn = 40f;
        }

        // ------------------------------------------------------------------ ring sections

        /// <summary>
        /// Sorting band per section (b .. b+6): fill, static, track, progress (masked), done flash, head, figure+label.
        /// The half-disc SpriteMask only affects order b+3, so each section's progress has its own mask.
        /// </summary>
        static BridgeSection BuildSection(Transform parent, string key, string label, float start, float end, float a0, float a1, int band)
        {
            Transform g = Group("Section_" + key + " (" + label + ")", parent);
            BridgeSection s = new BridgeSection();
            s.name = label;
            s.start = start;
            s.end = end;
            s.startAngle = a0 + Pad;
            s.sweep = (a1 - a0) - 2f * Pad;

            s.fill = Place(g, "rb_" + key + "_fill");
            Place(g, "rb_" + key + "_static");
            s.track = Place(g, "rb_" + key + "_track");

            s.progress = Place(g, "rb_" + key + "_prog");
            s.progress.maskInteraction = SpriteMaskInteraction.VisibleInsideMask;
            GameObject maskGo = new GameObject("ProgressMask (rotates to the leading edge)");
            maskGo.transform.SetParent(g, false);
            maskGo.transform.localPosition = DesignSpace.ToWorld(RingX, RingY);
            SpriteMask mask = maskGo.AddComponent<SpriteMask>();
            mask.sprite = S("rb_mask_half");
            mask.isCustomRangeActive = true;
            mask.backSortingOrder = band + 2;   // affects back < order <= front
            mask.frontSortingOrder = band + 3;
            s.progressMask = maskGo.transform;

            s.done = Hidden(Place(g, "rb_" + key + "_done"));

            GameObject pivot = new GameObject("HeadPivot (ring centre)");
            pivot.transform.SetParent(g, false);
            pivot.transform.localPosition = DesignSpace.ToWorld(RingX, RingY);
            LayerInfo h = BridgeLayout.Layers["rb_" + key + "_head"];
            SpriteRenderer head = PlaceAt(pivot.transform, "Head", "rb_" + key + "_head", h.x, h.y);
            head.transform.localPosition = DesignSpace.ToWorld(h.x, h.y) - DesignSpace.ToWorld(RingX, RingY);
            s.headPivot = pivot.transform;
            s.head = Hidden(head);

            s.ink = Place(g, "rb_" + key + "_ink");
            return s;
        }

        // ------------------------------------------------------------------ centre

        static void BuildCentre(Transform parent, RingBridgeChart chart)
        {
            Transform c = Group("Centre", parent);
            LayerInfo d = BridgeLayout.Layers["rb_disc"];
            GameObject discGo = new GameObject("Disc (beats on each completion)");
            discGo.transform.SetParent(c, false);
            discGo.transform.localPosition = DesignSpace.ToWorld(d.x, d.y);
            Transform disc = discGo.transform;
            PlaceAt(disc, "rb_disc", "rb_disc", DesignSpace.CenterX, DesignSpace.CenterY);

            Sprite[] frames = new Sprite[BridgeLayout.PoseFrames];
            for (int i = 0; i < frames.Length; i++) frames[i] = S("rb_pose_" + i.ToString("00"));
            LayerInfo p = BridgeLayout.Layers["rb_pose_00"];
            SpriteRenderer pose = PlaceAt(disc, "Pose (13 frames, hips down -> up)", "rb_pose_00",
                DesignSpace.CenterX + (p.x - d.x), DesignSpace.CenterY + (p.y - d.y));
            LayerInfo r = BridgeLayout.Layers["rb_rays"];
            SpriteRenderer rays = PlaceAt(disc, "HoldRays", "rb_rays",
                DesignSpace.CenterX + (r.x - d.x), DesignSpace.CenterY + (r.y - d.y));

            chart.disc = disc;
            chart.pose = pose;
            chart.poseFrames = frames;
            chart.rays = Hidden(rays);
        }

        // ------------------------------------------------------------------ HUD

        static Transform BuildHud(Transform parent)
        {
            Place(Group("HUD_Sprites", parent), "rb_hud_pause");

            GameObject canvasGo = new GameObject("HUD", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            canvasGo.transform.SetParent(parent, false);
            canvasGo.GetComponent<Canvas>().renderMode = RenderMode.ScreenSpaceOverlay;
            CanvasScaler scaler = canvasGo.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(844f, 390f); // 1 canvas unit = 1 design px
            scaler.screenMatchMode = CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
            scaler.matchWidthOrHeight = 1f;

            Font fredoka = AssetDatabase.LoadAssetAtPath<Font>(FontDir + "/Fredoka-SemiBold.ttf");
            AddText(canvasGo.transform, "Score", "0,524,180", fredoka, 24, Color.white, 826f, 10f, 260f, 28f);
            return AddText(canvasGo.transform, "Combo", "128 COMBO", fredoka, 11, Hex("#D8DCFF"), 826f, 36f, 160f, 16f);
        }

        // ------------------------------------------------------------------ helpers

        static Sprite S(string name)
        {
            Sprite s = AssetDatabase.LoadAssetAtPath<Sprite>(SpriteDir + "/" + name + ".png");
            if (s == null) Debug.LogError("[RingBridge] Missing sprite: " + name);
            return s;
        }

        static Transform Group(string name, Transform parent)
        {
            GameObject g = new GameObject(name);
            g.transform.SetParent(parent, false);
            return g.transform;
        }

        static SpriteRenderer Place(Transform parent, string sprite)
        {
            LayerInfo info = BridgeLayout.Layers[sprite];
            return PlaceAt(parent, sprite, sprite, info.x, info.y);
        }

        static SpriteRenderer PlaceAt(Transform parent, string name, string sprite, float x, float y)
        {
            GameObject g = new GameObject(name);
            g.transform.SetParent(parent, false);
            g.transform.localPosition = DesignSpace.ToWorld(x, y);
            SpriteRenderer r = g.AddComponent<SpriteRenderer>();
            r.sprite = S(sprite);
            LayerInfo info;
            r.sortingOrder = BridgeLayout.Layers.TryGetValue(sprite, out info) ? info.order : 0;
            return r;
        }

        static SpriteRenderer Hidden(SpriteRenderer r)
        {
            r.color = new Color(1f, 1f, 1f, 0f);
            return r;
        }

        /// <summary>Right-aligned UI text whose right edge sits at design x, top at design y.</summary>
        static Transform AddText(Transform parent, string name, string text, Font font, int size, Color color, float right, float top, float w, float h)
        {
            GameObject g = new GameObject(name, typeof(RectTransform));
            g.transform.SetParent(parent, false);
            RectTransform rt = (RectTransform)g.transform;
            rt.anchorMin = new Vector2(0.5f, 0.5f);
            rt.anchorMax = new Vector2(0.5f, 0.5f);
            rt.pivot = new Vector2(1f, 0.5f); // bumps scale from the right edge
            rt.anchoredPosition = new Vector2(right - DesignSpace.CenterX, -(top + h * 0.5f - DesignSpace.CenterY));
            rt.sizeDelta = new Vector2(w, h);
            Text t = g.AddComponent<Text>();
            t.font = font;
            t.fontSize = size;
            t.color = color;
            t.alignment = TextAnchor.UpperRight;
            t.horizontalOverflow = HorizontalWrapMode.Overflow;
            t.verticalOverflow = VerticalWrapMode.Overflow;
            t.raycastTarget = false;
            t.text = text;
            return rt;
        }

        static Color Hex(string hex)
        {
            Color c;
            return ColorUtility.TryParseHtmlString(hex, out c) ? c : Color.magenta;
        }
    }
}
