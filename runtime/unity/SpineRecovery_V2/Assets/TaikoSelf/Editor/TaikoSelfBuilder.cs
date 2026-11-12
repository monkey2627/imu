using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;

namespace TaikoSelf.EditorTools
{
    /// <summary>
    /// Menu: Tools > Taiko Self > Build Prefab + Demo Scene
    /// 1. Sets every texture in Assets/TaikoSelf/Sprites to Sprite, 200 PPU (2x art, 1 design px = 0.01 unit).
    /// 2. Assembles background, drum head sections, masks, stars, notes, dust, centre and HUD.
    /// 3. Saves Prefabs/TaikoSelf.prefab and Scenes/TaikoSelf_Demo.unity (driven by SimulatedPatient).
    /// </summary>
    public static class TaikoSelfBuilder
    {
        const string Root = "Assets/TaikoSelf";
        const string SpriteDir = Root + "/Sprites";
        const string FontDir = Root + "/Fonts";
        const string PrefabPath = Root + "/Prefabs/TaikoSelf.prefab";
        const string ScenePath = Root + "/Scenes/TaikoSelf_Demo.unity";

        [MenuItem("Tools/Taiko Self/Build Prefab + Demo Scene")]
        public static void Build()
        {
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
            ConfigureSprites();

            GameObject root = new GameObject("TaikoSelf");
            SelfPacedDrum drum = root.AddComponent<SelfPacedDrum>();
            root.AddComponent<SimulatedPatient>();

            BuildBackground(root.transform);
            Transform head = Group("DrumHead", root.transform);
            Place(head, "ts_bases");
            BuildLift(head, drum);
            BuildHold(head, drum);
            BuildReturn(head, drum);
            BuildCentre(head, drum);
            BuildHud(root.transform, drum);

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
            cam.backgroundColor = Hex("#0F1034");
            camGo.transform.position = new Vector3(0f, 0f, -10f);
            PrefabUtility.InstantiatePrefab(prefab);

            Directory.CreateDirectory(Root + "/Scenes");
            EditorSceneManager.SaveScene(scene, ScenePath);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            EditorUtility.DisplayDialog("Taiko Self", "Built:\n" + PrefabPath + "\n" + ScenePath +
                "\n\nPress Play: SimulatedPatient drives one ideal repetition.\nFor real use, disable SimulatedPatient and write SelfPacedDrum.height from the sensor.", "OK");
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
                    EditorUtility.DisplayProgressBar("Taiko Self", "Importing " + Path.GetFileName(path), (float)i / guids.Length);
                    TextureImporter ti = AssetImporter.GetAtPath(path) as TextureImporter;
                    if (ti == null) continue;
                    bool ok = ti.textureType == TextureImporterType.Sprite
                              && Mathf.Approximately(ti.spritePixelsPerUnit, 200f)
                              && !ti.mipmapEnabled
                              && ti.textureCompression == TextureImporterCompression.Uncompressed;
                    if (ok) continue;
                    ti.textureType = TextureImporterType.Sprite;
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
            Place(bg, "ts_bg");
            for (int i = 0; i < SelfLayout.Sparkles.Length; i++)
            {
                Twinkle tw = Place(bg, "ts_sparkle_" + i).gameObject.AddComponent<Twinkle>();
                tw.period = SelfLayout.Sparkles[i].period;
                tw.phase = SelfLayout.Sparkles[i].phase;
            }
            Place(bg, "ts_star_glow").gameObject.AddComponent<Twinkle>().period = 3f;
            MeteorLoop meteor = Place(bg, "ts_meteor").gameObject.AddComponent<MeteorLoop>();
            meteor.headToSpriteCenterPx = new Vector2(SelfLayout.MeteorCenterOffsetX, SelfLayout.MeteorCenterOffsetY);
            Hidden(meteor.GetComponent<SpriteRenderer>());
            Place(bg, "ts_lanterns").gameObject.AddComponent<FloatBob>();
            Place(bg, "ts_spin_ring").gameObject.AddComponent<SpinForever>().secondsPerTurn = 40f;
        }

        // ------------------------------------------------------------------ sections

        static void BuildLift(Transform parent, SelfPacedDrum drum)
        {
            Transform g = Group("Lift (抬起, pink)", parent);
            drum.liftFill = Hidden(Place(g, "ts_lift_fill"));
            drum.liftProgress = Hidden(Place(g, "ts_lift_prog"));
            drum.liftProgress.maskInteraction = SpriteMaskInteraction.VisibleInsideMask;
            // half-disc mask rotating to the leading edge reveals the progress arc (affects order 12 only)
            drum.liftMask = Mask(g, "LiftProgressMask", "ts_mask_half", SelfLayout.DrumX, SelfLayout.DrumY, 11, 12).transform;
        }

        static void BuildHold(Transform parent, SelfPacedDrum drum)
        {
            Transform g = Group("Hold (保持, gold)", parent);
            drum.holdFill = Hidden(Place(g, "ts_hold_fill"));
            drum.stars = new SpriteRenderer[SelfLayout.Stars.Length];
            for (int k = 0; k < SelfLayout.Stars.Length; k++)
            {
                Vector2 p = SelfLayout.Stars[k];
                SpriteRenderer slot = PlaceAt(g, "StarSlot" + k, "ts_star_slot", p.x, p.y);
                slot.sortingOrder = 13;
                SpriteRenderer lit = PlaceAt(g, "Star" + k, "ts_star_lit", p.x, p.y);
                lit.sortingOrder = 14;
                drum.stars[k] = Hidden(lit);
            }
        }

        static void BuildReturn(Transform parent, SelfPacedDrum drum)
        {
            Transform g = Group("Return (慢慢放下, lilac)", parent);
            drum.returnFill = Hidden(Place(g, "ts_ret_fill"));
            // comet spans the whole progress; it rotates with the dot and a static sector mask hides what lies before the arc start
            Mask(g, "ReturnClipMask", "ts_ret_clip", SelfLayout.DrumX, SelfLayout.DrumY, 14, 15);
            drum.comet = Hidden(Place(g, "ts_ret_comet"));
            drum.comet.maskInteraction = SpriteMaskInteraction.VisibleInsideMask;

            int n = SelfLayout.Notes.Length;
            drum.notes = new SpriteRenderer[n];
            drum.noteRings = new SpriteRenderer[n];
            for (int i = 0; i < n; i++)
            {
                Vector2 p = SelfLayout.Notes[i];
                drum.notes[i] = PlaceAt(g, "Note" + i, "ts_note", p.x, p.y);
                drum.notes[i].sortingOrder = 16;
                SpriteAlpha.Set(drum.notes[i], 0.25f);
                drum.noteRings[i] = Hidden(PlaceAt(g, "NoteRing" + i, "ts_note_ring", p.x, p.y));
                drum.noteRings[i].sortingOrder = 17;
            }

            GameObject pivot = new GameObject("DotPivot (drum centre)");
            pivot.transform.SetParent(g, false);
            pivot.transform.localPosition = DesignSpace.ToWorld(SelfLayout.DrumX, SelfLayout.DrumY);
            LayerInfo d = SelfLayout.Layers["ts_ret_dot"];
            SpriteRenderer dot = PlaceAt(pivot.transform, "Dot", "ts_ret_dot", d.x, d.y);
            dot.transform.localPosition = DesignSpace.ToWorld(d.x, d.y) - DesignSpace.ToWorld(SelfLayout.DrumX, SelfLayout.DrumY);
            drum.dotPivot = pivot.transform;
            drum.dot = Hidden(dot);

            int m = SelfLayout.Dust.Length;
            drum.dust = new SpriteRenderer[m];
            for (int i = 0; i < m; i++)
            {
                Vector2 p = SelfLayout.Dust[i];
                drum.dust[i] = Hidden(PlaceAt(g, "Dust" + i, "ts_dust_" + SelfLayout.DustVariant[i], p.x, p.y));
                drum.dust[i].sortingOrder = 19;
            }
        }

        static void BuildCentre(Transform parent, SelfPacedDrum drum)
        {
            Transform g = Group("Centre", parent);
            drum.coreGlow = Hidden(Place(g, "ts_core_glow"));
            drum.core = Place(g, "ts_core").transform;
            Place(g, "ts_sheen");
            drum.constellation = Hidden(Place(g, "ts_const"));
        }

        // ------------------------------------------------------------------ HUD

        static void BuildHud(Transform parent, SelfPacedDrum drum)
        {
            Transform s = Group("HUD_Sprites", parent);
            Place(s, "ts_hud_pause");
            drum.captionLift = Hidden(Place(s, "ts_cap_lift"));
            drum.captionHold = Hidden(Place(s, "ts_cap_hold"));
            drum.captionReturn = Hidden(Place(s, "ts_cap_ret"));

            GameObject canvasGo = new GameObject("HUD", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            canvasGo.transform.SetParent(parent, false);
            canvasGo.GetComponent<Canvas>().renderMode = RenderMode.ScreenSpaceOverlay;
            CanvasScaler scaler = canvasGo.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(844f, 390f);
            scaler.screenMatchMode = CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
            scaler.matchWidthOrHeight = 1f;
            Font fredoka = AssetDatabase.LoadAssetAtPath<Font>(FontDir + "/Fredoka-SemiBold.ttf");
            AddText(canvasGo.transform, "Score", "0,524,180", fredoka, 24, Color.white, 826f, 10f, 260f, 28f);
            AddText(canvasGo.transform, "Combo", "128 COMBO", fredoka, 11, Hex("#D8DCFF"), 826f, 36f, 160f, 16f);
        }

        // ------------------------------------------------------------------ helpers

        static Sprite S(string name)
        {
            Sprite s = AssetDatabase.LoadAssetAtPath<Sprite>(SpriteDir + "/" + name + ".png");
            if (s == null) Debug.LogError("[TaikoSelf] Missing sprite: " + name);
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
            LayerInfo info = SelfLayout.Layers[sprite];
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
            r.sortingOrder = SelfLayout.Layers.TryGetValue(sprite, out info) ? info.order : 0;
            return r;
        }

        static SpriteMask Mask(Transform parent, string name, string sprite, float x, float y, int back, int front)
        {
            GameObject g = new GameObject(name);
            g.transform.SetParent(parent, false);
            g.transform.localPosition = DesignSpace.ToWorld(x, y);
            SpriteMask m = g.AddComponent<SpriteMask>();
            m.sprite = S(sprite);
            m.isCustomRangeActive = true;
            m.backSortingOrder = back;   // affects back < order <= front
            m.frontSortingOrder = front;
            return m;
        }

        static SpriteRenderer Hidden(SpriteRenderer r)
        {
            r.color = new Color(1f, 1f, 1f, 0f);
            return r;
        }

        static void AddText(Transform parent, string name, string text, Font font, int size, Color color, float right, float top, float w, float h)
        {
            GameObject g = new GameObject(name, typeof(RectTransform));
            g.transform.SetParent(parent, false);
            RectTransform rt = (RectTransform)g.transform;
            rt.anchorMin = new Vector2(0.5f, 0.5f);
            rt.anchorMax = new Vector2(0.5f, 0.5f);
            rt.pivot = new Vector2(1f, 0.5f);
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
        }

        static Color Hex(string hex)
        {
            Color c;
            return ColorUtility.TryParseHtmlString(hex, out c) ? c : Color.magenta;
        }
    }
}
