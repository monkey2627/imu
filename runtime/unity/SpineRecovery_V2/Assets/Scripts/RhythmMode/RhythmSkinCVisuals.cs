using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

/// <summary>Presentation only: observes existing notes/results and draws the supplied Ring Skin C art.</summary>
public sealed class RhythmSkinCVisuals : MonoBehaviour
{
    public RhythmModeController controller;
    public RhythmRingLayout layout;
    public GameObject artwork;
    public Camera presentationCamera;
    public Sprite[] noteBodies = new Sprite[5];
    public Sprite[] noteGlows = new Sprite[5];
    public Sprite[] noteTrails = new Sprite[5];
    public Sprite[] hitRipples = new Sprite[5];
    public Sprite perfectText;
    public Sprite perfectCenterText;
    public SpriteRenderer[] zoneFlashes = new SpriteRenderer[4];
    public Transform core;
    public float designScale = 0.4335664f;
    [Tooltip("Foreground and interface scale. Leaves room for the glasses field of view without changing gameplay geometry; the background always covers the display.")]
    [Range(0.5f, 1f)] public float viewScale = 0.8f;
    public Material spriteMaterial;
    public Material displayMaterial;

    private RenderTexture displayTexture;
    private GameObject displayCanvas;
    private RawImage displayImage;
    private Transform background;
    private SpriteRenderer backgroundSky;
    private readonly List<InterfaceLayout> interfaceLayouts = new List<InterfaceLayout>();
    private bool interfaceCached;
    private float appliedInterfaceScale = -1f;

    private struct InterfaceLayout
    {
        public RectTransform rect;
        public Vector2 anchorMin, anchorMax, anchoredPosition;
        public Vector3 localScale;
    }

    private readonly float[] flashTimes = { -100f, -100f, -100f, -100f };
    private float coreHitTime = -100f;
    private bool wasVisible;

    private void LateUpdate()
    {
        bool visible = artwork != null && artwork.activeInHierarchy;
        if (presentationCamera != null)
        {
            EnsureDisplay();
            ApplyInterfaceScale();
            presentationCamera.enabled = visible;
            displayCanvas.SetActive(visible);
            // Keep the supplied 844 x 390 composition intact on wider and narrower displays.
            float scale = layout.transform.lossyScale.x * designScale;
            presentationCamera.orthographicSize = scale * Mathf.Max(1.95f, 4.22f / presentationCamera.aspect) / Mathf.Clamp(viewScale, 0.5f, 1f);
            presentationCamera.transform.position = layout.transform.position +
                new Vector3(0f, 0.2f * scale, -10f);
            FitBackgroundToView();
        }
        if (!visible && wasVisible)
        {
            StopAllCoroutines();
            foreach (Transform child in transform)
                if (child.name.StartsWith("SkinC_FX_")) Destroy(child.gameObject);
            for (int i = 0; i < flashTimes.Length; i++) flashTimes[i] = -100f;
            coreHitTime = -100f;
        }
        wasVisible = visible;
        for (int i = 0; i < zoneFlashes.Length; i++)
        {
            float age = Time.time - flashTimes[i];
            RingSkinC.SpriteAlpha.Set(zoneFlashes[i], age < 0.88f ? Mathf.Pow(1f - age / 0.88f, 2f) : 0f);
        }
        if (core != null)
        {
            float age = Time.time - coreHitTime;
            float bump = age < 0.04f ? Mathf.Lerp(1f, 1.22f, age / 0.04f) :
                Mathf.Lerp(1.22f, 1f, RingSkinC.Ease.OutQuad((age - 0.04f) / 0.56f));
            core.localScale = Vector3.one * bump;
        }
    }

    public void FitBackgroundToView()
    {
        if (artwork == null || presentationCamera == null) return;
        if (background == null)
        {
            background = artwork.transform.Find("Background");
            if (background != null)
                backgroundSky = background.Find("Sky")?.GetComponent<SpriteRenderer>();
        }
        if (backgroundSky == null || backgroundSky.sprite == null) return;

        // Cover the viewport with only the scenery. The ring, notes and UI retain View Scale.
        Vector3 sourceSize = backgroundSky.sprite.bounds.size;
        Vector3 parentScale = artwork.transform.lossyScale;
        Vector3 skyScale = backgroundSky.transform.localScale;
        float sourceWidth = Mathf.Abs(sourceSize.x * parentScale.x * skyScale.x);
        float sourceHeight = Mathf.Abs(sourceSize.y * parentScale.y * skyScale.y);
        if (sourceWidth <= 0f || sourceHeight <= 0f) return;
        float viewHeight = presentationCamera.orthographicSize * 2f;
        float coverScale = Mathf.Max(viewHeight * presentationCamera.aspect / sourceWidth, viewHeight / sourceHeight);
        background.localScale = Vector3.one * coverScale;
    }

    private void ApplyInterfaceScale()
    {
        if (!interfaceCached)
        {
            // Keep the original hierarchy and all button/controller references intact.
            foreach (Canvas canvas in FindObjectsOfType<Canvas>(true))
            {
                if (canvas.gameObject.scene != gameObject.scene || !canvas.isRootCanvas ||
                    canvas.gameObject == displayCanvas || canvas.renderMode != RenderMode.ScreenSpaceOverlay) continue;
                foreach (Transform child in canvas.transform)
                {
                    if (!(child is RectTransform rect)) continue;
                    interfaceLayouts.Add(new InterfaceLayout { rect = rect, anchorMin = rect.anchorMin,
                        anchorMax = rect.anchorMax, anchoredPosition = rect.anchoredPosition, localScale = rect.localScale });
                }
            }
            interfaceCached = true;
        }
        float scale = Mathf.Clamp(viewScale, 0.5f, 1f);
        if (Mathf.Approximately(appliedInterfaceScale, scale)) return;
        foreach (InterfaceLayout entry in interfaceLayouts)
        {
            if (entry.rect == null) continue;
            Vector2 anchorPivot = new Vector2(
                Mathf.Lerp(entry.anchorMin.x, entry.anchorMax.x, entry.rect.pivot.x),
                Mathf.Lerp(entry.anchorMin.y, entry.anchorMax.y, entry.rect.pivot.y));
            Vector2 inset = (Vector2.one * 0.5f - anchorPivot) * (1f - scale);
            // Shift the pivot toward the screen center while retaining the original anchor span.
            entry.rect.anchorMin = entry.anchorMin + inset;
            entry.rect.anchorMax = entry.anchorMax + inset;
            entry.rect.anchoredPosition = entry.anchoredPosition * scale;
            entry.rect.localScale = entry.localScale * scale;
        }
        appliedInterfaceScale = scale;
    }

    private void EnsureDisplay()
    {
        if (displayCanvas == null)
        {
            displayCanvas = new GameObject("RingSkinC_Display", typeof(Canvas));
            // Overlay canvases must not inherit the rhythm world's 1.5x transform scale.
            Canvas canvas = displayCanvas.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = -100;
            GameObject screen = new GameObject("Artwork", typeof(RectTransform), typeof(RawImage));
            screen.transform.SetParent(displayCanvas.transform, false);
            displayImage = screen.GetComponent<RawImage>();
            displayImage.raycastTarget = false;
            displayImage.material = displayMaterial;
            RectTransform rect = displayImage.rectTransform;
            rect.anchorMin = Vector2.zero;
            rect.anchorMax = Vector2.one;
            rect.offsetMin = rect.offsetMax = Vector2.zero;
        }
        int width = Mathf.Max(1, Screen.width), height = Mathf.Max(1, Screen.height);
        if (displayTexture == null || displayTexture.width != width || displayTexture.height != height)
        {
            ReleaseDisplayTexture();
            // The supplied preview composites translucent PNG layers in sRGB. Preserve that blending
            // in this isolated surface without changing the project's lighting or global color space.
            displayTexture = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
            displayTexture.name = "RingSkinC_sRGB_Composition";
            displayTexture.Create();
            displayImage.texture = displayTexture;
            presentationCamera.targetTexture = displayTexture;
            presentationCamera.aspect = (float)width / height;
        }
    }

    private void OnDestroy()
    {
        ReleaseDisplayTexture();
        if (displayCanvas != null) Destroy(displayCanvas);
    }

    private void OnDisable()
    {
        if (presentationCamera != null) presentationCamera.enabled = false;
        if (displayCanvas != null) displayCanvas.SetActive(false);
    }

    private void ReleaseDisplayTexture()
    {
        if (displayTexture == null) return;
        if (presentationCamera != null) presentationCamera.targetTexture = null;
        displayTexture.Release();
        Destroy(displayTexture);
        displayTexture = null;
    }

    public GameObject CreateNote(RhythmBodyTarget target, Vector3 gameplayScale)
    {
        int lane = (int)target;
        GameObject root = new GameObject("SkinC_Note");
        root.layer = gameObject.layer;
        Transform art = new GameObject("Visual").transform;
        art.SetParent(root.transform, false);
        art.gameObject.layer = root.layer;
        // Compensate inside the visual child; the original root scale and entry radius remain untouched.
        art.localScale = new Vector3(designScale / gameplayScale.x, designScale / gameplayScale.y, 1f);
        SpriteRenderer trail = MakeSprite("Trail", noteTrails[lane], art, 59);
        SpriteRenderer glow = MakeSprite("Glow", noteGlows[lane], art, 60);
        SpriteRenderer body = MakeSprite("Body", noteBodies[lane], art, 61);
        root.AddComponent<RhythmSkinCNoteVisual>().Initialize(controller, trail, glow, body);
        return root;
    }

    public void PlayResult(RhythmNoteResult result)
    {
        int lane = (int)result.Target;
        if (!result.IsHit)
        {
            GameObject miss = new GameObject("SkinC_FX_Miss");
            miss.layer = gameObject.layer;
            miss.transform.SetParent(transform, false);
            miss.transform.localPosition = result.LocalHitPosition;
            TextMeshPro text = miss.AddComponent<TextMeshPro>();
            text.text = "MISS";
            text.alignment = TextAlignmentOptions.Center;
            text.fontSize = 3f;
            text.color = new Color(0.85f, 0.86f, 1f);
            text.rectTransform.sizeDelta = new Vector2(1f, 0.3f);
            text.GetComponent<MeshRenderer>().sortingOrder = 80;
            StartCoroutine(FadeMiss(text));
            return;
        }
        if (lane < zoneFlashes.Length) flashTimes[lane] = Time.time;
        else coreHitTime = Time.time;
        StartCoroutine(Ripple(hitRipples[lane], result.LocalHitPosition));
        Vector2[] textPositions = { new Vector2(-1.68f, 1.35f), new Vector2(1.68f, 1.35f),
            new Vector2(-1.68f, -1.33f), new Vector2(1.68f, -1.33f), new Vector2(0f, 0.95f) };
        StartCoroutine(JudgementText(lane == 4 ? perfectCenterText : perfectText,
            textPositions[lane] * designScale));
    }

    private IEnumerator Ripple(Sprite sprite, Vector3 position)
    {
        SpriteRenderer r = MakeSprite("SkinC_FX_Ripple", sprite, transform, 70);
        r.transform.localPosition = position;
        for (float age = 0; age < 0.64f; age += Time.deltaTime)
        {
            float k = RingSkinC.Ease.OutQuad(age / 0.64f);
            r.transform.localScale = Vector3.one * (Mathf.Lerp(0.5f, 1.9f, k) * designScale);
            RingSkinC.SpriteAlpha.Set(r, 1f - k);
            yield return null;
        }
        Destroy(r.gameObject);
    }

    private IEnumerator JudgementText(Sprite sprite, Vector3 position)
    {
        SpriteRenderer r = MakeSprite("SkinC_FX_Perfect", sprite, transform, 80);
        float[,] keys = { { 0f, 0f, 0.8f, 6f }, { 0.08f, 1f, 1.15f, 0f },
            { 0.24f, 1f, 1f, 0f }, { 0.8f, 1f, 1f, -6f }, { 1.04f, 0f, 1f, -8f } };
        for (float age = 0; age < 1.04f; age += Time.deltaTime)
        {
            for (int i = 0; i < 4; i++)
            {
                if (age < keys[i, 0] || age > keys[i + 1, 0]) continue;
                float u = RingSkinC.Ease.OutQuad((age - keys[i, 0]) / (keys[i + 1, 0] - keys[i, 0]));
                RingSkinC.SpriteAlpha.Set(r, Mathf.Lerp(keys[i, 1], keys[i + 1, 1], u));
                r.transform.localScale = Vector3.one * (designScale * Mathf.Lerp(keys[i, 2], keys[i + 1, 2], u));
                r.transform.localPosition = position + Vector3.down *
                    (designScale * 0.01f * Mathf.Lerp(keys[i, 3], keys[i + 1, 3], u));
                break;
            }
            yield return null;
        }
        Destroy(r.gameObject);
    }

    private IEnumerator FadeMiss(TextMeshPro text)
    {
        for (float age = 0; age < 0.65f; age += Time.deltaTime)
        {
            text.alpha = 1f - age / 0.65f;
            yield return null;
        }
        Destroy(text.gameObject);
    }

    private SpriteRenderer MakeSprite(string name, Sprite sprite, Transform parent, int order)
    {
        GameObject go = new GameObject(name);
        go.layer = parent.gameObject.layer;
        go.transform.SetParent(parent, false);
        SpriteRenderer renderer = go.AddComponent<SpriteRenderer>();
        renderer.sprite = sprite;
        renderer.sharedMaterial = spriteMaterial;
        renderer.sortingOrder = order;
        return renderer;
    }
}
