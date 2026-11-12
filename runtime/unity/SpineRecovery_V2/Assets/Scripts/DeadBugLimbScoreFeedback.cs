using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

// Tints the four limb indicators from the calibrated anatomical limb scores.
// The color palette is owned by the road progress bar so both visuals always
// use the same score thresholds and colors.
public sealed class DeadBugLimbScoreFeedback : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private DeadBugGamingPoseScorer poseScorer;
    [SerializeField] private DeadBugScoreRoadProgressBar scoreColorSource;
    [SerializeField] private Canvas gamingCanvas;
    [SerializeField] private Transform leftArmStar;
    [SerializeField] private Transform rightArmStar;
    [SerializeField] private Transform leftLegStar;
    [SerializeField] private Transform rightLegStar;

    [Header("Renderer Color Properties")]
    [SerializeField] private string[] colorPropertyNames =
    {
        "_TintColor",
        "_BaseColor",
        "_Color"
    };

    [Header("Correction Hints")]
    [SerializeField] private Vector3 hintLocalOffset = new Vector3(0f, -72f, 0f);
    [SerializeField] private Vector2 hintSize = new Vector2(145f, 42f);
    [Min(10f)] [SerializeField] private float hintFontSize = 18f;
    [Min(0f)] [SerializeField] private float hintCanvasPadding = 12f;
    [SerializeField] private Color hintTextColor = Color.white;

    private sealed class RendererColorTarget
    {
        public Renderer renderer;
        public int materialIndex;
        public string colorPropertyName;
        public float baseAlpha;
        public bool setEmission;
        public float emissionMultiplier;
        public float emissionAlpha;
    }

    private readonly List<RendererColorTarget> leftArmTargets = new List<RendererColorTarget>();
    private readonly List<RendererColorTarget> rightArmTargets = new List<RendererColorTarget>();
    private readonly List<RendererColorTarget> leftLegTargets = new List<RendererColorTarget>();
    private readonly List<RendererColorTarget> rightLegTargets = new List<RendererColorTarget>();
    private ParticleSystem[] leftArmParticles = new ParticleSystem[0];
    private ParticleSystem[] rightArmParticles = new ParticleSystem[0];
    private ParticleSystem[] leftLegParticles = new ParticleSystem[0];
    private ParticleSystem[] rightLegParticles = new ParticleSystem[0];
    private MaterialPropertyBlock propertyBlock;
    private Color leftArmColor;
    private Color rightArmColor;
    private Color leftLegColor;
    private Color rightLegColor;
    private bool hasLeftArmColor;
    private bool hasRightArmColor;
    private bool hasLeftLegColor;
    private bool hasRightLegColor;
    private bool leftArmGreen;
    private bool rightArmGreen;
    private bool leftLegGreen;
    private bool rightLegGreen;
    private TMP_Text leftArmHint;
    private TMP_Text rightArmHint;
    private TMP_Text leftLegHint;
    private TMP_Text rightLegHint;

    private void Awake()
    {
        ResolveReferences();
        CacheTargets();
        CreateCorrectionHints();
    }

    private void OnEnable()
    {
        ResolveReferences();
        if (propertyBlock == null)
        {
            CacheTargets();
        }
        Subscribe();
    }

    private void OnDisable()
    {
        Unsubscribe();
        StopGreenParticles(leftArmParticles, ref leftArmGreen);
        StopGreenParticles(rightArmParticles, ref rightArmGreen);
        StopGreenParticles(leftLegParticles, ref leftLegGreen);
        StopGreenParticles(rightLegParticles, ref rightLegGreen);
    }

    private void LateUpdate()
    {
        UpdateHintPositions();
    }

    private void HandleSessionStarted()
    {
        ApplyAllScores(default(AnatomicalLimbScores));
    }

    private void HandleLimbScoresChanged(AnatomicalLimbScores scores)
    {
        ApplyAllScores(scores);
    }

    private void ApplyAllScores(AnatomicalLimbScores scores)
    {
        ApplyScore(
            leftArmTargets,
            scores.leftArm.score,
            ref hasLeftArmColor,
            ref leftArmColor);
        ApplyScore(
            rightArmTargets,
            scores.rightArm.score,
            ref hasRightArmColor,
            ref rightArmColor);
        ApplyScore(
            leftLegTargets,
            scores.leftLeg.score,
            ref hasLeftLegColor,
            ref leftLegColor);
        ApplyScore(
            rightLegTargets,
            scores.rightLeg.score,
            ref hasRightLegColor,
            ref rightLegColor);
        ApplyGreenParticles(leftArmParticles, IsGreenScore(scores.leftArm.score), ref leftArmGreen);
        ApplyGreenParticles(rightArmParticles, IsGreenScore(scores.rightArm.score), ref rightArmGreen);
        ApplyGreenParticles(leftLegParticles, IsGreenScore(scores.leftLeg.score), ref leftLegGreen);
        ApplyGreenParticles(rightLegParticles, IsGreenScore(scores.rightLeg.score), ref rightLegGreen);
        UpdateHint(leftArmHint, scores.leftArm, AnatomicalLimb.LeftArm);
        UpdateHint(rightArmHint, scores.rightArm, AnatomicalLimb.RightArm);
        UpdateHint(leftLegHint, scores.leftLeg, AnatomicalLimb.LeftLeg);
        UpdateHint(rightLegHint, scores.rightLeg, AnatomicalLimb.RightLeg);
    }

    private void ApplyScore(
        List<RendererColorTarget> targets,
        float score,
        ref bool hasAppliedColor,
        ref Color appliedColor)
    {
        Color color = scoreColorSource != null
            ? scoreColorSource.GetScoreColor(score)
            : GetFallbackScoreColor(score);
        if (hasAppliedColor && appliedColor == color)
        {
            return;
        }

        hasAppliedColor = true;
        appliedColor = color;

        foreach (RendererColorTarget target in targets)
        {
            if (target.renderer == null)
            {
                continue;
            }

            target.renderer.GetPropertyBlock(propertyBlock, target.materialIndex);
            Color rendererColor = color;
            rendererColor.a *= target.baseAlpha;
            propertyBlock.SetColor(target.colorPropertyName, rendererColor);

            if (target.setEmission)
            {
                Color emission = color * target.emissionMultiplier;
                emission.a = target.emissionAlpha;
                propertyBlock.SetColor("_EmissionColor", emission);
            }

            target.renderer.SetPropertyBlock(propertyBlock, target.materialIndex);
            propertyBlock.Clear();
        }
    }

    private void CacheTargets()
    {
        propertyBlock = new MaterialPropertyBlock();
        CacheTarget(leftArmStar, leftArmTargets);
        CacheTarget(rightArmStar, rightArmTargets);
        CacheTarget(leftLegStar, leftLegTargets);
        CacheTarget(rightLegStar, rightLegTargets);
        leftArmParticles = CacheParticles(leftArmStar);
        rightArmParticles = CacheParticles(rightArmStar);
        leftLegParticles = CacheParticles(leftLegStar);
        rightLegParticles = CacheParticles(rightLegStar);
    }

    private static ParticleSystem[] CacheParticles(Transform root)
    {
        return root != null
            ? root.GetComponentsInChildren<ParticleSystem>(true)
            : new ParticleSystem[0];
    }

    private void CacheTarget(Transform root, List<RendererColorTarget> destination)
    {
        destination.Clear();
        if (root == null)
        {
            return;
        }

        Renderer[] renderers = root.GetComponentsInChildren<Renderer>(true);
        foreach (Renderer targetRenderer in renderers)
        {
            Material[] materials = targetRenderer.sharedMaterials;
            for (int materialIndex = 0; materialIndex < materials.Length; materialIndex++)
            {
                Material material = materials[materialIndex];
                if (material == null)
                {
                    continue;
                }

                string colorPropertyName = FindColorProperty(material);
                if (string.IsNullOrEmpty(colorPropertyName))
                {
                    continue;
                }

                Color baseColor = material.GetColor(colorPropertyName);
                Color emission = material.HasProperty("_EmissionColor")
                    ? material.GetColor("_EmissionColor")
                    : Color.black;
                float emissionMultiplier = Mathf.Max(emission.r, Mathf.Max(emission.g, emission.b));
                destination.Add(new RendererColorTarget
                {
                    renderer = targetRenderer,
                    materialIndex = materialIndex,
                    colorPropertyName = colorPropertyName,
                    baseAlpha = baseColor.a,
                    setEmission = emissionMultiplier > 0.001f,
                    emissionMultiplier = emissionMultiplier,
                    emissionAlpha = emission.a
                });
            }
        }
    }

    private string FindColorProperty(Material material)
    {
        foreach (string propertyName in colorPropertyNames)
        {
            if (!string.IsNullOrEmpty(propertyName) && material.HasProperty(propertyName))
            {
                return propertyName;
            }
        }

        return null;
    }

    private void CreateCorrectionHints()
    {
        leftArmHint = CreateHint(leftArmHint, leftArmStar, "LeftArmCorrectionHint");
        rightArmHint = CreateHint(rightArmHint, rightArmStar, "RightArmCorrectionHint");
        leftLegHint = CreateHint(leftLegHint, leftLegStar, "LeftLegCorrectionHint");
        rightLegHint = CreateHint(rightLegHint, rightLegStar, "RightLegCorrectionHint");
    }

    private TMP_Text CreateHint(TMP_Text existing, Transform star, string objectName)
    {
        if (existing != null || star == null || star.parent == null)
        {
            return existing;
        }

        GameObject hintObject = new GameObject(
            objectName,
            typeof(RectTransform),
            typeof(CanvasRenderer),
            typeof(TextMeshProUGUI),
            typeof(Shadow));
        hintObject.layer = star.parent.gameObject.layer;
        RectTransform rect = hintObject.GetComponent<RectTransform>();
        rect.SetParent(star.parent, false);
        rect.anchorMin = Vector2.zero;
        rect.anchorMax = Vector2.zero;
        rect.pivot = new Vector2(0.5f, 0.5f);
        rect.sizeDelta = hintSize;
        rect.localPosition = star.localPosition + hintLocalOffset;
        rect.localRotation = Quaternion.identity;
        rect.localScale = Vector3.one;

        TextMeshProUGUI text = hintObject.GetComponent<TextMeshProUGUI>();
        text.text = string.Empty;
        text.color = hintTextColor;
        text.fontSize = hintFontSize;
        text.fontStyle = FontStyles.Bold;
        text.alignment = TextAlignmentOptions.Center;
        text.enableWordWrapping = false;
        text.enableAutoSizing = true;
        text.fontSizeMin = 12f;
        text.fontSizeMax = hintFontSize;
        text.raycastTarget = false;

        Shadow shadow = hintObject.GetComponent<Shadow>();
        shadow.effectColor = new Color(0f, 0f, 0f, 0.8f);
        shadow.effectDistance = new Vector2(1.5f, -1.5f);
        shadow.useGraphicAlpha = true;
        hintObject.SetActive(false);
        return text;
    }

    private void UpdateHint(
        TMP_Text hint,
        AnatomicalPoseScore score,
        AnatomicalLimb limb)
    {
        if (hint == null)
        {
            return;
        }

        bool show = score.validSegmentCount > 0 && IsRedScore(score.score);
        hint.gameObject.SetActive(show);
        if (show)
        {
            hint.text = GetCorrectionText(score, limb);
        }
    }

    private bool IsRedScore(float score)
    {
        return scoreColorSource != null
            ? scoreColorSource.IsRedScore(score)
            : score < 50f;
    }

    private bool IsGreenScore(float score)
    {
        return scoreColorSource != null
            ? scoreColorSource.IsGreenScore(score)
            : score >= 70f;
    }

    private static void ApplyGreenParticles(
        ParticleSystem[] particles,
        bool green,
        ref bool wasGreen)
    {
        if (green == wasGreen)
        {
            return;
        }

        wasGreen = green;
        foreach (ParticleSystem particle in particles)
        {
            if (particle == null)
            {
                continue;
            }

            if (green)
            {
                particle.Stop(false, ParticleSystemStopBehavior.StopEmittingAndClear);
                particle.Play(false);
            }
            else
            {
                particle.Stop(false, ParticleSystemStopBehavior.StopEmittingAndClear);
            }
        }
    }

    private static void StopGreenParticles(ParticleSystem[] particles, ref bool wasGreen)
    {
        if (!wasGreen)
        {
            return;
        }

        ApplyGreenParticles(particles, false, ref wasGreen);
    }

    private static string GetCorrectionText(AnatomicalPoseScore score, AnatomicalLimb limb)
    {
        string subject = GetCorrectionSubject(score.worstSegment, limb);
        Vector3 direction = score.correctionDirection;
        if (direction.sqrMagnitude < 1e-6f)
        {
            return subject + "\u52a0\u5927\u52a8\u4f5c\u5e45\u5ea6";
        }

        float absoluteX = Mathf.Abs(direction.x);
        float absoluteY = Mathf.Abs(direction.y);
        float absoluteZ = Mathf.Abs(direction.z);
        bool rotate = score.worstSegment == HumanBodyBones.LeftHand ||
            score.worstSegment == HumanBodyBones.RightHand;
        if (absoluteY >= absoluteX && absoluteY >= absoluteZ)
        {
            return subject + (direction.y > 0f
                ? rotate ? "\u5411\u4e0a\u8f6c" : "\u5411\u4e0a\u62ac"
                : rotate ? "\u5411\u4e0b\u8f6c" : "\u5411\u4e0b\u653e");
        }

        if (absoluteZ >= absoluteX)
        {
            return subject + (direction.z > 0f
                ? rotate ? "\u5411\u524d\u8f6c" : "\u5411\u524d\u4f38"
                : rotate ? "\u5411\u540e\u8f6c" : "\u5411\u540e\u6536");
        }

        return subject + (direction.x > 0f
            ? rotate ? "\u5411\u53f3\u8f6c" : "\u5411\u53f3\u79fb"
            : rotate ? "\u5411\u5de6\u8f6c" : "\u5411\u5de6\u79fb");
    }

    private static string GetCorrectionSubject(HumanBodyBones segment, AnatomicalLimb limb)
    {
        switch (segment)
        {
            case HumanBodyBones.LeftUpperArm: return "\u5de6\u4e0a\u81c2";
            case HumanBodyBones.LeftLowerArm: return "\u5de6\u524d\u81c2";
            case HumanBodyBones.LeftHand: return "\u5de6\u624b";
            case HumanBodyBones.RightUpperArm: return "\u53f3\u4e0a\u81c2";
            case HumanBodyBones.RightLowerArm: return "\u53f3\u524d\u81c2";
            case HumanBodyBones.RightHand: return "\u53f3\u624b";
            case HumanBodyBones.LeftUpperLeg: return "\u5de6\u5927\u817f";
            case HumanBodyBones.LeftLowerLeg: return "\u5de6\u5c0f\u817f";
            case HumanBodyBones.RightUpperLeg: return "\u53f3\u5927\u817f";
            case HumanBodyBones.RightLowerLeg: return "\u53f3\u5c0f\u817f";
        }

        switch (limb)
        {
            case AnatomicalLimb.LeftArm: return "\u5de6\u81c2";
            case AnatomicalLimb.RightArm: return "\u53f3\u81c2";
            case AnatomicalLimb.LeftLeg: return "\u5de6\u817f";
            case AnatomicalLimb.RightLeg: return "\u53f3\u817f";
            default: return "\u8be5\u80a2\u4f53";
        }
    }

    private void UpdateHintPositions()
    {
        PositionHint(leftArmHint, leftArmStar);
        PositionHint(rightArmHint, rightArmStar);
        PositionHint(leftLegHint, leftLegStar);
        PositionHint(rightLegHint, rightLegStar);
    }

    private void PositionHint(TMP_Text hint, Transform star)
    {
        if (hint == null || star == null || gamingCanvas == null)
        {
            return;
        }

        RectTransform canvasRect = gamingCanvas.transform as RectTransform;
        RectTransform hintRect = hint.rectTransform;
        if (canvasRect == null || canvasRect.rect.width <= 0f || canvasRect.rect.height <= 0f)
        {
            return;
        }

        Camera canvasCamera = gamingCanvas.renderMode == RenderMode.ScreenSpaceOverlay
            ? null
            : gamingCanvas.worldCamera;
        Vector2 screenPoint = RectTransformUtility.WorldToScreenPoint(canvasCamera, star.position);
        if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(
            canvasRect,
            screenPoint,
            canvasCamera,
            out Vector2 starLocalPoint))
        {
            return;
        }

        Vector2 desired = starLocalPoint + new Vector2(hintLocalOffset.x, hintLocalOffset.y);
        Rect bounds = canvasRect.rect;
        float halfWidth = hintRect.rect.width * 0.5f;
        float halfHeight = hintRect.rect.height * 0.5f;
        desired.x = Mathf.Clamp(
            desired.x,
            bounds.xMin + halfWidth + hintCanvasPadding,
            bounds.xMax - halfWidth - hintCanvasPadding);
        desired.y = Mathf.Clamp(
            desired.y,
            bounds.yMin + halfHeight + hintCanvasPadding,
            bounds.yMax - halfHeight - hintCanvasPadding);
        hintRect.localPosition = new Vector3(desired.x, desired.y, 0f);
    }

    private void ResolveReferences()
    {
        if (poseScorer == null)
        {
            poseScorer = FindObjectOfType<DeadBugGamingPoseScorer>(true);
        }
        if (scoreColorSource == null)
        {
            scoreColorSource = FindObjectOfType<DeadBugScoreRoadProgressBar>(true);
        }
        if (leftArmStar == null) leftArmStar = FindTransform("LeftArm");
        if (rightArmStar == null) rightArmStar = FindTransform("RightArm");
        if (leftLegStar == null) leftLegStar = FindTransform("LeftLeg");
        if (rightLegStar == null) rightLegStar = FindTransform("RightLeg");
        if (gamingCanvas == null && leftArmStar != null)
        {
            gamingCanvas = leftArmStar.GetComponentInParent<Canvas>();
        }
    }

    private static Transform FindTransform(string targetName)
    {
        Transform[] transforms = FindObjectsOfType<Transform>(true);
        foreach (Transform candidate in transforms)
        {
            if (candidate != null && candidate.name == targetName)
            {
                return candidate;
            }
        }

        return null;
    }

    private void Subscribe()
    {
        if (poseScorer == null)
        {
            Debug.LogWarning("DeadBugLimbScoreFeedback could not find the pose scorer.", this);
            return;
        }

        poseScorer.SessionStarted -= HandleSessionStarted;
        poseScorer.LimbScoresChanged -= HandleLimbScoresChanged;
        poseScorer.SessionStarted += HandleSessionStarted;
        poseScorer.LimbScoresChanged += HandleLimbScoresChanged;
    }

    private void Unsubscribe()
    {
        if (poseScorer == null)
        {
            return;
        }

        poseScorer.SessionStarted -= HandleSessionStarted;
        poseScorer.LimbScoresChanged -= HandleLimbScoresChanged;
    }

    private static Color GetFallbackScoreColor(float score)
    {
        if (score < 50f) return new Color(1f, 0.48f, 0.35f, 1f);
        if (score < 70f) return new Color(1f, 0.85f, 0.25f, 1f);
        return new Color(0.25f, 1f, 0.55f, 1f);
    }
}
