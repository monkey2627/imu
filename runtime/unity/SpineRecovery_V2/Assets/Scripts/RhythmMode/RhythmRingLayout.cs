using System.Collections.Generic;
using UnityEngine;

public sealed class RhythmRingLayout : MonoBehaviour
{
    [Header("Ring Geometry")]
    [Min(0.1f)] [SerializeField] private float judgementRadius = 0.62f;
    [Min(0.05f)] [SerializeField] private float waistRadius = 0.18f;
    [Min(0.01f)] [SerializeField] private float ringWidth = 0.018f;
    [Range(24, 160)] [SerializeField] private int ringSegments = 80;

    [Header("Authored Guide Ring")]
    [SerializeField] private bool useAuthoredGuideRing = true;
    [Tooltip("Optional authored outer ring. When assigned, it replaces the procedural limb ring.")]
    [SerializeField] private GameObject authoredGuideRingPrefab;
    [Min(0.01f)] [SerializeField] private float authoredGuideRingSourceRadius = 3f;
    [SerializeField] private Vector3 authoredGuideRingEulerAngles = Vector3.zero;
    [Tooltip("Complete scene-authored skin. Replaces only generated ring graphics; path geometry is unchanged.")]
    [SerializeField] private GameObject completeVisualRoot;
    [Range(0.8f, 1f)] [SerializeField] private float authoredInnerRingRadiusScale = 0.9667f;
    [Range(1f, 1.2f)] [SerializeField] private float authoredOuterRingRadiusScale = 1.03f;

    [Header("Spawn Geometry")]
    [Min(0.2f)] [SerializeField] private float limbSpawnRadiusMin = 1.65f;
    [Min(0.2f)] [SerializeField] private float limbSpawnRadiusMax = 1.95f;
    [Min(0.1f)] [SerializeField] private float waistSpawnRadiusMin = 0.84f;
    [Min(0.1f)] [SerializeField] private float waistSpawnRadiusMax = 1.05f;
    [Range(0f, 30f)] [SerializeField] private float sectorEdgePaddingDegrees = 12f;

    [Header("Colors")]
    [Tooltip("Optional authored material. Leave empty to use the runtime fallback.")]
    [SerializeField] private Material lineMaterialOverride;
    [SerializeField] private Color ringColor = new Color(0.82f, 0.92f, 1f, 0.88f);
    [SerializeField] private Color dividerColor = new Color(0.52f, 0.68f, 0.78f, 0.62f);
    [SerializeField] private Color leftHandColor = new Color(0.15f, 0.82f, 1f, 1f);
    [SerializeField] private Color rightHandColor = new Color(1f, 0.3f, 0.44f, 1f);
    [SerializeField] private Color leftFootColor = new Color(0.25f, 0.92f, 0.48f, 1f);
    [SerializeField] private Color rightFootColor = new Color(1f, 0.72f, 0.18f, 1f);
    [SerializeField] private Color waistColor = new Color(0.78f, 0.42f, 1f, 1f);

    private readonly List<GameObject> generatedVisuals = new List<GameObject>();
    private Material lineMaterial;
    private bool ownsLineMaterial;

    public float JudgementRadius => judgementRadius;
    public float WaistRadius => waistRadius;

    private void Awake()
    {
        BuildVisuals();
    }

    private void OnDestroy()
    {
        if (ownsLineMaterial && lineMaterial != null)
        {
            Destroy(lineMaterial);
        }
    }

    public void GetRandomPath(
        RhythmBodyTarget target,
        out Vector3 spawnLocalPosition,
        out Vector3 hitLocalPosition)
    {
        float angle = Random.Range(GetMinimumAngle(target), GetMaximumAngle(target));
        float radians = angle * Mathf.Deg2Rad;
        Vector3 direction = new Vector3(Mathf.Cos(radians), Mathf.Sin(radians), 0f);

        if (target == RhythmBodyTarget.Waist)
        {
            float spawnRadius = Random.Range(waistSpawnRadiusMin, waistSpawnRadiusMax);
            spawnLocalPosition = direction * spawnRadius;
            hitLocalPosition = direction * waistRadius;
            return;
        }

        float limbSpawnRadius = Random.Range(limbSpawnRadiusMin, limbSpawnRadiusMax);
        spawnLocalPosition = direction * limbSpawnRadius;
        hitLocalPosition = direction * judgementRadius;
    }

    public Color GetTargetColor(RhythmBodyTarget target)
    {
        switch (target)
        {
            case RhythmBodyTarget.LeftHand: return leftHandColor;
            case RhythmBodyTarget.RightHand: return rightHandColor;
            case RhythmBodyTarget.LeftFoot: return leftFootColor;
            case RhythmBodyTarget.RightFoot: return rightFootColor;
            default: return waistColor;
        }
    }

    public Color GetRandomAccentColor()
    {
        switch (Random.Range(0, 5))
        {
            case 0: return leftHandColor;
            case 1: return rightHandColor;
            case 2: return leftFootColor;
            case 3: return rightFootColor;
            default: return waistColor;
        }
    }

    public void SetVisible(bool visible)
    {
        foreach (GameObject visual in generatedVisuals)
        {
            if (visual != null)
            {
                visual.SetActive(visible);
            }
        }
    }

    private float GetMinimumAngle(RhythmBodyTarget target)
    {
        if (target == RhythmBodyTarget.Waist)
        {
            return 0f;
        }

        return GetSectorCenter(target) - 45f + sectorEdgePaddingDegrees;
    }

    private float GetMaximumAngle(RhythmBodyTarget target)
    {
        if (target == RhythmBodyTarget.Waist)
        {
            return 360f;
        }

        return GetSectorCenter(target) + 45f - sectorEdgePaddingDegrees;
    }

    private static float GetSectorCenter(RhythmBodyTarget target)
    {
        switch (target)
        {
            case RhythmBodyTarget.LeftHand: return 135f;
            case RhythmBodyTarget.RightHand: return 45f;
            case RhythmBodyTarget.LeftFoot: return 225f;
            case RhythmBodyTarget.RightFoot: return 315f;
            default: return 0f;
        }
    }

    private void BuildVisuals()
    {
        ClearVisuals();
        if (completeVisualRoot != null)
        {
            generatedVisuals.Add(completeVisualRoot);
            return;
        }
        lineMaterial = lineMaterialOverride != null ? lineMaterialOverride : CreateLineMaterial();
        ownsLineMaterial = lineMaterialOverride == null;
        if (!useAuthoredGuideRing || !CreateAuthoredGuideRing())
        {
            CreateRing("LimbJudgementRing", judgementRadius, ringColor, ringWidth);
        }
        CreateRing("WaistJudgementRing", waistRadius, waistColor, ringWidth * 1.15f);

        CreateDivider("UpperDivider", 90f);
        CreateDivider("RightDivider", 0f);
        CreateDivider("LowerDivider", 270f);
        CreateDivider("LeftDivider", 180f);
    }

    private bool CreateAuthoredGuideRing()
    {
        if (authoredGuideRingPrefab == null)
        {
            return false;
        }

        GameObject guideRing = Instantiate(authoredGuideRingPrefab, transform, false);
        guideRing.name = "AuthoredLimbJudgementRing";
        guideRing.transform.localPosition = Vector3.zero;
        guideRing.transform.localRotation = Quaternion.Euler(authoredGuideRingEulerAngles);
        float scale = judgementRadius / Mathf.Max(0.01f, authoredGuideRingSourceRadius);
        guideRing.transform.localScale = Vector3.one * scale;

        WidenAuthoredGuideRing(guideRing);

        foreach (Renderer ringRenderer in guideRing.GetComponentsInChildren<Renderer>(true))
        {
            ringRenderer.sortingOrder = 10;
        }

        generatedVisuals.Add(guideRing);
        return true;
    }

    private void WidenAuthoredGuideRing(GameObject guideRing)
    {
        MeshRenderer[] renderers = guideRing.GetComponentsInChildren<MeshRenderer>(true);
        bool hasSeparateRingRenderers = false;
        foreach (MeshRenderer meshRenderer in renderers)
        {
            GetRingMaterialFlags(meshRenderer, out bool hasInner, out bool hasOuter);
            if (hasInner != hasOuter)
            {
                hasSeparateRingRenderers = true;
                break;
            }
        }

        foreach (MeshRenderer meshRenderer in renderers)
        {
            GetRingMaterialFlags(meshRenderer, out bool hasInner, out bool hasOuter);
            if (hasInner && hasOuter && hasSeparateRingRenderers)
            {
                meshRenderer.enabled = false;
                continue;
            }

            if (hasInner && !hasOuter)
            {
                meshRenderer.transform.localScale *= authoredInnerRingRadiusScale;
            }
            else if (hasOuter && !hasInner)
            {
                meshRenderer.transform.localScale *= authoredOuterRingRadiusScale;
            }
        }
    }

    private static void GetRingMaterialFlags(
        MeshRenderer meshRenderer,
        out bool hasInner,
        out bool hasOuter)
    {
        hasInner = false;
        hasOuter = false;
        foreach (Material material in meshRenderer.sharedMaterials)
        {
            if (material == null)
            {
                continue;
            }

            hasInner |= material.name.Contains("GuideRing_Inner");
            hasOuter |= material.name.Contains("GuideRing_Outer");
        }
    }

    private void CreateRing(string objectName, float radius, Color color, float width)
    {
        GameObject ringObject = CreateVisualObject(objectName);
        LineRenderer line = ringObject.AddComponent<LineRenderer>();
        ConfigureLine(line, color, width);
        line.loop = true;
        line.positionCount = ringSegments;

        for (int index = 0; index < ringSegments; index++)
        {
            float angle = index * Mathf.PI * 2f / ringSegments;
            line.SetPosition(index, new Vector3(
                Mathf.Cos(angle) * radius,
                Mathf.Sin(angle) * radius,
                0f));
        }
    }

    private void CreateDivider(string objectName, float angleDegrees)
    {
        float angle = angleDegrees * Mathf.Deg2Rad;
        Vector3 direction = new Vector3(Mathf.Cos(angle), Mathf.Sin(angle), 0f);
        GameObject dividerObject = CreateVisualObject(objectName);
        LineRenderer line = dividerObject.AddComponent<LineRenderer>();
        ConfigureLine(line, dividerColor, ringWidth * 0.6f);
        line.positionCount = 2;
        line.SetPosition(0, direction * (waistRadius + ringWidth * 1.5f));
        line.SetPosition(1, direction * (judgementRadius - ringWidth * 1.5f));
    }

    private GameObject CreateVisualObject(string objectName)
    {
        GameObject visual = new GameObject(objectName);
        visual.transform.SetParent(transform, false);
        generatedVisuals.Add(visual);
        return visual;
    }

    private void ConfigureLine(LineRenderer line, Color color, float width)
    {
        line.useWorldSpace = false;
        line.sharedMaterial = lineMaterial;
        line.startColor = color;
        line.endColor = color;
        line.startWidth = width;
        line.endWidth = width;
        line.numCapVertices = 4;
        line.numCornerVertices = 3;
        line.textureMode = LineTextureMode.Stretch;
        line.sortingOrder = 10;
    }

    private static Material CreateLineMaterial()
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Unlit") ??
            Shader.Find("Sprites/Default") ??
            Shader.Find("Unlit/Color");
        Material material = new Material(shader)
        {
            name = "Runtime Rhythm Ring Material"
        };
        if (material.HasProperty("_Surface")) material.SetFloat("_Surface", 1f);
        if (material.HasProperty("_ZWrite")) material.SetFloat("_ZWrite", 0f);
        material.renderQueue = 3000;
        return material;
    }

    private void ClearVisuals()
    {
        foreach (GameObject visual in generatedVisuals)
        {
            if (visual != null)
            {
                Destroy(visual);
            }
        }
        generatedVisuals.Clear();
    }
}
