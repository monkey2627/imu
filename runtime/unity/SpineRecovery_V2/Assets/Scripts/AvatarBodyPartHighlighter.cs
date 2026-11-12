using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Builds skinned overlay meshes from the avatar's existing bone weights, so a
/// body part glows directly on the user model without destructively editing the
/// imported VRM or placing spheres at the joints.
/// </summary>
public sealed class AvatarBodyPartHighlighter : MonoBehaviour
{
    private const string GeneratedPrefix = "__BodyPartHighlight_";

    [SerializeField] private Animator targetAnimator;
    [SerializeField] private Material highlightMaterial;
    [SerializeField] private Color highlightColor = new Color(0f, 1f, 0f, 0.95f);
    [SerializeField, Range(0.05f, 0.9f)] private float minimumTriangleWeight = 0.35f;
    [Tooltip("Torso uses a stricter surface threshold so shoulder, neck, hip, " +
        "and clothing-edge triangles are not painted as one oversized block.")]
    [SerializeField, Range(0.05f, 0.95f)] private float torsoMinimumTriangleWeight = 0.55f;
    [SerializeField] private bool pulse = true;
    [SerializeField, Min(0f)] private float pulseSpeed = 3f;
    [SerializeField] private Vector2 pulseRange = new Vector2(0.85f, 1.35f);
    [Tooltip("Minimum delay before an automatic retry after no usable overlay " +
        "could be generated.")]
    [SerializeField, Min(0.1f)] private float failedBuildRetrySeconds = 1f;

    private readonly Dictionary<BodyPart, List<SkinnedMeshRenderer>> overlays =
        new Dictionary<BodyPart, List<SkinnedMeshRenderer>>();
    private readonly List<Mesh> generatedMeshes = new List<Mesh>();
    private readonly HashSet<BodyPart> visibleParts = new HashSet<BodyPart>();
    private readonly HashSet<BodyPart> requestedParts = new HashSet<BodyPart>();
    private readonly Dictionary<BodyPart, Color> activeColorOverrides =
        new Dictionary<BodyPart, Color>();
    private Material runtimeMaterial;
    private MaterialPropertyBlock colorPropertyBlock;
    private bool isBuilt;
    private float nextAutomaticBuildAttemptTime;
    private int buildAttemptCount;

    public Animator TargetAnimator => targetAnimator;
    public bool IsReady => isBuilt && OverlayRendererCount > 0;
    public BodyPart VisiblePart { get; private set; } = BodyPart.None;
    public IReadOnlyCollection<BodyPart> VisibleParts => visibleParts;
    public bool PulseEnabled => pulse;
    public int BuildAttemptCount => buildAttemptCount;

    public int OverlayRendererCount
    {
        get
        {
            int count = 0;
            foreach (List<SkinnedMeshRenderer> renderers in overlays.Values)
            {
                count += renderers.Count;
            }

            return count;
        }
    }

    public int VisibleRendererCount
    {
        get
        {
            int count = 0;
            foreach (List<SkinnedMeshRenderer> renderers in overlays.Values)
            {
                foreach (SkinnedMeshRenderer renderer in renderers)
                {
                    if (renderer != null && renderer.enabled && renderer.gameObject.activeInHierarchy)
                    {
                        count++;
                    }
                }
            }

            return count;
        }
    }

    private void Update()
    {
        if (runtimeMaterial == null || !pulse || visibleParts.Count == 0)
        {
            return;
        }

        float wave = (Mathf.Sin(Time.unscaledTime * pulseSpeed) + 1f) * 0.5f;
        runtimeMaterial.SetFloat("_Pulse", Mathf.Lerp(pulseRange.x, pulseRange.y, wave));
    }

    public void Configure(Animator animator)
    {
        if (animator == null)
        {
            return;
        }

        if (targetAnimator == animator && isBuilt)
        {
            return;
        }

        targetAnimator = animator;
        Rebuild();
    }

    public Color GetDisplayedColor(BodyPart part)
    {
        return activeColorOverrides.TryGetValue(part, out Color color)
            ? color
            : highlightColor;
    }

    public void SetPulseEnabled(bool enabled)
    {
        pulse = enabled;
        if (!pulse && runtimeMaterial != null)
        {
            runtimeMaterial.SetFloat("_Pulse", 1f);
        }
    }

    public void Rebuild()
    {
        buildAttemptCount++;
        ClearGeneratedOverlays();
        if (targetAnimator == null)
        {
            ScheduleAutomaticBuildRetry();
            return;
        }

        EnsureMaterial();
        InitializePartLists();

        SkinnedMeshRenderer[] sources =
            targetAnimator.GetComponentsInChildren<SkinnedMeshRenderer>(true);
        foreach (SkinnedMeshRenderer source in sources)
        {
            if (source == null || source.sharedMesh == null ||
                source.gameObject.name.StartsWith(GeneratedPrefix))
            {
                continue;
            }

            if (!source.sharedMesh.isReadable)
            {
#if UNITY_EDITOR || DEVELOPMENT_BUILD
                Debug.LogWarning(
                    $"[BodyPartHighlight] Skipping unreadable mesh " +
                    $"'{source.sharedMesh.name}'. Enable Read/Write or use " +
                    "the fallback body-part card.",
                    source);
#endif
                continue;
            }

            try
            {
                BuildOverlaysForRenderer(source);
            }
            catch (UnityException exception)
            {
#if UNITY_EDITOR || DEVELOPMENT_BUILD
                Debug.LogWarning(
                    $"[BodyPartHighlight] Could not build overlay for " +
                    $"'{source.name}': {exception.Message}",
                    source);
#endif
            }
        }

        isBuilt = OverlayRendererCount > 0;
        if (!isBuilt)
        {
            ScheduleAutomaticBuildRetry();
        }
        else
        {
            nextAutomaticBuildAttemptTime = 0f;
        }
        ApplyRendererColors();
        Hide();
    }

    public bool Show(BodyPart part)
    {
        return Show(new[] { part });
    }

    public bool Show(IEnumerable<BodyPart> parts)
    {
        activeColorOverrides.Clear();
        return ShowParts(parts);
    }

    /// <summary>
    /// Shows every supplied body part with its own diagnostic color.
    /// </summary>
    public bool Show(IReadOnlyDictionary<BodyPart, Color> partColors)
    {
        if (partColors == null)
        {
            Hide();
            return false;
        }

        EnsureBuiltForDisplay();

        activeColorOverrides.Clear();
        foreach (KeyValuePair<BodyPart, Color> pair in partColors)
        {
            if (pair.Key != BodyPart.None && overlays.ContainsKey(pair.Key))
            {
                activeColorOverrides[pair.Key] = pair.Value;
            }
        }

        return ShowParts(activeColorOverrides.Keys);
    }

    private bool ShowParts(IEnumerable<BodyPart> parts)
    {
        if (parts == null)
        {
            Hide();
            return false;
        }

        if (!EnsureBuiltForDisplay())
        {
            Hide();
            return false;
        }

        requestedParts.Clear();
        foreach (BodyPart part in parts)
        {
            if (part != BodyPart.None && overlays.ContainsKey(part))
            {
                requestedParts.Add(part);
            }
        }

        if (requestedParts.SetEquals(visibleParts))
        {
            ApplyRendererColors();
            return VisibleRendererCount > 0;
        }

        HideRenderersOnly();
        visibleParts.Clear();
        VisiblePart = BodyPart.None;

        int enabledCount = 0;
        foreach (BodyPart part in requestedParts)
        {
            if (!overlays.TryGetValue(part, out List<SkinnedMeshRenderer> renderers))
            {
                continue;
            }

            int partRendererCount = 0;
            foreach (SkinnedMeshRenderer renderer in renderers)
            {
                if (renderer == null)
                {
                    continue;
                }

                renderer.enabled = true;
                enabledCount++;
                partRendererCount++;
            }

            if (partRendererCount > 0)
            {
                visibleParts.Add(part);
                if (VisiblePart == BodyPart.None)
                {
                    VisiblePart = part;
                }
            }
        }

        ApplyRendererColors();

#if UNITY_EDITOR || DEVELOPMENT_BUILD
        Debug.Log(
            $"[BodyPartHighlight] visible={string.Join(", ", visibleParts)}, " +
            $"renderers={enabledCount}",
            this);
#endif
        return enabledCount > 0;
    }

    private bool EnsureBuiltForDisplay()
    {
        if (isBuilt && OverlayRendererCount > 0)
        {
            return true;
        }

        if (Time.realtimeSinceStartup < nextAutomaticBuildAttemptTime)
        {
            return false;
        }

        Rebuild();
        return isBuilt && OverlayRendererCount > 0;
    }

    private void ScheduleAutomaticBuildRetry()
    {
        nextAutomaticBuildAttemptTime = Time.realtimeSinceStartup +
            Mathf.Max(0.1f, failedBuildRetrySeconds);
    }

    public bool IsPartVisible(BodyPart part)
    {
        return visibleParts.Contains(part);
    }

    public void Hide()
    {
        bool hadVisibleParts = visibleParts.Count > 0;
        HideRenderersOnly();
        visibleParts.Clear();
        activeColorOverrides.Clear();
        VisiblePart = BodyPart.None;
#if UNITY_EDITOR || DEVELOPMENT_BUILD
        if (hadVisibleParts)
        {
            Debug.Log("[BodyPartHighlight] visible=None, renderers=0", this);
        }
#endif
    }

    private void BuildOverlaysForRenderer(SkinnedMeshRenderer source)
    {
        Mesh sourceMesh = source.sharedMesh;
        BoneWeight[] weights = sourceMesh.boneWeights;
        Transform[] bones = source.bones;
        if (weights == null || weights.Length != sourceMesh.vertexCount || bones == null || bones.Length == 0)
        {
            return;
        }

        BodyPart[] parts =
        {
            BodyPart.LeftArm,
            BodyPart.RightArm,
            BodyPart.LeftLeg,
            BodyPart.RightLeg,
            BodyPart.Torso
        };

        var weightsByPart = new Dictionary<BodyPart, float[]>();
        var trianglesByPart = new Dictionary<BodyPart, List<int>>();
        foreach (BodyPart part in parts)
        {
            weightsByPart[part] = BuildVertexWeights(weights, bones, part);
            trianglesByPart[part] = new List<int>();
        }

        SelectDominantTriangles(
            sourceMesh,
            parts,
            weightsByPart,
            trianglesByPart);
        foreach (BodyPart part in parts)
        {
            List<int> selectedTriangles = trianglesByPart[part];
            if (selectedTriangles.Count == 0)
            {
                continue;
            }

            Mesh overlayMesh = CopyMeshWithTriangles(sourceMesh, selectedTriangles, part);
            SkinnedMeshRenderer overlayRenderer = CreateOverlayRenderer(source, overlayMesh, part);
            generatedMeshes.Add(overlayMesh);
            overlays[part].Add(overlayRenderer);
        }
    }

    private float[] BuildVertexWeights(BoneWeight[] weights, Transform[] bones, BodyPart part)
    {
        float[] result = new float[weights.Length];
        for (int index = 0; index < weights.Length; index++)
        {
            BoneWeight weight = weights[index];
            result[index] =
                GetWeightForBone(weight.boneIndex0, weight.weight0, bones, part) +
                GetWeightForBone(weight.boneIndex1, weight.weight1, bones, part) +
                GetWeightForBone(weight.boneIndex2, weight.weight2, bones, part) +
                GetWeightForBone(weight.boneIndex3, weight.weight3, bones, part);
        }

        return result;
    }

    private float GetWeightForBone(int index, float weight, Transform[] bones, BodyPart part)
    {
        if (weight <= 0f || index < 0 || index >= bones.Length || bones[index] == null)
        {
            return 0f;
        }

        return ClassifyBone(bones[index]) == part ? weight : 0f;
    }

    private BodyPart ClassifyBone(Transform bone)
    {
        if (IsBoneOrInBranch(
                bone,
                HumanBodyBones.LeftShoulder,
                HumanBodyBones.LeftUpperArm))
        {
            return BodyPart.LeftArm;
        }

        if (IsBoneOrInBranch(
                bone,
                HumanBodyBones.RightShoulder,
                HumanBodyBones.RightUpperArm))
        {
            return BodyPart.RightArm;
        }

        if (IsInBranch(bone, HumanBodyBones.LeftUpperLeg))
        {
            return BodyPart.LeftLeg;
        }

        if (IsInBranch(bone, HumanBodyBones.RightUpperLeg))
        {
            return BodyPart.RightLeg;
        }

        if (IsExactHumanoidBone(
                bone,
                HumanBodyBones.Hips,
                HumanBodyBones.Spine,
                HumanBodyBones.Chest,
                HumanBodyBones.UpperChest))
        {
            return BodyPart.Torso;
        }

        string lowerName = bone.name.ToLowerInvariant();
        if (ContainsAny(lowerName, "humerus_l", "ulna_l", "radius_l", "arm_l", "hand_l"))
        {
            return BodyPart.LeftArm;
        }

        if (ContainsAny(lowerName, "humerus_r", "ulna_r", "radius_r", "arm_r", "hand_r"))
        {
            return BodyPart.RightArm;
        }

        if (ContainsAny(lowerName, "femur_l", "tibia_l", "leg_l", "foot_l", "toe_l"))
        {
            return BodyPart.LeftLeg;
        }

        if (ContainsAny(lowerName, "femur_r", "tibia_r", "leg_r", "foot_r", "toe_r"))
        {
            return BodyPart.RightLeg;
        }

        if (ContainsAny(lowerName, "pelvis", "lumbar", "thorax", "spine", "chest", "hips"))
        {
            return BodyPart.Torso;
        }

        return BodyPart.None;
    }

    private void ApplyRendererColors()
    {
        if (runtimeMaterial == null)
        {
            return;
        }

        if (colorPropertyBlock == null)
        {
            colorPropertyBlock = new MaterialPropertyBlock();
        }

        foreach (KeyValuePair<BodyPart, List<SkinnedMeshRenderer>> pair in overlays)
        {
            Color color = GetDisplayedColor(pair.Key);
            foreach (SkinnedMeshRenderer renderer in pair.Value)
            {
                if (renderer == null)
                {
                    continue;
                }

                colorPropertyBlock.Clear();
                colorPropertyBlock.SetColor("_Color", color);
                renderer.SetPropertyBlock(colorPropertyBlock);
            }
        }
    }

    private bool IsBoneOrInBranch(
        Transform bone,
        HumanBodyBones directBone,
        HumanBodyBones branchBone)
    {
        Transform direct = targetAnimator.GetBoneTransform(directBone);
        return (direct != null && bone == direct) ||
            IsInBranch(bone, branchBone);
    }

    private bool IsInBranch(Transform bone, HumanBodyBones branchBone)
    {
        Transform branch = targetAnimator.GetBoneTransform(branchBone);
        return branch != null && (bone == branch || bone.IsChildOf(branch));
    }

    private bool IsExactHumanoidBone(
        Transform bone,
        params HumanBodyBones[] humanoidBones)
    {
        if (targetAnimator == null || bone == null || humanoidBones == null)
        {
            return false;
        }

        foreach (HumanBodyBones humanoidBone in humanoidBones)
        {
            Transform mapped = targetAnimator.GetBoneTransform(humanoidBone);
            if (mapped != null && bone == mapped)
            {
                return true;
            }
        }

        return false;
    }

    private static bool ContainsAny(string value, params string[] fragments)
    {
        foreach (string fragment in fragments)
        {
            if (value.Contains(fragment))
            {
                return true;
            }
        }

        return false;
    }

    private void SelectDominantTriangles(
        Mesh mesh,
        IReadOnlyList<BodyPart> parts,
        IReadOnlyDictionary<BodyPart, float[]> weightsByPart,
        IReadOnlyDictionary<BodyPart, List<int>> trianglesByPart)
    {
        for (int subMesh = 0; subMesh < mesh.subMeshCount; subMesh++)
        {
            int[] triangles = mesh.GetTriangles(subMesh);
            for (int triangle = 0; triangle + 2 < triangles.Length; triangle += 3)
            {
                int a = triangles[triangle];
                int b = triangles[triangle + 1];
                int c = triangles[triangle + 2];
                BodyPart dominantPart = BodyPart.None;
                float dominantWeight = 0f;
                foreach (BodyPart part in parts)
                {
                    float[] vertexWeights = weightsByPart[part];
                    float averageWeight =
                        (vertexWeights[a] +
                         vertexWeights[b] +
                         vertexWeights[c]) / 3f;
                    if (averageWeight > dominantWeight)
                    {
                        dominantWeight = averageWeight;
                        dominantPart = part;
                    }
                }

                float requiredWeight = dominantPart == BodyPart.Torso
                    ? Mathf.Max(minimumTriangleWeight, torsoMinimumTriangleWeight)
                    : minimumTriangleWeight;
                if (dominantPart == BodyPart.None ||
                    dominantWeight < requiredWeight)
                {
                    continue;
                }

                List<int> destination = trianglesByPart[dominantPart];
                destination.Add(a);
                destination.Add(b);
                destination.Add(c);
            }
        }
    }

    private static Mesh CopyMeshWithTriangles(Mesh source, List<int> triangles, BodyPart part)
    {
        Mesh mesh = new Mesh
        {
            name = $"{source.name}_{part}_Highlight",
            indexFormat = source.indexFormat,
            vertices = source.vertices,
            normals = source.normals,
            tangents = source.tangents,
            colors = source.colors,
            uv = source.uv,
            uv2 = source.uv2,
            uv3 = source.uv3,
            uv4 = source.uv4,
            boneWeights = source.boneWeights,
            bindposes = source.bindposes,
            bounds = source.bounds
        };
        mesh.SetTriangles(triangles, 0, false);
        return mesh;
    }

    private SkinnedMeshRenderer CreateOverlayRenderer(
        SkinnedMeshRenderer source,
        Mesh mesh,
        BodyPart part)
    {
        GameObject overlayObject = new GameObject($"{GeneratedPrefix}{part}_{source.name}");
        Transform overlayTransform = overlayObject.transform;
        overlayTransform.SetParent(source.transform.parent, false);
        overlayTransform.localPosition = source.transform.localPosition;
        overlayTransform.localRotation = source.transform.localRotation;
        overlayTransform.localScale = source.transform.localScale;
        overlayObject.layer = source.gameObject.layer;

        SkinnedMeshRenderer overlay = overlayObject.AddComponent<SkinnedMeshRenderer>();
        overlay.sharedMesh = mesh;
        overlay.sharedMaterial = runtimeMaterial;
        overlay.bones = source.bones;
        overlay.rootBone = source.rootBone;
        overlay.localBounds = source.localBounds;
        overlay.quality = source.quality;
        overlay.updateWhenOffscreen = source.updateWhenOffscreen;
        overlay.skinnedMotionVectors = false;
        overlay.shadowCastingMode = ShadowCastingMode.Off;
        overlay.receiveShadows = false;
        overlay.lightProbeUsage = LightProbeUsage.Off;
        overlay.reflectionProbeUsage = ReflectionProbeUsage.Off;
        overlay.enabled = false;
        return overlay;
    }

    private void EnsureMaterial()
    {
        if (runtimeMaterial != null)
        {
            return;
        }

        if (highlightMaterial != null)
        {
            runtimeMaterial = new Material(highlightMaterial);
        }
        else
        {
            Shader shader = Shader.Find("SpineRecovery/BodyPartHighlightOverlay");
            if (shader == null)
            {
                shader = Shader.Find("Unlit/Color");
            }

            runtimeMaterial = new Material(shader);
        }

        runtimeMaterial.name = "Runtime Body Part Highlight";
        runtimeMaterial.SetColor("_Color", highlightColor);
        runtimeMaterial.SetFloat("_Pulse", 1f);
    }

    private void InitializePartLists()
    {
        overlays.Clear();
        overlays[BodyPart.LeftArm] = new List<SkinnedMeshRenderer>();
        overlays[BodyPart.RightArm] = new List<SkinnedMeshRenderer>();
        overlays[BodyPart.LeftLeg] = new List<SkinnedMeshRenderer>();
        overlays[BodyPart.RightLeg] = new List<SkinnedMeshRenderer>();
        overlays[BodyPart.Torso] = new List<SkinnedMeshRenderer>();
    }

    private void HideRenderersOnly()
    {
        foreach (List<SkinnedMeshRenderer> renderers in overlays.Values)
        {
            foreach (SkinnedMeshRenderer renderer in renderers)
            {
                if (renderer != null)
                {
                    renderer.enabled = false;
                }
            }
        }
    }

    private void ClearGeneratedOverlays()
    {
        HideRenderersOnly();
        SkinnedMeshRenderer[] renderers = GetComponentsInChildren<SkinnedMeshRenderer>(true);
        foreach (SkinnedMeshRenderer renderer in renderers)
        {
            if (renderer != null && renderer.gameObject.name.StartsWith(GeneratedPrefix))
            {
                DisposeUnityObject(renderer.gameObject);
            }
        }

        foreach (Mesh mesh in generatedMeshes)
        {
            DisposeUnityObject(mesh);
        }

        generatedMeshes.Clear();
        overlays.Clear();
        visibleParts.Clear();
        VisiblePart = BodyPart.None;
        isBuilt = false;
    }

    private static void DisposeUnityObject(Object target)
    {
        if (target == null)
        {
            return;
        }

        if (Application.isPlaying)
        {
            Destroy(target);
        }
        else
        {
            DestroyImmediate(target);
        }
    }

    private void OnDestroy()
    {
        ClearGeneratedOverlays();
        if (runtimeMaterial != null)
        {
            DisposeUnityObject(runtimeMaterial);
            runtimeMaterial = null;
        }
    }
}
