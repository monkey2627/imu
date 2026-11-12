using System.Collections.Generic;
using UnityEngine;

[ExecuteAlways]
[DisallowMultipleComponent]
public sealed class DoubleBrokenFourColorRing : MonoBehaviour
{
    [Header("Double broken ring model")]
    public float innerRadius = 10.35f;
    public float outerRadius = 11.2f;
    public float bandWidth = 0.28f;
    [Range(1, 6)] public int dashesPerQuarter = 3;
    [Range(0f, 12f)] public float gapDegrees = 5f;
    [Range(2, 16)] public int subdivisionsPerDash = 6;
    public float heightOffset = 0.04f;

    [Header("HDR colors")]
    public float intensity = 3f;
    public Color blue = new Color(0.05f, 0.3f, 1f, 1f);
    public Color purple = new Color(0.55f, 0.08f, 1f, 1f);
    public Color green = new Color(0.05f, 1f, 0.25f, 1f);
    public Color pink = new Color(1f, 0.04f, 0.45f, 1f);

    const string ModelRootName = "Double Broken Four Color Ring Model";
    static readonly int BaseColorId = Shader.PropertyToID("_BaseColor");
    static readonly int EmissionColorId = Shader.PropertyToID("_EmissionColor");

    void OnEnable() => Build();

    [ContextMenu("Rebuild Ring Model")]
    public void Build()
    {
        var particleRenderer = GetComponent<ParticleSystemRenderer>();
        if (particleRenderer != null)
            particleRenderer.enabled = false;

        Transform root = transform.Find(ModelRootName);
        if (root == null)
        {
            var rootObject = new GameObject(ModelRootName);
            rootObject.transform.SetParent(transform, false);
            root = rootObject.transform;
        }

        string[] names = { "Blue Quarter", "Purple Quarter", "Green Quarter", "Pink Quarter" };
        Color[] colors = { blue, purple, green, pink };
        for (int quarter = 0; quarter < 4; quarter++)
        {
            Transform child = root.Find(names[quarter]);
            if (child == null)
            {
                var childObject = new GameObject(names[quarter]);
                childObject.transform.SetParent(root, false);
                child = childObject.transform;
            }

            var filter = child.GetComponent<MeshFilter>();
            if (filter == null) filter = child.gameObject.AddComponent<MeshFilter>();
            var renderer = child.GetComponent<MeshRenderer>();
            if (renderer == null) renderer = child.gameObject.AddComponent<MeshRenderer>();

            filter.sharedMesh = CreateQuarterMesh(quarter);
            renderer.sharedMaterial = CreateMaterial(names[quarter], colors[quarter]);
            renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            renderer.receiveShadows = false;
        }
    }

    Mesh CreateQuarterMesh(int quarter)
    {
        var vertices = new List<Vector3>();
        var triangles = new List<int>();
        float slot = 90f / dashesPerQuarter;
        float dashAngle = Mathf.Max(0.1f, slot - gapDegrees);
        float quarterStart = quarter * 90f;

        for (int ring = 0; ring < 2; ring++)
        {
            float radius = ring == 0 ? innerRadius : outerRadius;
            for (int dash = 0; dash < dashesPerQuarter; dash++)
            {
                float start = quarterStart + dash * slot + gapDegrees * 0.5f;
                AddArc(vertices, triangles, radius, start, start + dashAngle);
            }
        }

        var mesh = new Mesh { name = $"Broken Ring Quarter {quarter + 1}" };
        mesh.SetVertices(vertices);
        mesh.SetTriangles(triangles, 0);
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        return mesh;
    }

    void AddArc(List<Vector3> vertices, List<int> triangles, float radius, float startDegrees, float endDegrees)
    {
        int first = vertices.Count;
        float halfWidth = bandWidth * 0.5f;
        for (int i = 0; i <= subdivisionsPerDash; i++)
        {
            float t = i / (float)subdivisionsPerDash;
            float angle = Mathf.Lerp(startDegrees, endDegrees, t) * Mathf.Deg2Rad;
            float x = Mathf.Cos(angle);
            float z = Mathf.Sin(angle);
            vertices.Add(new Vector3(x * (radius - halfWidth), heightOffset, z * (radius - halfWidth)));
            vertices.Add(new Vector3(x * (radius + halfWidth), heightOffset, z * (radius + halfWidth)));
        }

        for (int i = 0; i < subdivisionsPerDash; i++)
        {
            int a = first + i * 2;
            triangles.Add(a);
            triangles.Add(a + 1);
            triangles.Add(a + 2);
            triangles.Add(a + 1);
            triangles.Add(a + 3);
            triangles.Add(a + 2);
        }
    }

    Material CreateMaterial(string materialName, Color color)
    {
        Shader shader = Shader.Find("Music/Emissive Ring");
        if (shader == null) shader = Shader.Find("Unlit/Color");
        var material = new Material(shader) { name = materialName + " Emissive" };
        Color hdr = color * intensity;
        material.color = color;
        if (material.HasProperty(BaseColorId)) material.SetColor(BaseColorId, hdr);
        if (material.HasProperty("_Color")) material.SetColor("_Color", hdr);
        if (material.HasProperty("_Cull")) material.SetFloat("_Cull", 0f);
        if (material.HasProperty("_CullMode")) material.SetFloat("_CullMode", 0f);
        if (material.HasProperty("_Surface")) material.SetFloat("_Surface", 0f);
        if (material.HasProperty(EmissionColorId))
        {
            material.EnableKeyword("_EMISSION");
            material.SetColor(EmissionColorId, hdr);
        }
        return material;
    }
}
