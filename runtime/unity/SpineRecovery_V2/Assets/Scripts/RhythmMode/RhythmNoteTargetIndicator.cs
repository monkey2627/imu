using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public sealed class RhythmNoteTargetIndicator : MonoBehaviour
{
    private float radius;
    private Mesh indicatorMesh;
    private Material indicatorMaterial;
    private Material particleMaterial;
    private Texture2D particleTexture;

    public static RhythmNoteTargetIndicator Create(
        Transform parent,
        Vector3 localPosition,
        float ringRadius,
        float ringWidth,
        int dashCount,
        float dashFill,
        Color color,
        GameObject authoredVisualPrefab,
        bool showParticles,
        float particleRate,
        float particleSize)
    {
        GameObject indicatorObject = new GameObject("NoteArrivalIndicator");
        indicatorObject.transform.SetParent(parent, false);
        indicatorObject.transform.localPosition = localPosition + new Vector3(0f, 0f, 0.006f);

        RhythmNoteTargetIndicator indicator = indicatorObject.AddComponent<RhythmNoteTargetIndicator>();
        if (authoredVisualPrefab != null)
        {
            indicator.BuildAuthored(authoredVisualPrefab, ringRadius, color);
        }
        else
        {
            indicator.BuildProcedural(ringRadius, ringWidth, dashCount, dashFill, color);
        }
        if (showParticles)
        {
            indicator.BuildRingParticles(color, particleRate, particleSize);
        }
        return indicator;
    }

    public bool ContainsNote(Vector3 noteLocalPosition, float noteRadius)
    {
        Vector2 noteCenter = new Vector2(noteLocalPosition.x, noteLocalPosition.y);
        Vector2 ringCenter = new Vector2(transform.localPosition.x, transform.localPosition.y);
        float overlapRadius = radius + noteRadius * 0.25f;
        return Vector2.Distance(noteCenter, ringCenter) <= overlapRadius;
    }

    private void BuildAuthored(GameObject visualPrefab, float ringRadius, Color color)
    {
        radius = Mathf.Max(0.02f, ringRadius);
        GameObject visual = Instantiate(visualPrefab, transform, false);
        visual.name = "AuthoredArrivalRing";
        visual.transform.localPosition = Vector3.zero;
        visual.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
        visual.transform.localScale = Vector3.one;

        MeshRenderer[] renderers = visual.GetComponentsInChildren<MeshRenderer>(true);
        if (renderers.Length == 0)
        {
            visual.SetActive(false);
            BuildProcedural(radius, 0.018f, 16, 0.55f, color);
            return;
        }

        Bounds bounds = renderers[0].bounds;
        for (int index = 1; index < renderers.Length; index++)
        {
            bounds.Encapsulate(renderers[index].bounds);
        }

        Vector3 localCenter = transform.InverseTransformPoint(bounds.center);
        float visibleDiameter = Mathf.Max(bounds.size.x, bounds.size.y);
        if (visibleDiameter > 0.000001f)
        {
            float scale = radius * 2f / visibleDiameter;
            visual.transform.localScale = Vector3.one * scale;
            visual.transform.localPosition = -localCenter * scale;
        }

        indicatorMaterial = CreateIndicatorMaterial(color);
        foreach (MeshRenderer meshRenderer in renderers)
        {
            meshRenderer.sharedMaterial = indicatorMaterial;
            meshRenderer.sortingOrder = 18;
        }
    }

    private void BuildProcedural(float ringRadius, float ringWidth, int dashCount, float dashFill, Color color)
    {
        radius = Mathf.Max(0.02f, ringRadius);
        float halfWidth = Mathf.Max(0.001f, ringWidth) * 0.5f;
        int safeDashCount = Mathf.Clamp(dashCount, 6, 32);
        float safeDashFill = Mathf.Clamp(dashFill, 0.2f, 0.85f);

        indicatorMesh = BuildDashedRingMesh(radius, halfWidth, safeDashCount, safeDashFill);
        indicatorMesh.name = "Runtime Note Arrival Dashed Ring";
        GetComponent<MeshFilter>().sharedMesh = indicatorMesh;

        indicatorMaterial = CreateIndicatorMaterial(color);

        MeshRenderer meshRenderer = GetComponent<MeshRenderer>();
        meshRenderer.sharedMaterial = indicatorMaterial;
        meshRenderer.sortingOrder = 18;
    }

    private static Material CreateIndicatorMaterial(Color color)
    {
        Shader shader = Shader.Find("Standard") ??
            Shader.Find("Sprites/Default") ??
            Shader.Find("Unlit/Color");
        Material material = new Material(shader)
        {
            name = "Runtime Note Arrival Indicator",
            color = color,
            renderQueue = 3010
        };
        if (material.HasProperty("_BaseColor"))
        {
            material.SetColor("_BaseColor", color);
        }
        if (material.HasProperty("_EmissionColor"))
        {
            material.EnableKeyword("_EMISSION");
            material.SetColor("_EmissionColor", color * 2.5f);
        }
        return material;
    }

    private void BuildRingParticles(Color color, float emissionRate, float particleSize)
    {
        GameObject particleObject = new GameObject("ArrivalRingParticles");
        particleObject.transform.SetParent(transform, false);
        particleObject.transform.localPosition = new Vector3(0f, 0f, -0.004f);

        ParticleSystem particles = particleObject.AddComponent<ParticleSystem>();
        particles.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);

        ParticleSystem.MainModule main = particles.main;
        main.loop = true;
        main.playOnAwake = true;
        main.duration = 1f;
        main.startLifetime = new ParticleSystem.MinMaxCurve(0.35f, 0.65f);
        main.startSpeed = new ParticleSystem.MinMaxCurve(0.005f, 0.025f);
        main.startSize = new ParticleSystem.MinMaxCurve(
            Mathf.Max(0.002f, particleSize) * 0.65f,
            Mathf.Max(0.002f, particleSize) * 1.35f);
        main.startColor = new ParticleSystem.MinMaxGradient(
            new Color(color.r, color.g, color.b, 0.45f),
            new Color(1f, 1f, 1f, 0.95f));
        main.simulationSpace = ParticleSystemSimulationSpace.Local;
        main.scalingMode = ParticleSystemScalingMode.Hierarchy;
        main.maxParticles = 48;

        ParticleSystem.EmissionModule emission = particles.emission;
        emission.rateOverTime = Mathf.Max(1f, emissionRate);

        ParticleSystem.ShapeModule shape = particles.shape;
        shape.enabled = true;
        shape.shapeType = ParticleSystemShapeType.Circle;
        shape.radius = radius;
        shape.radiusThickness = 0.08f;
        shape.arc = 360f;

        ParticleSystem.ColorOverLifetimeModule colorOverLifetime = particles.colorOverLifetime;
        colorOverLifetime.enabled = true;
        Gradient alphaGradient = new Gradient();
        alphaGradient.SetKeys(
            new[]
            {
                new GradientColorKey(Color.white, 0f),
                new GradientColorKey(Color.white, 1f)
            },
            new[]
            {
                new GradientAlphaKey(0f, 0f),
                new GradientAlphaKey(1f, 0.18f),
                new GradientAlphaKey(0f, 1f)
            });
        colorOverLifetime.color = alphaGradient;

        ParticleSystemRenderer particleRenderer = particles.GetComponent<ParticleSystemRenderer>();
        particleRenderer.renderMode = ParticleSystemRenderMode.Billboard;
        particleRenderer.sharedMaterial = CreateParticleMaterial();
        particleRenderer.sortingOrder = 19;
        particles.Play();
    }

    private Material CreateParticleMaterial()
    {
        Shader shader = Shader.Find("Legacy Shaders/Particles/Additive") ??
            Shader.Find("Particles/Standard Unlit") ??
            Shader.Find("Sprites/Default");
        particleTexture = CreateSoftParticleTexture();
        particleMaterial = new Material(shader)
        {
            name = "Runtime Arrival Ring Particles",
            mainTexture = particleTexture,
            renderQueue = 3020
        };
        if (particleMaterial.HasProperty("_TintColor"))
        {
            particleMaterial.SetColor("_TintColor", Color.white);
        }
        return particleMaterial;
    }

    private static Texture2D CreateSoftParticleTexture()
    {
        const int size = 16;
        Texture2D texture = new Texture2D(size, size, TextureFormat.RGBA32, false)
        {
            name = "Runtime Soft Particle",
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
        Color[] pixels = new Color[size * size];
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float nx = (x + 0.5f) / size * 2f - 1f;
                float ny = (y + 0.5f) / size * 2f - 1f;
                float alpha = Mathf.Clamp01(1f - Mathf.Sqrt(nx * nx + ny * ny));
                alpha *= alpha;
                pixels[y * size + x] = new Color(1f, 1f, 1f, alpha);
            }
        }
        texture.SetPixels(pixels);
        texture.Apply(false, true);
        return texture;
    }

    private static Mesh BuildDashedRingMesh(
        float ringRadius,
        float halfWidth,
        int dashCount,
        float dashFill)
    {
        Vector3[] vertices = new Vector3[dashCount * 4];
        int[] triangles = new int[dashCount * 6];
        float angleStep = Mathf.PI * 2f / dashCount;
        float halfDashAngle = angleStep * dashFill * 0.5f;
        float innerRadius = Mathf.Max(0.001f, ringRadius - halfWidth);
        float outerRadius = ringRadius + halfWidth;

        for (int index = 0; index < dashCount; index++)
        {
            float centerAngle = index * angleStep;
            float startAngle = centerAngle - halfDashAngle;
            float endAngle = centerAngle + halfDashAngle;
            int vertexIndex = index * 4;
            int triangleIndex = index * 6;

            vertices[vertexIndex] = PointOnCircle(startAngle, innerRadius);
            vertices[vertexIndex + 1] = PointOnCircle(startAngle, outerRadius);
            vertices[vertexIndex + 2] = PointOnCircle(endAngle, outerRadius);
            vertices[vertexIndex + 3] = PointOnCircle(endAngle, innerRadius);

            triangles[triangleIndex] = vertexIndex;
            triangles[triangleIndex + 1] = vertexIndex + 1;
            triangles[triangleIndex + 2] = vertexIndex + 2;
            triangles[triangleIndex + 3] = vertexIndex;
            triangles[triangleIndex + 4] = vertexIndex + 2;
            triangles[triangleIndex + 5] = vertexIndex + 3;
        }

        Mesh mesh = new Mesh();
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateBounds();
        return mesh;
    }

    private static Vector3 PointOnCircle(float angle, float radius)
    {
        return new Vector3(Mathf.Cos(angle) * radius, Mathf.Sin(angle) * radius, 0f);
    }

    private void OnDestroy()
    {
        if (indicatorMesh != null)
        {
            Destroy(indicatorMesh);
        }
        if (indicatorMaterial != null)
        {
            Destroy(indicatorMaterial);
        }
        if (particleMaterial != null)
        {
            Destroy(particleMaterial);
        }
        if (particleTexture != null)
        {
            Destroy(particleTexture);
        }
    }
}
