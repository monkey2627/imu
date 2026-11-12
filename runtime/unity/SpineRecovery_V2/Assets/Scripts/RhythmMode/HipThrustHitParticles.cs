using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

// Canvas-space particles stay visible over the taiko artwork in both desktop and glasses UI.
[RequireComponent(typeof(CanvasRenderer))]
public sealed class HipThrustHitParticles : MaskableGraphic
{
    struct Spark { public Vector2 position, velocity; public float age, life, size; public Color tint; }
    readonly List<Spark> sparks = new List<Spark>(128);
    Vector2 ringCenter;
    float ringAge = 1f;
    public bool Paused { get; set; }
    public int ParticleCount => sparks.Count;
    public bool RingVisible => ringAge < 0.45f;

    public void Burst(Vector2 center, Vector2 tailStart, Vector2 tailEnd)
    {
        ringCenter = center;
        ringAge = 0f;
        for (int i = 0; i < 28; i++)
        {
            float angle = i * Mathf.PI * 2f / 28;
            Vector2 direction = new Vector2(Mathf.Cos(angle), Mathf.Sin(angle));
            Add(center + direction * 58f, direction * Random.Range(70f, 165f), new Color(1f, 0.87f, 0.32f), 7f, 0.55f);
        }
        Trail(tailStart, tailEnd, 22);
        SetVerticesDirty();
    }

    public void Trail(Vector2 from, Vector2 to, int count = 3)
    {
        for (int i = 0; i < count; i++)
            Add(Vector2.Lerp(from, to, Random.value) + new Vector2(0, Random.Range(-12f, 12f)),
                new Vector2(Random.Range(-25f, 25f), Random.Range(-65f, 65f)),
                new Color(0.55f, 0.95f, 1f), Random.Range(4f, 8f), 0.4f);
        SetVerticesDirty();
    }

    void Add(Vector2 position, Vector2 velocity, Color tint, float size, float life)
    {
        if (sparks.Count >= 160) sparks.RemoveAt(0);
        sparks.Add(new Spark { position = position, velocity = velocity, tint = tint, size = size, life = life });
    }

    void Update()
    {
        if (Paused || (sparks.Count == 0 && !RingVisible)) return;
        ringAge += Time.deltaTime;
        for (int i = sparks.Count - 1; i >= 0; i--)
        {
            Spark spark = sparks[i];
            spark.age += Time.deltaTime;
            if (spark.age >= spark.life) { sparks.RemoveAt(i); continue; }
            spark.position += spark.velocity * Time.deltaTime;
            sparks[i] = spark;
        }
        SetVerticesDirty();
    }

    public void Clear()
    {
        sparks.Clear();
        ringAge = 1f;
        Paused = false;
        SetVerticesDirty();
    }

    protected override void OnPopulateMesh(VertexHelper mesh)
    {
        mesh.Clear();
        if (RingVisible)
        {
            float t = ringAge / 0.45f, radius = 70f + 45f * t;
            Color tint = new Color(1f, 0.88f, 0.35f, 1f - t);
            for (int i = 0; i < 64; i++)
            {
                float a = i * Mathf.PI / 32f, b = (i + 1) * Mathf.PI / 32f;
                Vector2 da = new Vector2(Mathf.Cos(a), Mathf.Sin(a)), db = new Vector2(Mathf.Cos(b), Mathf.Sin(b));
                Quad(mesh, ringCenter + da * radius, ringCenter + db * radius,
                    ringCenter + db * (radius + 7f), ringCenter + da * (radius + 7f), tint);
            }
        }
        foreach (Spark spark in sparks)
        {
            float fade = 1f - spark.age / spark.life, size = spark.size * (0.4f + fade * 0.6f);
            Color tint = spark.tint; tint.a = fade;
            Quad(mesh, spark.position + Vector2.up * size * 1.6f, spark.position + Vector2.right * size,
                spark.position + Vector2.down * size * 1.6f, spark.position + Vector2.left * size, tint);
        }
    }

    static void Quad(VertexHelper mesh, Vector2 a, Vector2 b, Vector2 c, Vector2 d, Color tint)
    {
        int index = mesh.currentVertCount;
        mesh.AddVert(a, tint, Vector2.zero); mesh.AddVert(b, tint, Vector2.zero);
        mesh.AddVert(c, tint, Vector2.zero); mesh.AddVert(d, tint, Vector2.zero);
        mesh.AddTriangle(index, index + 1, index + 2); mesh.AddTriangle(index, index + 2, index + 3);
    }
}
