using System.Collections.Generic;
using UnityEngine;

// Lives on the moving EnergyNova star. It owns all score-driven visual changes
// so the road/progress controller only has to move the star and pass in scores.
public class DeadBugScoreStarFeedback : MonoBehaviour
{
    [Header("Score Opacity")]
    [Range(0f, 1f)] [SerializeField] private float minimumOpacity = 0.2f;
    [Range(0f, 1f)] [SerializeField] private float maximumOpacity = 1f;

    [Header("Score Particles")]
    [Tooltip("Manual particles emitted per second by every child particle system at score 0.")]
    [Min(0f)] [SerializeField] private float minimumParticlesPerSecond = 1f;
    [Tooltip("Manual particles emitted per second by every child particle system at score 100.")]
    [Min(0f)] [SerializeField] private float maximumParticlesPerSecond = 12f;

    [Header("Renderer Color Properties")]
    [SerializeField] private string[] colorPropertyNames =
    {
        "_TintColor",
        "_BaseColor",
        "_Color"
    };

    private readonly List<RendererColorTarget> rendererTargets = new List<RendererColorTarget>();
    private ParticleSystem[] particleSystems;
    private MaterialPropertyBlock propertyBlock;
    private float score01;
    private Color scoreColor = Color.red;
    private float particleAccumulator;
    private bool sessionActive;

    private sealed class RendererColorTarget
    {
        public Renderer renderer;
        public string propertyName;
        public Color baseColor;
    }

    public float Score => score01 * 100f;

    private void Awake()
    {
        CacheVisuals();
        foreach (ParticleSystem particleSystem in particleSystems)
        {
            if (particleSystem == null)
            {
                continue;
            }

            particleSystem.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
            ParticleSystem.EmissionModule emission = particleSystem.emission;
            emission.enabled = false;
        }
        ApplyVisuals();
    }

    private void Update()
    {
        if (!sessionActive || particleSystems == null || particleSystems.Length == 0)
        {
            return;
        }

        float particlesPerSecond = Mathf.Lerp(
            minimumParticlesPerSecond,
            maximumParticlesPerSecond,
            score01);
        particleAccumulator += Mathf.Max(0f, particlesPerSecond) * Time.deltaTime;
        int emitCount = Mathf.FloorToInt(particleAccumulator);
        if (emitCount <= 0)
        {
            return;
        }

        particleAccumulator -= emitCount;
        foreach (ParticleSystem particleSystem in particleSystems)
        {
            if (particleSystem != null)
            {
                particleSystem.Emit(emitCount);
            }
        }
    }

    public void BeginSession(float initialScore = 0f)
    {
        BeginSession(initialScore, scoreColor);
    }

    public void BeginSession(float initialScore, Color color)
    {
        if (particleSystems == null)
        {
            CacheVisuals();
        }

        sessionActive = true;
        particleAccumulator = 0f;
        SetScore(initialScore, color);

        foreach (ParticleSystem particleSystem in particleSystems)
        {
            if (particleSystem == null)
            {
                continue;
            }

            particleSystem.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
            ParticleSystem.EmissionModule emission = particleSystem.emission;
            emission.enabled = false;
            particleSystem.Play();
        }
    }

    public void EndSession(float finalScore)
    {
        EndSession(finalScore, scoreColor);
    }

    public void EndSession(float finalScore, Color color)
    {
        SetScore(finalScore, color);
        sessionActive = false;

        if (particleSystems == null)
        {
            return;
        }

        foreach (ParticleSystem particleSystem in particleSystems)
        {
            if (particleSystem != null)
            {
                particleSystem.Stop(true, ParticleSystemStopBehavior.StopEmitting);
            }
        }
    }

    public void SetScore(float score)
    {
        SetScore(score, scoreColor);
    }

    public void SetScore(float score, Color color)
    {
        score01 = Mathf.Clamp01(score / 100f);
        scoreColor = color;
        ApplyVisuals();
    }

    private void CacheVisuals()
    {
        particleSystems = GetComponentsInChildren<ParticleSystem>(true);
        propertyBlock = new MaterialPropertyBlock();
        rendererTargets.Clear();

        Renderer[] renderers = GetComponentsInChildren<Renderer>(true);
        foreach (Renderer targetRenderer in renderers)
        {
            if (targetRenderer == null || targetRenderer.sharedMaterial == null)
            {
                continue;
            }

            Material material = targetRenderer.sharedMaterial;
            foreach (string propertyName in colorPropertyNames)
            {
                if (!string.IsNullOrEmpty(propertyName) && material.HasProperty(propertyName))
                {
                    rendererTargets.Add(new RendererColorTarget
                    {
                        renderer = targetRenderer,
                        propertyName = propertyName,
                        baseColor = material.GetColor(propertyName)
                    });
                    break;
                }
            }
        }
    }

    private void ApplyVisuals()
    {
        if (propertyBlock == null)
        {
            return;
        }

        float opacity = Mathf.Lerp(minimumOpacity, maximumOpacity, score01);
        foreach (RendererColorTarget target in rendererTargets)
        {
            if (target.renderer == null)
            {
                continue;
            }

            target.renderer.GetPropertyBlock(propertyBlock);
            Color color = scoreColor;
            color.a *= opacity;
            propertyBlock.SetColor(target.propertyName, color);
            target.renderer.SetPropertyBlock(propertyBlock);
            propertyBlock.Clear();
        }
    }
}
