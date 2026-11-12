using System.Collections;
using TMPro;
using UnityEngine;

public sealed class RhythmHitFeedback : MonoBehaviour
{
    [SerializeField] private RhythmSkinCVisuals visualSkin;
    [Header("Replaceable Visual Assets")]
    [Tooltip("Optional success effect prefab. Its particle systems are played automatically.")]
    [SerializeField] private GameObject successEffectPrefab;
    [Tooltip("Optional MISS prefab. Leave empty to use generated white TextMeshPro text.")]
    [SerializeField] private GameObject missFeedbackPrefab;

    [Header("Audio")]
    [SerializeField] private AudioSource hitAudioSource;
    [SerializeField] private AudioClip hitSound;
    [SerializeField] private AudioClip shatterSound;
    [Range(0f, 1f)] [SerializeField] private float hitVolume = 1f;
    [Range(0f, 1f)] [SerializeField] private float shatterVolume = 0.9f;
    [Range(0f, 0.2f)] [SerializeField] private float pitchVariation = 0.035f;

    [Range(4, 40)] [SerializeField] private int burstCount = 16;
    [Min(0.05f)] [SerializeField] private float particleLifetime = 0.42f;
    [Min(0.01f)] [SerializeField] private float particleSize = 0.045f;
    [Min(0.05f)] [SerializeField] private float particleSpeed = 0.48f;

    [Header("Hit Feedback")]
    [Min(0.1f)] [SerializeField] private float hitDuration = 0.65f;
    [Min(0.1f)] [SerializeField] private float hitFontSize = 6f;
    [Min(0.001f)] [SerializeField] private float hitTextScale = 0.24f;

    [Header("Miss Feedback")]
    [Min(0.1f)] [SerializeField] private float missDuration = 0.65f;
    [Min(0.1f)] [SerializeField] private float missFontSize = 6f;
    [Min(0.001f)] [SerializeField] private float missTextScale = 0.24f;

    private Material particleMaterial;

    private void Awake()
    {
        if (hitAudioSource == null)
        {
            hitAudioSource = gameObject.AddComponent<AudioSource>();
        }

        hitAudioSource.playOnAwake = false;
        hitAudioSource.loop = false;
        hitAudioSource.spatialBlend = 0f;
    }

    private void OnDestroy()
    {
        if (particleMaterial != null)
        {
            Destroy(particleMaterial);
        }
    }

    public void Play(RhythmNoteResult result, Color targetColor, Color hitTextColor)
    {
        if (visualSkin != null)
        {
            if (result.IsHit) PlayHitAudio();
            visualSkin.PlayResult(result);
            return;
        }
        if (result.Judgement == RhythmJudgement.Miss)
        {
            PlayMiss(result.LocalHitPosition);
            return;
        }

        PlayHitAudio();
        PlayHit(result.LocalHitPosition, hitTextColor);

        if (successEffectPrefab != null)
        {
            PlaySuccessPrefab(result.LocalHitPosition);
            return;
        }

        GameObject effectObject = new GameObject($"{result.Target}_{result.Judgement}_Burst");
        effectObject.transform.SetParent(transform, false);
        effectObject.transform.localPosition = result.LocalHitPosition;

        ParticleSystem particles = effectObject.AddComponent<ParticleSystem>();
        particles.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
        ParticleSystem.MainModule main = particles.main;
        main.loop = false;
        main.playOnAwake = false;
        main.duration = 0.05f;
        main.startLifetime = particleLifetime;
        main.startSpeed = particleSpeed;
        main.startSize = particleSize;
        main.startColor = targetColor;
        main.simulationSpace = ParticleSystemSimulationSpace.Local;
        main.maxParticles = burstCount;

        ParticleSystem.EmissionModule emission = particles.emission;
        emission.rateOverTime = 0f;
        emission.SetBursts(new[]
        {
            new ParticleSystem.Burst(0f, (short)burstCount)
        });

        ParticleSystem.ShapeModule shape = particles.shape;
        shape.enabled = true;
        shape.shapeType = ParticleSystemShapeType.Circle;
        shape.radius = 0.035f;
        shape.radiusThickness = 1f;

        ParticleSystemRenderer renderer = particles.GetComponent<ParticleSystemRenderer>();
        renderer.sharedMaterial = GetParticleMaterial();
        renderer.sortingOrder = 30;
        particles.Play();
        Destroy(effectObject, particleLifetime + 0.25f);
    }

    private void PlayHitAudio()
    {
        if (hitAudioSource == null)
        {
            return;
        }

        hitAudioSource.pitch = 1f + Random.Range(-pitchVariation, pitchVariation);
        if (hitSound != null)
        {
            hitAudioSource.PlayOneShot(hitSound, hitVolume);
        }

        if (shatterSound != null)
        {
            hitAudioSource.PlayOneShot(shatterSound, shatterVolume);
        }
    }

    private void PlayHit(Vector3 localPosition, Color textColor)
    {
        GameObject hitObject = new GameObject("RhythmHit");
        hitObject.transform.SetParent(transform, false);
        hitObject.transform.localPosition = localPosition + new Vector3(0f, 0.035f, -0.012f);
        hitObject.transform.localRotation = Quaternion.identity;
        hitObject.transform.localScale = Vector3.one * hitTextScale;

        TextMeshPro text = hitObject.AddComponent<TextMeshPro>();
        text.text = "HIT";
        text.color = textColor;
        text.fontSize = hitFontSize;
        text.fontStyle = FontStyles.Bold;
        text.alignment = TextAlignmentOptions.Center;
        text.enableWordWrapping = false;
        text.rectTransform.sizeDelta = new Vector2(3f, 1f);
        text.GetComponent<MeshRenderer>().sortingOrder = 41;
        StartCoroutine(AnimateHit(text, textColor));
    }

    private void PlayMiss(Vector3 localPosition)
    {
        if (missFeedbackPrefab != null)
        {
            GameObject customMiss = Instantiate(missFeedbackPrefab, transform);
            customMiss.name = "RhythmMiss";
            customMiss.transform.localPosition = localPosition;
            customMiss.transform.localRotation = Quaternion.identity;
            customMiss.SetActive(true);
            Destroy(customMiss, missDuration);
            return;
        }

        GameObject missObject = new GameObject("RhythmMiss");
        missObject.transform.SetParent(transform, false);
        missObject.transform.localPosition = localPosition + new Vector3(0f, 0.035f, -0.01f);
        missObject.transform.localRotation = Quaternion.identity;
        missObject.transform.localScale = Vector3.one * missTextScale;

        TextMeshPro text = missObject.AddComponent<TextMeshPro>();
        text.text = "MISS";
        text.color = Color.white;
        text.fontSize = missFontSize;
        text.fontStyle = FontStyles.Bold;
        text.alignment = TextAlignmentOptions.Center;
        text.enableWordWrapping = false;
        text.rectTransform.sizeDelta = new Vector2(3f, 1f);
        text.GetComponent<MeshRenderer>().sortingOrder = 40;
        StartCoroutine(AnimateMiss(text));
    }

    private void PlaySuccessPrefab(Vector3 localPosition)
    {
        GameObject effect = Instantiate(successEffectPrefab, transform);
        effect.name = "RhythmSuccessEffect";
        effect.transform.localPosition = localPosition;
        effect.transform.localRotation = Quaternion.identity;
        effect.SetActive(true);

        float lifetime = particleLifetime + 0.5f;
        foreach (ParticleSystem particles in effect.GetComponentsInChildren<ParticleSystem>(true))
        {
            particles.Play(true);
            ParticleSystem.MainModule main = particles.main;
            lifetime = Mathf.Max(lifetime, main.duration + main.startLifetime.constantMax);
        }
        Destroy(effect, lifetime);
    }

    private IEnumerator AnimateMiss(TextMeshPro text)
    {
        float elapsed = 0f;
        Vector3 startPosition = text.transform.localPosition;
        while (text != null && elapsed < missDuration)
        {
            elapsed += Time.unscaledDeltaTime;
            float progress = Mathf.Clamp01(elapsed / missDuration);
            Color color = Color.white;
            color.a = 1f - progress;
            text.color = color;
            text.transform.localPosition = startPosition + Vector3.up * (0.08f * progress);
            yield return null;
        }

        if (text != null)
        {
            Destroy(text.gameObject);
        }
    }

    private IEnumerator AnimateHit(TextMeshPro text, Color textColor)
    {
        float elapsed = 0f;
        Vector3 startPosition = text.transform.localPosition;
        while (text != null && elapsed < hitDuration)
        {
            elapsed += Time.unscaledDeltaTime;
            float progress = Mathf.Clamp01(elapsed / hitDuration);
            Color color = textColor;
            color.a = 1f - progress;
            text.color = color;
            text.transform.localPosition = startPosition + Vector3.up * (0.08f * progress);
            yield return null;
        }

        if (text != null)
        {
            Destroy(text.gameObject);
        }
    }

    private Material GetParticleMaterial()
    {
        if (particleMaterial != null)
        {
            return particleMaterial;
        }

        Shader shader = Shader.Find("Universal Render Pipeline/Particles/Unlit") ??
            Shader.Find("Particles/Standard Unlit") ??
            Shader.Find("Sprites/Default");
        particleMaterial = new Material(shader)
        {
            name = "Runtime Rhythm Particle Material",
            renderQueue = 3100
        };
        return particleMaterial;
    }
}
