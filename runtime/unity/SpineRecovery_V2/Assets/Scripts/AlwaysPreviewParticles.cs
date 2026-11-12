using UnityEngine;

[ExecuteAlways]
[DisallowMultipleComponent]
public sealed class AlwaysPreviewParticles : MonoBehaviour
{
    ParticleSystem[] particleSystems;

    void OnEnable()
    {
        particleSystems = GetComponentsInChildren<ParticleSystem>(true);
        PlayAll();
    }

    void Update()
    {
        if (particleSystems == null || particleSystems.Length == 0)
            particleSystems = GetComponentsInChildren<ParticleSystem>(true);

        foreach (ParticleSystem particleSystem in particleSystems)
        {
            if (particleSystem == null)
                continue;

            if (IsStaticMusicFace(particleSystem))
            {
                if (!particleSystem.isPaused)
                    FreezeAtVisibleFrame(particleSystem);
                continue;
            }

            if (!particleSystem.isPlaying && !particleSystem.isPaused)
                particleSystem.Play(true);
        }
    }

    [ContextMenu("Play All Particles")]
    public void PlayAll()
    {
        if (particleSystems == null || particleSystems.Length == 0)
            particleSystems = GetComponentsInChildren<ParticleSystem>(true);

        foreach (ParticleSystem particleSystem in particleSystems)
        {
            if (particleSystem == null)
                continue;

            if (IsStaticMusicFace(particleSystem))
                FreezeAtVisibleFrame(particleSystem);
            else
                particleSystem.Play(true);
        }
    }

    static bool IsStaticMusicFace(ParticleSystem particleSystem)
    {
        string objectName = particleSystem.gameObject.name;
        return objectName == "Spherical_Explosion_Disco"
            || objectName == "Spherical_Explosion_Disco_Add";
    }

    static void FreezeAtVisibleFrame(ParticleSystem particleSystem)
    {
        particleSystem.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
        particleSystem.Simulate(0.5f, true, true, true);
        particleSystem.Pause(true);
    }
}
