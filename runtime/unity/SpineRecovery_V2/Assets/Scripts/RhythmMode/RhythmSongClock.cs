using UnityEngine;

public sealed class RhythmSongClock : MonoBehaviour
{
    [SerializeField] private AudioSource musicSource;
    [SerializeField] private AudioClip song;
    [Range(0f, 1f)] [SerializeField] private float musicVolume = 0.32f;
    [Min(0f)] [SerializeField] private float scheduleLeadIn = 0.15f;
    [SerializeField] private double calibrationOffsetSeconds;

    private double startDspTime;
    private bool running;

    public bool IsRunning => running;
    public double SongTime => running
        ? AudioSettings.dspTime - startDspTime + calibrationOffsetSeconds
        : 0d;

    private void Awake()
    {
        if (musicSource == null)
        {
            musicSource = gameObject.AddComponent<AudioSource>();
        }

        musicSource.playOnAwake = false;
        musicSource.loop = true;
        musicSource.spatialBlend = 0f;
        musicSource.volume = musicVolume;
    }

    public void StartMusicImmediately()
    {
        AudioClip clip = song != null ? song : musicSource.clip;
        if (clip == null)
        {
            return;
        }

        musicSource.clip = clip;
        musicSource.loop = true;
        if (!musicSource.isPlaying)
        {
            musicSource.Play();
        }
    }

    public void StartClock()
    {
        double now = AudioSettings.dspTime;
        startDspTime = now;
        running = true;

        AudioClip clip = song != null ? song : musicSource.clip;
        if (clip != null && !musicSource.isPlaying)
        {
            musicSource.clip = clip;
            musicSource.loop = true;
            startDspTime = now + scheduleLeadIn;
            musicSource.PlayScheduled(startDspTime);
        }
    }

    public void StopClock()
    {
        if (musicSource != null)
        {
            musicSource.Stop();
        }
        running = false;
    }
}
