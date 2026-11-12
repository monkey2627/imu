using System;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public sealed class ReplayUIController : MonoBehaviour
{
    [SerializeField] private ReplayPlayer player;
    [SerializeField] private TMP_Text stateText;
    [SerializeField] private TMP_Text timeText;
    [SerializeField] private TMP_Text errorText;
    [SerializeField] private Button playResumeButton;
    [SerializeField] private Button pauseButton;
    [SerializeField] private Button restartButton;
    [SerializeField] private Button backFiveButton;
    [SerializeField] private Button forwardFiveButton;
    [SerializeField] private Slider timelineSlider;

    private string sourceNotice;

    public void Configure(
        ReplayPlayer replayPlayer,
        TMP_Text status,
        TMP_Text time,
        TMP_Text errorLabel,
        Button playResume,
        Button pause,
        Button restart,
        Button backFive,
        Button forwardFive,
        Slider timeline,
        bool loadedFromLatest)
    {
        Unbind();
        player = replayPlayer;
        stateText = status;
        timeText = time;
        errorText = errorLabel;
        playResumeButton = playResume;
        pauseButton = pause;
        restartButton = restart;
        backFiveButton = backFive;
        forwardFiveButton = forwardFive;
        timelineSlider = timeline;
        sourceNotice = loadedFromLatest ? " (latest completed)" : string.Empty;
        Bind();
        Refresh();
    }

    public void ShowLoadError(string message)
    {
        if (errorText != null)
        {
            errorText.text = message;
            errorText.gameObject.SetActive(true);
        }

        Refresh();
    }

    public void PlayOrResume() => player?.Play();
    public void Pause() => player?.Pause();
    public void Restart() => player?.Restart();
    public void BackFiveSeconds() => player?.SeekRelative(-5f);
    public void ForwardFiveSeconds() => player?.SeekRelative(5f);

    private void Awake()
    {
        Bind();
    }

    private void OnDestroy()
    {
        Unbind();
    }

    private void Update()
    {
        RefreshTime();
    }

    private void Bind()
    {
        if (player != null)
        {
            player.StateChanged -= HandlePlayerStateChanged;
            player.StateChanged += HandlePlayerStateChanged;
        }

        BindButton(playResumeButton, PlayOrResume);
        BindButton(pauseButton, Pause);
        BindButton(restartButton, Restart);
        BindButton(backFiveButton, BackFiveSeconds);
        BindButton(forwardFiveButton, ForwardFiveSeconds);
        // PlaybackProgressSlider is the sole owner of timeline value changes.
        // Registering a second listener here caused programmatic progress
        // updates to be interpreted as seeks, pausing playback every frame.
    }

    private void Unbind()
    {
        if (player != null)
        {
            player.StateChanged -= HandlePlayerStateChanged;
        }

        UnbindButton(playResumeButton, PlayOrResume);
        UnbindButton(pauseButton, Pause);
        UnbindButton(restartButton, Restart);
        UnbindButton(backFiveButton, BackFiveSeconds);
        UnbindButton(forwardFiveButton, ForwardFiveSeconds);
    }

    private void HandlePlayerStateChanged(ReplayPlayerState state)
    {
        Refresh();
    }

    private void Refresh()
    {
        if (player == null)
        {
            return;
        }

        if (stateText != null)
        {
            stateText.text = $"Replay: {player.State}{sourceNotice}";
        }

        bool controls = player.ControlsEnabled;
        SetInteractable(playResumeButton, controls && player.State != ReplayPlayerState.Playing);
        SetInteractable(pauseButton, controls && player.State == ReplayPlayerState.Playing);
        SetInteractable(restartButton, controls);
        SetInteractable(backFiveButton, controls);
        SetInteractable(forwardFiveButton, controls);
        if (timelineSlider != null)
        {
            timelineSlider.interactable = controls;
        }

        bool hasError = player.State == ReplayPlayerState.Error ||
            player.State == ReplayPlayerState.Unavailable;
        if (errorText != null)
        {
            errorText.text = hasError ? player.Error : string.Empty;
            errorText.gameObject.SetActive(hasError);
        }

        RefreshTime();
    }

    private void RefreshTime()
    {
        if (player == null)
        {
            return;
        }

        if (timeText != null)
        {
            timeText.text = $"{FormatTime(player.CurrentTime)} / {FormatTime(player.Duration)}";
        }

        // PlaybackProgressSlider updates the authored Slider without feeding
        // its own programmatic write back into ReplayPlayer.SeekNormalized.
    }

    private static string FormatTime(double seconds)
    {
        int totalSeconds = Mathf.Max(0, Mathf.FloorToInt((float)seconds));
        return $"{totalSeconds / 60:00}:{totalSeconds % 60:00}";
    }

    private static void BindButton(Button button, UnityEngine.Events.UnityAction action)
    {
        if (button == null)
        {
            return;
        }

        button.onClick.RemoveListener(action);
        button.onClick.AddListener(action);
    }

    private static void UnbindButton(Button button, UnityEngine.Events.UnityAction action)
    {
        button?.onClick.RemoveListener(action);
    }

    private static void SetInteractable(Button button, bool interactable)
    {
        if (button != null)
        {
            button.interactable = interactable;
        }
    }
}
