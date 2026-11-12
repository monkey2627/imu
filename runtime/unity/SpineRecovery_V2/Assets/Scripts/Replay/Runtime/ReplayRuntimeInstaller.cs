using System;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public static class ReplayRuntimeInstaller
{
    private const string PlaybackSceneName = "PlayBack";

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void Register()
    {
        SceneManager.sceneLoaded -= HandleSceneLoaded;
        SceneManager.sceneLoaded += HandleSceneLoaded;
    }

    private static void HandleSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        if (string.Equals(scene.name, PlaybackSceneName, StringComparison.OrdinalIgnoreCase))
        {
            Install();
        }
    }

    public static void Install()
    {
        ReplaySceneBootstrap existing =
            UnityEngine.Object.FindObjectOfType<ReplaySceneBootstrap>(true);
        if (existing != null)
        {
            return;
        }

        MotionPlaybackController legacyPlayback =
            UnityEngine.Object.FindObjectOfType<MotionPlaybackController>(true);
        Animator userAnimator = legacyPlayback != null
            ? legacyPlayback.UserAnimator
            : null;
        Animator coachAnimator = legacyPlayback != null
            ? legacyPlayback.CoachAnimator
            : null;
        if (legacyPlayback != null)
        {
            legacyPlayback.enabled = false;
        }

        foreach (CoachActionController controller in
                 UnityEngine.Object.FindObjectsOfType<CoachActionController>(true))
        {
            controller.enabled = false;
        }

        foreach (AutoGroundRootFromAvatar grounding in
                 UnityEngine.Object.FindObjectsOfType<AutoGroundRootFromAvatar>(true))
        {
            grounding.enabled = false;
        }

        var rootObject = new GameObject("ReplayRoot");
        Transform replayRoot = rootObject.transform;
        replayRoot.SetPositionAndRotation(
            CalculateAuthoredReplayOrigin(
                userAnimator != null ? userAnimator.transform : null,
                coachAnimator != null ? coachAnimator.transform : null),
            Quaternion.identity);

        GameObject menu = GameObject.Find("ActionMenu");
        if (menu != null)
        {
            foreach (Button button in menu.GetComponentsInChildren<Button>(true))
            {
                string buttonName = button.gameObject.name;
                if (string.Equals(buttonName, "PlayAction1Button", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(buttonName, "ReturnIdle", StringComparison.OrdinalIgnoreCase))
                {
                    button.gameObject.SetActive(false);
                }
            }
        }

        ReplayPlayer player = rootObject.AddComponent<ReplayPlayer>();
        ReplayPlacementController placement =
            rootObject.AddComponent<ReplayPlacementController>();
        ReplayUIController ui = InstallUi(menu, player);
        ReplaySceneBootstrap bootstrap = rootObject.AddComponent<ReplaySceneBootstrap>();

        placement.Configure(
            replayRoot,
            player,
            Camera.main);
        placement.ConfigureContent(
            userAnimator != null ? userAnimator.transform : null,
            coachAnimator != null ? coachAnimator.transform : null,
            menu != null ? menu.transform : null);
        if (Application.isEditor)
        {
            // The editor is the authored scene preview, not a spatial
            // session. Keep the exact scene positions so entering PlayBack
            // cannot move the models or the menu outside the Game view.
            placement.UseConfiguredPoseWithoutSpatialRelocation();
        }
        bootstrap.Configure(
            player,
            placement,
            ui,
            replayRoot,
            userAnimator,
            coachAnimator);
    }

    public static Vector3 CalculateAuthoredReplayOrigin(
        Transform userRoot,
        Transform coachRoot)
    {
        if (userRoot == null && coachRoot == null)
        {
            return Vector3.zero;
        }

        Vector3 userPosition = userRoot != null
            ? userRoot.position
            : coachRoot.position;
        Vector3 coachPosition = coachRoot != null
            ? coachRoot.position
            : userRoot.position;
        Vector3 midpoint = (userPosition + coachPosition) * 0.5f;
        return new Vector3(midpoint.x, 0f, midpoint.z);
    }

    private static ReplayUIController InstallUi(GameObject menu, ReplayPlayer player)
    {
        GameObject host = menu != null ? menu : new GameObject("ReplayUI");
        ReplayUIController ui = host.GetComponent<ReplayUIController>();
        if (ui == null)
        {
            ui = host.AddComponent<ReplayUIController>();
        }

        TMP_Text error = CreateLabel(host.transform, "ReplayError", new Vector2(0f, 125f), 20f);
        error.color = new Color(1f, 0.35f, 0.3f, 1f);
        error.gameObject.SetActive(false);

        // Preserve the authored page. Replay logic is attached to the existing
        // controls instead of generating a second row of debug-style buttons.
        Button play = FindButton(host.transform, "Play");
        Button pause = FindButton(host.transform, "Pause");
        PlaybackProgressSlider progress =
            host.GetComponentInChildren<PlaybackProgressSlider>(true);
        Slider slider = progress != null ? progress.GetComponent<Slider>() : null;
        progress?.Configure(player);

        ui.Configure(
            player,
            null,
            null,
            error,
            play,
            pause,
            null,
            null,
            null,
            slider,
            string.IsNullOrWhiteSpace(ReplaySessionContext.SessionId));
        return ui;
    }

    private static TMP_Text CreateLabel(
        Transform parent,
        string name,
        Vector2 anchoredPosition,
        float fontSize)
    {
        Transform existing = parent.Find(name);
        TextMeshProUGUI label;
        if (existing != null)
        {
            label = existing.GetComponent<TextMeshProUGUI>();
        }
        else
        {
            var labelObject = new GameObject(name, typeof(RectTransform));
            labelObject.transform.SetParent(parent, false);
            label = labelObject.AddComponent<TextMeshProUGUI>();
        }

        RectTransform rect = label.rectTransform;
        rect.anchorMin = rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.anchoredPosition = anchoredPosition;
        rect.sizeDelta = new Vector2(620f, 44f);
        label.alignment = TextAlignmentOptions.Center;
        label.fontSize = fontSize;
        label.enableWordWrapping = false;
        return label;
    }

    private static Button FindButton(Transform root, string name)
    {
        foreach (Button button in root.GetComponentsInChildren<Button>(true))
        {
            if (string.Equals(button.gameObject.name, name, StringComparison.OrdinalIgnoreCase))
            {
                return button;
            }
        }

        return null;
    }
}
