using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

[RequireComponent(typeof(Button))]
public sealed class GamingGuideButtonController : MonoBehaviour
{
    [SerializeField] private string guideSceneName = "gamingguide";

    private Button guideButton;

    private void Awake()
    {
        guideButton = GetComponent<Button>();
        guideButton.onClick.AddListener(OpenGuide);
    }

    private void OnDestroy()
    {
        if (guideButton != null)
        {
            guideButton.onClick.RemoveListener(OpenGuide);
        }
    }

    public void OpenGuide()
    {
        GamingGuideNavigation.SetReturnScene(SceneManager.GetActiveScene().name);
        SceneManager.LoadScene(guideSceneName);
    }
}

public static class GamingGuideNavigation
{
    private static string returnSceneName;
    private static string calibrationRestartSceneName;

    public static bool HasReturnScene => !string.IsNullOrEmpty(returnSceneName);

    public static void SetReturnScene(string sceneName)
    {
        returnSceneName = sceneName;
        calibrationRestartSceneName = null;
    }

    public static void ReturnToOrigin()
    {
        string targetScene = returnSceneName;
        returnSceneName = null;

        if (!string.IsNullOrEmpty(targetScene))
        {
            calibrationRestartSceneName = targetScene;
            SceneManager.LoadScene(targetScene);
        }
    }

    public static bool ConsumeCalibrationRestart(string sceneName)
    {
        if (string.IsNullOrEmpty(sceneName) ||
            !string.Equals(
                calibrationRestartSceneName,
                sceneName,
                System.StringComparison.Ordinal))
        {
            return false;
        }

        calibrationRestartSceneName = null;
        return true;
    }
}
