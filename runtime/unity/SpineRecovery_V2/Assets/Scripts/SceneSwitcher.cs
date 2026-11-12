using UnityEngine;
using UnityEngine.SceneManagement;

public class SceneSwitcher : MonoBehaviour
{
    private const string GameScene = "Gamescene";
    private const string ActionScene = "ActionScene";
    private const string BirdDogScene = "BirdDog";
    private const string BirdDogGamingScene = "BirdDogGaming";
    private const string BirdDogPracticeScene = "BirdDogPractice";
    private const string DeadBugScene = "DeadBug";
    private const string DeadBugGamingScene = "DeadBugGaming";
    private const string DeadBugPracticeScene = "DeadBugPractice";
    private const string TestGamingScene = "testGaming";
    public static string TestGamingSceneName => TestGamingScene;
    private const string HipThrustScene = "HipThrust";
    private const string PlayBackScene = "PlayBack";
    private const string CoachingChooseScene = "CoachingChoose";
    private const string GamingChooseScene = "GamingChoose";

    public void ReturnToMobileApp()
    {
        if (SpineFlowTrainingSession.TryReturnToMobile())
        {
            return;
        }

        Debug.LogWarning(
            "[SpineFlow] Could not return to the mobile app. " +
            "The Home scene is intentionally not used in this release.",
            this);
    }

    public void LoadGameScene()
    {
        LoadSceneForCurrentMobileSession(GameScene);
    }

    public void LoadActionScene()
    {
        LoadSceneForCurrentMobileSession(ActionScene);
    }

    public void LoadBirdDogScene()
    {
        LoadSceneForCurrentMobileSession(BirdDogScene);
    }

    public void LoadBirdDogGamingScene()
    {
        LoadSceneForCurrentMobileSession(ResolveReplayReturnScene(
            BirdDogGamingScene,
            ReplaySessionContext.ReturnGamingScene));
    }

    public void LoadBirdDogPracticeScene()
    {
        LoadSceneForCurrentMobileSession(ResolveReplayReturnScene(
            BirdDogPracticeScene,
            ReplaySessionContext.ReturnPracticeScene));
    }

    public void LoadDeadBugScene()
    {
        LoadSceneForCurrentMobileSession(DeadBugScene);
    }

    public void LoadDeadBugGamingScene()
    {
        LoadSceneForCurrentMobileSession(ResolveReplayReturnScene(
            DeadBugGamingScene,
            ReplaySessionContext.ReturnGamingScene));
    }

    public void LoadTestGamingScene()
    {
        LoadSceneForCurrentMobileSession(TestGamingScene);
    }

    public void LoadHipThrustScene()
    {
        LoadSceneForCurrentMobileSession(HipThrustScene);
    }

    public void LoadDeadBugPracticeScene()
    {
        LoadSceneForCurrentMobileSession(ResolveReplayReturnScene(
            DeadBugPracticeScene,
            ReplaySessionContext.ReturnPracticeScene));
    }

    public void LoadPlayBackScene()
    {
        LoadSceneForCurrentMobileSession(PlayBackScene);
    }

    public void LoadCoachingChooseScene()
    {
        LoadSceneForCurrentMobileSession(CoachingChooseScene);
    }

    public void LoadGamingChooseScene()
    {
        LoadSceneForCurrentMobileSession(GamingChooseScene);
    }

    /// <summary>
    /// 加载统一练习场景（testPractice），传入动作 ID 进行配置。
    /// 用于从其他场景（如 CoachingChoose）跳转到数据驱动的练习场景。
    /// </summary>
    /// <param name="actionId">动作 ID，如 "deadbug", "birddog"</param>
    public void LoadUnifiedPracticeScene(string actionId)
    {
        PracticeSceneController.LoadPracticeScene(actionId, "testPractice");
    }

    /// <summary>
    /// 加载统一练习场景，使用服务器下发的动作包。
    /// 用于服务器动态下发新动作的场景。
    /// </summary>
    /// <param name="package">服务器下发的动作数据包</param>
    public void LoadUnifiedPracticeSceneWithPackage(CoachMotionPackage package)
    {
        string error = "";
        if (package == null || !package.IsValid(out error))
        {
            Debug.LogError($"[SceneSwitcher] 无效的动作包: {error}");
            return;
        }

        // 将 package 序列化为 JSON，暂存到临时文件
        string tempPath = System.IO.Path.Combine(Application.persistentDataPath, "temp_motion_package.json");
        try
        {
            string json = JsonUtility.ToJson(package, prettyPrint: false);
            System.IO.File.WriteAllText(tempPath, json);
            PracticeSceneInitializer.PendingMotionPackageUri = tempPath;
            LoadScene("testPractice");
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"[SceneSwitcher] 保存临时动作包失败: {ex.Message}");
        }
    }

    public void LoadScene(string sceneName)
    {
        LoadSceneForCurrentMobileSession(sceneName);
    }

    private static void LoadSceneForCurrentMobileSession(string sceneName)
    {
        SceneManager.LoadScene(SpineFlowTrainingSession.ResolveSceneForMobileSession(sceneName));
    }

    private static string ResolveReplayReturnScene(string fallback, string replayTarget)
    {
        return string.Equals(
                   SceneManager.GetActiveScene().name,
                   PlayBackScene,
                   System.StringComparison.OrdinalIgnoreCase) &&
               !string.IsNullOrWhiteSpace(replayTarget)
            ? replayTarget
            : fallback;
    }
}
