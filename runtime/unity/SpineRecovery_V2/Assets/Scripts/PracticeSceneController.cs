using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// 统一练习场景的控制器。提供按钮调用的场景跳转方法，根据当前动作配置路由到正确的目标场景。
/// 这是一个静态工具类，按钮可以直接调用其方法（无需序列化引用）。
/// </summary>
public static class PracticeSceneController
{
    /// <summary>
    /// 重新开始当前练习（刷新场景）
    /// </summary>
    public static void RestartPractice()
    {
        string currentScene = SceneManager.GetActiveScene().name;
        SceneManager.LoadScene(currentScene);
        Debug.Log($"[PracticeSceneController] 重新加载练习场景: {currentScene}");
    }

    /// <summary>
    /// 返回到教练选择场景
    /// </summary>
    public static void LoadCoachingScene()
    {
        var config = PracticeSceneInitializer.GetCurrentConfig();
        if (config == null)
        {
            Debug.LogWarning("[PracticeSceneController] 无当前配置，无法返回教练场景");
            LoadFallbackScene("CoachingChoose");
            return;
        }

        string targetScene = !string.IsNullOrWhiteSpace(config.coachingSceneName)
            ? config.coachingSceneName
            : config.legacyPracticeSceneName;

        if (string.IsNullOrWhiteSpace(targetScene))
        {
            Debug.LogWarning("[PracticeSceneController] 配置中未设置教练场景，使用默认场景");
            LoadFallbackScene("CoachingChoose");
            return;
        }

        LoadSceneForCurrentSession(targetScene);
        Debug.Log($"[PracticeSceneController] 返回教练场景: {targetScene}");
    }

    /// <summary>
    /// 进入游戏模式场景
    /// </summary>
    public static void LoadGamingScene()
    {
        var config = PracticeSceneInitializer.GetCurrentConfig();
        if (config == null)
        {
            Debug.LogWarning("[PracticeSceneController] 无当前配置，无法进入游戏场景");
            LoadFallbackScene("DeadBugGaming");
            return;
        }

        if (string.IsNullOrWhiteSpace(config.gamingSceneName))
        {
            Debug.LogError("[PracticeSceneController] 配置中未设置游戏场景");
            return;
        }

        LoadSceneForCurrentSession(config.gamingSceneName);
        Debug.Log($"[PracticeSceneController] 进入游戏场景: {config.gamingSceneName}");
    }

    /// <summary>
    /// 加载练习场景（供其他场景调用，如 CoachingChoose）
    /// </summary>
    /// <param name="actionId">动作 ID</param>
    /// <param name="targetSceneName">目标场景名（默认 testPractice）</param>
    public static void LoadPracticeScene(string actionId, string targetSceneName = "testPractice")
    {
        if (string.IsNullOrWhiteSpace(actionId))
        {
            Debug.LogWarning("[PracticeSceneController] actionId 为空，使用默认动作");
            actionId = PracticeActionLibrary.GetDefaultConfig()?.actionId ?? "deadbug";
        }

        // 设置待加载的动作 ID
        PracticeSceneInitializer.PendingActionId = actionId;

        // 加载场景
        LoadSceneForCurrentSession(targetSceneName);
        Debug.Log($"[PracticeSceneController] 加载练习场景: {targetSceneName}, actionId: {actionId}");
    }

    /// <summary>
    /// 使用 SpineFlowTrainingSession 的场景解析逻辑（支持性别切换等）
    /// </summary>
    private static void LoadSceneForCurrentSession(string sceneName)
    {
        string resolvedScene = SpineFlowTrainingSession.ResolveSceneForMobileSession(sceneName);
        SceneManager.LoadScene(resolvedScene);
    }

    /// <summary>
    /// 加载回退场景（当配置缺失时）
    /// </summary>
    private static void LoadFallbackScene(string fallbackSceneName)
    {
        LoadSceneForCurrentSession(fallbackSceneName);
    }
}
