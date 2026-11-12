using System;
using System.Linq;
using UnityEngine;

/// <summary>
/// 管理所有练习动作配置的中央注册表。
/// 必须放在 Resources/ 目录下以便运行时加载（命名为 PracticeActionLibrary.asset）。
/// </summary>
[CreateAssetMenu(fileName = "PracticeActionLibrary", menuName = "Spine Recovery/Practice Action Library")]
public class PracticeActionLibrary : ScriptableObject
{
    private static PracticeActionLibrary instance;

    /// <summary>
    /// 单例访问。自动从 Resources 加载。
    /// </summary>
    public static PracticeActionLibrary Instance
    {
        get
        {
            if (instance == null)
            {
                instance = Resources.Load<PracticeActionLibrary>("PracticeActionLibrary");
                if (instance == null)
                {
                    Debug.LogError(
                        "[PracticeActionLibrary] 未找到 Resources/PracticeActionLibrary.asset。" +
                        "请在 Assets/Resources/ 目录下创建此资产。");
                }
            }
            return instance;
        }
    }

    [SerializeField]
    [Tooltip("所有可用的练习动作配置")]
    private PracticeActionConfig[] actions = Array.Empty<PracticeActionConfig>();

    [SerializeField]
    [Tooltip("默认动作 ID（当无法推断时使用）")]
    private string defaultActionId = "deadbug";

    /// <summary>
    /// 根据 actionId 获取配置
    /// </summary>
    public static PracticeActionConfig GetConfig(string actionId)
    {
        if (Instance == null || string.IsNullOrWhiteSpace(actionId))
        {
            return null;
        }

        return Instance.actions.FirstOrDefault(
            config => config != null &&
                      string.Equals(config.actionId, actionId, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// 从场景名称推断 actionId
    /// </summary>
    public static PracticeActionConfig InferFromSceneName(string sceneName)
    {
        if (Instance == null || string.IsNullOrWhiteSpace(sceneName))
        {
            return null;
        }

        // 直接匹配 legacyPracticeSceneName
        foreach (var config in Instance.actions)
        {
            if (config != null &&
                !string.IsNullOrWhiteSpace(config.legacyPracticeSceneName) &&
                string.Equals(sceneName, config.legacyPracticeSceneName, StringComparison.OrdinalIgnoreCase))
            {
                return config;
            }
        }

        // 模糊匹配：场景名包含 actionId
        string lowerScene = sceneName.ToLowerInvariant();
        foreach (var config in Instance.actions)
        {
            if (config != null &&
                !string.IsNullOrWhiteSpace(config.actionId) &&
                lowerScene.Contains(config.actionId.ToLowerInvariant()))
            {
                return config;
            }
        }

        return null;
    }

    /// <summary>
    /// 获取默认配置（fallback）
    /// </summary>
    public static PracticeActionConfig GetDefaultConfig()
    {
        if (Instance == null)
        {
            return null;
        }

        // 先尝试用默认 ID
        if (!string.IsNullOrWhiteSpace(Instance.defaultActionId))
        {
            var config = GetConfig(Instance.defaultActionId);
            if (config != null)
            {
                return config;
            }
        }

        // 兜底：返回第一个有效配置
        return Instance.actions.FirstOrDefault(c => c != null);
    }

    /// <summary>
    /// 获取所有注册的动作 ID
    /// </summary>
    public static string[] GetAllActionIds()
    {
        if (Instance == null)
        {
            return Array.Empty<string>();
        }

        return Instance.actions
            .Where(c => c != null && !string.IsNullOrWhiteSpace(c.actionId))
            .Select(c => c.actionId)
            .ToArray();
    }

    /// <summary>
    /// 验证库的完整性
    /// </summary>
    private void OnValidate()
    {
        if (actions == null || actions.Length == 0)
        {
            Debug.LogWarning("[PracticeActionLibrary] actions 数组为空", this);
            return;
        }

        // 检查重复 ID
        var ids = actions
            .Where(c => c != null && !string.IsNullOrWhiteSpace(c.actionId))
            .Select(c => c.actionId.ToLowerInvariant())
            .ToList();

        var duplicates = ids.GroupBy(id => id)
            .Where(g => g.Count() > 1)
            .Select(g => g.Key)
            .ToArray();

        if (duplicates.Length > 0)
        {
            Debug.LogError(
                $"[PracticeActionLibrary] 发现重复的 actionId: {string.Join(", ", duplicates)}",
                this);
        }

        // 验证每个配置
        foreach (var config in actions)
        {
            if (config != null && !config.Validate(out string error))
            {
                Debug.LogError(
                    $"[PracticeActionLibrary] 配置 '{config.name}' 无效: {error}",
                    config);
            }
        }
    }
}
