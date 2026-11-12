using UnityEngine;

/// <summary>
/// 难度等级枚举
/// </summary>
public enum DifficultyLevel
{
    Relaxed = 0,    // 宽松
    Medium = 1,     // 中等
    Strict = 2      // 严格
}

/// <summary>
/// 难度配置 - 定义不同难度下的稳定性和评分阈值
/// </summary>
[System.Serializable]
public class DifficultyConfig
{
    [Header("稳定性判断阈值")]
    [Tooltip("旋转漂移阈值（度/秒）- 用户姿势需保持的稳定程度")]
    public float stabilityThresholdDegPerSec = 5f;

    [Tooltip("对齐角度阈值（度）- 一般骨骼与教练的对齐要求")]
    public float stabilityAlignmentThreshold = 35f;

    [Tooltip("腿部对齐角度阈值（度）- 腿部骨骼与教练的对齐要求")]
    public float stabilityAlignmentThresholdLegs = 50f;

    [Header("评分误差阈值")]
    [Tooltip("最大角度误差（度）- 超过此值得0分")]
    public float maxAngleError = 30f;

    [Tooltip("完美角度阈值（度）- 小于此值得100分")]
    public float perfectAngleThreshold = 5f;
}

/// <summary>
/// 难度设置管理器 - 存储和管理三个难度档位的配置
/// </summary>
[CreateAssetMenu(fileName = "DifficultySettings", menuName = "Spine Recovery/Difficulty Settings")]
public class DifficultySettings : ScriptableObject
{
    public const float RelaxedStabilityThresholdDegPerSec = 8f;
    public const float RelaxedAlignmentThreshold = 45f;
    public const float RelaxedLegAlignmentThreshold = 60f;
    public const float RelaxedMaxAngleError = 40f;
    public const float RelaxedPerfectAngleThreshold = 8f;

    public const float MediumStabilityThresholdDegPerSec = 5f;
    public const float MediumAlignmentThreshold = 35f;
    public const float MediumLegAlignmentThreshold = 50f;
    public const float MediumMaxAngleError = 30f;
    public const float MediumPerfectAngleThreshold = 5f;

    public const float StrictStabilityThresholdDegPerSec = 3f;
    public const float StrictAlignmentThreshold = 25f;
    public const float StrictLegAlignmentThreshold = 35f;
    public const float StrictMaxAngleError = 20f;
    public const float StrictPerfectAngleThreshold = 3f;

    [Header("宽松模式（适合初学者）")]
    [SerializeField] private DifficultyConfig relaxedConfig = new DifficultyConfig
    {
        // 稳定性要求更宽松
        stabilityThresholdDegPerSec = 8f,      // 允许更大的晃动
        stabilityAlignmentThreshold = 45f,      // 对齐角度更宽松
        stabilityAlignmentThresholdLegs = 60f,  // 腿部对齐更宽松

        // 评分更宽容
        maxAngleError = 40f,                    // 更大的容错范围
        perfectAngleThreshold = 8f              // 完美分数的要求降低
    };

    [Header("中等模式（推荐）")]
    [SerializeField] private DifficultyConfig mediumConfig = new DifficultyConfig
    {
        // 当前的默认值
        stabilityThresholdDegPerSec = 5f,
        stabilityAlignmentThreshold = 35f,
        stabilityAlignmentThresholdLegs = 50f,

        maxAngleError = 30f,
        perfectAngleThreshold = 5f
    };

    [Header("严格模式（适合熟练者）")]
    [SerializeField] private DifficultyConfig strictConfig = new DifficultyConfig
    {
        // 稳定性要求更严格
        stabilityThresholdDegPerSec = 3f,       // 需要更稳定的姿势
        stabilityAlignmentThreshold = 25f,      // 对齐角度更严格
        stabilityAlignmentThresholdLegs = 35f,  // 腿部对齐更严格

        // 评分更严格
        maxAngleError = 20f,                    // 更小的容错范围
        perfectAngleThreshold = 3f              // 完美分数的要求更高
    };

    // 当前选中的难度
    private static DifficultyLevel currentDifficulty = DifficultyLevel.Medium;

    /// <summary>
    /// 获取当前难度等级
    /// </summary>
    public static DifficultyLevel CurrentDifficulty
    {
        get => currentDifficulty;
        set => currentDifficulty = value;
    }

    /// <summary>
    /// 根据难度等级获取配置
    /// </summary>
    public DifficultyConfig GetConfig(DifficultyLevel level)
    {
        switch (level)
        {
            case DifficultyLevel.Relaxed:
                return relaxedConfig;
            case DifficultyLevel.Medium:
                return mediumConfig;
            case DifficultyLevel.Strict:
                return strictConfig;
            default:
                return mediumConfig;
        }
    }

    /// <summary>
    /// 获取当前选中难度的配置
    /// </summary>
    public DifficultyConfig GetCurrentConfig()
    {
        return GetConfig(currentDifficulty);
    }

    public static DifficultyConfig CreateBuiltInConfig(DifficultyLevel level)
    {
        switch (level)
        {
            case DifficultyLevel.Relaxed:
                return new DifficultyConfig
                {
                    stabilityThresholdDegPerSec = RelaxedStabilityThresholdDegPerSec,
                    stabilityAlignmentThreshold = RelaxedAlignmentThreshold,
                    stabilityAlignmentThresholdLegs = RelaxedLegAlignmentThreshold,
                    maxAngleError = RelaxedMaxAngleError,
                    perfectAngleThreshold = RelaxedPerfectAngleThreshold
                };
            case DifficultyLevel.Strict:
                return new DifficultyConfig
                {
                    stabilityThresholdDegPerSec = StrictStabilityThresholdDegPerSec,
                    stabilityAlignmentThreshold = StrictAlignmentThreshold,
                    stabilityAlignmentThresholdLegs = StrictLegAlignmentThreshold,
                    maxAngleError = StrictMaxAngleError,
                    perfectAngleThreshold = StrictPerfectAngleThreshold
                };
            case DifficultyLevel.Medium:
            default:
                return new DifficultyConfig
                {
                    stabilityThresholdDegPerSec = MediumStabilityThresholdDegPerSec,
                    stabilityAlignmentThreshold = MediumAlignmentThreshold,
                    stabilityAlignmentThresholdLegs = MediumLegAlignmentThreshold,
                    maxAngleError = MediumMaxAngleError,
                    perfectAngleThreshold = MediumPerfectAngleThreshold
                };
        }
    }

    /// <summary>
    /// 获取难度的显示名称
    /// </summary>
    public static string GetDifficultyName(DifficultyLevel level)
    {
        switch (level)
        {
            case DifficultyLevel.Relaxed:
                return "宽松";
            case DifficultyLevel.Medium:
                return "中等";
            case DifficultyLevel.Strict:
                return "严格";
            default:
                return "未知";
        }
    }

    /// <summary>
    /// 获取难度的描述
    /// </summary>
    public static string GetDifficultyDescription(DifficultyLevel level)
    {
        switch (level)
        {
            case DifficultyLevel.Relaxed:
                return "适合初学者\n更宽松的姿势要求";
            case DifficultyLevel.Medium:
                return "推荐选择\n标准的姿势要求";
            case DifficultyLevel.Strict:
                return "适合熟练者\n更严格的姿势要求";
            default:
                return "";
        }
    }
}
