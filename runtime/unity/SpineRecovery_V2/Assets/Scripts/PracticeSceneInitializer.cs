using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// 统一练习场景的初始化器。在场景加载时根据传入的动作配置初始化教练、分段、按钮等。
/// 挂载到 testPractice 场景的某个 GameObject 上，executionOrder 应设为负数以确保在其他组件之前运行。
///
/// 新增：支持从本地 JSON 文件或 content URI 加载服务器下发的动作包（CoachMotionPackage）。
/// </summary>
public class PracticeSceneInitializer : MonoBehaviour
{
    /// <summary>
    /// 跳转到练习场景前设置此字段，场景加载后会读取并应用对应配置。
    /// </summary>
    public static string PendingActionId { get; set; }

    /// <summary>
    /// 服务器下发动作的本地缓存路径或 content:// URI。
    /// 如果非空，优先从此加载 CoachMotionPackage，走关键帧播放后端。
    /// </summary>
    public static string PendingMotionPackageUri
    {
        get
        {
#if UNITY_EDITOR
            // 编辑器模式下从 EditorPrefs 读取（防止场景加载时丢失）
            string editorUri = UnityEditor.EditorPrefs.GetString("PracticeSceneInitializer.PendingMotionPackageUri", null);
            if (!string.IsNullOrEmpty(editorUri))
            {
                return editorUri;
            }
#endif
            return pendingMotionPackageUri;
        }
        set
        {
            pendingMotionPackageUri = value;
#if UNITY_EDITOR
            // 编辑器模式下同时保存到 EditorPrefs
            if (!string.IsNullOrEmpty(value))
            {
                UnityEditor.EditorPrefs.SetString("PracticeSceneInitializer.PendingMotionPackageUri", value);
            }
            else
            {
                UnityEditor.EditorPrefs.DeleteKey("PracticeSceneInitializer.PendingMotionPackageUri");
            }
#endif
        }
    }

    private static string pendingMotionPackageUri;

    [Header("References (Optional - Auto-Find if Empty)")]
    [Tooltip("教练 avatar 的 Animator 组件。留空则自动查找。")]
    [SerializeField] private Animator coachAnimator;

    [Tooltip("SegmentedCoachController 组件。留空则自动查找。")]
    [SerializeField] private SegmentedCoachController segmentedCoachController;
    private JsonPracticeCoachController jsonPracticeCoachController;
    private bool jsonMotionMode;

    [Header("Buttons (Optional - Auto-Find by Name)")]
    [Tooltip("\"PracAgain\" 按钮（再次练习）")]
    [SerializeField] private Button pracAgainButton;

    [Tooltip("\"Back\" 按钮（返回选择）")]
    [SerializeField] private Button backButton;

    [Tooltip("\"Game\" 按钮（进入游戏模式）")]
    [SerializeField] private Button gameButton;

    [Header("Debug")]
    [Tooltip("是否输出详细日志")]
    [SerializeField] private bool verboseLogging = true;

    private PracticeActionConfig currentConfig;

    private void Start()
    {
        if (jsonMotionMode)
        {
            BindJsonStartButtons();
        }
    }

    private void Awake()
    {
        // 优先级1: 检查是否有服务器下发的动作包
        if (!string.IsNullOrEmpty(PendingMotionPackageUri))
        {
            string uriToLoad = PendingMotionPackageUri;
            LoadMotionPackageFromUri(uriToLoad);
            PendingMotionPackageUri = null; // 清空，包括 EditorPrefs
            return; // JSON 模式下不需要继续加载传统配置
        }

        // 优先级2: 静态变量指定的 actionId（配置资产路径）
        string actionId = ResolveActionId();
        if (verboseLogging)
        {
            Debug.Log($"[PracticeSceneInitializer] Resolved actionId: '{actionId}'");
        }

        // 2. 加载配置
        currentConfig = PracticeActionLibrary.GetConfig(actionId);
        if (currentConfig == null)
        {
            Debug.LogWarning($"[PracticeSceneInitializer] 未找到 actionId '{actionId}' 的配置，尝试使用默认配置");
            currentConfig = PracticeActionLibrary.GetDefaultConfig();
        }

        if (currentConfig == null)
        {
            Debug.LogWarning(
                "[PracticeSceneInitializer] 无可用配置。" +
                "如需使用传统 AnimatorController 模式，请在 Assets/Resources/ 目录下创建 PracticeActionLibrary.asset。" +
                "如需使用 JSON 驱动模式，请通过 TestJsonLoader 或手机端启动。");
            return;
        }

        if (!currentConfig.Validate(out string error))
        {
            Debug.LogWarning($"[PracticeSceneInitializer] 配置无效: {error}，场景可能无法正常工作");
            return;
        }

        // 3. 应用配置
        ConfigureCoach();
        ConfigureButtons();

        if (verboseLogging)
        {
            Debug.Log($"[PracticeSceneInitializer] 场景已初始化为动作: {currentConfig.displayName}");
        }

        // 4. 清空静态变量，避免下次启动场景时误用
        PendingActionId = null;
    }

    /// <summary>
    /// 从本地路径或 content URI 加载服务器下发的动作包，切换到关键帧播放后端。
    /// </summary>
    private void LoadMotionPackageFromUri(string uri)
    {
        try
        {
            string jsonText = ReadJsonFromUri(uri);
            if (string.IsNullOrEmpty(jsonText))
            {
                Debug.LogError($"[PracticeSceneInitializer] 无法读取动作包: {uri}");
                return;
            }

            CoachMotionPackage package = JsonUtility.FromJson<CoachMotionPackage>(jsonText);
            if (package == null)
            {
                Debug.LogError($"[PracticeSceneInitializer] 反序列化失败: {uri}");
                return;
            }

            if (!package.IsValid(out string error))
            {
                Debug.LogError($"[PracticeSceneInitializer] 动作包无效: {error}");
                return;
            }

            // 查找组件
            if (coachAnimator == null)
            {
                coachAnimator = FindCoachAnimator();
            }

            if (segmentedCoachController == null)
            {
                segmentedCoachController = FindObjectOfType<SegmentedCoachController>();
            }

            SpineFlowTrainingSession.CacheMotionPackage(package);

            // 不要禁用 PracticeSessionController - 让它处理介绍和倒数
            // 只禁用传统的 SegmentedCoachController
            if (segmentedCoachController != null)
            {
                segmentedCoachController.enabled = false;
            }

            if (coachAnimator == null)
            {
                Debug.LogError("[PracticeSceneInitializer] 未找到 JSON 教练 Animator");
                return;
            }

            jsonPracticeCoachController = coachAnimator.gameObject.AddComponent<JsonPracticeCoachController>();
            jsonPracticeCoachController.Initialize(
                coachAnimator,
                package,
                FindObjectOfType<MotionRecorder>(true));
            jsonMotionMode = true;

            // 配置按钮（从 package 读场景名）
            ConfigureButtonsForPackage(package);

            if (verboseLogging)
            {
                Debug.Log(
                    $"[PracticeSceneInitializer] 已加载服务器动作包: {package.actionId}, " +
                    $"{package.segments.Length} 段, {package.setCount} 组");
            }
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"[PracticeSceneInitializer] 加载动作包异常: {ex.Message}\n{ex.StackTrace}");
        }
    }

    /// <summary>
    /// 从 URI 读取 JSON 文本。支持：
    /// - 本地文件路径（如 /sdcard/cache/deadbug.motion.json）
    /// - content:// URI（Android FileProvider）
    /// </summary>
    private string ReadJsonFromUri(string uri)
    {
        // content:// URI 走 Android 接口
        if (uri.StartsWith("content://"))
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            try
            {
                using (AndroidJavaClass unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
                using (AndroidJavaObject currentActivity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity"))
                using (AndroidJavaObject contentResolver = currentActivity.Call<AndroidJavaObject>("getContentResolver"))
                using (AndroidJavaClass uriClass = new AndroidJavaClass("android.net.Uri"))
                {
                    AndroidJavaObject androidUri = uriClass.CallStatic<AndroidJavaObject>("parse", uri);
                    using (AndroidJavaObject inputStream = contentResolver.Call<AndroidJavaObject>("openInputStream", androidUri))
                    {
                        if (inputStream == null)
                        {
                            Debug.LogError($"[PracticeSceneInitializer] 无法打开 content URI: {uri}");
                            return null;
                        }

                        using (AndroidJavaObject reader = new AndroidJavaObject("java.io.InputStreamReader", inputStream, "UTF-8"))
                        using (AndroidJavaObject bufferedReader = new AndroidJavaObject("java.io.BufferedReader", reader))
                        {
                            try
                            {
                                System.Text.StringBuilder sb = new System.Text.StringBuilder();
                                string line;
                                while ((line = bufferedReader.Call<string>("readLine")) != null)
                                {
                                    sb.AppendLine(line);
                                }
                                return sb.ToString();
                            }
                            finally
                            {
                                // Dispose 只释放 JNI 引用，不关闭 Java 流；需显式 close
                                // 才能归还文件描述符。
                                bufferedReader.Call("close");
                                inputStream.Call("close");
                            }
                        }
                    }
                }
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[PracticeSceneInitializer] 读取 content URI 异常: {ex.Message}");
                return null;
            }
#else
            Debug.LogWarning($"[PracticeSceneInitializer] content URI 仅在 Android 设备上支持: {uri}");
            return null;
#endif
        }

        // 普通文件路径
        if (File.Exists(uri))
        {
            return File.ReadAllText(uri);
        }

        Debug.LogError($"[PracticeSceneInitializer] 文件不存在: {uri}");
        return null;
    }

    /// <summary>
    /// 为服务器下发的动作配置按钮（没有配置资产，用默认场景名）
    /// </summary>
    private void ConfigureButtonsForPackage(CoachMotionPackage package)
    {
        if (pracAgainButton == null || backButton == null || gameButton == null)
        {
            FindButtons();
        }

        if (pracAgainButton != null)
        {
            pracAgainButton.onClick = new Button.ButtonClickedEvent();
            pracAgainButton.onClick.AddListener(() =>
            {
                // 重新加载当前场景，保持 package
                string tempPath = System.IO.Path.Combine(Application.persistentDataPath, "temp_motion_package.json");
                string json = JsonUtility.ToJson(package, prettyPrint: false);
                System.IO.File.WriteAllText(tempPath, json);
                PendingMotionPackageUri = tempPath;
                UnityEngine.SceneManagement.SceneManager.LoadScene(
                    UnityEngine.SceneManagement.SceneManager.GetActiveScene().name);
            });
        }

        if (backButton != null)
        {
            backButton.onClick = new Button.ButtonClickedEvent();
            backButton.onClick.AddListener(() =>
            {
                UnityEngine.SceneManagement.SceneManager.LoadScene("UserMenu");
            });
        }

        if (gameButton != null)
        {
            gameButton.onClick = new Button.ButtonClickedEvent();
            gameButton.onClick.AddListener(() =>
                UnityEngine.SceneManagement.SceneManager.LoadScene("testGaming"));
        }

        BindJsonStartButtons();
    }

    private void BindJsonStartButtons()
    {
        // JSON 模式下，PracticeSessionController 会处理整个流程（包括介绍和倒数）
        // 只需要绑定 Next/Continue 按钮用于推进动作段
        foreach (Button button in FindObjectsOfType<Button>(true))
        {
            string buttonName = button.gameObject.name;

            if (string.Equals(buttonName, "Next", System.StringComparison.OrdinalIgnoreCase) ||
                string.Equals(buttonName, "NextStep", System.StringComparison.OrdinalIgnoreCase) ||
                string.Equals(buttonName, "Continue", System.StringComparison.OrdinalIgnoreCase))
            {
                button.onClick.RemoveAllListeners();
                button.onClick.AddListener(AdvanceJsonAction);
            }
        }
    }

    private void PlayJsonAction()
    {
        if (jsonPracticeCoachController == null)
        {
            Debug.LogError("[PracticeSceneInitializer] JSON 教练尚未初始化，无法开始训练", this);
            return;
        }

        jsonPracticeCoachController.PlayAction();
    }

    private void AdvanceJsonAction()
    {
        jsonPracticeCoachController?.AdvanceToNext();
    }

    /// <summary>
    /// 解析当前应该使用的动作 ID
    /// </summary>
    private string ResolveActionId()
    {
        // 优先级1: 显式传入的 PendingActionId
        if (!string.IsNullOrWhiteSpace(PendingActionId))
        {
            return PendingActionId;
        }

        // 优先级2: 从场景名称推断
        string sceneName = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
        var inferredConfig = PracticeActionLibrary.InferFromSceneName(sceneName);
        if (inferredConfig != null)
        {
            return inferredConfig.actionId;
        }

        // 优先级3: 使用默认配置
        var defaultConfig = PracticeActionLibrary.GetDefaultConfig();
        return defaultConfig != null ? defaultConfig.actionId : "deadbug";
    }

    /// <summary>
    /// 配置教练：设置 Animator Controller 和 Segments
    /// </summary>
    private void ConfigureCoach()
    {
        // 自动查找 Animator
        if (coachAnimator == null)
        {
            coachAnimator = FindCoachAnimator();
        }

        if (coachAnimator != null)
        {
            coachAnimator.runtimeAnimatorController = currentConfig.animatorController;
            if (verboseLogging)
            {
                Debug.Log($"[PracticeSceneInitializer] 已设置 Animator Controller: {currentConfig.animatorController.name}");
            }
        }
        else
        {
            Debug.LogWarning("[PracticeSceneInitializer] 未找到 Coach Animator");
        }

        // 自动查找 SegmentedCoachController
        if (segmentedCoachController == null)
        {
            segmentedCoachController = FindObjectOfType<SegmentedCoachController>();
        }

        if (segmentedCoachController != null)
        {
            var configuredSegments = new List<ActionSegment>(currentConfig.segments ?? new ActionSegment[0]);
            SetPrivateField(segmentedCoachController, "segments", configuredSegments);
            SetPrivateField(segmentedCoachController, "setCount", Mathf.Max(1, currentConfig.setCount));

            // 设置 Dead Bug 特殊参数（通过反射，因为这些是私有字段）
            if (currentConfig.composeDeadBugPreparationArms)
            {
                SetPrivateField(segmentedCoachController, "composeDeadBugPreparationArms", true);
                SetPrivateField(segmentedCoachController, "deadBugPreparationStateName", currentConfig.deadBugPreparationStateName);
                SetPrivateField(segmentedCoachController, "deadBugRightArmExtensionStateName", currentConfig.deadBugRightArmExtensionStateName);
                SetPrivateField(segmentedCoachController, "deadBugLeftArmExtensionStateName", currentConfig.deadBugLeftArmExtensionStateName);
            }

            if (verboseLogging)
            {
                Debug.Log($"[PracticeSceneInitializer] 已设置 {currentConfig.segments.Length} 个 segments，{currentConfig.setCount} 组");
            }
        }
        else
        {
            Debug.LogWarning("[PracticeSceneInitializer] 未找到 SegmentedCoachController");
        }
    }

    /// <summary>
    /// 配置按钮：重新绑定 onClick 到当前配置
    /// </summary>
    private void ConfigureButtons()
    {
        // 自动查找按钮
        if (pracAgainButton == null || backButton == null || gameButton == null)
        {
            FindButtons();
        }

        // 绑定按钮事件（添加到现有 listeners 之后，或替换持久化调用）
        if (pracAgainButton != null)
        {
            pracAgainButton.onClick.RemoveAllListeners();
            pracAgainButton.onClick.AddListener(() => PracticeSceneController.RestartPractice());
        }

        if (backButton != null)
        {
            backButton.onClick.RemoveAllListeners();
            backButton.onClick.AddListener(() => PracticeSceneController.LoadCoachingScene());
        }

        if (gameButton != null)
        {
            gameButton.onClick.RemoveAllListeners();
            gameButton.onClick.AddListener(() => PracticeSceneController.LoadGamingScene());
        }

        if (verboseLogging)
        {
            Debug.Log("[PracticeSceneInitializer] 按钮已重新绑定");
        }
    }

    /// <summary>
    /// 查找 Coach Animator（从场景中所有 Animator 里筛选）
    /// </summary>
    private Animator FindCoachAnimator()
    {
        // 策略：找到名字包含 "coach" 或 "avatar" 的 Animator，且不是用户的
        Animator[] animators = FindObjectsOfType<Animator>(true);
        foreach (var animator in animators)
        {
            string objName = animator.gameObject.name.ToLowerInvariant();
            // 排除用户 avatar（通常名字包含 "user" 或 "player"）
            if (objName.Contains("user") || objName.Contains("player"))
            {
                continue;
            }

            // 匹配 coach 或 model
            if (objName.Contains("coach") || objName.Contains("avatar") || objName.Contains("model") || objName.Contains("mesh_edit"))
            {
                return animator;
            }
        }

        // 兜底：返回第一个 Animator（假设教练先于用户创建）
        return animators.Length > 0 ? animators[0] : null;
    }

    /// <summary>
    /// 按名称查找按钮
    /// </summary>
    private void FindButtons()
    {
        Button[] buttons = FindObjectsOfType<Button>(true);
        foreach (var button in buttons)
        {
            string btnName = button.gameObject.name;
            if (string.Equals(btnName, "PracAgain", System.StringComparison.OrdinalIgnoreCase))
            {
                pracAgainButton = button;
            }
            else if (string.Equals(btnName, "Back", System.StringComparison.OrdinalIgnoreCase))
            {
                backButton = button;
            }
            else if (string.Equals(btnName, "Game", System.StringComparison.OrdinalIgnoreCase))
            {
                gameButton = button;
            }
        }
    }

    /// <summary>
    /// 通过反射设置私有字段（用于 Dead Bug 特殊参数）
    /// </summary>
    private void SetPrivateField(object obj, string fieldName, object value)
    {
        var field = obj.GetType().GetField(fieldName, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        if (field != null)
        {
            field.SetValue(obj, value);
        }
        else if (verboseLogging)
        {
            Debug.LogWarning($"[PracticeSceneInitializer] 未找到私有字段: {fieldName}");
        }
    }

    /// <summary>
    /// 获取当前配置（供其他组件使用）
    /// </summary>
    public static PracticeActionConfig GetCurrentConfig()
    {
        var initializer = FindObjectOfType<PracticeSceneInitializer>();
        return initializer != null ? initializer.currentConfig : null;
    }
}
